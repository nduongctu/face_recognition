import cv2
import json
import asyncio
import numpy as np
from datetime import datetime
from app.utils.postgres import *
from qdrant_client import QdrantClient
from app.config.settings import QDRANT_HOST
from qdrant_client.models import SearchRequest
from app.service.extract_vector import extract_vector
from app.config.settings import COLLECTION_NAME, threshold


async def save_to_postgres(app, user_id, bbox, confidence, frame_idx):
    try:
        async with app.db.acquire() as conn:
            async with conn.transaction():
                # Chuyển bbox thành định dạng JSON
                bbox_json = json.dumps(bbox)

                # Thêm kết quả nhận dạng vào bảng recognize_results
                result_id = await conn.fetchval('''
                                                INSERT INTO recognize_results(user_id, bbox, confidence)
                                                VALUES ($1, $2, $3) RETURNING result_id
                                                ''', user_id, bbox_json, confidence)

                # Kiểm tra nếu result_id có giá trị (không bị lọc bởi trigger)
                if result_id:
                    # Lưu frame_idx vào bảng face_frames
                    await conn.execute('''
                                       INSERT INTO face_frames(result_id, idx_frame)
                                       VALUES ($1, $2)
                                       ''', result_id, frame_idx)

                    return result_id
            return None
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào PostgreSQL: {e}")
        return None


async def crop_face(img_np, bbox):
    """
    Cắt khuôn mặt từ ảnh gốc dựa trên bbox
    """
    try:
        height, width, _ = img_np.shape
        x_min, y_min, x_max, y_max = bbox

        # Chuyển từ tọa độ chuẩn hóa về pixel
        x_min_px = int(x_min * width)
        y_min_px = int(y_min * height)
        x_max_px = int(x_max * width)
        y_max_px = int(y_max * height)

        # Đảm bảo tọa độ nằm trong giới hạn ảnh
        x_min_px = max(0, x_min_px)
        y_min_px = max(0, y_min_px)
        x_max_px = min(width, x_max_px)
        y_max_px = min(height, y_max_px)

        # Cắt ảnh
        face_crop = img_np[y_min_px:y_max_px, x_min_px:x_max_px]
        return face_crop
    except Exception as e:
        print(f"Lỗi khi cắt khuôn mặt: {e}")
        return None


async def face_recognize(app, img_np, frame_idx, top_k=1, score_threshold=threshold):
    """
    Nhận diện nhiều khuôn mặt: trả về list user_id hoặc thông báo cho mỗi khuôn mặt trong ảnh (kèm bbox).
    Chỉ lưu vào PostgreSQL nếu nhận dạng được người dùng.
    """
    face_list = extract_vector(img_np)

    if isinstance(face_list, str):
        return {"detail": face_list}
    if not face_list:
        return {"detail": "Không tìm thấy khuôn mặt"}

    client = QdrantClient(QDRANT_HOST)
    results = []
    height, width, _ = img_np.shape

    for face in face_list:
        bbox = face.get("bbox")
        normalized_bbox = normalize_bbox(bbox, width, height)

        user_id = face.get("user_id")
        confidence = face.get("confidence", 0.0)

        # Nếu đã có user_id từ pipeline trước (Deep SORT hoặc gán tay)
        if user_id:
            results.append({"user_id": user_id, "bbox": normalized_bbox})
            await save_to_postgres(app, user_id, normalized_bbox, confidence, frame_idx)
            continue

        # Nếu đã report quá số lần cho phép
        if face.get("reported", False) and face.get("frame_count", 0) >= 10:
            results.append({"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"})
            continue

        embedding = face.get("embedding")
        if embedding is None:
            results.append({"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"})
            continue

        try:
            search_result = client.search_groups(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k,
                group_by="user_id",
                group_size=1,
                with_payload=True,
                score_threshold=score_threshold,
            )

            if not search_result.groups or not search_result.groups[0].hits:
                results.append({"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"})
                continue

            hit = search_result.groups[0].hits[0]
            user_id = hit.payload.get('user_id') if hit.payload else None
            confidence = hit.score

            if user_id:
                face["user_id"] = user_id
                face["is_identified"] = True
                results.append({"user_id": user_id, "bbox": normalized_bbox})
                await save_to_postgres(app, user_id, normalized_bbox, confidence, frame_idx)
            else:
                results.append({"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"})

        except Exception as e:
            results.append({
                "bbox": normalized_bbox,
                "detail": f"Lỗi kết nối với Qdrant: {str(e)}"
            })

    return results


def normalize_bbox(bbox, img_width, img_height):
    """
    Chuẩn hóa bounding box từ (x_min, y_min, x_max, y_max) thành [0, 1].
    """
    x_min, y_min, x_max, y_max = bbox
    normalized_bbox = [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height
    ]
    return normalized_bbox
