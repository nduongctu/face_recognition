import cv2
import asyncio
import numpy as np
from app.service.upload_r2 import *
from qdrant_client import QdrantClient
from app.service.save_to_postgres import *
from app.config.settings import QDRANT_HOST
from app.service.extract_vector import extract_vector
from app.utils.preprocess import crop_face, normalize_bbox
from app.config.settings import COLLECTION_NAME, threshold


async def face_recognize(app, img_np, frame_idx, top_k=1, score_threshold=threshold):
    """ Nhận diện nhiều khuôn mặt: trả về danh sách user_id hoặc thông báo cho mỗi khuôn mặt trong ảnh (kèm bbox).
    Chỉ lưu vào PostgreSQL và R2 nếu nhận dạng được người dùng. """
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
        user_id = face.get("user_id")
        confidence = face.get("confidence", 0.0)

        if bbox:
            face_crop = await crop_face(img_np, bbox)
            if face_crop is None:
                continue

            normalized_bbox = normalize_bbox(bbox, width, height)

        else:
            continue

        if user_id:
            results.append({"user_id": user_id, "bbox": normalized_bbox})

            object_name = upload_face_crop_to_r2(face_crop, user_id, frame_idx)
            await save_to_postgres(app, user_id, normalized_bbox, confidence, frame_idx, object_name)
            continue

        if face.get("reported", False) and face.get("frame_count", 0) >= 8:
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

                object_name = upload_face_crop_to_r2(face_crop, user_id, frame_idx)
                await save_to_postgres(app, user_id, normalized_bbox, confidence, frame_idx, object_name)
            else:
                results.append({"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"})

        except Exception as e:
            results.append({
                "bbox": normalized_bbox,
                "detail": f"Lỗi kết nối với Qdrant: {str(e)}"
            })

    return results
