import pytz
import asyncio
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from functools import lru_cache
from qdrant_client import QdrantClient
from typing import Dict, List, Any, Optional, Union
from face_recognize.extract_vector import extract_vector
from face_recognize.upload_r2 import upload_face_crop_to_r2
from face_recognize.save_to_postgres import save_to_postgres
from face_recognize.preprocess import crop_face, normalize_bbox
from face_recognize.config import QDRANT_HOST, COLLECTION_NAME, threshold


@lru_cache(maxsize=1)
def get_vn_timezone():
    return pytz.timezone("Asia/Ho_Chi_Minh")


client = QdrantClient(QDRANT_HOST)


async def process_single_face(
        cam_id: str,
        img_np: np.ndarray,
        face: Dict[str, Any],
        frame_idx: int,
        height: int,
        width: int,
        score_threshold: float
) -> Optional[Dict[str, Any]]:
    """Xử lý một khuôn mặt riêng lẻ để phát hiện và nhận dạng"""

    bbox = face.get("bbox")
    user_id = face.get("user_id")
    confidence = face.get("confidence", 0.0)

    if not bbox:
        return None

    face_crop = await crop_face(img_np, bbox)
    if face_crop is None:
        return None

    normalized_bbox = normalize_bbox(bbox, width, height)

    if user_id:
        object_name = upload_face_crop_to_r2(face_crop, user_id, frame_idx)
        vn_time = datetime.now(get_vn_timezone()).replace(tzinfo=None)
        await save_to_postgres(user_id, normalized_bbox, confidence, cam_id, frame_idx, vn_time, object_name)
        return {"user_id": user_id, "bbox": normalized_bbox}

    if face.get("reported", False) and face.get("frame_count", 0) >= 8:
        return {"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"}

    # Kiểm tra xem có vector nhúng để tìm kiếm không
    embedding = face.get("embedding")
    if embedding is None:
        return {"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"}

    try:
        search_result = client.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=1,
            group_by="user_id",
            group_size=1,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Không tìm thấy kết quả khớp
        if not search_result.groups or not search_result.groups[0].hits:
            return {"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"}

        # Tìm thấy kết quả khớp
        hit = search_result.groups[0].hits[0]
        user_id = hit.payload.get('user_id') if hit.payload else None
        confidence = hit.score

        if user_id:
            face["user_id"] = user_id
            face["is_identified"] = True

            object_name = upload_face_crop_to_r2(face_crop, user_id, frame_idx)
            vn_time = datetime.now(get_vn_timezone()).replace(tzinfo=None)
            await save_to_postgres(user_id, normalized_bbox, confidence, cam_id, frame_idx, vn_time, object_name)
            return {"user_id": user_id, "bbox": normalized_bbox}
        else:
            return {"bbox": normalized_bbox, "detail": "Không tìm thấy người phù hợp"}

    except Exception as e:
        return {"bbox": normalized_bbox, "detail": f"Lỗi kết nối với Qdrant: {str(e)}"}


async def face_recognize(
        cam_id: str,
        img_np: np.ndarray,
        frame_idx: int,
        score_threshold: float = threshold
) -> Union[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    # Trích xuất vector khuôn mặt
    face_list = extract_vector(img_np)

    # Xử lý các trường hợp lỗi
    if isinstance(face_list, str):
        return {"detail": face_list}
    if not face_list:
        return {"detail": "Không tìm thấy khuôn mặt"}

    height, width, _ = img_np.shape

    results = []
    for face in face_list:
        result = await process_single_face(cam_id, img_np, face, frame_idx, height, width, score_threshold)
        if result is not None:
            results.append(result)

    return results
