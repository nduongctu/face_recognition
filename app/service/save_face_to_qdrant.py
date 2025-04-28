import uuid
import numpy as np
from app.utils.qdrant import client
from app.config.settings import COLLECTION_NAME
from app.service.extract_vector_save_qdrant import extract_vector


async def save_face_to_qdrant(img_np, user_id, pose):
    """
    Nhận ảnh đầu vào (np.array), trích xuất vector và lưu vào Qdrant kèm theo user_id và pose.
    Thông báo thành công/thất bại.
    """
    if img_np is None:
        raise ValueError("Thiếu ảnh đầu vào!")

    if not user_id:
        raise ValueError("Thiếu user_id!")

    if not pose:
        raise ValueError("Thiếu pose!")

    item = extract_vector(img_np)
    if isinstance(item, str):
        raise ValueError(f"Lỗi khi trích xuất vector khuôn mặt: {item}")
    if item is None or item.get("embedding") is None:
        raise ValueError("Không trích xuất được vector khuôn mặt nào!")

    embedding = item["embedding"]
    if not isinstance(embedding, (np.ndarray, list)):
        raise ValueError("Embedding không hợp lệ!")

    point = {
        "id": str(uuid.uuid4()),
        "vector": embedding if isinstance(embedding, list) else embedding.tolist(),
        "payload": {
            "user_id": user_id,
            "pose": pose
        }
    }

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi lưu vào Qdrant: {str(e)}")

    return {"message": "Lưu thành công"}
