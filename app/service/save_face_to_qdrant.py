import uuid
import numpy as np
from app.utils.qdrant import client
from app.config.settings import COLLECTION_NAME
from app.service.extract_vector import extract_vector


async def save_face_to_qdrant(img_np, user_id):
    """
    Nhận ảnh đầu vào (np.array), trích xuất vector và lưu vào Qdrant kèm theo user_id.
    :param img_np: Ảnh đầu vào dạng numpy array (RGB)
    :param user_id: ID người dùng cần lưu
    :return: Danh sách ID của các vector đã lưu (uuid)
    """
    if img_np is None:
        raise ValueError("Thiếu ảnh đầu vào!")

    if not user_id:
        raise ValueError("Thiếu user_id!")

    # Trích xuất list vector từ ảnh
    face_list = extract_vector(img_np)

    if isinstance(face_list, str):
        raise ValueError(f"Lỗi khi trích xuất vector khuôn mặt: {face_list}")
    if face_list is None or len(face_list) == 0:
        raise ValueError("Không trích xuất được vector khuôn mặt nào!")

    ids = []
    points = []
    for item in face_list:
        embedding = item.get("embedding")
        if embedding is None or not isinstance(embedding, (np.ndarray, list)):
            continue
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": embedding if isinstance(embedding, list) else embedding.tolist(),
            "payload": {"user_id": user_id}
        })
        ids.append(point_id)

    if not points:
        raise ValueError("Không có vector embedding hợp lệ để lưu!")

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi lưu vào Qdrant: {str(e)}")

    return ids
