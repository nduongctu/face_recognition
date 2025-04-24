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
    :return: ID của vector đã lưu (uuid)
    """
    if img_np is None:
        raise ValueError("Thiếu ảnh đầu vào!")

    if not user_id:
        raise ValueError("Thiếu user_id!")

    # Trích xuất vector từ ảnh
    vector = await extract_vector(img_np)

    # Kiểm tra kỹ kiểu dữ liệu vector, chỉ nhận numpy.ndarray hoặc list số thực
    if isinstance(vector, str):
        raise ValueError(f"Lỗi khi trích xuất vector khuôn mặt: {vector}")
    if vector is None or (hasattr(vector, '__len__') and len(vector) == 0):
        raise ValueError("Không trích xuất được vector khuôn mặt!")
    if not (isinstance(vector, (np.ndarray, list))):
        raise ValueError("Kết quả trích xuất vector không hợp lệ!")

    metadata = {"user_id": user_id}
    point_id = str(uuid.uuid4())

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": point_id,
                    "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
                    "payload": metadata
                }
            ]
        )
    except Exception as e:
        raise ValueError(f"Lỗi khi lưu vào Qdrant: {str(e)}")

    return point_id
