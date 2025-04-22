from app.utils.qdrant import client
from app.config.settings import COLLECTION_NAME
import uuid


def save_face_to_qdrant(face_vector, metadata: dict):
    """
    Lưu vector khuôn mặt và metadata vào Qdrant.
    :param face_vector: np.array hoặc list số (vector embedding)
    :param metadata: dict (chứa thông tin bổ sung: tên, v.v.)
    :return: id của point đã lưu (uuid)
    """
    if face_vector is None or metadata is None:
        raise ValueError("Vector hoặc metadata không hợp lệ!")

    # Tạo id duy nhất cho mỗi khuôn mặt
    point_id = str(uuid.uuid4())

    # Qdrant point format
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {
                "id": point_id,
                "vector": face_vector.tolist() if hasattr(face_vector, "tolist") else face_vector,
                "payload": metadata
            }
        ]
    )
    return point_id
