import numpy as np
from qdrant_client import QdrantClient
from app.config.settings import QDRANT_HOST
from qdrant_client.models import SearchRequest
from app.service.extract_vector import extract_vector
from app.config.settings import COLLECTION_NAME, threshold


async def face_recognize(img_np, top_k=1, score_threshold=threshold):
    """
    Nhận diện khuôn mặt: trả về user_id trong Qdrant của vector gần nhất với khuôn mặt trên ảnh (nếu tìm được).
    Nếu không tìm thấy khuôn mặt hoặc không có kết quả phù hợp sẽ trả về thông báo phù hợp.
    """
    # Trích xuất vector từ ảnh
    query_vector = await extract_vector(img_np)

    # Kiểm tra nếu không trích xuất được vector
    if isinstance(query_vector, str):
        return query_vector
    if query_vector is None or len(query_vector) == 0:
        return "Không tìm thấy khuôn mặt"

    # Kết nối với Qdrant
    client = QdrantClient(QDRANT_HOST)

    try:
        # Tìm kiếm trong Qdrant với vector trích xuất
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )
    except Exception as e:
        return f"Lỗi kết nối với Qdrant: {str(e)}"

    # Kiểm tra kết quả tìm kiếm
    if not search_result or len(search_result) == 0:
        return "Không tìm thấy người phù hợp"

    # Lấy user_id từ payload
    user_id = search_result[0].payload.get('user_id') if search_result[0].payload else None

    # Trả về kết quả
    if user_id:
        return user_id
    else:
        return "Không tìm thấy người phù hợp"
