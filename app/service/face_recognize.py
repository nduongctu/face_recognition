import numpy as np
from qdrant_client import QdrantClient
from app.config.settings import QDRANT_HOST
from qdrant_client.models import SearchRequest
from app.service.extract_vector import extract_vector
from app.config.settings import COLLECTION_NAME, threshold


async def face_recognize(img_np, top_k=1, score_threshold=threshold):
    """
    Nhận diện nhiều khuôn mặt: trả về list user_id hoặc thông báo cho mỗi khuôn mặt trong ảnh (kèm bbox).
    """
    face_list = extract_vector(img_np)

    if isinstance(face_list, str):
        return {"detail": face_list}
    if face_list is None or len(face_list) == 0:
        return {"detail": "Không tìm thấy khuôn mặt"}

    client = QdrantClient(QDRANT_HOST)
    results = []

    for face in face_list:
        bbox = face.get("bbox")

        if face.get("user_id") is not None:
            # Nếu đã có user_id, trả về kết quả ngay không cần truy vấn Qdrant
            results.append({"user_id": face.get("user_id"), "bbox": bbox})
            continue

        # Kiểm tra nếu khuôn mặt đã báo cáo là không tìm thấy người phù hợp
        if face.get("reported", False) and face.get("frame_count", 0) >= 30:
            results.append({"bbox": bbox, "detail": "Không tìm thấy người phù hợp"})
            continue

        embedding = face.get("embedding")
        if embedding is None:
            results.append({"bbox": bbox, "detail": "Không tìm thấy người phù hợp"})
            continue

        # Chỉ truy vấn Qdrant cho khuôn mặt chưa nhận diện hoặc chưa báo cáo
        try:
            search_result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold,
            )

            if not search_result or len(search_result) == 0:
                if face.get("frame_count", 0) >= 30:
                    results.append({"bbox": bbox, "detail": "Không tìm thấy người phù hợp"})
                else:
                    results.append({"bbox": bbox, "detail": f"Đang xử lý..."})
            else:
                user_id = search_result[0].payload.get('user_id') if search_result[0].payload else None
                if user_id:
                    face["user_id"] = user_id
                    face["is_identified"] = True

                    results.append({"user_id": user_id, "bbox": bbox})
                else:
                    results.append({"bbox": bbox, "detail": "Không tìm thấy người phù hợp"})

        except Exception as e:
            results.append({
                "bbox": bbox,
                "detail": f"Lỗi kết nối với Qdrant: {str(e)}"
            })

    return results
