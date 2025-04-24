import numpy as np
from app.service.insightface_wrapper import model_app


async def extract_vector(img_np):
    if img_np is None or not isinstance(img_np, np.ndarray):
        return "Ảnh đầu vào không hợp lệ"

    # Bước 1: Detect faces
    faces = model_app.get(img_np)
    if not faces or len(faces) == 0:
        return "Không tìm thấy khuôn mặt"

    embedding = faces[0].normed_embedding
    if embedding is not None:
        return embedding
    else:
        return "Không tìm thấy embedding của khuôn mặt"
