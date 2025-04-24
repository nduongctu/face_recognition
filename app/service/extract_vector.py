import numpy as np
from app.service.fas_model import is_real_face
from app.service.insightface_wrapper import model_app


async def extract_vector(img_np):
    if img_np is None or not isinstance(img_np, np.ndarray):
        return "Ảnh đầu vào không hợp lệ"

    # Detect face
    faces = model_app.get(img_np)
    if not faces or len(faces) == 0:
        return "Không tìm thấy khuôn mặt"

    # Chọn mặt đầu tiên và crop vùng mặt nếu cần trích xuất embedding
    face = faces[0]
    x1, y1, x2, y2 = [int(i) for i in face.bbox]
    x1 = max(x1 - 5, 0)
    y1 = max(y1 - 5, 0)
    x2 = min(x2 + 5, img_np.shape[1])
    y2 = min(y2 + 5, img_np.shape[0])
    face_crop = img_np[y1:y2, x1:x2]

    if not is_real_face(face_crop):
        return "Phát hiện khuôn mặt giả mạo"

    # Trích xuất embedding của khuôn mặt
    embedding = face.normed_embedding
    if embedding is not None:
        return embedding
    else:
        return "Không tìm thấy embedding của khuôn mặt"
