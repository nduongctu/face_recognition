import tempfile
import numpy as np
from PIL import Image
from deepface import DeepFace
from app.config import settings
from app.service.resize_img import resize_image
from app.service.detect_face import detect_and_crop_face_np


async def extract_vector(img_np, model_name=settings.model_name):
    """
    Nhận numpy array, detect & crop khuôn mặt bằng detect_and_crop_face_np, resize ảnh rồi extract vector embedding.
    Nếu không tìm thấy khuôn mặt, trả về thông báo.
    """
    if img_np is None:
        return None

    # Phát hiện và crop khuôn mặt
    face_crop = detect_and_crop_face_np(img_np)
    if isinstance(face_crop, str):
        return face_crop
    if face_crop is None:
        return "Không tìm thấy khuôn mặt"

    try:
        # Resize ảnh crop về kích thước phù hợp (112, 112)
        resized_face_crop = resize_image(face_crop, target_size=(112, 112))

        # Lưu ảnh crop đã resize vào file tạm thời
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            Image.fromarray(resized_face_crop).save(temp_path)

        # Trích xuất vector từ ảnh đã crop và resize
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend='skip'
        )

        # Kiểm tra kết quả embedding và trả về
        if isinstance(embedding, dict) and 'embedding' in embedding:
            return np.array(embedding['embedding'])  # Đảm bảo là numpy array
        elif isinstance(embedding, list) and len(embedding) > 0 and 'embedding' in embedding[0]:
            return np.array(embedding[0]['embedding'])  # Đảm bảo là numpy array
        else:
            return None
    except Exception as e:
        return f"Lỗi trong quá trình trích xuất vector: {str(e)}"
