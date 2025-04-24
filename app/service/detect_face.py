import numpy as np
from deepface import DeepFace
from app.config import settings


def detect_and_crop_face_np(img_np):
    try:
        detection = DeepFace.extract_faces(
            img_np,
            detector_backend=settings.model_detect,
            enforce_detection=True,
            align=True
        )
        if detection and len(detection) > 0:
            face_info = detection[0]
            face_crop = face_info["face"]
            # Kiểm tra kích thước crop và kiểu dữ liệu
            if (
                    not isinstance(face_crop, np.ndarray) or
                    face_crop.shape[0] < 10 or face_crop.shape[1] < 10
            ):
                return "Khuôn mặt được crop quá nhỏ hoặc không hợp lệ"
            if face_crop.dtype != np.uint8:
                face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)
            return face_crop
        else:
            return "Không tìm thấy khuôn mặt"
    except Exception as e:
        return f"Lỗi DeepFace detect: {str(e)}"
