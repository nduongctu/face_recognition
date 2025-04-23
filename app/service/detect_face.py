import numpy as np
from retinaface import RetinaFace


def detect_and_crop_face_np(img_np):
    """
    Nhận ảnh numpy array, detect & crop khuôn mặt đầu tiên, trả về ảnh khuôn mặt đã crop (numpy array RGB).
    Nếu không phát hiện khuôn mặt, trả về None.
    """
    if img_np is None or not isinstance(img_np, np.ndarray):
        return 'Không tìm thấy khuôn mặt'

    try:
        results = RetinaFace.extract_faces(img_np, align=True)
    except Exception as e:
        return f"Lỗi trong quá trình phát hiện khuôn mặt: {str(e)}"

    if isinstance(results, list) and len(results) > 0:
        face_crop = results[0]
        return face_crop
    else:
        return None
