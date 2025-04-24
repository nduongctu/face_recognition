import numpy as np
import cv2
from fastapi import Request


def detect_and_crop_face_np(img_np: np.ndarray, request: Request):
    model = request.app.state.insight_model
    faces = model.get(img_np)
    if not faces:
        return "Không tìm thấy khuôn mặt"
    # Chọn khuôn mặt có diện tích lớn nhất
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(img_np.shape[1], x2)
    y2 = min(img_np.shape[0], y2)
    face_crop = img_np[y1:y2, x1:x2]
    return face_crop
