from retinaface import RetinaFace
import numpy as np
from PIL import Image


def detect_and_crop_face_np(img_np):
    if img_np is None or not isinstance(img_np, np.ndarray):
        return None

    results = RetinaFace.extract_faces(img_np, align=True)

    if isinstance(results, dict) and len(results) > 0:
        face_key = list(results.keys())[0]
        facial_area = results[face_key]['facial_area']
        x1, y1, x2, y2 = facial_area
        face_crop = img_np[y1:y2, x1:x2]

        face_pil = Image.fromarray(face_crop[..., ::-1])
        return face_pil

    else:
        return None
