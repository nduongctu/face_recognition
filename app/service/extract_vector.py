from deepface import DeepFace
import numpy as np
from PIL import Image
from app.config import settings


def extract_vector(face_img, model_name=settings.model_name, enforce_detection=False):
    if face_img is None:
        return None

    img_np = np.array(face_img)

    embedding = DeepFace.represent(
        img_path=img_np,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend='skip'
    )

    if isinstance(embedding, dict):
        return np.array(embedding['embedding'])
    elif isinstance(embedding, list) and len(embedding) > 0:
        return np.array(embedding[0]['embedding'])
    else:
        return None
