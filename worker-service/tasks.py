from celery_app import celery_app
from face_recognize.face_recognize import face_recognize
from PIL import Image
import numpy as np
from io import BytesIO
import asyncio


@celery_app.task
def recognize_face_task_non_stream(cam_id, frame_bytes, frame_idx):
    try:
        print(f"Starting task: {frame_idx}")  # Logging
        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        img_np = np.array(image)

        if img_np.ndim != 3 or img_np.shape[2] != 3:
            return {"error": "Invalid image shape"}

        # Gọi face_recognize như một hàm đồng bộ
        result = face_recognize(cam_id, img_np, frame_idx)
        print(f"Task completed: {frame_idx}")  # Logging task completion
        return result

    except Exception as e:
        return {"error": str(e)}


@celery_app.task
def recognize_face_task_stream(cam_id, frame_bytes, frame_idx):
    try:
        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        img_np = np.array(image)

        if img_np.ndim != 3 or img_np.shape[2] != 3:
            return {"error": "Invalid image shape"}

        result = face_recognize(cam_id, img_np, frame_idx)
        return result

    except Exception as e:
        return {"error": str(e)}
