import cv2
import numpy as np
from insightface.app.common import Face
from app.utils.model import det_model, rec_model


async def warmup_models():
    try:
        # Tạo ảnh giả
        img_resize = np.ones((640, 640, 3), dtype=np.uint8) * 128

        # Warm-up mô hình phát hiện
        bboxes, kpss = det_model.detect(img_resize, max_num=0, metric='default')

        # Tạo dữ liệu giả
        bbox_resize_expanded = np.array([100, 100, 200, 200], dtype=np.float32)
        kps = np.array([
            [130, 130],
            [170, 130],
            [150, 150],
            [130, 170],
            [170, 170]
        ], dtype=np.float32)
        det_score = 0.99

        x1, y1, x2, y2 = [int(i) for i in bbox_resize_expanded[:4]]
        aligned_face_img = img_resize[y1:y2, x1:x2]

        if aligned_face_img.size == 0:
            aligned_face_img = np.ones((112, 112, 3), dtype=np.uint8) * 128

        # Tạo đối tượng Face
        face = Face(bbox=bbox_resize_expanded, kps=kps, det_score=det_score)

        # Warm-up mô hình nhận diện
        rec_model.get(aligned_face_img, face)

        print("Warm-up thành công!")

    except Exception as e:
        print(f"Lỗi warm-up: {e}")
        import traceback
        traceback.print_exc()
