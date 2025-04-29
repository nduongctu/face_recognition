import cv2
import asyncio
import numpy as np


def resize_with_padding(img, target_size=(640, 640), bg_color=(255, 255, 255), return_info=False):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Tạo ảnh nền trắng cùng kích thước target_size
    new_img = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # Tính vị trí dán ảnh đã resize lên nền trắng
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    new_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    if return_info:
        # scale: tỉ lệ resize, (x_offset, y_offset): padding trái-trên
        return new_img, (scale, (x_offset, y_offset))
    else:
        return new_img


async def crop_face(img_np, bbox):
    try:
        height, width, _ = img_np.shape
        x_min, y_min, x_max, y_max = bbox

        x_min = max(0, min(x_min, width))
        y_min = max(0, min(y_min, height))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))

        face_crop = img_np[y_min:y_max, x_min:x_max]
        return face_crop
    except Exception as e:
        print(f"Lỗi khi cắt khuôn mặt: {e}")
        return None


# Chuẩn hóa về [0,1]
def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    normalized_bbox = [
        x_min / img_width,
        y_min / img_height,
        x_max / img_width,
        y_max / img_height
    ]
    return normalized_bbox
