import cv2
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
