import cv2
import numpy as np


def align_face(img, kps):
    left_eye = kps[0]  # Mắt trái (x, y)
    right_eye = kps[1]  # Mắt phải (x, y)

    # Tính toán góc xoay giữa mắt trái và mắt phải
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

    # Tính toán điểm giữa hai mắt
    center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # Tạo ma trận xoay với điểm giữa hai mắt làm trung tâm
    M = cv2.getRotationMatrix2D(center, angle, 1)  # Sử dụng scale=1, không thay đổi kích thước

    aligned_face = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned_face
