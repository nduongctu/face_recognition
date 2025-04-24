import onnxruntime as ort
import numpy as np
import cv2

# Khởi tạo ONNX session
session = ort.InferenceSession(
    "app/weights/AntiSpoofing_bin_1.5_128.onnx",
    providers=["CUDAExecutionProvider"]
)


def preprocess(img, size=128):
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img


def is_real_face(face_crop, threshold=0.5):
    input_tensor = preprocess(face_crop)

    pred = session.run(None, {"input": input_tensor})[0][0]
    score = float(pred[1])
    return score > threshold
