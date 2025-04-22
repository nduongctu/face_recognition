import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QDRANT_HOST = "http://qdrant:6333"
COLLECTION_NAME = "face_recognition"

VECTOR_SIZE = 128
DISTANCE_METRIC = "cosine"

model_name = 'ArcFace'
# model_name = 'FaceNet'
