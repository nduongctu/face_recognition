QDRANT_HOST = "http://qdrant:6333"
COLLECTION_NAME = "face_recognition"

VECTOR_SIZE = 512
model_name = 'ArcFace'
# VECTOR_SIZE = 128
# model_name = 'FaceNet'
DISTANCE_METRIC = "cosine"

threshold = 0.75
shape = (112, 112, 3)