from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
from face_recognize.config import QDRANT_HOST, COLLECTION_NAME, VECTOR_SIZE

distance_metric = Distance.COSINE

client = QdrantClient(QDRANT_HOST)

collections = client.get_collections().collections
existing_collection_names = [col.name for col in collections]

if COLLECTION_NAME not in existing_collection_names:
    print(f"Tạo collection '{COLLECTION_NAME}' cho vector khuôn mặt...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=distance_metric
        )
    )

    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=200
        )
    )
    print(f"Collection '{COLLECTION_NAME}' đã tạo.")
else:
    print(f"Collection '{COLLECTION_NAME}' đã tồn tại để lưu vector khuôn mặt.")
