import numpy as np
from PIL import Image
from fastapi import FastAPI
from deepface import DeepFace
from app.router import face_recognition
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def warmup_deepface_model():
    print("Warming up ArcFace model...")
    dummy_image = np.zeros(settings.shape, dtype=np.uint8)
    dummy_path = "/tmp/deepface_dummy.jpg"
    Image.fromarray(dummy_image).save(dummy_path)

    DeepFace.represent(
        img_path=dummy_path,
        model_name=settings.model_name,
        enforce_detection=False,
        detector_backend="skip"
    )
    print("ArcFace model is warmed up and ready!")


@app.on_event("startup")
def on_startup():
    warmup_deepface_model()


app.include_router(face_recognition.router, prefix="/face", tags=["Face Recognition"])

if __name__ == "__main__":
    import uvicorn

    warmup_deepface_model()

    uvicorn.run(app, host="0.0.0.0", port=8000)
