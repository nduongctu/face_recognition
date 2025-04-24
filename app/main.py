from fastapi import FastAPI
from deepface import DeepFace
from app.config import settings
from app.router import face_recognition
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    print(f"Warming up DeepFace model ({settings.model_name})...")
    _ = DeepFace.build_model(settings.model_name)
    print("DeepFace model is warmed up and ready!")


app.include_router(face_recognition.router, prefix="/face", tags=["Face Recognition"])

if __name__ == "__main__":
    import uvicorn

    print(f"Warming up DeepFace model ({settings.model_name})...")
    _ = DeepFace.build_model(settings.model_name)
    print("DeepFace model is warmed up and ready!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
