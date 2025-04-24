from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import face_recognition
from app.service.insightface_wrapper import model_app

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
    _ = model_app.models
    print("InsightFace model is warmed up and ready!")


app.include_router(face_recognition.router, prefix="/face", tags=["Face Recognition"])

if __name__ == "__main__":
    import uvicorn
    _ = model_app.models
    print("InsightFace model is warmed up and ready!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
