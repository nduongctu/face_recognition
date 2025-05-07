import os
from fastapi import FastAPI
from app.utils.postgres import *
from app.router import face_recognition, report
from fastapi.middleware.cors import CORSMiddleware
from app.utils.warmup_model import warmup_models

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    await init_db_pool()
    await warmup_models()


@app.on_event("shutdown")
async def on_shutdown():
    await close_db_pool()


app.include_router(face_recognition.router, prefix="/face", tags=["Face Recognition"])
app.include_router(report.router, prefix="/report", tags=["Report"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
