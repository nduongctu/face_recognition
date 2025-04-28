import os
from fastapi import FastAPI
from app.utils.postgres import *
from app.router import face_recognition
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Kết nối tới PostgreSQL khi ứng dụng khởi động
@app.on_event("startup")
async def startup_db():
    app.db = await connect_to_db()  # Kết nối PostgreSQL


# Ngắt kết nối khi ứng dụng tắt
@app.on_event("shutdown")
async def shutdown_db():
    await close_db_connection(app.db)  # Đóng kết nối PostgreSQL


app.include_router(face_recognition.router, prefix="/face", tags=["Face Recognition"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
