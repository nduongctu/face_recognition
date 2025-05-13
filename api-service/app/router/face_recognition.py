import uuid
import json
import redis
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from face_recognize.face_recognize import face_recognize
from service.save_face_to_qdrant import save_face_to_qdrant
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request

router = APIRouter()


@router.post("/save_qdrant", summary="Lưu vector khuôn mặt vào Qdrant")
async def save_qdrant(
        frontal: UploadFile = File(...),
        up: UploadFile = File(...),
        left: UploadFile = File(...),
        right: UploadFile = File(...),
        user_id: str = Form(...),
):
    try:
        # Kiểm tra xem tất cả ảnh đã được tải lên hay chưa
        if not all([frontal, up, left, right]):
            raise HTTPException(status_code=400, detail="Cần đủ 4 ảnh (frontal, up, left, right)!")

        # Tạo dictionary cho các ảnh và kiểm tra định dạng
        upload_files = {"frontal": frontal, "up": up, "left": left, "right": right}
        images_np = {}

        # Đọc và xử lý các tệp ảnh
        for pose, file in upload_files.items():
            try:
                img_bytes = await file.read()
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images_np[pose] = np.array(img)

                # Kiểm tra ảnh có đúng định dạng
                if images_np[pose].ndim != 3 or images_np[pose].shape[2] != 3:
                    raise ValueError(f"Ảnh {pose} không đúng định dạng RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Lỗi xử lý ảnh {pose}: {str(e)}")

        # Lưu mỗi ảnh vào Qdrant với user_id và pose
        results = {}
        for pose, img_np in images_np.items():
            results[pose] = await save_face_to_qdrant(img_np, user_id=user_id, pose=pose)

        return JSONResponse({
            "success": True,
            "message": "Lưu thành công vào Qdrant",
            "results": results
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


@router.post("/recognize")
async def recognize_face(
        file: UploadFile = File(...),  # Nhận tệp ảnh
        cam_id: str = Form(...),  # Nhận cam_id (UUID4)
        frame_idx: int = Form(...),  # Nhận frame_idx dưới dạng integer
        request: Request = None
):
    try:
        # Kiểm tra và chuyển đổi ảnh từ file
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)

        # Kiểm tra kích thước của img_np
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Ảnh đầu vào không hợp lệ!")

        # Tiến hành nhận diện và trả kết quả
        result = await face_recognize(request.app, cam_id, img_np, frame_idx)

        return JSONResponse({"success": True, "message": "Nhận diện thành công", "result": result})
    except Exception as e:
        return {"detail": f"Lỗi khi xử lý ảnh: {str(e)}"}


@router.post("/process_video")
async def process_video(url_stream: str):
    cam_id = str(uuid.uuid4())
    redis_host = "redis"

    try:
        redis_client = redis.Redis(host=redis_host, port=6379, db=0)
        redis_client.ping()
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Redis: {e}")

    data = {
        "stream_url": url_stream,
        "cam_id": cam_id
    }

    try:
        redis_client.set(f"video_stream:{cam_id}", json.dumps(data))
        print(f"Stream URL {url_stream} pushed to Redis with cam_id {cam_id}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to push stream info to Redis: {e}")

    return {"message": "Video processing started successfully", "cam_id": cam_id}


@router.post("/stop_video")
async def stop_video():
    redis_host = "redis"

    try:
        redis_client = redis.Redis(host=redis_host, port=6379, db=0)
        redis_client.ping()
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Redis: {e}")

    keys = redis_client.keys('video_stream:*')
    for key in keys:
        redis_client.delete(key)
        print(f"Deleted key {key.decode()}")

    return {"message": "Video streams stopped successfully"}
