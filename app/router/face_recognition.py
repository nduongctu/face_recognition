import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from app.service.face_recognize import face_recognize
from app.service.save_face_to_qdrant import save_face_to_qdrant
from fastapi import APIRouter, HTTPException, File, UploadFile, Form

router = APIRouter()


@router.post("/save_qdrant", summary="Lưu vector khuôn mặt vào Qdrant")
async def save_qdrant(image: UploadFile = File(...), user_id: str = Form(...)):
    try:
        # Đọc file ảnh
        img_bytes = await image.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)

        # Lưu ảnh vào Qdrant
        result = await save_face_to_qdrant(img_np, user_id=user_id)
        return JSONResponse({"success": True, "message": "Lưu thành công vào Qdrant", "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


@router.post("/recognize", summary="Nhận diện khuôn mặt từ Qdrant")
async def recognize(image: UploadFile = File(...)):
    try:
        # Đọc file ảnh
        img_bytes = await image.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)

        # Kiểm tra kích thước của img_np
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Ảnh đầu vào không hợp lệ!")

        # Nhận diện khuôn mặt từ Qdrant
        result = await face_recognize(img_np)
        return JSONResponse({"success": True, "message": "Nhận diện thành công", "result": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")
