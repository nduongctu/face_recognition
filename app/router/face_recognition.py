from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from app.service.detect_face import detect_and_crop_face_np
from app.service.extract_vector import extract_vector
from app.config.settings import COLLECTION_NAME
from app.utils.qdrant import client

router = APIRouter()


@router.post("/upload", summary="Upload ảnh vào memory, convert sang numpy array")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ nhận file ảnh!")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        return JSONResponse({"message": "Upload thành công! Đã convert sang numpy array."})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")


@router.post("/recognize", summary="Nhận diện khuôn mặt user")
async def face_recognize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ nhận file ảnh!")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Detect & crop face
        face_img = detect_and_crop_face_np(img_np)
        if face_img is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy khuôn mặt trong ảnh.")

        # Extract embedding vector
        face_vec = extract_vector(face_img)
        if face_vec is None:
            raise HTTPException(status_code=500, detail="Không thể trích xuất vector khuôn mặt.")

        # Tìm vector gần nhất trong Qdrant
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=face_vec.tolist(),
            limit=1
        )

        if not hits:
            return JSONResponse({"message": "Không tìm thấy user trùng khớp."})

        hit = hits[0]
        user_info = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
        score = hit.score if hasattr(hit, "score") else hit.get("score", None)

        return JSONResponse({
            "message": "Đã nhận diện, có user trùng khớp.",
            "user": user_info,
            "score": score
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi nhận diện: {str(e)}")
