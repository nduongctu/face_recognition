from fastapi import APIRouter, HTTPException
from datetime import date
import asyncpg

from app.utils.postgres import get_db_pool

router = APIRouter()


@router.get("/total_recognitions", summary="Thống kê tổng số nhận diện trong khoảng thời gian")
async def total_recognitions(start_date: date, end_date: date):
    try:
        pool = get_db_pool()
        query = """
                SELECT COUNT(*) AS total
                FROM recognize_results
                WHERE DATE(detection_time) BETWEEN $1 AND $2
                """
        total = await pool.fetchval(query, start_date, end_date)
        return {"total_recognitions": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy tổng số nhận diện: {e}")


@router.get("/by_user/{user_id}", summary="Lấy tất cả nhận diện theo user_id")
async def recognitions_by_user(user_id: str):
    try:
        pool = get_db_pool()
        query = """
                SELECT result_id, user_id, bbox, detection_time, confidence, object_name
                FROM recognize_results
                WHERE user_id = $1
                ORDER BY detection_time DESC
                """
        rows = await pool.fetch(query, user_id)
        if not rows:
            raise HTTPException(status_code=404, detail="Không tìm thấy kết quả nhận diện cho người dùng này.")
        return {"data": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy kết quả nhận diện theo user_id: {e}")


@router.get("/by_user_in_time/{user_id}", summary="Lấy nhận diện theo user_id trong khoảng thời gian")
async def recognitions_by_user_time(user_id: str, start_date: date, end_date: date):
    try:
        pool = get_db_pool()
        query = """
                SELECT result_id, user_id, bbox, detection_time, confidence, object_name
                FROM recognize_results
                WHERE user_id = $1
                  AND DATE(detection_time) BETWEEN $2 AND $3
                ORDER BY detection_time DESC
                """
        rows = await pool.fetch(query, user_id, start_date, end_date)
        if not rows:
            raise HTTPException(status_code=404, detail="Không có kết quả nhận diện nào trong khoảng thời gian này.")
        return {"data": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu theo thời gian: {e}")


@router.get("/unique_users", summary="Thống kê số lượng user duy nhất trong khoảng thời gian")
async def unique_user_count(start_date: date, end_date: date):
    try:
        pool = get_db_pool()
        query = """
                SELECT COUNT(DISTINCT user_id) AS user_count
                FROM recognize_results
                WHERE DATE(detection_time) BETWEEN $1 AND $2
                """
        count = await pool.fetchval(query, start_date, end_date)
        return {"unique_user_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thống kê số lượng user duy nhất: {e}")


@router.get("/recognition_details/{user_id}", summary="Lấy chi tiết nhận diện theo user_id")
async def recognition_details(user_id: str):
    try:
        pool = get_db_pool()
        query = """
                SELECT rr.result_id,
                       rr.user_id,
                       rr.bbox,
                       rr.detection_time,
                       rr.confidence,
                       rr.object_name,
                       ff.frame_id,
                       ff.idx_frame
                FROM recognize_results rr
                         LEFT JOIN face_frames ff ON rr.result_id = ff.result_id
                WHERE rr.user_id = $1
                ORDER BY rr.detection_time DESC
                """
        rows = await pool.fetch(query, user_id)
        if not rows:
            raise HTTPException(status_code=404, detail="Không tìm thấy chi tiết nhận diện cho người dùng này.")
        return {"data": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy chi tiết nhận diện: {e}")


@router.get("/recognition_count/{user_id}", summary="Thống kê số lần nhận diện theo user_id")
async def recognition_count(user_id: str):
    try:
        pool = get_db_pool()
        query = """
                SELECT COUNT(*) AS recognition_count
                FROM recognize_results
                WHERE user_id = $1
                """
        count = await pool.fetchval(query, user_id)
        if count is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu nhận diện cho người dùng này.")
        return {"user_id": user_id, "recognition_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thống kê số lần nhận diện: {e}")
