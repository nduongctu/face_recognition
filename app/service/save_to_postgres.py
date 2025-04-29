import json
import asyncio
import numpy as np
from app.utils.postgres import *


async def save_to_postgres(app, user_id, bbox, confidence, frame_idx, object_name=None):
    try:
        async with app.db.acquire() as conn:
            async with conn.transaction():
                # Chuyển bbox thành định dạng JSON
                bbox_json = json.dumps(bbox)

                # Thêm kết quả nhận dạng vào bảng recognize_results
                result_id = await conn.fetchval('''
                                                INSERT INTO recognize_results(user_id, bbox, confidence, image_r2)
                                                VALUES ($1, $2, $3, $4) RETURNING result_id
                                                ''', user_id, bbox_json, confidence, object_name)

                # Kiểm tra nếu result_id có giá trị (không bị lọc bởi trigger)
                if result_id:
                    # Lưu frame_idx vào bảng face_frames
                    await conn.execute('''
                                       INSERT INTO face_frames(result_id, idx_frame)
                                       VALUES ($1, $2)
                                       ''', result_id, frame_idx)

                    return result_id
            return None
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào PostgreSQL: {e}")
        return None
