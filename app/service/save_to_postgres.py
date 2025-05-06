import json
from uuid import uuid1, UUID
from app.utils.postgres import *


async def save_to_postgres(app, user_id, bbox, confidence, cam_id, frame_idx, detection_time, object_name=None):
    try:
        frame_id = uuid1(node=int(UUID(cam_id)))

        async with app.db.acquire() as conn:
            async with conn.transaction():
                bbox_json = json.dumps(bbox)

                result_id = await conn.fetchval('''
                                                INSERT INTO recognize_results(user_id, bbox, confidence, object_name, detection_time)
                                                VALUES ($1, $2, $3, $4, $5)
                                                RETURNING result_id
                                                ''', user_id, bbox_json, confidence, object_name, detection_time)

                if result_id:
                    await conn.execute('''
                                       INSERT INTO face_frames(result_id, frame_id, idx_frame)
                                       VALUES ($1, $2, $3)
                                       ''', result_id, frame_id, frame_idx)

                    return result_id
            return None
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào PostgreSQL: {e}")
        return None
