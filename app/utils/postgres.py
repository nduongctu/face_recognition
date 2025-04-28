import os
import asyncpg
from datetime import datetime

# Lấy thông tin kết nối từ biến môi trường
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")


async def connect_to_db():
    try:
        pool = await asyncpg.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            min_size=5,
            max_size=20
        )
        print(f"Đã kết nối đến PostgreSQL tại {DB_HOST}:{DB_PORT}")
        return pool
    except Exception as e:
        print(f"Lỗi kết nối đến PostgreSQL: {e}")
        raise


async def close_db_connection(pool):
    await pool.close()


async def cleanup_old_face_data(app):
    try:
        async with app.db.acquire() as conn:
            await conn.execute("SELECT cleanup_old_face_data()")
            print(f"Đã dọn dẹp dữ liệu thành công lúc {datetime.now()}")
    except Exception as e:
        print(f"Lỗi khi dọn dẹp dữ liệu: {e}")
