import os
import asyncpg
from typing import Optional

# Biến toàn cục để giữ pool
_pool: Optional[asyncpg.Pool] = None

# Lấy thông tin kết nối từ biến môi trường
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = int(os.getenv("POSTGRES_PORT", 5432))
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")


async def init_db_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            min_size=5,
            max_size=20
        )
        print(f"Đã kết nối đến PostgreSQL tại {DB_HOST}:{DB_PORT}")


async def close_db_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_db_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool has not been initialized.")
    return _pool
