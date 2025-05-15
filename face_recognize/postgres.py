import os
import asyncpg
import asyncio
from typing import Optional

_pool: Optional[asyncpg.Pool] = None

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
            min_size=1,
            max_size=2
        )
        print(f"[PID {os.getpid()}] PostgreSQL pool initialized.")


def get_db_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool has not been initialized.")
    return _pool


def init_db_pool_sync():
    """
    Hàm sync dùng trong signal của dramatiq.
    """
    asyncio.run(init_db_pool())
