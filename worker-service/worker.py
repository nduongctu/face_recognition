import os
import dramatiq
import redis
import time
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from redis_queue import RedisQueue
from face_recognize.face_recognize import face_recognize
from face_recognize.postgres import _pool, init_db_pool, get_db_pool
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import AsyncIO

redis_host = os.environ.get("REDIS_HOST", "redis")
redis_port = int(os.environ.get("REDIS_PORT", 6379))

redis_client = redis.Redis(host=redis_host, port=redis_port)

redis_broker = RedisBroker(host=redis_host, port=redis_port)
redis_broker.add_middleware(AsyncIO())
dramatiq.set_broker(redis_broker)


@dramatiq.actor(queue_name="default")
async def process_frame_task(frame_key):
    global _pool
    try:
        # Lazy init pool trong worker process khi task đầu tiên chạy
        if _pool is None:
            await init_db_pool()

        raw = redis_client.get(frame_key)
        if not raw:
            print(f"[ERROR] Redis key {frame_key} not found")
            return

        frame_data = pickle.loads(raw)
        frame_idx = frame_data['frame_idx']
        frame_bytes = frame_data['frame']
        cam_id = frame_data.get('cam_id', 'default_cam')

        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        img_np = np.array(image)

        pool = get_db_pool()

        result = await face_recognize(cam_id, img_np, frame_idx)
        print(f"[DONE] Frame {frame_idx} (cam_id={cam_id}) - Result: {result}")

    except Exception as e:
        print(f"[ERROR] Processing frame failed: {e}")


def main():
    queue = RedisQueue(name="video_frames", redis_client=redis_client)

    try:
        while True:
            frame_data = queue.get(block=True, timeout=1)
            if frame_data:
                try:
                    frame_idx = frame_data.get("frame_idx")
                    redis_key = f"frame:{frame_idx}"
                    redis_client.setex(redis_key, 60, pickle.dumps(frame_data))
                    process_frame_task.send(redis_key)
                    print(f"[SEND] Frame {frame_idx} enqueued to dramatiq with key {redis_key}")
                except Exception as e:
                    print(f"[ERROR] Khi gửi frame lên Dramatiq: {e}")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping worker loop...")


if __name__ == "__main__":
    main()
