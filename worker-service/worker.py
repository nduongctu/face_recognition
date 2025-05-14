import redis
import time
import numpy as np
from PIL import Image
from io import BytesIO
from redis_queue import RedisQueue
from face_recognize.face_recognize import face_recognize
from face_recognize.postgres import init_db_pool, close_db_pool
import asyncio


def connect_redis(redis_host='redis', redis_port=6379, retries=5, delay=3):
    attempt = 0
    while attempt < retries:
        try:
            client = redis.Redis(host=redis_host, port=redis_port)
            client.ping()
            print("Connected to Redis!")
            return client
        except redis.exceptions.ConnectionError:
            attempt += 1
            print(f"Redis not available. Retry {attempt}/{retries}...")
            time.sleep(delay)
    raise Exception(f"Failed to connect to Redis after {retries} attempts")


async def handle_frame(frame_data):
    try:
        frame_idx = frame_data['frame_idx']
        frame_bytes = frame_data['frame']
        cam_id = frame_data.get('cam_id', 'default_cam')

        if not isinstance(frame_bytes, (bytes, bytearray)):
            print(f"[Worker] Invalid frame data type: {type(frame_bytes)}")
            return

        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        img_np = np.array(image)

        if img_np.ndim != 3 or img_np.shape[2] != 3:
            print(f"[Worker] Invalid image shape in frame {frame_idx}. Skipping.")
            return

        result = await face_recognize(cam_id, img_np, frame_idx)
        print(f"[Worker] Frame {frame_idx} (cam_id={cam_id}) recognized result: {result}")
    except KeyError as ke:
        print(f"[Worker] Missing key in frame data: {ke}")
    except Exception as e:
        print(f"[Worker] Error processing frame {frame_data.get('frame_idx')} from camera {frame_data.get('cam_id')}: {e}")


async def process_frames(redis_host='redis', redis_port=6379, queue_name='video_frames', batch_size=4):
    redis_client = connect_redis(redis_host, redis_port)
    queue = RedisQueue(name=queue_name, redis_client=redis_client)

    while True:
        frames_batch = []
        for _ in range(batch_size):
            try:
                frame_data = queue.get()
                if frame_data:
                    frames_batch.append(frame_data)
            except Exception as e:
                print(f"[Worker] Error getting frame from queue: {e}")
                time.sleep(1)

        if frames_batch:
            await asyncio.gather(*(handle_frame(f) for f in frames_batch))
        else:
            await asyncio.sleep(0.1)


async def start_worker():
    await init_db_pool()
    try:
        await process_frames()
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(start_worker())
