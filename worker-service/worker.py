import redis
import time
from redis_queue import RedisQueue
from tasks import recognize_face_task_stream, recognize_face_task_non_stream
from face_recognize.postgres import init_db_pool, close_db_pool
import os
from dotenv import load_dotenv

load_dotenv()

MODE = os.getenv("MODE", "non-stream")


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


def process_frames(redis_host='redis', redis_port=6379, queue_name='video_frames'):
    redis_client = connect_redis(redis_host, redis_port)
    queue = RedisQueue(name=queue_name, redis_client=redis_client)

    print(f"[Worker] {MODE} mode selected.")

    while True:
        try:
            frame_data = queue.get()
        except Exception as e:
            print(f"[Worker] Error getting frame from queue: {e}")
            time.sleep(1)
            continue

        if frame_data is None:
            print("[Worker] No frame in queue, sleeping...")
            time.sleep(1)
            continue

        try:
            frame_idx = frame_data['frame_idx']
            frame_bytes = frame_data['frame']
            cam_id = frame_data.get('cam_id', 'default_cam')

            if not isinstance(frame_bytes, (bytes, bytearray)):
                print(f"[Worker] Invalid frame data type: {type(frame_bytes)}")
                continue

            if MODE == "non-stream":
                result = recognize_face_task_non_stream.delay(cam_id, frame_bytes, frame_idx)
                print(f"[Worker] Non-stream task sent to Celery: Frame {frame_idx} (cam_id={cam_id})")

            elif MODE == "stream":
                result = recognize_face_task_stream.delay(cam_id, frame_bytes, frame_idx)
                print(f"[Worker] Stream task sent to Celery: Frame {frame_idx} (cam_id={cam_id})")

            else:
                print(f"[Worker] Unknown mode: {MODE}")

        except KeyError as ke:
            print(f"[Worker] Missing key in frame data: {ke}")
        except Exception as e:
            print(f"[Worker] Error sending frame to Celery: {e}")


def start_worker():
    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_until_complete(init_db_pool())
    try:
        process_frames()
    finally:
        loop.run_until_complete(close_db_pool())


if __name__ == "__main__":
    start_worker()
