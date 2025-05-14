import time
import redis
import cv2
import json
import os
import uuid
import requests
from urllib.parse import quote
from redis_queue import RedisQueue
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

URL_STREAM = os.getenv("URL_STREAM")
MODE = os.getenv("MODE")


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


def process_video_stream(redis_host='redis', redis_port=6379, queue_name='video_frames'):
    redis_client = connect_redis(redis_host, redis_port)
    queue = RedisQueue(name=queue_name, redis_client=redis_client)

    if not URL_STREAM:
        print("URL_STREAM is not set in .env")
        return

    # Tạo cam_id bằng uuid4
    cam_id = str(uuid.uuid4())
    print(f"Generated cam_id: {cam_id}")

    cap = cv2.VideoCapture(URL_STREAM)
    if not cap.isOpened():
        print(f"Cannot open stream: {URL_STREAM}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from stream.")
            break

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            frame_data = buffer.getvalue()

            # Đẩy frame vào Redis queue
            queue.put({
                'frame_idx': frame_idx,
                'frame': frame_data,
                'cam_id': cam_id
            })

            print(f"[{cam_id}] Frame {frame_idx} pushed to Redis queue.")
            frame_idx += 1

        except Exception as e:
            print(f"Error encoding/sending frame: {e}")

    cap.release()


def generate_fastapi_stream_url(camera_url):
    encoded_camera_url = quote(camera_url)
    stream_url = f"http://localhost:8000/video/?camera_url={encoded_camera_url}"
    print(f"URL truy cập từ trình duyệt: {stream_url}")
    return stream_url


if __name__ == "__main__":
    if MODE == "non-stream":
        process_video_stream()
    else:
        generate_fastapi_stream_url(URL_STREAM)
