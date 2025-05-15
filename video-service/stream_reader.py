import time
import redis
import cv2
import os
import uuid
import threading
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from urllib.parse import quote
from redis_queue import RedisQueue

load_dotenv()


def get_env_vars():
    env_vars = {}
    with open(".env", "r") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                env_vars[key] = value
    return env_vars.get("MODE"), env_vars.get("URL_STREAM"), env_vars.get("API_URL")


def connect_redis(redis_host='redis', redis_port=6379, retries=5, delay=3):
    attempt = 0
    while attempt < retries:
        try:
            client = redis.Redis(host=redis_host, port=redis_port)
            client.ping()
            print("Đã kết nối Redis!")
            return client
        except redis.exceptions.ConnectionError:
            attempt += 1
            print(f"Redis chưa sẵn sàng. Thử lại {attempt}/{retries}...")
            time.sleep(delay)
    raise Exception(f"Không kết nối được Redis sau {retries} lần thử")


stop_thread = threading.Event()


def process_video_stream(URL_STREAM, redis_host='redis', redis_port=6379, queue_name='video_frames'):
    redis_client = connect_redis(redis_host, redis_port)
    queue = RedisQueue(name=queue_name, redis_client=redis_client)

    if not URL_STREAM:
        print("URL_STREAM chưa được thiết lập trong .env")
        return

    cam_id = str(uuid.uuid4())
    print(f"Tạo cam_id: {cam_id}")

    cap = cv2.VideoCapture(URL_STREAM)
    if not cap.isOpened():
        print(f"Không mở được stream: {URL_STREAM}")
        return

    frame_idx = 0
    while not stop_thread.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ stream.")
            break

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            frame_data = buffer.getvalue()

            queue.put({
                'frame_idx': frame_idx,
                'frame': frame_data,
                'cam_id': cam_id
            })

            print(f"[{cam_id}] Frame {frame_idx} đã đẩy vào Redis queue.")
            frame_idx += 1

        except Exception as e:
            print(f"Lỗi khi mã hóa/gửi frame: {e}")

    cap.release()
    print("Dừng xử lý video stream.")


def generate_fastapi_stream_url(camera_url, api_url):
    encoded_camera_url = quote(camera_url)
    stream_url = f"{api_url}/?camera_url={encoded_camera_url}"
    print(f"URL: {stream_url}")
    return stream_url


if __name__ == "__main__":
    last_mode = None
    current_thread = None

    while True:
        current_mode, url_stream, api_url = get_env_vars()

        if current_mode != last_mode:
            print(f"[INFO] MODE thay đổi thành: {current_mode}")
            last_mode = current_mode

            if current_thread and current_thread.is_alive():
                stop_thread.set()
                current_thread.join()
                stop_thread.clear()

            if current_mode == "non-stream":
                current_thread = threading.Thread(target=process_video_stream, args=(url_stream,))
                current_thread.start()

            elif current_mode == "stream":
                if url_stream and api_url:
                    generate_fastapi_stream_url(url_stream, api_url)
                else:
                    print("[ERROR] URL_STREAM hoặc API_URL chưa được thiết lập hoặc rỗng")

            else:
                print(f"[WARN] MODE không xác định: {current_mode}")

        time.sleep(5)
