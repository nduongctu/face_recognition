import time
import redis
import cv2
import json
from redis_queue import RedisQueue
from PIL import Image
from io import BytesIO


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

    while True:
        # Lấy tất cả các stream từ Redis
        keys = redis_client.keys('video_stream:*')
        for key in keys:
            try:
                cam_id = key.decode().split(':')[1]

                stream_info = redis_client.get(key)
                if not stream_info:
                    print(f"Stream info not found for cam_id={cam_id}. It might have been deleted.")
                    continue

                data = json.loads(stream_info)
                stream_url = data.get('stream_url')

                if not stream_url:
                    print(f"No stream_url for cam_id={cam_id}, skipping.")
                    continue

                print(f"Processing stream for cam_id={cam_id}, url={stream_url}")

                # Mở video stream từ URL
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    print(f"Cannot open stream: {stream_url}. Skipping.")
                    continue

                frame_idx = 0
                while True:
                    # Kiểm tra lại xem stream còn tồn tại trong Redis hay không
                    if not redis_client.exists(key):
                        print(f"Stream for cam_id={cam_id} has been stopped or deleted. Stopping capture.")
                        cap.release()
                        break

                    ret, frame = cap.read()
                    if not ret:
                        print(f"Stream read failed for cam_id={cam_id}, stopping.")
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
                        continue

                # Đóng stream sau khi kết thúc
                cap.release()

                # Xóa thông tin stream từ Redis khi stream kết thúc
                redis_client.delete(key)
                print(f"Finished stream for cam_id={cam_id}, key deleted.")

            except Exception as e:
                print(f"Error processing key {key}: {e}")

        # Kiểm tra lại sau 2 giây
        time.sleep(2)


if __name__ == "__main__":
    process_video_stream()
