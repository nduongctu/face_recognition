import cv2
import requests
import numpy as np
import time
import uuid
import threading
import queue

API_URL = "http://localhost:8000/face/recognize"
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS

SEND_EVERY_N_FRAMES = 1
MAX_DELAY = 3
NUM_WORKERS = 3

cam_id = str(uuid.uuid4())

frame_queue = queue.Queue(maxsize=5)
results = {}
results_lock = threading.Lock()

ready_flags = [threading.Event() for _ in range(NUM_WORKERS)]


def api_worker(worker_idx):
    ready_flags[worker_idx].set()
    while True:
        try:
            idx, frame = frame_queue.get()
            _, jpg_img = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', jpg_img.tobytes(), 'image/jpeg')}
            data = {
                'frame_idx': idx,
                'cam_id': cam_id
            }
            resp = requests.post(API_URL, files=files, data=data, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                if "result" in result:
                    with results_lock:
                        results[idx] = result["result"]
        except Exception as e:
            print(f"API worker exception: {e}")


for i in range(NUM_WORKERS):
    threading.Thread(target=api_worker, args=(i,), daemon=True).start()

# Đợi toàn bộ worker đã sẵn sàng
for flag in ready_flags:
    flag.wait()

cap = cv2.VideoCapture(0)

frame_idx = 0
frame_count = 0
start_time = time.time()
fps = 0
last_frame_time = time.time()

last_result = None
last_result_frame_idx = -1

while True:
    now = time.time()
    elapsed = now - last_frame_time
    if elapsed < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - elapsed)
    last_frame_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_count += 1
    display_frame = frame.copy()

    if frame_idx % SEND_EVERY_N_FRAMES == 0 and not frame_queue.full():
        frame_queue.put((frame_idx, frame.copy()))

    matched_result = None
    with results_lock:
        for idx in sorted(results.keys(), reverse=True):
            if idx <= frame_idx and frame_idx - idx <= MAX_DELAY:
                matched_result = results.pop(idx)
                last_result = matched_result
                last_result_frame_idx = idx
                break

    if matched_result is None and last_result and frame_idx - last_result_frame_idx <= MAX_DELAY:
        matched_result = last_result

    if matched_result:
        height, width, _ = frame.shape
        for item in matched_result:
            # Kiểm tra nếu item là dictionary và có "bbox"
            if isinstance(item, dict) and "bbox" in item:
                bbox = item["bbox"]
                if bbox:
                    detail = item.get("detail", "")
                    user_id = item.get("user_id")

                    if detail == "Không tìm thấy người phù hợp" or detail == "Đang xử lý...":
                        label = "Khong xac dinh"
                        color = (0, 0, 255)
                    else:
                        label = str(user_id) if user_id is not None else "Khong xac dinh"
                        color = (0, 255, 0)

                    x1, y1, x2, y2 = bbox
                    x1 = int(x1 * width)
                    y1 = int(y1 * height)
                    x2 = int(x2 * width)
                    y2 = int(y2 * height)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    current_time = time.time()
    if current_time - start_time >= 1.0:
        fps = frame_count / (current_time - start_time)
        frame_count = 0
        start_time = current_time

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(display_frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition (ESC de thoat)", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
