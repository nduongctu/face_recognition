import cv2
import requests
import numpy as np
import time

API_URL = "http://localhost:8000/face/recognize"
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS

cap = cv2.VideoCapture(0)

frame_idx = 0
frame_count = 0
start_time = time.time()
fps = 0
last_frame_time = time.time()

while True:
    now = time.time()
    elapsed = now - last_frame_time
    if elapsed < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - elapsed)
    last_frame_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_idx += 1  # Tăng frame_idx

    # Gửi ảnh và frame_idx lên API
    _, jpg_img = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', jpg_img.tobytes(), 'image/jpeg')}  # Sửa thành 'file' thay vì 'image'
    data = {'frame_idx': str(frame_idx)}  # Đảm bảo frame_idx là kiểu string

    try:
        resp = requests.post(API_URL, files=files, data=data, timeout=5)

        if resp.status_code == 200:
            result = resp.json()

            if "result" in result:
                result_data = result["result"]
                if isinstance(result_data, list):
                    for item in result_data:
                        bbox = item.get("bbox")
                        if bbox:
                            detail = item.get("detail", "")
                            user_id = item.get("user_id")

                            if detail == "Không tìm thấy người phù hợp" or detail == "Đang xử lý...":
                                # Nếu không nhận diện được người, màu đỏ và nhãn "Không xác định"
                                label = "Không xác định"
                                color = (0, 0, 255)  # Màu đỏ
                            else:
                                # Nếu nhận diện được người, hiển thị user_id và màu xanh
                                label = str(user_id) if user_id is not None else "Không xác định"
                                color = (0, 255, 0)  # Màu xanh

                            # Chuyển đổi bbox từ tỉ lệ 0-1 về tọa độ pixel
                            height, width, _ = frame.shape
                            x1, y1, x2, y2 = bbox
                            x1 = int(x1 * width)  # Chuyển đổi từ tỉ lệ sang pixel
                            y1 = int(y1 * height)
                            x2 = int(x2 * width)
                            y2 = int(y2 * height)

                            # Vẽ bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        else:
            print(f"Error: Received status code {resp.status_code}")
            cv2.putText(frame, f"API Error: {resp.status_code}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2)
    except Exception as e:
        print(f"Exception: {e}")
        cv2.putText(frame, f"API Error: {str(e)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Tính toán FPS thực tế (cập nhật mỗi giây)
    current_time = time.time()
    time_elapsed = current_time - start_time
    if time_elapsed >= 1.0:
        fps = frame_count / time_elapsed
        frame_count = 0
        start_time = current_time

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition (ESC de thoat)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()
