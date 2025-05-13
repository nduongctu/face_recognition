import asyncio
from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
import cv2
import threading
import numpy as np
from typing import AsyncGenerator, Optional, Union
import uuid
from face_recognize.face_recognize import face_recognize
import time

router = APIRouter()


class Camera:
    def __init__(self, url: Optional[Union[str, int]] = 0) -> None:
        if isinstance(url, int) or url == 0:
            self.cap = cv2.VideoCapture(url)
        else:
            self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()

    def get_frame(self) -> bytes:
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                return b''  # Nếu không lấy được frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                return b''
            return jpeg.tobytes()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def release(self) -> None:
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()


async def gen_frames(camera: Camera, cam_id: str) -> AsyncGenerator[bytes, None]:
    frame_idx = 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = camera.get_raw_frame()
            if frame is None:
                break

            # Gọi hàm nhận diện khuôn mặt
            result = await face_recognize(cam_id, frame, frame_idx)
            frame_idx += 1
            frame_count += 1

            height, width, _ = frame.shape

            # Vẽ bounding boxes nếu có trong kết quả nhận diện
            for face in result:  # result giờ là một danh sách
                if 'bbox' in face:
                    # bbox trả về [xmin, ymin, xmax, ymax] chuẩn hóa [0, 1]
                    xmin, ymin, xmax, ymax = face["bbox"]

                    # Chuyển đổi tọa độ bounding box về pixel thực
                    x1 = int(xmin * width)
                    y1 = int(ymin * height)
                    x2 = int(xmax * width)
                    y2 = int(ymax * height)

                    # Kiểm tra nếu không có user_id
                    user_id = face.get("user_id", None)
                    if user_id:
                        label = str(user_id)
                        color = (0, 255, 0)  # Màu xanh lá cho khuôn mặt có user_id
                    else:
                        label = "Khong xac dinh"
                        color = (0, 0, 255)  # Màu đỏ cho khuôn mặt không xác định

                    # Vẽ bounding box với màu sắc tương ứng
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Vẽ nhãn với màu sắc tương ứng
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Tính FPS (frames per second)
            current_time = time.time()
            if current_time - start_time >= 1.0:  # Tính FPS mỗi giây
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
            else:
                fps = frame_count / (current_time - start_time)

            # Vẽ FPS ở góc phải
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Encode frame thành JPEG để gửi qua StreamingResponse
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                break

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

            await asyncio.sleep(0)
    except (asyncio.CancelledError, GeneratorExit):
        print("Frame generation cancelled.")
    finally:
        print("Frame generator exited.")


@router.get("/")
async def video_feed(
        camera_url: Optional[str] = None,
        cam_id: Optional[str] = Query(default_factory=lambda: str(uuid.uuid4()))
) -> StreamingResponse:
    camera = Camera(camera_url or 0)
    return StreamingResponse(
        gen_frames(camera, cam_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
