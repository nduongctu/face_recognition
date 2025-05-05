import os
import boto3
import uuid
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError
from concurrent.futures import ThreadPoolExecutor

# Load các biến từ file .env
load_dotenv()

# Lấy thông tin từ biến môi trường
r2_access_key = os.getenv('R2_ACCESS_KEY')
r2_secret_key = os.getenv('R2_SECRET_KEY')
r2_endpoint_url = os.getenv('R2_ENDPOINT_URL')
bucket_name = os.getenv('R2_BUCKET_NAME')

# Cấu hình kết nối với R2
r2_client = boto3.client(
    's3',
    endpoint_url=r2_endpoint_url,
    aws_access_key_id=r2_access_key,
    aws_secret_access_key=r2_secret_key,
    region_name='auto'
)

# Tạo thread pool để xử lý bất đồng bộ
executor = ThreadPoolExecutor(max_workers=4)


def upload_face_crop_to_r2(face_crop: np.ndarray, user_id: str, frame_idx: int) -> str:
    try:
        # Chuyển numpy array thành PIL Image
        pil_img = Image.fromarray(face_crop)

        # Chuẩn bị byte stream
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG')
        buffer.seek(0)

        unique_id = uuid.uuid4().hex
        object_name = f"face-recognize/user_{user_id}_frame_{frame_idx}_{unique_id}.jpg"

        def upload_to_r2():
            try:
                r2_client.upload_fileobj(buffer, bucket_name, object_name)
            except Exception as e:
                print(f"Lỗi khi upload ảnh lên R2: {e}")

        # Gọi hàm upload trong thread riêng biệt
        executor.submit(upload_to_r2)

        # Trả về object_name ngay lập tức
        return object_name

    except NoCredentialsError:
        print("Lỗi: Không có credentials R2")
        return None
    except Exception as e:
        print(f"Lỗi khi upload ảnh lên R2: {e}")
        return None
