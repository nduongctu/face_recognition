FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Cài đặt các công cụ cần thiết
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt Python 3.9 trở lên từ PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN if [ ! -e /usr/bin/python3 ] || [ ! -L /usr/bin/python3 ]; then ln -s /usr/bin/python3.9 /usr/bin/python3; fi

RUN python3 --version && pip3 install --upgrade pip

WORKDIR /app

# Copy requirements.txt và cài đặt các phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy các file của ứng dụng vào container
COPY . /app

# Cấu hình command để chạy Celery worker
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=debug"]
