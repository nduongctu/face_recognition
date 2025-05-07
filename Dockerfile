FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các gói cần thiết cho hệ thống
RUN apt-get update && apt-get install -y \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt uv từ ghcr.io (Astral)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Sao chép pyproject.toml vào container
COPY pyproject.toml .

# Tạo uv.lock và cài đặt các dependencies
RUN uv lock && uv sync --frozen --no-cache

COPY . /app

EXPOSE 8000

CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]