FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 g++ && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "stream_reader.py"]
