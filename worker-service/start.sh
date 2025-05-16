#!/bin/bash
set -e

# Khởi động dramatiq worker trong background
echo "[STARTUP] Starting Dramatiq worker..."
dramatiq -p 2 -t 4 worker &
DRAMATIQ_PID=$!

sleep 1

# Kiểm tra xem worker có chạy không
if ! ps -p $DRAMATIQ_PID > /dev/null; then
    echo "[ERROR] Dramatiq worker failed to start"
    exit 1
fi

echo "[STARTUP] Dramatiq worker started successfully with PID $DRAMATIQ_PID"

# Khởi động frame processor
echo "[STARTUP] Starting frame processor..."
python3 worker.py