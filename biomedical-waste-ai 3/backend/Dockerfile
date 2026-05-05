FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS deps for OpenCV / PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
      build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Make sure SQLite + weights dirs exist
RUN mkdir -p /app/database /app/weights

EXPOSE 8000

# Prefetch YOLO pretrained weights at build-time so first-boot is instant
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
