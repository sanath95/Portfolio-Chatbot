FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local

COPY configs/ ./configs/
COPY src/ ./src/
COPY static/ ./static/
COPY main.py .

EXPOSE 8080
CMD ["python", "main.py"]