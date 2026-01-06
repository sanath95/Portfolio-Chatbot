# ============================
# Stage 1: Index builder
# ============================
FROM python:3.11-slim AS indexer

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6  && \
    rm -rf /var/lib/apt/lists/*

# Install only indexing dependencies
COPY index_requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r index_requirements.txt

# Copy indexing inputs
COPY index.py .
COPY data/ ./data/
COPY configs/ ./configs/

# Run indexing ONCE at build time
RUN --mount=type=secret,id=QDRANT_URL \
    --mount=type=secret,id=QDRANT_API_KEY \
    --mount=type=secret,id=OPENAI_API_KEY \
    python index.py

# ============================
# Stage 2: Runtime
# ============================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

# Runtime Python deps only (NO torch / transformers)
COPY chatbot_requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r chatbot_requirements.txt

# Copy app code
COPY src/ ./src/
COPY static/ ./static/
COPY main.py .

EXPOSE 8080
CMD ["python", "main.py"]