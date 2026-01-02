# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# 1. Install OS Dependencies (Cached unless you add packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Environment for Model Cache
ENV HF_HOME=/app/model_cache

# 3. Copy Requirements ONLY
COPY requirements.txt .

# 4. Install Pip Packages with Caching
# --mount=type=cache stores the pip cache on the host, so it persists across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 5. Pre-download Embedding Model
# We copy ONLY the preload script. If main.py changes, this layer is NOT invalidated.
COPY preload.py .
RUN python preload.py

# 6. Copy Application Code (This is the only layer that rebuilds often)
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]