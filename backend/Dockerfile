# syntax=docker/dockerfile:1
FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY main.py .
COPY database.py .
COPY audio_inference.py .
COPY feature_utils.py .
COPY train_xgb.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]