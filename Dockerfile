# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# Install dependencies for OCR, PDF rendering, and OpenCV
RUN apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --fix-missing \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script into the container
COPY fraud_detect.py .

# Default command (runs automatically on container start)
CMD ["python", "fraud_detect.py"]
