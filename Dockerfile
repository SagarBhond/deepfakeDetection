
# ============================
# 1. BUILDER STAGE
# ============================
FROM python:3.10-slim AS builder

# Prevent timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, dlib, torch, face-recognition, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ============================
# 2. FINAL RUNTIME STAGE
# ============================
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Only install runtime libs (lighter and faster)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /usr/local /usr/local

# Copy project files
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
