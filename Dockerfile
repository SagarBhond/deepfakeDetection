# Base image (full Debian, NOT slim)
FROM python:3.10

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by TF, Torch, and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Upgrade pip (important for torch/tf wheels)
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Copy the rest of the project
COPY . .

# Expose port for Flask/Gunicorn
EXPOSE 5000

# Run Flask or Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
