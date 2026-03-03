# Use official Python slim image
FROM python:3.11-slim

# ---------------------------
# System dependencies
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# Upgrade pip
# ---------------------------
RUN pip install --upgrade pip

# ---------------------------
# Install CPU-only PyTorch and torchvision
# ---------------------------
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 torchvision==0.16.0

# ---------------------------
# Install other Python dependencies including MLflow
# ---------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir mlflow==2.8.0

# Ensure pip executables are in PATH
ENV PATH="/root/.local/bin:$PATH"

# ---------------------------
# Copy project files
# ---------------------------
COPY . .

# ---------------------------
# Default command (used only if container is run without override)
# ---------------------------
CMD ["python", "train.py"]
