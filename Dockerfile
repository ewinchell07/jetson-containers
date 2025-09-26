FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Copy your application code
COPY . /app/

# Set the entrypoint
ENTRYPOINT ["python3", "main.py"] 