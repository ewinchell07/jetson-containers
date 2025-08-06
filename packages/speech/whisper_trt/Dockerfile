#---
# name: whisper_trt
# group: audio
# depends: [pytorch, torch2trt, onnxruntime]
# requires: '>=36'
# test: test.py
# notes: TensorRT optimized Whisper ASR with real-time audio processing
#---
FROM dustynv/whisper_trt:r36.3.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /opt/whisper_trt

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/opt/whisper_trt
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Command to run the container
CMD ["bash"]

