# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Whisper-TRT speech transcription system optimized for NVIDIA Jetson devices. The project provides GPU-accelerated speech-to-text functionality using TensorRT-optimized Whisper models with real-time audio processing capabilities.

## Key Architecture Components

### Core Files
- `simple_transcribe.py`: Main application with three operational modes (file, continuous, batch)
- `test.py`: Simple test script for TensorRT model loading and transcription
- `Dockerfile`: Container build configuration based on `dustynv/whisper_trt:r36.3.0`
- `requirements.txt`: Python dependencies including whisper-trt, torch, pyannote.audio

### Operational Modes
1. **File Mode**: Single audio file transcription
2. **Continuous Mode**: Real-time recording and transcription with configurable chunk duration
3. **Batch Mode**: Process multiple recordings within specified time ranges
4. **Record-Only Mode**: Optimized recording without transcription overhead (uses `SimpleAudioRecorder`)

### Audio Processing Architecture
- **TranscriptionManager**: Full-featured class handling recording, processing, and transcription
- **SimpleAudioRecorder**: Lightweight class for record-only mode with minimal memory footprint
- Queue-based processing for continuous mode
- Support for speaker diarization via pyannote.audio

## Development Commands

### Building the Container
```bash
# Build the whisper-trt container
docker build -t whisper-trt:latest .
```

### Running the Container
```bash
# Run with GPU access and audio device support
docker run --gpus all -it \
    --network=host \
    --device /dev/snd \
    -v $(pwd):/opt/whisper_trt \
    -v ~/.cache/whisper_trt/logs:/root/.cache/whisper_trt/logs \
    --name whisper-trt \
    whisper-trt:latest
```

### Using the Jetson Containers Framework
This package is part of the larger jetson-containers ecosystem:

```bash
# Build using the framework (from repository root)
./build.sh whisper_trt

# Run using the framework
./run.sh whisper_trt
```

### Testing
```bash
# Test TensorRT model loading and basic transcription
python3 test.py

# Test different modes
python3 simple_transcribe.py --mode file --audio_file /data/audio/dusty.wav --model tiny.en
python3 simple_transcribe.py --mode continuous --model tiny.en --chunk_duration 600
python3 simple_transcribe.py --mode continuous --record_only --chunk_duration 600
```

## Configuration Details

### Audio Configuration
- Sample rate: 16000 Hz (mono)
- Default chunk duration: 600 seconds (10 minutes)
- Block size: 1024 samples
- Low latency configuration for real-time processing

### Model Options
- Available models: tiny.en, base.en, small.en, medium.en, large
- TensorRT optimization with fallback to regular Whisper
- Automatic CUDA memory management with TF32 support

### Key Parameters
- `--temperature`: Controls transcription randomness (0.0 = deterministic)
- `--no_speech_threshold`: Silence detection threshold (0.0-1.0, default: 0.3)
- `--logprob_threshold`: Token validity threshold (default: -0.7)
- `--compression_ratio`: Segment compression limit (default: 1.8)
- `--hf_token`: HuggingFace token for speaker diarization
- `--num_speakers`: Number of speakers for diarization (default: 2)

## File Structure

### Input/Output Directories
- `recordings/`: Audio chunks and transcription files
- `~/.cache/whisper_trt/logs/`: Detailed application logs
- `/data/audio/`: Sample audio files (dusty.wav, commands.wav)

### File Naming Conventions
- Audio: `recording_YYYYMMDD_HHMMSS.wav`
- Partial chunks: `recording_YYYYMMDD_HHMMSS_partial.wav`
- Transcripts: `transcript_*.json` (with speaker diarization data)

## Performance Optimization

### Memory Management
- Use `--record_only` for minimal memory footprint
- Automatic CUDA memory cleanup after transcription
- Queue-based processing prevents memory buildup in continuous mode

### Resource Usage Modes
- **Record-Only**: No GPU operations, minimal CPU overhead
- **Transcription**: Full GPU acceleration with memory monitoring
- **Batch Processing**: Automatic cleanup between files

### Recommended Settings by Use Case
- **Data Collection**: `--record_only --chunk_duration 600`
- **High Accuracy**: `--model base.en --no_speech_threshold 0.3`
- **Speaker Diarization**: `--hf_token TOKEN --num_speakers 2`
- **Long-term Recording**: `--record_only` to prevent memory issues

## Container Environment

### Base Image
- `dustynv/whisper_trt:r36.3.0` (NVIDIA L4T PyTorch base)
- JetPack 5.1+ required for Jetson devices
- NVIDIA Container Toolkit for GPU access

### Environment Variables
- `PYTHONPATH=/opt/whisper_trt`
- `CUDA_VISIBLE_DEVICES=0`
- `TF_FORCE_GPU_ALLOW_GROWTH=true`

### Device Access
- `/dev/snd`: Audio device access
- GPU access via `--gpus all` flag
- Network host mode for optimal performance