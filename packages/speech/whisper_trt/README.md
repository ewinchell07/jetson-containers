# Whisper-TRT Speech Transcription System

A robust, production-ready speech transcription system optimized for NVIDIA Jetson devices. Features GPU-accelerated transcription with TensorRT optimization, speaker diarization, and comprehensive audio processing capabilities.

## 🚀 Key Features

- **GPU-Accelerated Transcription**: Uses Whisper-TRT (TensorRT optimized) with automatic fallback to regular Whisper
- **Speaker Diarization**: Resemblyzer-based speaker identification optimized for Jetson devices
- **Smart Model Selection**: Automatic model selection with fallback based on available system resources
- **Memory Management**: Comprehensive GPU memory management and cleanup
- **Batch Processing**: Process single files or entire directories
- **Audio Preprocessing**: Automatic noise suppression, filtering, and normalization
- **Jetson Optimized**: Specifically designed for Jetson Nano and other Jetson devices
- **Comprehensive Testing**: Full test suite with Jetson-specific tests

## 📋 Prerequisites

- NVIDIA Jetson device with JetPack 5.1 or later
- Docker installed
- NVIDIA Container Toolkit installed
- Audio input device (microphone)
- FFmpeg with RNNoise support (for audio pre-processing)

## 🏗️ Building the Container

```bash
# Build the container
docker build -t whisper-trt:latest .
```

## 🚀 Running the Container

### Memory Configuration

The container is configured for optimal performance on Jetson devices:

```bash
docker run --gpus all -it \
    --network=host \
    --device /dev/snd \
    --memory=6g \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/opt/whisper_trt \
    -v ~/.cache/whisper_trt/logs:/root/.cache/whisper_trt/logs \
    --name whisper-trt \
    whisper-trt:latest
```

**Note**: Adjust `--memory` value based on your system (e.g., `--memory=14g` for 16GB systems).

### Container Management

```bash
# View logs
docker logs whisper-trt

# Follow logs in real-time
docker logs -f whisper-trt

# Restart container
docker start -i whisper-trt
```

## ⚙️ Configuration

### Environment Variables

```bash
# Model selection (default: small)
export TRANSCRIBE_MODEL=medium          # Options: tiny, base, small, medium, large-v2

# Swap usage on Jetson Nano (default: true)
export ALLOW_SWAP=true                  # Set to false to disable swap usage

# Maximum audio duration in minutes (optional)
export MAX_DURATION_MIN=30              # Skip files longer than this

# Custom transcription prompt (optional)
export TRANSCRIBE_PROMPT="Your custom prompt here"
```

### Jetson Nano Swap Setup

For larger Whisper models on Jetson Nano:

```bash
# Quick setup - 16GB swap (recommended for large-v2 models)
./scripts/setup_swap.sh 16

# Or 8GB swap (sufficient for medium models)
./scripts/setup_swap.sh 8
```

**Model Requirements:**
- **Whisper medium**: ≥8GB swap recommended
- **Whisper large-v2**: ≥16GB swap recommended
- **Smaller models**: No swap required

## 📖 Usage

### 1. Single File Transcription

```bash
# Basic transcription
python3 transcriber.py audio.wav

# With specific model
python3 transcriber.py audio.wav --model medium

# With speaker diarization
python3 transcriber.py audio.wav --model medium --num-speakers 2

# Custom output directory
python3 transcriber.py audio.wav --output-dir my_transcriptions
```

### 2. Batch Processing

```bash
# Process all WAV files in a directory
python3 transcriber.py recordings/ --batch

# Process with specific pattern
python3 transcriber.py recordings/ --batch --pattern "*.mp3"

# Batch with speaker diarization
python3 transcriber.py recordings/ --batch --num-speakers 2
```

### 3. Continuous Recording

```bash
# Basic continuous recording
python3 continuous_recorder.py --chunk-duration 600 --output-dir recordings

# Optimized for stability
python3 continuous_recorder.py --chunk-duration 600 --buffer-size 4096 --latency high
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `audio_file` | Audio file to transcribe (or directory for batch mode) | Required |
| `--model` | Whisper model to use | From config |
| `--output-dir` | Output directory for transcriptions | `transcriptions` |
| `--num-speakers` | Number of speakers for diarization | Auto-detect |
| `--no-diarization` | Disable speaker diarization | False |
| `--batch` | Process all audio files in directory | False |
| `--pattern` | File pattern for batch processing | `*.wav` |

## 🧪 Testing

### Run Tests

```bash
# Basic tests (works without dependencies)
python3 run_tests.py

# Comprehensive tests (requires dependencies)
python3 test_jetson.py

# Unit tests with pytest (if available)
python3 test_transcriber.py
```

### Test Coverage

- **Import Tests**: Verify all modules can be imported
- **Configuration Tests**: Test configuration loading and validation
- **GPU Memory Tests**: Test GPU memory management
- **Model Loading Tests**: Test model loading with fallback
- **Audio Processing Tests**: Test audio loading and preprocessing
- **Diarization Tests**: Test speaker diarization functionality
- **Integration Tests**: End-to-end transcription pipeline
- **Batch Processing Tests**: Test batch processing functionality

## 🏗️ Architecture

### Core Components

- **`Transcriber`**: Main transcription processor with clean architecture
- **`ModelManager`**: Handles model loading with Whisper-TRT fallback
- **`DiarizationManager`**: Resemblyzer-based speaker diarization
- **`GPUManager`**: GPU memory management and cleanup
- **`AudioProcessor`**: Audio preprocessing and format conversion
- **`SpeakerMerger`**: Merges transcription with speaker information

### Data Flow

1. **Audio Input** → Audio preprocessing and normalization
2. **Model Loading** → Whisper model with TensorRT optimization
3. **Transcription** → GPU-accelerated speech-to-text
4. **Diarization** → Speaker identification and clustering
5. **Merging** → Combine transcription with speaker information
6. **Output** → JSON results with comprehensive metadata

## 📁 File Structure

```
whisper_trt/
├── transcriber.py              # Main transcription system
├── config.py                   # Configuration management
├── continuous_recorder.py      # Continuous audio recording
├── audio/
│   └── preprocess.py          # Audio preprocessing utilities
├── scripts/
│   └── setup_swap.sh          # Swap setup for Jetson Nano
├── test_transcriber.py        # Comprehensive unit tests
├── test_jetson.py             # Jetson-specific tests
├── run_tests.py               # Simple test runner
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
└── README.md                  # This file
```

## 📊 Output Format

Transcription results are saved as JSON files with the following structure:

```json
{
  "timestamp": "2025-01-14T10:30:00",
  "audio_file": "/path/to/audio.wav",
  "model": "medium",
  "transcription": {
    "text": "Transcribed text here",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello world",
        "speaker": "SPEAKER_00",
        "speaker_confidence": 0.95
      }
    ]
  },
  "speaker_segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "speaker": "SPEAKER_00"
    }
  ],
  "merged_segments": [...],
  "gpu_memory_used_gb": 1.2,
  "processing_time_seconds": 15.3,
  "config": {
    "temperature": 0.0,
    "language": "en",
    "task": "transcribe"
  }
}
```

## 🔧 Advanced Configuration

### Custom Transcription Parameters

```python
from transcriber import TranscriptionConfig, Transcriber

# Create custom configuration
config = TranscriptionConfig(
    temperature=0.0,
    no_speech_threshold=0.7,
    logprob_threshold=-1.0,
    compression_ratio=2.6,
    language="en",
    task="transcribe"
)

# Use with transcriber
transcriber = Transcriber("medium", enable_diarization=True)
```

### Programmatic Usage

```python
from transcriber import Transcriber

# Initialize transcriber
transcriber = Transcriber("medium", enable_diarization=True)

# Process single file
result = transcriber.process_file("audio.wav", "output_dir")

# Process batch
results = transcriber.process_batch("recordings/", "output_dir")

# Access results
print(f"Text: {result.text}")
print(f"Speakers: {len(result.speaker_segments)}")
print(f"Processing time: {result.processing_time:.2f}s")
```

## 🐛 Troubleshooting

### Common Issues

1. **No Audio Input**
   - Check microphone connection and permissions
   - Verify audio device selection in logs

2. **Transcription Issues**
   - Verify GPU memory availability
   - Check audio file format and quality
   - Review logs for detailed error messages

3. **Memory Issues**
   - Use smaller models (tiny, base) for limited memory
   - Set up swap space for larger models
   - Monitor GPU memory usage in logs

4. **Model Loading Failures**
   - Check available memory and swap space
   - Verify model files are downloaded
   - Try fallback to smaller models

5. **Performance Issues**
   - Monitor swap usage (slower than RAM)
   - Consider using smaller models
   - Check GPU utilization

### Logging

Logs are stored in `~/.cache/whisper_trt/logs/` and include:
- Audio processing details
- GPU memory usage
- Transcription results
- Error messages and warnings
- System status information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the jetson-containers repository. Please refer to the main repository for licensing information.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the base transcription model
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for GPU optimization
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) for speaker diarization
- [jetson-containers](https://github.com/dusty-nv/jetson-containers) for the container framework