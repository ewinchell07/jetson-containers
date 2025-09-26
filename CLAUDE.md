# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the jetson-containers project - a modular container build system providing AI/ML packages for NVIDIA Jetson devices. **The repository has been extended with a Family AI system** that implements a "bicycle for families" concept using whisper_trt, nano_llm, and vllm to analyze parent-child interactions and provide parenting insights through asynchronous coaching.

## Family AI System Architecture

### Core Components
- **family_analysis.py**: Main analysis orchestrator that uses nano_llm via jetson-containers to analyze family transcription data and provide parenting insights
- **whisper_trt package**: TensorRT-optimized speech transcription with continuous recording capabilities 
- **nano_llm package**: Lightweight LLM inference for family interaction analysis
- **vllm package**: High-performance LLM serving for more compute-intensive analysis

### System Workflow
1. **Audio Capture**: Continuous recording via whisper_trt with configurable chunk durations
2. **Transcription**: Real-time speech-to-text with speaker diarization using whisperx/pyannote
3. **Analysis**: Batch analysis using nano_llm with family-specific system prompts
4. **Coaching Output**: Structured reports with inspiration, coaching recommendations, safety alerts, and family bonding suggestions

### Key Integration Points
- family_analysis.py uses `./autotag nano_llm` to find compatible containers
- Transcription files are processed from `~/recordings/transcriptions/` directory
- Analysis results are saved as markdown reports with timestamps

## Build System Commands

### Core Container Commands
```bash
# Install the container tools (run once)
bash install.sh

# Build containers for Family AI components
./build.sh whisper_trt
./build.sh nano_llm  
./build.sh vllm

# Build with specific versions
CUDA_VERSION=12.8 PYTORCH_VERSION=2.6 ./build.sh nano_llm

# List available packages
./build.sh --list-packages
./build.sh --show-packages nano_llm
```

### Running Containers
```bash
# Run with automatic image selection
./run.sh $(./autotag nano_llm)
./run.sh $(./autotag whisper_trt)

# Run with CSI camera support (custom extension)
./run.sh --csi2webcam --csi-capture-res=1640x1232@30 $(./autotag whisper_trt)

# Run with volume mounts for data persistence
./run.sh -v ~/recordings:/data/recordings $(./autotag whisper_trt)
```

### Family AI Usage
```bash
# Analyze recent family conversations (24 hours)
python3 family_analysis.py --hours-back 24

# Analyze specific date pattern
python3 family_analysis.py --date 20250805

# Generate custom report
python3 family_analysis.py --output family_report.md --transcriptions-dir ~/recordings/transcriptions --verbose

# Record continuous audio without transcription (memory efficient)
# Inside whisper_trt container:
python3 simple_transcribe.py --mode continuous --record_only --chunk_duration 600
```

## Key Architecture Components

### Container Package System
- **Modular packages** in `packages/` directory organized by category (speech, llm, cv, etc.)
- **Dependency management** via config.py files that specify build chains
- **Version control** through environment variables (CUDA_VERSION, PYTORCH_VERSION, etc.)
- **Auto-tagging system** that finds compatible images based on JetPack/L4T versions

### Build Pipeline
- **build.sh**: Main entry point that launches jetson_containers/build.py
- **run.sh**: Enhanced docker run wrapper with device mounting and CSI camera support
- **autotag**: Script that resolves container compatibility for the current system

### Family AI Integration Architecture
- **Asynchronous analysis**: family_analysis.py processes batched transcription data
- **Container orchestration**: Uses jetson-containers run commands to launch nano_llm for analysis
- **Fallback analysis**: Provides basic keyword analysis when LLM containers are unavailable
- **Safety monitoring**: System prompts include safety alert generation for dangerous conversations

### Package Dependencies
- **nano_llm**: Depends on awq, whisper_trt (on L4T >= 36), includes ROS variants
- **whisper_trt**: TensorRT-optimized Whisper with speaker diarization support
- **vllm**: High-performance serving with flashinfer dependencies for newer versions

## Development Commands

### Testing
```bash
# Test specific packages
./build.sh --skip-tests=all nano_llm  # Skip all tests
./build.sh --skip-tests=intermediate whisper_trt  # Only final tests

# Test Family AI components
python3 packages/speech/whisper_trt/test.py
python3 packages/llm/nano_llm/test.py
```

### Container Management
```bash
# Build without cache
./build.sh --build-flags="--no-cache" nano_llm

# Build multiple containers
./build.sh --multiple nano_llm vllm whisper_trt

# Build with custom base image
./build.sh --base=my_container:latest --name=my_container:pytorch pytorch
```

### Family AI Development
```bash
# Direct transcription testing
# Inside whisper_trt container:
python3 simple_transcribe.py --mode file --audio_file /data/audio/dusty.wav --model base.en

# Batch processing recordings
python3 simple_transcribe.py --mode batch --start_time "2025-01-01 00:00:00" --end_time "2025-01-02 00:00:00"

# Test analysis with specific transcription directory
python3 family_analysis.py --transcriptions-dir ./test_transcriptions --jetson-root /path/to/jetson-containers
```

## Critical File Locations

### Family AI System Files
- `family_analysis.py`: Main family analysis orchestrator
- `packages/speech/whisper_trt/`: Speech transcription package with continuous recording
- `packages/llm/nano_llm/`: Lightweight LLM package for analysis
- `packages/speech/whisper/continuous_record_transcribe.py`: Alternative continuous recording
- `packages/speech/whisperx/continuous_record_transcribe.py`: WhisperX with speaker diarization

### Core Build System Files
- `jetson_containers/build.py`: Core build orchestration logic
- `jetson_containers/container.py`: Container management functionality
- `build.sh`: Main build entry point
- `run.sh`: Enhanced container runner with device support
- `autotag`: Container compatibility resolver

### Configuration Files
- `packages/*/config.py`: Package definitions with dependencies and versions
- `requirements.txt`: Top-level Python dependencies for build system
- `data/nano_llm/presets/*.json`: Pre-configured model presets for nano_llm