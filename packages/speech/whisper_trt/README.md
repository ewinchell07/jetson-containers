# Whisper-TRT Speech Transcription System

A robust speech transcription system using Whisper-TRT (TensorRT optimized) with real-time audio processing capabilities.

## Prerequisites

- NVIDIA Jetson device with JetPack 5.1 or later
- Docker installed
- NVIDIA Container Toolkit installed
- Audio input device (microphone)

## Building the Container

```bash
# Build the container
docker build -t whisper-trt:latest .
```

## Running the Container

To run the container with GPU access and sound device support:

```bash
docker run --gpus all -it \
    --network=host \
    --device /dev/snd \
    -v $(pwd):/opt/whisper_trt \
    -v ~/.cache/whisper_trt/logs:/root/.cache/whisper_trt/logs \
    --name whisper-trt \
    whisper-trt:latest
```

To view container logs:
```bash
# View logs of running container
docker logs whisper-trt

# View logs of stopped container
docker logs whisper-trt 2>&1 | tee whisper-trt.log

# Follow logs in real-time
docker logs -f whisper-trt
```

To restart a stopped container:
```bash
docker start -i whisper-trt
```

## Usage

The system supports multiple modes of operation:

### 1. Simple Continuous Recording (Recommended for Memory Efficiency)

For lightweight, memory-efficient continuous audio recording without transcription:

```bash
# Basic usage with auto-detected device settings
python3 continuous_recorder.py --chunk-duration 600 --output-dir recordings

# Optimized for stability (recommended for most devices)
python3 continuous_recorder.py --chunk-duration 600 --buffer-size 4096 --latency high

# For high-end devices experiencing buffer overflows
python3 continuous_recorder.py --chunk-duration 600 --buffer-size 8192 --latency high
```

Options:
- `--chunk-duration`: Duration of each audio chunk in seconds (default: 600)
- `--sample-rate`: Target sample rate for transcription files (default: 16000)
- `--output-dir`: Output directory for recordings (default: recordings)
- `--device`: Audio input device ID (default: auto-detect)
- `--buffer-size`: Audio buffer size for stability (default: 4096, increase if getting overflows)
- `--latency`: Audio latency mode - 'high' for stability, 'low' for responsiveness (default: high)

**Key Features:**
- **Auto Sample Rate Detection**: Automatically detects and uses your device's optimal sample rate
- **Dual File Output**: Saves both original quality (e.g., 48kHz) and transcription-ready (16kHz) versions
- **Smart Device Compatibility**: Works with various audio devices including Wireless GO II, USB mics, etc.
- **Buffer Overflow Protection**: Optimized settings to prevent audio dropouts
- **Memory Efficient**: Lightweight design avoids memory issues of the full transcription system

**File Output:**
- Original: `recording_YYYYMMDD_HHMMSS_48k.wav` (native device quality)
- Transcription: `recording_YYYYMMDD_HHMMSS.wav` (16kHz for Whisper compatibility)

### 2. Simple Transcription (Recommended for Processing Audio Files)

For focused, memory-efficient transcription of audio files without the complexity of the full system:

```bash
# Single file transcription
python3 transcriber.py audio.wav --model base.en

# With speaker diarization
python3 transcriber.py audio.wav --model base.en --hf-token YOUR_HF_TOKEN --num-speakers 2

# Batch processing of multiple files
python3 transcriber.py recordings/ --batch --model base.en --pattern "*.wav"
```

Options:
- `audio_file`: Audio file to transcribe (or directory for batch mode)
- `--model`: Whisper model to use (tiny.en, base.en, small.en, medium.en, large) (default: base.en)
- `--output-dir`: Output directory for transcriptions (default: transcriptions)
- `--hf-token`: HuggingFace token for speaker diarization
- `--num-speakers`: Number of speakers for diarization
- `--batch`: Process all audio files in the specified directory
- `--pattern`: File pattern for batch processing (default: *.wav)

This transcriber is optimized for processing existing audio files with proper memory management and essential configuration options. Advanced settings can be edited as constants in the script file.

### 3. Single File Transcription (Full System)

```bash
python3 simple_transcribe.py --mode file --audio_file /path/to/your/audio.wav --model tiny.en
```

Options:
- `--mode`: Set to 'file' for single file transcription
- `--audio_file`: Path to the audio file to transcribe
- `--model`: Whisper model to use (tiny.en, base.en, small.en, medium.en, large)
- `--language`: Language code for transcription (default: 'en')
- `--temperature`: Sampling temperature (0.0 = deterministic, higher = more random)
- `--no_speech_threshold`: Threshold for silence detection (0.0-1.0, default: 0.3)
- `--logprob_threshold`: Minimum log probability for tokens (default: -0.7)
- `--compression_ratio`: Maximum compression ratio for segments (default: 1.8)
- `--initial_prompt`: Optional initial prompt to guide transcription
- `--hf_token`: HuggingFace token for speaker diarization
- `--num_speakers`: Number of speakers for diarization (default: 2)

### 4. Continuous Recording and Transcription (Full System)

```bash
python3 simple_transcribe.py --mode continuous --model tiny.en --chunk_duration 600
```

Options:
- `--mode`: Set to 'continuous' for real-time recording
- `--model`: Whisper model to use (tiny.en, base.en, small.en, medium.en, large)
- `--chunk_duration`: Duration of audio chunks in seconds (default: 300)
- `--language`: Language code for transcription (default: 'en')
- `--temperature`: Sampling temperature (0.0 = deterministic, higher = more random)
- `--no_speech_threshold`: Threshold for silence detection (0.0-1.0, default: 0.3)
- `--logprob_threshold`: Minimum log probability for tokens (default: -0.7)
- `--compression_ratio`: Maximum compression ratio for segments (default: 1.8)
- `--initial_prompt`: Optional initial prompt to guide transcription
- `--hf_token`: HuggingFace token for speaker diarization
- `--num_speakers`: Number of speakers for diarization (default: 2)
- `--record_only`: Flag to only record audio without transcription (useful for collecting data or reducing resource usage)

### 5. Batch Processing (Full System)

Process multiple recordings within a specified time range:

```bash
python3 simple_transcribe.py --mode batch \
    --start_time "2025-05-30 22:00:00" \
    --end_time "2025-05-31 00:00:00" \
    --model base.en \
    --hf_token YOUR_HF_TOKEN \
    --num_speakers 2
```

Options:
- `--mode`: Set to 'batch' for processing multiple recordings
- `--start_time`: Start time for processing (format: YYYY-MM-DD HH:MM:SS)
- `--end_time`: End time for processing (format: YYYY-MM-DD HH:MM:SS)
- `--model`: Whisper model to use (tiny.en, base.en, small.en, medium.en, large)
- `--language`: Language code for transcription (default: 'en')
- `--temperature`: Sampling temperature (0.0 = deterministic, higher = more random)
- `--no_speech_threshold`: Threshold for silence detection (0.0-1.0, default: 0.3)
- `--logprob_threshold`: Minimum log probability for tokens (default: -0.7)
- `--compression_ratio`: Maximum compression ratio for segments (default: 1.8)
- `--initial_prompt`: Optional initial prompt to guide transcription
- `--hf_token`: HuggingFace token for speaker diarization
- `--num_speakers`: Number of speakers for diarization (default: 2)

Features:
- Processes all recordings within the specified time range
- Skips files that already have transcripts
- Processes files in chronological order
- Handles errors for individual files without stopping the entire batch
- Cleans up resources after each file
- Shows progress and completion status

### Recommended Settings

For optimal performance in different scenarios:

1. **Ultra-Low Resource Usage (Record Only)**:
```bash
python3 simple_transcribe.py --mode continuous --record_only --chunk_duration 600
```
*Note: No model parameter needed in record-only mode - uses optimized SimpleAudioRecorder*

2. **High Accuracy Transcription**:
```bash
python3 simple_transcribe.py --mode continuous --model base.en --chunk_duration 600 --no_speech_threshold 0.3 --logprob_threshold -0.7
```

3. **Speaker Diarization**:
```bash
python3 simple_transcribe.py --mode continuous --model base.en --chunk_duration 600 --hf_token YOUR_HF_TOKEN --num_speakers 2
```

4. **Parent-Child Conversation**:
```bash
python3 simple_transcribe.py --mode continuous --model base.en --chunk_duration 600 --initial_prompt "This is a conversation between parents and children in a home environment." --num_speakers 2
```

5. **Batch Processing with Diarization**:
```bash
python3 simple_transcribe.py --mode batch \
    --start_time "2025-05-30 22:00:00" \
    --end_time "2025-05-31 00:00:00" \
    --model base.en \
    --hf_token YOUR_HF_TOKEN \
    --num_speakers 2 \
    --initial_prompt "This is a conversation between parents and children in a home environment."
```

### Record-Only Mode

The `--record_only` flag enables a **highly optimized recording mode** that eliminates all transcription overhead:

#### **Performance Benefits:**
- **Minimal memory usage** - No audio queues or processing buffers
- **No GPU operations** - No CUDA initialization or memory cleanup
- **Reduced CPU overhead** - No transcription processing threads
- **Stable long-term recording** - Less memory fragmentation
- **Faster startup** - Skips CUDA compatibility checks

#### **Use Cases:**
1. **Data Collection**:
   - Record audio without transcription overhead
   - Save raw audio files for later processing
   - Reduce resource usage during recording

2. **Resource Management**:
   - Minimize memory usage
   - Reduce GPU utilization
   - Prevent OOM (Out of Memory) errors

3. **Long-term Recording**:
   - Extended audio capture sessions
   - High-quality audio preservation
   - Testing audio quality

#### **Technical Implementation:**
- Uses `SimpleAudioRecorder` class (no `TranscriptionManager`)
- **Direct file saving** - Audio chunks saved immediately to disk
- **No queue processing** - Eliminates memory buildup
- **Immediate cleanup** - Audio data deleted after each chunk
- **Single cleanup message** - No duplicate "Recording stopped" logs

#### **File Output:**
- Audio files saved in WAV format
- Files named with timestamps: `recording_YYYYMMDD_HHMMSS.wav`
- Partial chunks saved as `recording_YYYYMMDD_HHMMSS_partial.wav` on shutdown
- No transcription or diarization files created
- Can be processed later using batch mode

### Audio Configuration

The system uses the following default audio settings:
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Format: 16-bit PCM
- Chunk duration: 300 seconds (5 minutes)
- Block size: 1024 samples
- Latency: Low

## Features

- **GPU-Accelerated Transcription**
  - Uses Whisper-TRT for optimized performance
  - Falls back to regular Whisper model if TensorRT model fails
  - Automatic CUDA memory management
  - TF32 support for better performance on compatible GPUs

- **Real-time Audio Processing**
  - Continuous recording with configurable chunk duration
  - Immediate transcription of audio chunks
  - Automatic audio validation and preprocessing
  - Support for multiple audio input devices

- **Optimized Record-Only Mode**
  - **SimpleAudioRecorder** class for minimal overhead
  - Direct file saving without queue processing
  - No GPU operations or CUDA initialization
  - Stable long-term recording with minimal memory usage

- **Robust Error Handling**
  - Automatic model fallback mechanisms
  - Comprehensive error logging
  - Audio validation and quality checks
  - Memory management and cleanup

- **Output and Logging**
  - Detailed transcription results in JSON format
  - Comprehensive logging with timestamps
  - GPU memory usage monitoring (transcription mode only)
  - Audio statistics and quality metrics

## Architecture

### **Record-Only Mode (Optimized)**
- **SimpleAudioRecorder**: Direct audio recording without transcription overhead
- **No queues**: Audio saved directly to disk
- **No GPU operations**: Skips CUDA initialization and memory cleanup
- **Minimal memory footprint**: Immediate cleanup after each chunk
- **Single cleanup message**: No duplicate logging

### **Transcription Mode (Full Feature)**
- **TranscriptionManager**: Handles audio recording, processing, and transcription
- **Queue-based processing**: Audio chunks processed through queues
- **GPU acceleration**: CUDA operations for Whisper models
- **Speaker diarization**: Optional speaker identification
- **Comprehensive cleanup**: Memory management and resource cleanup

## File Structure

- `recordings/`: Directory where audio chunks and transcriptions are saved
- `~/.cache/whisper_trt/logs/`: Directory containing detailed logs
- `transcript_*.json`: Transcription results in JSON format (transcription mode only)
- `recording_*.wav`: Recorded audio chunks
- `recording_*_partial.wav`: Partial audio chunks saved on shutdown (record-only mode)

## Notes

- The container uses the NVIDIA L4T PyTorch base image for optimal performance
- Sound device access is enabled through the `--device /dev/snd` flag
- GPU access is enabled through the `--gpus all` flag
- The current directory is mounted to `/opt/whisper_trt` in the container
- Audio chunks are processed immediately after recording
- System automatically handles audio resampling to 16kHz
- Memory usage is optimized for Jetson devices

## Troubleshooting

1. **No Audio Input**
   - Check if the microphone is properly connected
   - Verify audio device permissions in the container
   - Check audio input device selection in logs

2. **Transcription Issues**
   - Verify GPU memory availability
   - Check audio file format and quality
   - Review logs for detailed error messages

3. **Performance Issues**
   - **For memory issues**: Use `--record_only` mode for minimal overhead
   - **For transcription**: Monitor GPU memory usage in logs
   - **For long recordings**: Consider using a smaller model (tiny.en)
   - **For stability**: Adjust chunk duration if needed

4. **Audio Recording Issues (continuous_recorder.py)**
   - **"Input overflow" warnings**: Increase buffer size (`--buffer-size 8192`) or use high latency (`--latency high`)
   - **"Invalid sample rate" errors**: Script auto-detects supported rates, but manually specify device if needed (`--device N`)
   - **"Invalid flag" errors**: Script uses compatible settings for all PortAudio versions
   - **No audio recorded**: Check device permissions and that microphone is not in use by other applications

4. **Multiple "Recording stopped" Messages**
   - **Fixed in record-only mode**: Uses optimized SimpleAudioRecorder
   - **For transcription mode**: Normal behavior due to multiple cleanup processes

## Logging

Logs are stored in `~/.cache/whisper_trt/logs/` with timestamps. They include:
- Audio processing details
- GPU memory usage
- Transcription results
- Error messages and warnings
- System status information
