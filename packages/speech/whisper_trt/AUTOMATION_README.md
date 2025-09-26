# Whisper-TRT Automation Manager

A comprehensive automation system for continuous audio recording and transcription using Whisper-TRT on Jetson devices.

## 🎯 Overview

This automation system provides:

- **Scheduled Recording**: Records 10-minute audio chunks from 6:30 AM to 8:00 PM daily
- **Health Monitoring**: Automatically detects and restarts recording if issues occur
- **Automated Transcription**: Transcribes all recordings at 8:00 PM using the small Whisper model with 3-speaker diarization
- **Docker Integration**: Runs recording and transcription inside the Whisper-TRT Docker container
- **Systemd Integration**: Runs as a system service with automatic startup and restart
- **Comprehensive Logging**: Detailed logs for monitoring and troubleshooting

## 🐳 Architecture

The system uses a **hybrid architecture**:

- **Automation Manager**: Runs on the host system (outside container)
- **Recording & Transcription**: Run inside the Docker container where all dependencies are available
- **File Sharing**: Uses Docker volumes to share recordings and transcriptions between host and container
- **Container Management**: Automatically starts/stops the container as needed

## 🚀 Quick Start

### 1. Build the Container

```bash
# Build the Whisper-TRT container first
docker build -t whisper-trt:latest .
```

### 2. Setup

```bash
# Make setup script executable and run it
chmod +x setup_automation.sh
sudo ./setup_automation.sh
```

### 3. Start the Service

```bash
# Start the automation service
sudo systemctl start whisper-automation

# Check status
sudo systemctl status whisper-automation

# View logs
sudo journalctl -u whisper-automation -f
```

### 4. Monitor the System

```bash
# Check current status
python3 monitor_automation.py

# View recent activity
python3 monitor_automation.py --activity 12

# Check logs
python3 monitor_automation.py --logs 100
```

## 📋 Features

### Recording Management
- **Schedule**: 6:30 AM to 8:00 PM daily
- **Chunk Duration**: 10 minutes per recording
- **Audio Quality**: 16kHz sample rate, amplified and normalized
- **Health Monitoring**: Checks every 5 minutes for missing recordings
- **Auto-Restart**: Restarts recording if 2+ chunks are missing

### Transcription Management
- **Schedule**: Runs at 8:00 PM daily
- **Model**: Uses small Whisper model for efficiency
- **Diarization**: Identifies 3 speakers automatically
- **Batch Processing**: Transcribes all unprocessed recordings
- **Output**: JSON files with full transcription and speaker information

### System Integration
- **Systemd Service**: Runs as a system service
- **Auto-Start**: Starts automatically on boot
- **Auto-Restart**: Restarts if the service crashes
- **Resource Limits**: Memory and CPU limits for stability
- **Security**: Runs with restricted privileges

## ⏰ Timezone Handling

**Important**: The automation system uses **local time** (not UTC) for all operations:

- **Scheduling**: Recording and transcription schedules use your computer's local time
- **Recording filenames**: Use local time timestamps (e.g., `recording_20250114_063000.wav`)
- **Health monitoring**: Compares local time schedules with local time filenames
- **Transcription timestamps**: Include timezone information in JSON output

This ensures consistency between your schedule preferences and the actual recording files. If you need to change your system timezone, the automation will automatically adjust to the new local time.

## 🐳 Container Configuration

The system can run in two modes:

### Container Mode (Recommended)
- **Recording**: Runs `continuous_recorder.py` inside the Docker container
- **Transcription**: Runs `transcriber.py` inside the Docker container
- **Benefits**: All dependencies available, GPU access, consistent environment
- **Configuration**: Set `use_container: true` in config

### Direct Mode (Development/Testing)
- **Recording**: Runs `continuous_recorder.py` directly on host
- **Transcription**: Runs `transcriber.py` directly on host
- **Benefits**: Easier debugging, no Docker overhead
- **Configuration**: Set `use_container: false` in config

### Container Management
The automation system automatically:
- Starts the container when needed using the same configuration as the README.md
- Stops recording/transcription processes
- Manages container lifecycle
- Mounts necessary volumes for file sharing

**Container Configuration** (matches README.md):
```bash
docker run -d \
    --gpus all \
    --network=host \
    --device /dev/snd \
    --memory=6g \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/opt/whisper_trt \
    -v ~/.cache/whisper_trt/logs:/root/.cache/whisper_trt/logs \
    -v ~/recordings:/opt/whisper_trt/recordings \
    -v ~/transcriptions:/opt/whisper_trt/transcriptions \
    --name whisper-trt \
    whisper-trt:latest
```

**Note**: The automation uses `-d` (detached mode) instead of `-it` (interactive) and adds additional volume mounts for recordings and transcriptions.

## ⚙️ Configuration

The system is configured via `automation_config.yaml`:

```yaml
# Recording Schedule (LOCAL TIME - not UTC)
recording_start_time: "06:30"    # Start recording at 6:30 AM local time
recording_end_time: "20:00"      # Stop recording at 8:00 PM local time
chunk_duration: 600              # 10 minutes per recording chunk

# Recording Settings
output_dir: "~/recordings"       # Directory to save recordings
sample_rate: 16000              # Audio sample rate
amplify: true                   # Enable audio amplification
gain: 1.5                       # Audio gain boost

# Transcription Settings
transcription_model: "small"     # Whisper model
num_speakers: 3                 # Number of speakers
transcription_start_time: "20:00"  # Start transcription at 8:00 PM

# Health Monitoring
health_check_interval: 300      # Check every 5 minutes
max_missing_chunks: 2           # Restart if 2+ chunks missing
```

## 📁 File Structure

```
whisper_trt/
├── automation_manager.py       # Main automation script
├── automation_config.yaml      # Configuration file
├── monitor_automation.py       # Monitoring script
├── setup_automation.sh         # Setup script
├── whisper-automation.service  # Systemd service file
├── continuous_recorder.py      # Recording module
├── transcriber.py              # Transcription module
└── config.py                   # Configuration module
```

## 🎛️ Usage

### Manual Operation

```bash
# Run with custom config
python3 automation_manager.py --config custom_config.yaml

# Run in daemon mode (no console output)
python3 automation_manager.py --daemon

# Check status only
python3 automation_manager.py --status

# Run transcription now
python3 automation_manager.py --transcribe-now
```

### Service Management

```bash
# Start service
sudo systemctl start whisper-automation

# Stop service
sudo systemctl stop whisper-automation

# Restart service
sudo systemctl restart whisper-automation

# Check status
sudo systemctl status whisper-automation

# Enable auto-start on boot
sudo systemctl enable whisper-automation

# Disable auto-start
sudo systemctl disable whisper-automation
```

### Monitoring

```bash
# Basic status
python3 monitor_automation.py

# Detailed activity (last 12 hours)
python3 monitor_automation.py --activity 12

# View logs (last 100 lines)
python3 monitor_automation.py --logs 100

# JSON output for scripts
python3 monitor_automation.py --json
```

## 📊 Output Files

### Recordings
- **Location**: `~/recordings/`
- **Format**: `recording_YYYYMMDD_HHMMSS.wav`
- **Duration**: 10 minutes each
- **Quality**: 16kHz, mono, amplified

### Transcriptions
- **Location**: `~/transcriptions/`
- **Format**: `transcript_recording_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.json`
- **Content**: Full transcription with speaker diarization
- **Structure**:
  ```json
  {
    "timestamp": "2025-01-14T20:30:00",
    "audio_file": "/path/to/recording.wav",
    "model": "small",
    "transcription": {
      "text": "Full transcribed text...",
      "segments": [...]
    },
    "speaker_segments": [...],
    "merged_segments": [...],
    "processing_time_seconds": 45.2
  }
  ```

### Logs
- **Location**: `~/.cache/whisper_trt/logs/`
- **Format**: `automation_YYYYMMDD_HHMMSS.log`
- **Retention**: 30 days (configurable)

## 🔧 Troubleshooting

### Common Issues

1. **No Audio Input**
   ```bash
   # Check audio devices
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   
   # Test recording manually
   python3 continuous_recorder.py --chunk-duration 60
   ```

2. **Service Won't Start**
   ```bash
   # Check service status
   sudo systemctl status whisper-automation
   
   # View detailed logs
   sudo journalctl -u whisper-automation -n 50
   
   # Check configuration
   python3 automation_manager.py --status
   ```

3. **Missing Recordings**
   ```bash
   # Check health status
   python3 monitor_automation.py
   
   # View recent activity
   python3 monitor_automation.py --activity 24
   
   # Check audio permissions
   ls -la /dev/snd/
   ```

4. **Transcription Issues**
   ```bash
   # Test transcription manually
   python3 automation_manager.py --transcribe-now
   
   # Check GPU memory
   nvidia-smi
   
   # Check model files
   ls -la ~/.cache/whisper/
   ```

### Log Analysis

```bash
# View real-time logs
sudo journalctl -u whisper-automation -f

# Search for errors
sudo journalctl -u whisper-automation | grep -i error

# View logs from specific time
sudo journalctl -u whisper-automation --since "2025-01-14 06:00:00"
```

## 🔒 Security

The systemd service runs with restricted privileges:
- **NoNewPrivileges**: Cannot gain additional privileges
- **PrivateTmp**: Private temporary directory
- **ProtectSystem**: Read-only system directories
- **ProtectHome**: Read-only home directory (except allowed paths)
- **Resource Limits**: Memory and CPU limits

## 📈 Performance

### Resource Usage
- **CPU**: ~5-10% during recording, ~50-80% during transcription
- **Memory**: ~2-4GB during transcription (depends on model)
- **Storage**: ~10MB per 10-minute recording, ~1MB per transcription
- **GPU**: Used during transcription only

### Optimization Tips
1. Use SSD storage for better I/O performance
2. Ensure adequate swap space for larger models
3. Monitor disk space (recordings accumulate quickly)
4. Consider using smaller models for faster transcription

## 🔄 Maintenance

### Regular Tasks
- **Monitor disk space**: Recordings can accumulate quickly
- **Check logs**: Review logs weekly for issues
- **Update models**: Periodically update Whisper models
- **Backup transcriptions**: Important transcriptions should be backed up

### Cleanup
```bash
# Remove old recordings (older than 30 days)
find ~/recordings -name "*.wav" -mtime +30 -delete

# Remove old logs (older than 30 days)
find ~/.cache/whisper_trt/logs -name "*.log" -mtime +30 -delete
```

## 🤝 Support

For issues and questions:
1. Check the logs: `sudo journalctl -u whisper-automation -f`
2. Run diagnostics: `python3 monitor_automation.py`
3. Test components individually
4. Review the configuration file
5. Check system resources (CPU, memory, disk space)

## 📄 License

This automation system is part of the jetson-containers repository. Please refer to the main repository for licensing information.

