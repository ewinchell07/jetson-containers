# Quick Reference Guide

## 🎤 Multi-Channel Recording Commands

### Basic Recording
```bash
# Record 4 Rode microphones
python3 continuous_recorder.py --channels 4

# Record with custom names
python3 continuous_recorder.py --channels 4 --channel-names "Host" "Guest" "Room" "Backup"
```

### Channel Splitting
```bash
# Split 4-channel file into individual mono files (new syntax)
python3 split_channels.py recording.wav

# With custom output directory
python3 split_channels.py recording.wav --output-dir ./split_audio

# With custom channel names
python3 split_channels.py recording.wav --channel-names mic1 mic2 mic3 mic4

# With custom prefix
python3 split_channels.py recording.wav --prefix my_recording
```

### Testing Audio
```bash
# Test 4-channel recording
arecord -D plughw:0,0 -f S32_LE -r 48000 -c 4 -t wav -d 5 test.wav

# Check audio levels
python3 split_channels.py test.wav debug
```

## 🔧 Common Configurations

### Focusrite 4i4 Setup
```bash
# Optimal settings for Focusrite 4i4
python3 continuous_recorder.py \
    --channels 4 \
    --sample-rate 48000 \
    --buffer-size 8192 \
    --latency high \
    --gain 1.5
```

### High-Quality Recording
```bash
# Maximum quality settings
python3 continuous_recorder.py \
    --channels 4 \
    --sample-rate 48000 \
    --gain 2.0 \
    --normalize \
    --chunk-duration 300 \
    --save-combined \
    --save-individual
```

### Stable Recording
```bash
# Stable settings for long recordings
python3 continuous_recorder.py \
    --channels 4 \
    --buffer-size 8192 \
    --latency high \
    --chunk-duration 1800
```

## 🐛 Quick Troubleshooting

### Silent Recordings
```bash
# 1. Test with arecord
arecord -D plughw:0,0 -f S32_LE -r 48000 -c 4 -t wav -d 5 test.wav

# 2. Check levels
python3 split_channels.py test.wav debug

# 3. Verify device
arecord -l
```

### Audio Dropouts
```bash
# Increase buffer size
python3 continuous_recorder.py --buffer-size 8192

# Use high latency
python3 continuous_recorder.py --latency high
```

### Device Not Found
```bash
# List devices
arecord -l

# Check permissions
ls -la /dev/snd/
```

## 📁 Output Files

### Separate Channels (Default)
```
recordings/
├── recording_20250127_143022_Rode_Mic_1.wav
├── recording_20250127_143022_Rode_Mic_2.wav
├── recording_20250127_143022_Rode_Mic_3.wav
└── recording_20250127_143022_Rode_Mic_4.wav
```

### Single Multi-Channel File
```bash
# Use --no-separate-channels
python3 continuous_recorder.py --channels 4 --no-separate-channels
```

## 🎯 Channel Mapping

### Focusrite 4i4
- **Channel 1 (FL)**: Front Left - Input 1
- **Channel 2 (FR)**: Front Right - Input 2  
- **Channel 3 (FC)**: Front Center - Input 3
- **Channel 4 (LFE)**: Low Frequency - Input 4

### Rode Wireless Mics
- **Channel 1**: First Rode mic
- **Channel 2**: Second Rode mic
- **Channel 3**: Third Rode mic
- **Channel 4**: Fourth Rode mic

## ⚡ Performance Tips

### For Stability
- Use `--latency high`
- Increase `--buffer-size` to 8192
- Use `--chunk-duration 1800` for longer chunks

### For Quality
- Use `--sample-rate 48000`
- Enable `--amplify` with `--gain 2.0`
- Use `--normalize` for consistent levels

### For Storage
- Use `--chunk-duration 300` for shorter files
- Use `--no-separate-channels` for single files
- Monitor disk space with `df -h`

## 🔍 Monitoring

### Check Recording Status
```bash
# View logs
tail -f ~/.cache/whisper_trt/logs/continuous_recorder_*.log

# Check disk space
df -h

# Monitor audio levels
python3 split_channels.py latest_recording.wav monitor
```

### Analyze Recordings
```bash
# Check all channel levels
python3 -c "
import numpy as np, wave, glob
for file in glob.glob('recordings/*.wav'):
    with wave.open(file, 'rb') as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int32)
        rms = np.sqrt(np.mean(data**2))
        print(f'{file}: RMS={rms:.2f}')
"
```

## 📋 Command Reference

### continuous_recorder.py
```bash
python3 continuous_recorder.py [OPTIONS]

Options:
  --channels INT              Number of audio channels (default: 4)
  --chunk-duration INT        Chunk duration in seconds (default: 600)
  --sample-rate INT           Audio sample rate (default: 16000)
  --output-dir STR            Output directory (default: ~/recordings)
  --channel-names STR...     Names for each channel
  --separate-channels         Save each channel separately (default: True)
  --no-separate-channels      Save as single multi-channel file
  --amplify                   Enable audio amplification (default: True)
  --gain FLOAT                Amplification factor (default: 1.5)
  --normalize                 Normalize audio levels (default: True)
  --buffer-size INT           Audio buffer size (default: 4096)
  --latency {low,high}        Audio latency mode (default: high)
```

### split_channels.py
```bash
python3 split_channels.py INPUT_FILE [OPTIONS]

Options:
  --output-dir DIR           Output directory (default: current directory)
  --channel-names NAMES...   Names for each channel
  --prefix PREFIX           Prefix for output filenames
  --verbose                Enable verbose logging

Examples:
  python3 split_channels.py recording.wav
  python3 split_channels.py recording.wav --output-dir ./split_audio
  python3 split_channels.py recording.wav --channel-names mic1 mic2 mic3 mic4
```

## 🚀 Quick Start Workflow

1. **Test Audio Setup**:
   ```bash
   arecord -D plughw:0,0 -f S32_LE -r 48000 -c 4 -t wav -d 5 test.wav
   ```

2. **Start Recording**:
   ```bash
   python3 continuous_recorder.py --channels 4 --channel-names "Mic1" "Mic2" "Mic3" "Mic4"
   ```

3. **Split Channels** (if needed):
   ```bash
   python3 split_channels.py recording.wav mic
   ```

4. **Check Results**:
   ```bash
   ls -la recordings/
   python3 split_channels.py latest_recording.wav debug
   ```

This quick reference provides the essential commands and configurations for multi-channel audio recording with the Whisper-TRT system.





