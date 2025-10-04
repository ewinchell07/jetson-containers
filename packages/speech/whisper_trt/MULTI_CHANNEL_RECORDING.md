# Multi-Channel Audio Recording Guide

This guide covers the multi-channel audio recording capabilities of the Whisper-TRT system, including support for Focusrite 4i4 and other multi-channel audio interfaces.

## 🎤 Overview

The system supports recording from multiple microphones simultaneously, with automatic channel splitting and individual file output. This is particularly useful for:

- **Podcast Recording**: Multiple hosts/guests with separate microphones
- **Interview Recording**: Individual microphone tracks for each participant
- **Room Recording**: Multiple microphones for different areas/people
- **Backup Recording**: Redundant audio channels for reliability

## 🔧 Hardware Setup

### Focusrite 4i4 Setup

1. **Connect Audio Interface**:
   - Connect Focusrite 4i4 to Jetson via USB
   - Connect microphones to inputs 1-4
   - Ensure proper gain levels on the interface

2. **Verify Device Detection**:
   ```bash
   # List available audio devices
   arecord -l
   
   # Should show: "Scarlett 4i4 4th Gen: USB Audio"
   ```

3. **Test Audio Input**:
   ```bash
   # Test 4-channel recording
   arecord -D plughw:0,0 -f S32_LE -r 48000 -c 4 -t wav -d 5 test.wav
   
   # Check audio levels
   python3 -c "
   import numpy as np, wave
   with wave.open('test.wav', 'rb') as f:
       data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int32)
       data = data.reshape(-1, 4)
       for ch in range(4):
           rms = np.sqrt(np.mean(data[:, ch]**2))
           print(f'Channel {ch+1}: RMS={rms:.2f}')
   "
   ```

### Channel Mapping

For Focusrite 4i4, the channel mapping is:
- **Channel 1 (FL)**: Front Left - Input 1
- **Channel 2 (FR)**: Front Right - Input 2  
- **Channel 3 (FC)**: Front Center - Input 3
- **Channel 4 (LFE)**: Low Frequency - Input 4

## 🚀 Usage

### Basic Multi-Channel Recording

```bash
# Record 4 channels with automatic splitting
python3 continuous_recorder.py --channels 4

# Record with custom channel names
python3 continuous_recorder.py --channels 4 --channel-names "Host" "Guest" "Room" "Backup"

# Record without channel splitting (single multi-channel file)
python3 continuous_recorder.py --channels 4 --no-separate-channels
```

### Advanced Configuration

```bash
# High-quality recording with custom settings
python3 continuous_recorder.py \
    --channels 4 \
    --sample-rate 48000 \
    --chunk-duration 300 \
    --channel-names "Mic1" "Mic2" "Mic3" "Mic4" \
    --gain 2.0 \
    --buffer-size 8192 \
    --output-dir "/path/to/recordings"
```

### Channel Splitting

```bash
# Split existing 4-channel file (new syntax)
python3 split_channels.py recording_4channel.wav

# With custom output directory
python3 split_channels.py recording_4channel.wav --output-dir ./split_audio

# With custom channel names
python3 split_channels.py recording_4channel.wav --channel-names mic1 mic2 mic3 mic4

# With custom prefix
python3 split_channels.py recording_4channel.wav --prefix my_recording

# This creates:
# - recording_4channel_ch1.wav (Channel 1)
# - recording_4channel_ch2.wav (Channel 2) 
# - recording_4channel_ch3.wav (Channel 3)
# - recording_4channel_ch4.wav (Channel 4)
```

## 📁 Output Files

### Separate Channel Files (Default)

When `--separate-channels` is enabled (default), the system creates individual mono files for each channel:

```
recordings/
├── recording_20250127_143022_Rode_Mic_1.wav
├── recording_20250127_143022_Rode_Mic_2.wav
├── recording_20250127_143022_Rode_Mic_3.wav
└── recording_20250127_143022_Rode_Mic_4.wav
```

### Single Multi-Channel File

When `--no-separate-channels` is used, creates one multi-channel file:

```
recordings/
└── recording_20250127_143022.wav  # 4-channel file
```

## ⚙️ Configuration Options

### Audio Settings

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `--channels` | Number of audio channels | `4` | 1-8 |
| `--sample-rate` | Audio sample rate | `16000` | 8000-48000 |
| `--chunk-duration` | Chunk duration (seconds) | `600` | 60-3600 |
| `--buffer-size` | Audio buffer size | `4096` | 1024-16384 |
| `--latency` | Audio latency mode | `high` | low/high |

### Channel Settings

| Option | Description | Default |
|--------|-------------|---------|
| `--save-combined` | Save combined multi-channel file | `False` |
| `--save-individual` | Save individual channel files | `True` |
| `--no-save-individual` | Disable saving individual files | `False` |
| `--channel-names` | Custom channel names | Auto-generated |
| `--apply-gain-to` | Channel indices to apply gain to (0-indexed) | `2 3` (channels 3,4) |

### Audio Processing

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `--amplify` | Enable audio amplification | `True` | - |
| `--gain` | Amplification factor | `1.5` | 1.0-3.0 |
| `--normalize` | Normalize audio levels | `True` | - |
| `--no-amplify` | Disable amplification | `False` | - |
| `--no-normalize` | Disable normalization | `False` | - |

## 🔍 Troubleshooting

### Common Issues

#### 1. Silent Recordings

**Symptoms**: Files are created but contain no audio or very quiet audio.

**Solutions**:
```bash
# Test with arecord first
arecord -D plughw:0,0 -f S32_LE -r 48000 -c 4 -t wav -d 5 test.wav

# Check audio levels
python3 split_channels.py test.wav debug
```

**Causes**:
- Incorrect channel mapping
- Microphone not connected
- Low input levels
- Wrong audio device

#### 2. Device Not Found

**Symptoms**: "No suitable multi-channel device found"

**Solutions**:
```bash
# List available devices
arecord -l

# Check device permissions
ls -la /dev/snd/

# Test specific device
arecord -D hw:0,0 -f S32_LE -r 48000 -c 4 -t wav -d 5 test.wav
```

#### 3. Audio Dropouts

**Symptoms**: Choppy or interrupted audio

**Solutions**:
```bash
# Increase buffer size
python3 continuous_recorder.py --buffer-size 8192

# Use high latency mode
python3 continuous_recorder.py --latency high

# Reduce sample rate
python3 continuous_recorder.py --sample-rate 44100
```

#### 4. Channel Mapping Issues

**Symptoms**: Audio appears on wrong channels

**Solutions**:
- Check physical microphone connections
- Verify Focusrite 4i4 input routing
- Test each channel individually
- Use `arecord` to verify channel mapping

### Debugging Commands

```bash
# Check audio device capabilities
cat /proc/asound/card0/stream0

# Test individual channels
arecord -D plughw:0,0 -f S32_LE -r 48000 -c 1 -t wav -d 5 ch1.wav

# Analyze channel levels
python3 -c "
import numpy as np, wave
with wave.open('test.wav', 'rb') as f:
    data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int32)
    data = data.reshape(-1, 4)
    for ch in range(4):
        rms = np.sqrt(np.mean(data[:, ch]**2))
        max_val = np.max(np.abs(data[:, ch]))
        print(f'Channel {ch+1}: Max={max_val}, RMS={rms:.2f}')
"
```

## 📊 Performance Tips

### Optimize for Stability

```bash
# Use high latency for stability
python3 continuous_recorder.py --latency high

# Increase buffer size
python3 continuous_recorder.py --buffer-size 8192

# Use lower sample rate if needed
python3 continuous_recorder.py --sample-rate 44100
```

### Optimize for Quality

```bash
# Use higher sample rate
python3 continuous_recorder.py --sample-rate 48000

# Enable amplification
python3 continuous_recorder.py --amplify --gain 2.0

# Use normalization
python3 continuous_recorder.py --normalize
```

### Memory Management

```bash
# Shorter chunks for less memory usage
python3 continuous_recorder.py --chunk-duration 300

# Longer chunks for fewer files
python3 continuous_recorder.py --chunk-duration 1800
```

## 🔄 Workflow Examples

### Podcast Recording

```bash
# Record podcast with 4 microphones
python3 continuous_recorder.py \
    --channels 4 \
    --channel-names "Host" "Guest" "Room" "Backup" \
    --chunk-duration 1800 \
    --gain 1.8 \
    --output-dir "podcast_recordings"
```

### Interview Recording

```bash
# Record interview with 2 microphones
python3 continuous_recorder.py \
    --channels 2 \
    --channel-names "Interviewer" "Interviewee" \
    --chunk-duration 600 \
    --output-dir "interview_recordings"
```

### Room Recording

```bash
# Record room with 4 microphones
python3 continuous_recorder.py \
    --channels 4 \
    --channel-names "Front" "Back" "Left" "Right" \
    --chunk-duration 1200 \
    --output-dir "room_recordings"
```

## 📈 Monitoring

### Real-time Monitoring

```bash
# Monitor recording progress
tail -f ~/.cache/whisper_trt/logs/continuous_recorder_*.log

# Check disk space
df -h

# Monitor audio levels
python3 split_channels.py latest_recording.wav monitor
```

### Quality Analysis

```bash
# Analyze channel levels
python3 split_channels.py recording.wav analysis

# Check for silent channels
python3 -c "
import numpy as np, wave, glob
for file in glob.glob('recordings/*.wav'):
    with wave.open(file, 'rb') as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int32)
        rms = np.sqrt(np.mean(data**2))
        print(f'{file}: RMS={rms:.2f}')
"
```

## 🎯 Best Practices

1. **Test First**: Always test with `arecord` before using the Python recorder
2. **Monitor Levels**: Check audio levels regularly during recording
3. **Backup Channels**: Use multiple microphones for redundancy
4. **Regular Chunks**: Use appropriate chunk duration for your use case
5. **Quality Control**: Monitor recording quality and adjust settings as needed
6. **Storage Management**: Plan for storage requirements of multi-channel recordings
7. **Channel Naming**: Use descriptive channel names for easy identification

## 🔧 Advanced Usage

### Custom Channel Processing

```python
from continuous_recorder import ContinuousRecorder, AudioConfig

# Custom configuration
config = AudioConfig(
    channels=4,
    separate_channels=True,
    channel_names=['Mic1', 'Mic2', 'Mic3', 'Mic4'],
    enable_amplification=True,
    gain_boost=2.0,
    normalize_audio=True
)

# Start recording
recorder = ContinuousRecorder(config, "recordings")
recorder.start_recording()
```

### Batch Channel Splitting

```python
from split_channels import split_channels
import glob

# Split multiple files
for file in glob.glob("recordings/*.wav"):
    split_channels(file, f"split_{file.stem}")
```

This guide provides comprehensive coverage of multi-channel audio recording with the Whisper-TRT system. For additional support, refer to the main README.md or check the troubleshooting section.





