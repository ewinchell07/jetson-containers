# Adaptive Quality-Based Transcription Guide

This guide explains how to use the new adaptive quality-based model selection system for Whisper transcription.

## 🎯 Overview

The adaptive quality system automatically retries transcription with larger models when quality is insufficient, based on configurable quality metrics.

## 🚀 Quick Start

### 1. Enable Adaptive Quality

```bash
# Enable adaptive quality (default: enabled)
export ENABLE_QUALITY_RETRY=true

# Set quality threshold (0.0 = perfect, 1.0 = terrible)
export QUALITY_THRESHOLD=0.3

# Set maximum retries with larger models
export MAX_QUALITY_RETRIES=2
```

### 2. Run Transcription with Adaptive Quality

```bash
# Standard transcription (now uses adaptive quality)
python transcriber.py --input audio.wav

# Or explicitly enable adaptive quality
python transcriber.py --input audio.wav --adaptive-quality
```

### 3. Test the System

```bash
# Test adaptive transcription on a single file
python test_adaptive.py --audio-file audio.wav --verbose

# Analyze quality of existing transcript
python test_adaptive.py --transcript-file transcript.json

# Compare original vs improved transcript
python test_adaptive.py --compare original.json improved.json
```

## 📊 Quality Metrics

The system uses four key metrics to assess transcription quality:

### 1. Average Log Probability (Confidence)
- **What it measures**: Model confidence in transcription
- **Good values**: -0.5 to -1.0
- **Poor values**: Below -1.5
- **Weight**: 40% of overall quality score

### 2. No Speech Probability (False Speech Detection)
- **What it measures**: Likelihood of detecting speech where there's none
- **Good values**: 0.1 to 0.3
- **Poor values**: Above 0.5
- **Weight**: 30% of overall quality score

### 3. Compression Ratio (Repetition)
- **What it measures**: How repetitive the transcription is
- **Good values**: 1.0 to 2.0
- **Poor values**: Above 3.0
- **Weight**: 20% of overall quality score

### 4. Fragmentation (Segment Count)
- **What it measures**: How well-segmented the transcription is
- **Good values**: 3-8 segments per minute
- **Poor values**: Below 2 or above 20 per minute
- **Weight**: 10% of overall quality score

## ⚙️ Configuration Options

### Environment Variables

```bash
# Core settings
ENABLE_QUALITY_RETRY=true          # Enable/disable adaptive quality
QUALITY_THRESHOLD=0.3              # Quality threshold (0.0=perfect, 1.0=terrible)
MAX_QUALITY_RETRIES=2              # Maximum retries with larger models

# Individual metric thresholds
AVG_LOGPROB_THRESHOLD=-1.2         # Log probability threshold
NO_SPEECH_PROB_THRESHOLD=0.4       # No-speech probability threshold
COMPRESSION_RATIO_THRESHOLD=3.0    # Compression ratio threshold
```

### Configuration File

Edit `adaptive_config.yaml` for detailed configuration:

```yaml
quality:
  threshold: 0.3
  max_retries: 2
  metrics:
    avg_logprob_threshold: -1.2
    no_speech_prob_threshold: 0.4
    compression_ratio_threshold: 3.0
  weights:
    logprob: 0.4
    no_speech: 0.3
    compression: 0.2
    fragmentation: 0.1
```

## 🔄 Reprocessing Existing Transcripts

### Analyze Quality of Existing Transcripts

```bash
# Analyze quality of a single transcript
python test_adaptive.py --transcript-file transcript.json

# Analyze all transcripts in a directory
python reprocess_adaptive.py transcriptions/ --analyze-only
```

### Reprocess Low-Quality Transcripts

```bash
# Reprocess transcripts with quality below 0.3
python reprocess_adaptive.py transcriptions/ \
  --quality-threshold 0.3 \
  --min-improvement 0.1 \
  --output-dir improved_transcripts/

# More aggressive reprocessing (higher quality requirements)
python reprocess_adaptive.py transcriptions/ \
  --quality-threshold 0.2 \
  --min-improvement 0.05 \
  --max-retries 3
```

### Batch Reprocessing with Custom Settings

```bash
# For your specific gain issues (more strict)
export QUALITY_THRESHOLD=0.25
export NO_SPEECH_PROB_THRESHOLD=0.3
export COMPRESSION_RATIO_THRESHOLD=2.5

python reprocess_adaptive.py transcriptions/ \
  --quality-threshold 0.25 \
  --min-improvement 0.1
```

## 📈 Recommended Settings for Different Use Cases

### High Quality Requirements (Slower, More Accurate)
```bash
export QUALITY_THRESHOLD=0.2
export MAX_QUALITY_RETRIES=3
export AVG_LOGPROB_THRESHOLD=-1.0
export NO_SPEECH_PROB_THRESHOLD=0.3
export COMPRESSION_RATIO_THRESHOLD=2.5
```

### Fast Processing (Faster, Less Strict)
```bash
export QUALITY_THRESHOLD=0.4
export MAX_QUALITY_RETRIES=1
export AVG_LOGPROB_THRESHOLD=-1.5
export NO_SPEECH_PROB_THRESHOLD=0.5
export COMPRESSION_RATIO_THRESHOLD=4.0
```

### For Your Specific Gain Issues
```bash
# More strict due to gain-related quality issues
export QUALITY_THRESHOLD=0.25
export MAX_QUALITY_RETRIES=2
export AVG_LOGPROB_THRESHOLD=-1.0
export NO_SPEECH_PROB_THRESHOLD=0.3
export COMPRESSION_RATIO_THRESHOLD=2.5
```

## 🔍 Monitoring and Debugging

### View Quality Reports

```bash
# Get detailed quality report for a transcript
python test_adaptive.py --transcript-file transcript.json --verbose

# Compare before/after quality
python test_adaptive.py --compare original.json improved.json
```

### Log Analysis

Quality decisions are logged to:
```
~/.cache/whisper_trt/logs/
```

Key log messages:
- `Quality assessment for {model}: {score:.3f}`
- `Quality acceptable with {model} (score: {score:.3f})`
- `Quality insufficient ({score:.3f}), retrying with {next_model}`

## 📋 Example Workflows

### 1. Reprocess Your Gain-Affected Transcripts

```bash
# Step 1: Analyze current quality
python reprocess_adaptive.py transcriptions/20250917/ --analyze-only

# Step 2: Reprocess with strict quality requirements
export QUALITY_THRESHOLD=0.25
export NO_SPEECH_PROB_THRESHOLD=0.3
export COMPRESSION_RATIO_THRESHOLD=2.5

python reprocess_adaptive.py transcriptions/20250917/ \
  --quality-threshold 0.25 \
  --min-improvement 0.1 \
  --output-dir improved_20250917/
```

### 2. Test New Audio with Adaptive Quality

```bash
# Test with your current gain settings
python test_adaptive.py --audio-file new_recording.wav --verbose

# If quality is poor, the system will automatically retry with larger models
```

### 3. Batch Process Multiple Directories

```bash
# Process all date directories
for dir in transcriptions/202509*; do
  echo "Processing $dir"
  python reprocess_adaptive.py "$dir" \
    --quality-threshold 0.3 \
    --min-improvement 0.1 \
    --output-dir "improved_$(basename $dir)/"
done
```

## 🎯 Expected Results

Based on your analysis, the adaptive quality system should:

1. **Detect Korean text issues**: High no_speech_prob + compression_ratio
2. **Automatically retry with medium.en**: For better accuracy
3. **Preserve good quality**: Skip reprocessing for acceptable transcripts
4. **Improve overall quality**: While maintaining efficiency

## 🚨 Troubleshooting

### Common Issues

1. **"Adaptive transcriber not available"**
   - Ensure `adaptive_transcriber.py` is in the same directory
   - Check Python path

2. **"Audio file not found"**
   - Check audio file paths in transcript JSON
   - Ensure recordings directory exists

3. **"Quality not improving"**
   - Try stricter thresholds
   - Check if larger models are available
   - Verify audio quality

### Performance Tips

1. **Start with small batches** for testing
2. **Monitor GPU memory usage** with larger models
3. **Use appropriate quality thresholds** for your use case
4. **Check logs** for quality decisions

## 📊 Quality Score Interpretation

| Score Range | Quality Level | Action |
|-------------|---------------|---------|
| 0.0 - 0.2   | Excellent     | No retry needed |
| 0.2 - 0.3   | Good          | No retry needed |
| 0.3 - 0.5   | Fair          | Consider retry |
| 0.5 - 0.7   | Poor          | Retry recommended |
| 0.7 - 1.0   | Terrible      | Retry required |

## 🔧 Advanced Configuration

For custom quality assessment, modify the weights in `adaptive_config.yaml`:

```yaml
# For your gain issues, emphasize false speech detection
weights:
  no_speech: 0.4      # Higher weight on false speech
  compression: 0.3    # Higher weight on repetition
  logprob: 0.2         # Lower weight on confidence
  fragmentation: 0.1
```

This configuration is optimized for detecting the specific quality issues you observed with the increased gain setting.



