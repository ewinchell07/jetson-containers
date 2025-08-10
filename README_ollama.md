# Family AI Analysis System for Mac (Ollama Version)

A Mac-compatible version of the Family AI Analysis System that uses Ollama instead of nano_llm to analyze family conversation transcripts and provide parenting insights.

## Overview

This system analyzes family conversation transcripts to provide:
- 🎯 **Inspiration**: Activity suggestions based on conversation topics
- 📚 **Coaching Recommendations**: Parenting strategies and communication tips
- ⚠️ **Safety Alerts**: Detection of concerning patterns
- 👨‍👩‍👧‍👦 **Family Bonding Suggestions**: Ideas for strengthening family connections

## Prerequisites

- macOS (tested on macOS 12+)
- Python 3.8 or higher
- Ollama installed and running
- Transcription files in JSON format

## Installation

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

Or use Homebrew:
```bash
brew install ollama
```

### 2. Start Ollama Service

```bash
ollama serve
```

### 3. Pull a Language Model

Pull a model suitable for analysis (recommended: llama3.2:3b):
```bash
ollama pull llama3.2:3b
```

Other recommended models:
- `mistral:7b` - Good balance of speed and quality
- `mixtral:8x7b` - Higher quality but slower
- `phi3:medium` - Faster, smaller model

### 4. Install Python Dependencies

```bash
pip install -r requirements_ollama.txt
```

Or install manually:
```bash
pip install requests
```

## Usage

### Basic Usage

Analyze all transcripts in the default directory:
```bash
python family_analysis_ollama.py
```

### Analyze Specific Date

```bash
python family_analysis_ollama.py --date 20250614
```

### Analyze Time Range

Last 24 hours:
```bash
python family_analysis_ollama.py --hours-back 24
```

Specific time range:
```bash
python family_analysis_ollama.py --start-time "2025-06-14 00:00:00" --end-time "2025-06-14 23:59:59"
```

### Custom Transcriptions Directory

```bash
python family_analysis_ollama.py --transcriptions-dir ~/my_recordings/transcripts
```

### Use Different Model

```bash
python family_analysis_ollama.py --model mistral:7b
```

### Save to Specific File

```bash
python family_analysis_ollama.py --output my_analysis_report.md
```

### Verbose Output

```bash
python family_analysis_ollama.py --verbose
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--transcriptions-dir` | Directory containing transcript JSON files | `~/recordings/transcriptions` |
| `--model` | Ollama model to use | `llama3.2:3b` |
| `--ollama-host` | Ollama API host URL | `http://localhost:11434` |
| `--date` | Analyze specific date (YYYYMMDD format) | None |
| `--start-time` | Start time for analysis range | None |
| `--end-time` | End time for analysis range | None |
| `--hours-back` | Analyze last N hours from now | None |
| `--output` | Output file path for report | Auto-generated |
| `--verbose`, `-v` | Enable verbose logging | False |

## Expected Transcript Format

The system expects JSON transcript files with this structure:
```json
{
  "timestamp": "2025-06-14T14:43:44",
  "audio_file": "recording_20250614_144344.wav",
  "transcription": {
    "text": "Full transcript text here..."
  },
  "segments": [
    {
      "speaker": "SPEAKER_01",
      "text": "Segment text",
      "start": 0.0,
      "end": 2.5
    }
  ]
}
```

File naming convention: `transcript_YYYYMMDD_HHMMSS.json`

## Output

The system generates a comprehensive markdown report including:
- Analysis results with family insights
- Metadata about the analysis
- Summary statistics
- Actionable recommendations

Reports are saved as: `family_analysis_ollama_YYYYMMDD_HHMMSS.md`

## Troubleshooting

### "Ollama is not running"
- Start Ollama: `ollama serve`
- Check if it's running: `curl http://localhost:11434/api/tags`

### "No models available"
- Pull a model: `ollama pull llama3.2:3b`
- List available models: `ollama list`

### "No transcription files found"
- Check the transcriptions directory path
- Ensure files match the naming pattern: `transcript_*.json`
- Use `--verbose` flag for detailed logging

### Analysis Takes Too Long
- Use a smaller/faster model like `phi3:medium`
- Reduce the number of transcripts analyzed
- Check Ollama performance: `ollama ps`

## Examples

### Daily Family Review
```bash
# Analyze today's conversations
python family_analysis_ollama.py --date $(date +%Y%m%d)
```

### Weekly Summary
```bash
# Analyze last 7 days
python family_analysis_ollama.py --hours-back 168 --output weekly_summary.md
```

### High-Quality Analysis
```bash
# Use larger model for better insights
python family_analysis_ollama.py --model mixtral:8x7b --verbose
```

## Privacy & Security

- All analysis is performed locally on your Mac
- No data is sent to external servers
- Transcripts remain on your local filesystem
- Ollama runs entirely offline

## Comparison with Original (nano_llm) Version

| Feature | nano_llm (Jetson) | Ollama (Mac) |
|---------|-------------------|--------------|
| Platform | NVIDIA Jetson | macOS |
| LLM Backend | nano_llm | Ollama |
| Model Selection | Limited | Many options |
| Performance | Optimized for edge | Depends on Mac specs |
| Installation | Complex | Simple |

## Contributing

Feel free to submit issues or pull requests to improve the system.

## License

Same as the original jetson-containers project.