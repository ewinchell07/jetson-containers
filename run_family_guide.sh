#!/bin/bash
# Simple wrapper script to run FamilyGuide inside the nano_llm container

set -e

# Default values
TRANSCRIPT_FOLDER="./packages/speech/whisper_trt/transcriptions/transcriptions_dev"
QUESTION="What's a useful summary of the conversation?"
DAYS=7
MAX_SEGMENTS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--folder)
            TRANSCRIPT_FOLDER="$2"
            shift 2
            ;;
        -q|--question)
            QUESTION="$2"
            shift 2
            ;;
        -d|--days)
            DAYS="$2"
            shift 2
            ;;
        -s|--segments)
            MAX_SEGMENTS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -f, --folder PATH     Transcript folder path (default: ./packages/speech/whisper_trt/transcriptions/transcriptions_dev)"
            echo "  -q, --question TEXT   Your question about family dynamics"
            echo "  -d, --days N          Days of transcripts to analyze (default: 7)"
            echo "  -s, --segments N      Max transcript segments to include (default: 20)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -q \"What patterns do you notice in our family conversations?\""
            echo "  $0 -f /path/to/transcripts -q \"How can I help my child express emotions?\" -d 3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if question is provided
if [ -z "$QUESTION" ]; then
    echo "❌ Error: Question is required. Use -q or --question to provide your question."
    echo "Use -h or --help for usage information."
    exit 1
fi

# Convert host path to container path
CONTAINER_TRANSCRIPT_FOLDER="$TRANSCRIPT_FOLDER"

# If it's a relative path, make it absolute
if [[ "$TRANSCRIPT_FOLDER" == ./* ]]; then
    TRANSCRIPT_FOLDER="$(pwd)/$TRANSCRIPT_FOLDER"
fi

# Convert host path to container path
if [[ "$TRANSCRIPT_FOLDER" == *"/packages/speech/whisper_trt/transcriptions"* ]]; then
    # Replace the host path with container path
    CONTAINER_TRANSCRIPT_FOLDER="/data/transcriptions/transcriptions_dev"
elif [[ "$TRANSCRIPT_FOLDER" == *"/packages/speech/whisper_trt/transcriptions/"* ]]; then
    # Handle case where path ends with /
    CONTAINER_TRANSCRIPT_FOLDER="/data/transcriptions/transcriptions_dev"
else
    # For other paths, assume they're already container paths or custom mounts
    CONTAINER_TRANSCRIPT_FOLDER="$TRANSCRIPT_FOLDER"
fi

echo "🚀 Starting FamilyGuide..."
echo "📁 Host transcript folder: $TRANSCRIPT_FOLDER"
echo "📁 Container transcript folder: $CONTAINER_TRANSCRIPT_FOLDER"
echo "❓ Question: $QUESTION"
echo "📅 Analyzing last $DAYS days"
echo "📝 Max segments: $MAX_SEGMENTS"
echo ""

# Run the nano_llm container with FamilyGuide
jetson-containers run \
    --volume "$(pwd):/workspace" \
    --volume "/home/ethan/jetson-containers/packages/speech/whisper_trt/transcriptions:/data/transcriptions" \
    $(autotag nano_llm) \
    python3 /workspace/family_guide.py \
    "$CONTAINER_TRANSCRIPT_FOLDER" \
    "$QUESTION" \
    --days "$DAYS" \
    --max-segments "$MAX_SEGMENTS"
