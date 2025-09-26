#!/bin/bash
# Simplified FamilyGuide runner - easier to debug

set -e

# Default question
QUESTION="${1:-What's a useful summary of the conversation?}"

echo "🚀 Starting FamilyGuide with question: '$QUESTION'"
echo ""

# First, let's test if we can run the container at all
echo "📦 Testing nano_llm container..."
if jetson-containers run $(autotag nano_llm) --help > /dev/null 2>&1; then
    echo "✅ nano_llm container is available"
else
    echo "❌ nano_llm container not available. Trying to build it..."
    jetson-containers build nano_llm
fi

echo ""
echo "🔍 Running FamilyGuide analysis..."

# Run with explicit volume mounts
jetson-containers run \
    --volume "$(pwd):/workspace" \
    --volume "/home/ethan/jetson-containers/packages/speech/whisper_trt/transcriptions:/data/transcriptions" \
    $(autotag nano_llm) \
    bash -c "
        echo '📁 Available transcript files:'
        ls -la /data/transcriptions/transcriptions_dev/ | head -5
        echo ''
        echo '🤖 Running FamilyGuide...'
        python3 /workspace/family_guide.py \
            '/data/transcriptions/transcriptions_dev' \
            '$QUESTION' \
            --days 7 \
            --max-segments 50
    "
