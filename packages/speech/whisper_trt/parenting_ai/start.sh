#!/bin/bash

# Parenting AI SMS System Startup Script (Non-Docker)

set -e

echo "🚀 Starting Parenting AI SMS System..."

# Load .env file if it exists
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env..."
    set -a  # automatically export all variables
    source .env
    set +a  # stop automatically exporting
fi

# Check if config.yaml exists
if [ ! -f config.yaml ]; then
    echo "❌ config.yaml not found. Please ensure the configuration file exists."
    exit 1
fi

# Check if required environment variables are set
if [ -z "$TWILIO_ACCOUNT_SID" ] || [ -z "$TWILIO_AUTH_TOKEN" ]; then
    echo "⚠️  Twilio environment variables not set."
    echo "   Set them with:"
    echo "   export TWILIO_ACCOUNT_SID='your_sid'"
    echo "   export TWILIO_AUTH_TOKEN='your_token'"
    echo "   export TWILIO_PHONE_NUMBER='your_number'"
    echo "   export TWILIO_WEBHOOK_URL='http://your-server:5000/sms/incoming'"
    echo "   Or create a .env file with these variables"
    echo ""
    echo "   Continuing anyway (server will fail if Twilio config is needed)..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs parenting_ai_index transcriptions

# Optional: Check and start Ollama if available
echo ""
echo "🤖 Ollama Setup (Optional)"
echo "=========================="
if command -v ollama &> /dev/null; then
    # Check if Ollama server is responding
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        # Ollama server is not running
        echo "⚠️  Ollama server is not running"
        read -p "   Start Ollama server now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "   🚀 Starting Ollama server..."
            ollama serve > logs/ollama.log 2>&1 &
            OLLAMA_PID=$!
            echo "   ✅ Ollama server started (PID: $OLLAMA_PID)"
            echo "   ⏳ Waiting for Ollama to initialize..."
            sleep 5
            
            # Verify Ollama started
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "   ✅ Ollama server is responding"
            else
                echo "   ⚠️  Ollama may still be starting up"
            fi
        else
            echo "   ℹ️  Skipping Ollama startup - AI features will not be available"
        fi
    else
        # Ollama server is running, check for running models
        RUNNING_MODELS=$(ollama ps 2>/dev/null | tail -n +2 | wc -l)
        if [ "$RUNNING_MODELS" -eq 0 ]; then
            echo "⚠️  Ollama server is running but no models are loaded"
            
            # Get available models
            AVAILABLE_MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v "^$" || echo "")
            
            if [ -z "$AVAILABLE_MODELS" ]; then
                echo "   ⚠️  No models available. Pull a model first (e.g., ollama pull llama3.2:3b)"
                echo "   Continuing without Ollama models..."
            else
                echo "   📋 Available models:"
                MODEL_LIST=()
                COUNT=1
                while IFS= read -r model; do
                    if [ -n "$model" ]; then
                        MODEL_LIST+=("$model")
                        echo "      $COUNT) $model"
                        COUNT=$((COUNT + 1))
                    fi
                done <<< "$AVAILABLE_MODELS"
                
                read -p "   Start a model now? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    read -p "   Enter number (1-$((COUNT - 1))): " MODEL_CHOICE
                    if [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]] && [ "$MODEL_CHOICE" -ge 1 ] && [ "$MODEL_CHOICE" -lt "$COUNT" ]; then
                        SELECTED_MODEL="${MODEL_LIST[$((MODEL_CHOICE - 1))]}"
                        echo "   🚀 Preloading model: $SELECTED_MODEL"
                        # Preload model by making a simple API call
                        curl -s -X POST http://localhost:11434/api/generate \
                            -d "{\"model\": \"$SELECTED_MODEL\", \"prompt\": \"test\", \"stream\": false}" \
                            > /dev/null 2>&1 &
                        PRELOAD_PID=$!
                        echo "   ⏳ Model is loading in background..."
                        sleep 2
                        
                        # Check if model is now running
                        sleep 3
                        RUNNING_NOW=$(ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -q "^${SELECTED_MODEL}$" && echo "yes" || echo "no")
                        if [ "$RUNNING_NOW" = "yes" ]; then
                            echo "   ✅ Model $SELECTED_MODEL is loaded and ready"
                        else
                            echo "   ℹ️  Model will be loaded on first request (this may be slower)"
                        fi
                    else
                        echo "   ⚠️  Invalid selection. Continuing without loading a model..."
                    fi
                else
                    echo "   ℹ️  Skipping model startup - server will run without Ollama models"
                fi
            fi
        else
            echo "✅ Ollama is running with $RUNNING_MODELS model(s) loaded"
        fi
    fi
else
    echo "⚠️  Ollama not found (optional - install for AI features)"
    echo "   Install with: curl -fsSL https://ollama.ai/install.sh | sh"
fi

# Optional: Check and build RAG index
echo ""
echo "📚 RAG Index Setup (Optional)"
echo "============================"
if [ -d "parenting_ai_index/storage" ] && [ -n "$(ls -A parenting_ai_index/storage 2>/dev/null)" ]; then
    echo "✅ RAG index exists"
else
    echo "⚠️  No RAG index found"
    
    # Check for transcript files in multiple possible locations
    TRANSCRIPT_DIR=""
    if [ -d "transcriptions" ] && [ -n "$(find transcriptions -name '*.json' -type f 2>/dev/null | head -1)" ]; then
        TRANSCRIPT_DIR="transcriptions"
    elif [ -d "../transcriptions" ] && [ -n "$(find ../transcriptions -name '*.json' -type f 2>/dev/null | head -1)" ]; then
        TRANSCRIPT_DIR="../transcriptions"
    fi
    
    if [ -n "$TRANSCRIPT_DIR" ]; then
        TRANSCRIPT_COUNT=$(find "$TRANSCRIPT_DIR" -name '*.json' -type f 2>/dev/null | wc -l)
        echo "   📄 Found $TRANSCRIPT_COUNT transcript file(s) in $TRANSCRIPT_DIR/"
        read -p "   Build RAG index now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "   🔨 Building RAG index (this may take a few minutes)..."
            python3 transcript_indexer.py --config config.yaml --verbose
            if [ $? -eq 0 ]; then
                echo "   ✅ RAG index built successfully"
            else
                echo "   ⚠️  RAG index build failed - server will attempt to build on startup (may be slow)"
            fi
        else
            echo "   ℹ️  Skipping RAG index build - server will create index on startup (may be slow)"
        fi
    else
        echo "   ⚠️  No transcript files found in transcriptions/ or ../transcriptions/ directories"
        echo "   ℹ️  RAG functionality will not work until transcripts are added"
    fi
fi

# Check if SMS server is already running
if pgrep -f "sms_server.py" > /dev/null; then
    echo "⚠️  SMS server appears to be already running"
    echo "   PID: $(pgrep -f 'sms_server.py')"
    read -p "   Kill existing process and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "sms_server.py" || true
        sleep 2
    else
        echo "   Keeping existing process. Exiting."
        exit 0
    fi
fi

# Start SMS server in background
echo "🚀 Starting SMS server..."
cd "$(dirname "$0")"
nohup python3 sms_server.py --host 0.0.0.0 --port 5000 > logs/sms_server.log 2>&1 &
SMS_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if ps -p $SMS_PID > /dev/null; then
    echo "✅ SMS server started (PID: $SMS_PID)"
    echo "📝 Logs: tail -f logs/sms_server.log"
else
    echo "❌ SMS server failed to start"
    echo "📋 Check logs: cat logs/sms_server.log"
    exit 1
fi

# Test endpoint
echo "🔍 Testing SMS server..."
sleep 2
if curl -s http://localhost:5000/test > /dev/null 2>&1; then
    echo "✅ SMS server is responding"
else
    echo "⚠️  SMS server may not be fully ready yet"
fi

echo ""
echo "✅ Parenting AI SMS System started!"
echo ""
echo "📱 SMS Server: http://localhost:5000"
if command -v ollama &> /dev/null && curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🤖 Ollama: http://localhost:11434 (running)"
fi
if [ -d "parenting_ai_index/storage" ] && [ -n "$(ls -A parenting_ai_index/storage 2>/dev/null)" ]; then
    echo "📚 RAG Index: Available"
fi
echo "📋 To view logs: tail -f logs/sms_server.log"
echo "🛑 To stop: ./stop.sh"
echo "📊 To check status: ./status.sh"