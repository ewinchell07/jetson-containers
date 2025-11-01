#!/bin/bash
set -euo pipefail

# Parenting AI SMS System Setup Script
# Sets up Ollama, dependencies, and initializes the system

echo "🤖 Parenting AI SMS System Setup"
echo "================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PARENTING_AI_DIR="$PROJECT_DIR/parenting_ai"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "This script should not be run as root"
    exit 1
fi

# Check Python version
log_info "Checking Python version..."
python3 --version || {
    log_error "Python 3 is required but not installed"
    exit 1
}

# Check if we're in the right directory
if [[ ! -f "$PROJECT_DIR/transcriber.py" ]]; then
    log_error "Please run this script from the whisper_trt directory"
    exit 1
fi

# Step 1: Install Python dependencies
log_info "Installing Python dependencies..."
cd "$PROJECT_DIR"

if [[ -f "requirements_parenting_ai.txt" ]]; then
    pip3 install -r requirements_parenting_ai.txt
    log_success "Python dependencies installed"
else
    log_error "requirements_parenting_ai.txt not found"
    exit 1
fi

# Step 2: Check and install Ollama
log_info "Checking Ollama installation..."

if ! command -v ollama &> /dev/null; then
    log_warning "Ollama not found. Installing..."
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Add to PATH for current session
    export PATH="$PATH:/usr/local/bin"
    
    log_success "Ollama installed"
else
    log_success "Ollama already installed"
fi

# Step 3: Start Ollama service
log_info "Starting Ollama service..."
if ! pgrep -f "ollama serve" > /dev/null; then
    log_info "Starting Ollama server..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
    
    # Check if Ollama started successfully
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        log_success "Ollama server started"
    else
        log_error "Failed to start Ollama server"
        exit 1
    fi
else
    log_success "Ollama server already running"
fi

# Step 4: Download LLM model
log_info "Downloading LLM model (llama3.2:3b-instruct)..."
log_warning "This may take several minutes depending on your internet connection..."

ollama pull llama3.2:3b-instruct

if [[ $? -eq 0 ]]; then
    log_success "LLM model downloaded successfully"
else
    log_error "Failed to download LLM model"
    exit 1
fi

# Step 5: Create environment file template
log_info "Creating environment file template..."
cat > "$PROJECT_DIR/parenting_ai/.env.example" << 'EOF'
# Parenting AI SMS System Environment Variables
# Copy this file to .env and fill in your values

# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_WEBHOOK_URL=https://your-domain.com/sms/incoming

# Optional: Override default settings
# OLLAMA_MODEL=llama3.2:3b-instruct
# SMS_MAX_LENGTH=160
EOF

log_success "Environment template created: parenting_ai/.env.example"

# Step 6: Initialize transcript index
log_info "Initializing transcript index..."
cd "$PROJECT_DIR"

if [[ -d "transcriptions" && $(find transcriptions -name "*.json" | wc -l) -gt 0 ]]; then
    log_info "Found transcript files. Building index..."
    python3 -m parenting_ai.transcript_indexer --verbose
    
    if [[ $? -eq 0 ]]; then
        log_success "Transcript index created successfully"
    else
        log_warning "Failed to create transcript index. You can run it manually later."
    fi
else
    log_warning "No transcript files found. Index will be created when transcripts are available."
fi

# Step 7: Test system
log_info "Testing system components..."

# Test Ollama
log_info "Testing Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log_success "Ollama connection test passed"
else
    log_error "Ollama connection test failed"
fi

# Test Python imports
log_info "Testing Python imports..."
python3 -c "
import sys
sys.path.append('.')
try:
    from parenting_ai.utils import load_config
    from parenting_ai.llm_service import LLMService
    from parenting_ai.rag_engine import RAGEngine
    print('All imports successful')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    log_success "Python imports test passed"
else
    log_error "Python imports test failed"
fi

# Step 8: Create startup scripts
log_info "Creating startup scripts..."

# Start SMS server script
cat > "$PROJECT_DIR/scripts/start_sms_server.sh" << 'EOF'
#!/bin/bash
# Start Parenting AI SMS Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🤖 Starting Parenting AI SMS Server..."
echo "====================================="

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Please copy .env.example to .env and configure your settings"
    echo "   cp .env.example .env"
    echo ""
fi

# Start the server
python3 -m parenting_ai.sms_server --host 0.0.0.0 --port 5000
EOF

chmod +x "$PROJECT_DIR/scripts/start_sms_server.sh"

# Index transcripts script
cat > "$PROJECT_DIR/scripts/index_transcripts.sh" << 'EOF'
#!/bin/bash
# Index transcripts for Parenting AI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "📚 Indexing transcripts for Parenting AI..."
echo "=========================================="

python3 -m parenting_ai.transcript_indexer --verbose
EOF

chmod +x "$PROJECT_DIR/scripts/index_transcripts.sh"

log_success "Startup scripts created"

# Step 9: Final instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Configure your environment:"
echo "   cp parenting_ai/.env.example parenting_ai/.env"
echo "   # Edit parenting_ai/.env with your Twilio credentials"
echo ""
echo "2. Update phone whitelist in parenting_ai/config.yaml"
echo ""
echo "3. Start the SMS server:"
echo "   ./scripts/start_sms_server.sh"
echo ""
echo "4. Configure Twilio webhook:"
echo "   URL: http://your-server-ip:5000/sms/incoming"
echo ""
echo "5. Test the system:"
echo "   curl http://localhost:5000/status"
echo ""
echo "6. Monitor the system:"
echo "   # Optional: Start monitoring dashboard"
echo "   python3 -m parenting_ai.monitoring"
echo ""
echo "For more information, see README_PARENTING_AI.md"
echo ""
log_success "Parenting AI SMS System setup complete!"
