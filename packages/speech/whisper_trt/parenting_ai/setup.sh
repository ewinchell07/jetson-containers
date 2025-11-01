#!/bin/bash

# Parenting AI SMS System Setup Script (Non-Docker)

set -e

echo "🔧 Setting up Parenting AI SMS System..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip3 is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs parenting_ai_index transcriptions

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if Ollama is installed (optional)
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
else
    echo "⚠️  Ollama is not installed. Install it for LLM functionality:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
fi

# Set permissions
echo "🔐 Setting permissions..."
chmod +x start.sh stop.sh status.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Set environment variables (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, etc.)"
echo "2. Edit config.yaml with your phone whitelist"
echo "3. Run ./start.sh to start the SMS server"
echo "4. Run ./status.sh to check system health"
echo ""
echo "📚 For more information, see README.md"