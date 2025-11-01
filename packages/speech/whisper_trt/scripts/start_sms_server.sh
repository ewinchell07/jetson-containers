#!/bin/bash
# Start Parenting AI SMS Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🤖 Starting Parenting AI SMS Server..."
echo "====================================="

# Check if .env file exists
if [[ ! -f "parenting_ai/.env" ]]; then
    echo "⚠️  Warning: parenting_ai/.env file not found"
    echo "   Please copy .env.example to .env and configure your settings"
    echo "   cp parenting_ai/.env.example parenting_ai/.env"
    echo ""
fi

# Start the server
python3 -m parenting_ai.sms_server --host 0.0.0.0 --port 5000
