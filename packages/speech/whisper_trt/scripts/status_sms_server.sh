#!/bin/bash
# Check Parenting AI SMS Server Status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "📊 Parenting AI SMS Server Status"
echo "================================="

# Check for running processes
PROCESSES=$(ps aux | grep -E "(parenting_ai|sms_server)" | grep -v grep | grep -v "status_sms_server")
if [ ! -z "$PROCESSES" ]; then
    echo "🟢 Server is RUNNING"
    echo ""
    echo "Processes:"
    echo "$PROCESSES"
else
    echo "🔴 Server is STOPPED"
fi

echo ""

# Check port 5000
PORT_PID=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "🟢 Port 5000 is IN USE (PID: $PORT_PID)"
else
    echo "🔴 Port 5000 is FREE"
fi

echo ""

# Test server endpoint if running
if [ ! -z "$PORT_PID" ]; then
    echo "🌐 Testing server endpoints..."
    
    # Test status endpoint
    STATUS_RESPONSE=$(curl -s -w "%{http_code}" http://localhost:5000/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        HTTP_CODE="${STATUS_RESPONSE: -3}"
        if [ "$HTTP_CODE" = "200" ]; then
            echo "✅ Status endpoint responding (HTTP $HTTP_CODE)"
        else
            echo "⚠️  Status endpoint returned HTTP $HTTP_CODE"
        fi
    else
        echo "❌ Cannot reach status endpoint"
    fi
else
    echo "ℹ️  Server not running - skipping endpoint test"
fi

echo ""
echo "📋 Quick commands:"
echo "  Start:  ./scripts/start_sms_server.sh"
echo "  Stop:   ./scripts/stop_sms_server.sh"
echo "  Status: ./scripts/status_sms_server.sh"
