#!/bin/bash
# Stop Parenting AI SMS Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🛑 Stopping Parenting AI SMS Server..."
echo "====================================="

# Find and kill all parenting_ai processes
echo "🔍 Looking for running parenting_ai processes..."

# Kill by process name
pkill -f "parenting_ai.sms_server" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Killed parenting_ai.sms_server processes"
else
    echo "ℹ️  No parenting_ai.sms_server processes found"
fi

# Kill by port (if something is using port 5000)
PORT_PID=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "🔍 Found process using port 5000 (PID: $PORT_PID)"
    kill -TERM $PORT_PID 2>/dev/null
    sleep 2
    # Force kill if still running
    if kill -0 $PORT_PID 2>/dev/null; then
        echo "⚠️  Force killing process $PORT_PID"
        kill -KILL $PORT_PID 2>/dev/null
    fi
    echo "✅ Freed port 5000"
else
    echo "ℹ️  Port 5000 is free"
fi

# Check for any remaining processes
REMAINING=$(ps aux | grep -E "(parenting_ai|sms_server)" | grep -v grep | grep -v "stop_sms_server")
if [ ! -z "$REMAINING" ]; then
    echo "⚠️  Some processes may still be running:"
    echo "$REMAINING"
    echo ""
    echo "To force kill all remaining processes:"
    echo "pkill -f parenting_ai"
    echo "pkill -f sms_server"
else
    echo "✅ All parenting_ai processes stopped"
fi

echo ""
echo "📊 Final status:"
echo "Port 5000: $(lsof -ti:5000 >/dev/null 2>&1 && echo "IN USE" || echo "FREE")"
echo "Processes: $(ps aux | grep -E "(parenting_ai|sms_server)" | grep -v grep | wc -l) running"

echo ""
echo "🛑 SMS Server shutdown complete!"
