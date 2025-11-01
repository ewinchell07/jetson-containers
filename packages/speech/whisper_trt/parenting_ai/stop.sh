#!/bin/bash

# Parenting AI SMS System Stop Script (Non-Docker)

echo "🛑 Stopping Parenting AI SMS System..."

# Find and kill SMS server processes
if pgrep -f "sms_server.py" > /dev/null; then
    echo "🛑 Stopping SMS server..."
    pkill -f "sms_server.py"
    sleep 2
    
    # Check if still running
    if pgrep -f "sms_server.py" > /dev/null; then
        echo "⚠️  Force killing SMS server..."
        pkill -9 -f "sms_server.py"
        sleep 1
    fi
    
    echo "✅ SMS server stopped"
else
    echo "⚠️  SMS server is not running"
fi

# Check for other related processes
if pgrep -f "monitoring.py" > /dev/null; then
    echo "🛑 Stopping monitoring service..."
    pkill -f "monitoring.py"
    echo "✅ Monitoring service stopped"
fi

echo ""
echo "✅ All services stopped."