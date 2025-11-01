#!/bin/bash

# Parenting AI SMS System Status Script (Non-Docker)

echo "🔍 Parenting AI SMS System Status"
echo "================================="
echo ""

# Check SMS server
echo "📱 SMS Server:"
if pgrep -f "sms_server.py" > /dev/null; then
    SMS_PID=$(pgrep -f "sms_server.py")
    echo "   ✅ Running (PID: $SMS_PID)"
    
    # Test endpoint
    if curl -s http://localhost:5000/test > /dev/null 2>&1; then
        echo "   ✅ Responding to requests"
        
        # Check status endpoint
        echo "   📊 Status:"
        curl -s http://localhost:5000/status 2>/dev/null | python3 -m json.tool 2>/dev/null || \
        curl -s http://localhost:5000/status 2>/dev/null || \
        echo "   ⚠️  Could not fetch status"
    else
        echo "   ⚠️  Running but not responding to requests"
    fi
else
    echo "   ❌ Not running"
fi
echo ""

# Check Ollama (optional)
echo "🤖 Ollama:"
if command -v ollama &> /dev/null; then
    if pgrep -f "ollama" > /dev/null || curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Running"
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "   ✅ Responding to requests"
            MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print('   Models:', ', '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null || echo "   (available)")
        fi
    else
        echo "   ⚠️  Installed but not running"
        echo "   Start with: ollama serve"
    fi
else
    echo "   ❌ Not installed (optional)"
fi
echo ""

# Check monitoring (optional)
echo "📊 Monitoring:"
if pgrep -f "monitoring.py" > /dev/null; then
    MON_PID=$(pgrep -f "monitoring.py")
    echo "   ✅ Running (PID: $MON_PID)"
    if curl -s http://localhost:5001 > /dev/null 2>&1; then
        echo "   ✅ Responding to requests"
    fi
else
    echo "   ⚠️  Not running (optional)"
fi
echo ""

# Show recent logs
echo "📋 Recent SMS server logs:"
if [ -f logs/sms_server.log ]; then
    tail -10 logs/sms_server.log 2>/dev/null || echo "   No recent logs"
else
    echo "   No log file found"
fi
echo ""

# Resource usage
echo "💻 Resource Usage:"
if pgrep -f "sms_server.py" > /dev/null; then
    SMS_PID=$(pgrep -f "sms_server.py")
    ps -p $SMS_PID -o pid,rss,vsz,pcpu,comm 2>/dev/null | tail -1 | awk '{printf "   SMS Server: PID %s, Memory: %.1f MB, CPU: %.1f%%\n", $1, $2/1024, $4}'
fi
echo ""

# Summary
if pgrep -f "sms_server.py" > /dev/null && curl -s http://localhost:5000/test > /dev/null 2>&1; then
    echo "✅ System is operational"
else
    echo "⚠️  System may not be fully operational"
    echo "   Start with: ./start.sh"
fi