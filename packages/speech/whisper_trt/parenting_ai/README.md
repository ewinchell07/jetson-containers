# Parenting AI SMS System

A parenting advice system that provides SMS-based family guidance using AI, RAG (Retrieval-Augmented Generation), and transcript analysis.

## 🎯 Overview

This system provides:
- **SMS Integration**: Twilio webhook for receiving parenting questions via SMS
- **AI Responses**: Ollama-powered LLM for generating contextual parenting advice (optional)
- **RAG Engine**: Semantic search over family conversation transcripts (optional)
- **Transcript Indexing**: Automated indexing of conversation data for retrieval
- **Monitoring**: Health checks and system monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip3
- Twilio account with SMS capabilities
- (Optional) Ollama for LLM functionality
- (Optional) Transcript JSON files in `transcriptions/` directory

### Setup

1. **Run the setup script:**
   ```bash
   cd parenting_ai/
   ./setup.sh
   ```

2. **Configure environment variables:**
   ```bash
   export TWILIO_ACCOUNT_SID="your_account_sid"
   export TWILIO_AUTH_TOKEN="your_auth_token"
   export TWILIO_PHONE_NUMBER="your_twilio_phone_number"
   export TWILIO_WEBHOOK_URL="http://your-server-ip:5000/sms/incoming"
   ```

3. **Edit config.yaml:**
   Update the phone whitelist with your authorized phone numbers:
   ```yaml
   whitelist:
     phones:
       - "+1234567890"  # Your phone number
       - "+0987654321"  # Spouse's phone number
   ```

4. **Start the SMS server:**
   ```bash
   ./start.sh
   ```

### Optional: Install Ollama

For AI-powered responses, install and run Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull the required model
ollama pull llama3.2:3b
```

The SMS server will work without Ollama but will only provide basic responses.

## 📋 Management Commands

### Start Services
```bash
./start.sh
```

### Stop Services
```bash
./stop.sh
```

### Check Status
```bash
./status.sh
```

### View Logs
```bash
tail -f logs/sms_server.log
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TWILIO_ACCOUNT_SID` | Twilio account SID | Yes |
| `TWILIO_AUTH_TOKEN` | Twilio auth token | Yes |
| `TWILIO_PHONE_NUMBER` | Twilio phone number | Yes |
| `TWILIO_WEBHOOK_URL` | Webhook URL for Twilio | Yes |

### Config File (config.yaml)

Key settings:
- `whitelist.phones`: Authorized phone numbers
- `llm.model`: Ollama model name (default: "llama3.2:3b")
- `rag.top_k`: Number of results to retrieve
- `logging.level`: Log level (INFO, DEBUG, etc.)

## 🏗️ Architecture

```
SMS Server (Flask) ←→ Ollama (LLM) ←→ Transcript Indexer (RAG)
     ↓                    ↓                    ↓
Twilio Webhook    Vector Store (FAISS)    Transcript Files
```

## 📁 Directory Structure

```
parenting_ai/
├── sms_server.py          # Main SMS webhook server
├── llm_service.py          # Ollama LLM service wrapper
├── rag_engine.py          # RAG engine for semantic search
├── query_handler.py        # Query processing logic
├── transcript_indexer.py   # Transcript indexing tool
├── config.yaml            # Application configuration
├── requirements.txt       # Python dependencies
├── start.sh               # Start script
├── stop.sh                # Stop script
├── status.sh              # Status check script
├── setup.sh               # Setup script
├── logs/                  # Application logs
├── parenting_ai_index/    # Vector store data
└── transcriptions/        # Transcript JSON files
```

## 🔍 Testing

### Test SMS Server
```bash
# Test endpoint
curl http://localhost:5000/test

# Check status
curl http://localhost:5000/status

# Test health
curl http://localhost:5000/status | python3 -m json.tool
```

### Test Ollama
```bash
# Check if running
curl http://localhost:11434/api/tags

# List models
ollama list
```

## 🐛 Troubleshooting

### SMS Server not starting
```bash
# Check if port is in use
lsof -i :5000

# Check logs
tail -f logs/sms_server.log

# Verify Python dependencies
pip3 install -r requirements.txt
```

### Ollama not connecting
```bash
# Check if Ollama is running
pgrep -f ollama

# Start Ollama
ollama serve

# Verify model is installed
ollama list
```

### Service crashes
- Check system resources: `free -h`, `htop`
- Review logs: `tail -f logs/sms_server.log`
- Check for memory issues on Jetson devices

## 📊 Monitoring

The system includes built-in health checks:
- SMS server health: `http://localhost:5000/status`
- Service status: `./status.sh`
- Resource usage: `./status.sh` (includes CPU/memory)

## 🔒 Security Considerations

1. **Phone Whitelist**: Only whitelisted phone numbers can use the service
2. **Twilio Validation**: All webhook requests are validated using Twilio signatures
3. **Environment Variables**: Sensitive data stored in environment variables

## 📚 Additional Resources

- [Twilio SMS Documentation](https://www.twilio.com/docs/sms)
- [Ollama Documentation](https://ollama.ai/docs)
- [Llama-Index Documentation](https://docs.llamaindex.ai/)

## 💡 Tips for Jetson Devices

- Start with SMS server only (no Ollama) to reduce memory usage
- Monitor system resources regularly
- Consider using swap space for memory-intensive operations
- Use lighter models (llama3.2:3b recommended)
