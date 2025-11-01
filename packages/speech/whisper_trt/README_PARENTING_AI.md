# 🤖 Parenting AI SMS System

A local prototype parenting advice system that analyzes family conversations and provides insights via SMS. Built for privacy-first, local processing on Jetson Orin Nano Super.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Your Phone    │    │   Wife's Phone  │    │   Twilio SMS    │
│                 │    │                 │    │   Gateway       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     SMS Webhook Server     │
                    │     (Flask + Twilio)       │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Query Handler        │
                    │   (Orchestrates RAG+LLM)  │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────────▼─────────┐
│   RAG Engine     │    │   LLM Service    │    │  Transcript      │
│ (llama-index +   │    │    (Ollama)      │    │   Indexer        │
│  FAISS +         │    │                  │    │                  │
│  Embeddings)     │    │                  │    │                  │
└───────────────────┘    └───────────────────┘    └───────────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Whisper Transcripts    │
                    │    (JSON files from       │
                    │     whisper_trt system)   │
                    └───────────────────────────┘
```

## 📊 Data Flow Diagram

```
SMS Query → Phone Validation → Query Processing
    ↓
Time Range Extraction → RAG Search → Context Retrieval
    ↓
Context + Query → LLM Service → Parenting Advice
    ↓
Response Formatting → SMS Truncation → Twilio Response
```

## 🔍 RAG Flow Explanation

### 1. Transcript Indexing
- **Input**: JSON transcript files from whisper_trt system
- **Processing**: 
  - Parse JSON files to extract segments
  - Create embeddings using sentence-transformers
  - Store in FAISS vector database
- **Output**: Searchable vector index

### 2. Query Processing
- **User Query**: "Tell me about bedtime struggles"
- **Time Extraction**: "yesterday" → last 24 hours
- **Semantic Search**: Find relevant conversation segments
- **Context Assembly**: Format segments with timestamps, speakers, locations

### 3. LLM Generation
- **System Prompt**: Parenting coach persona
- **Context**: Relevant conversation segments
- **User Query**: Original question
- **Response**: Contextual parenting advice

## 🚀 Quick Start

### Prerequisites
- Jetson Orin Nano Super (8GB RAM)
- Python 3.8+
- Docker (for whisper_trt container)
- Twilio account

### 1. Setup System
```bash
# Run the setup script
./scripts/setup_parenting_ai.sh
```

### 2. Configure Environment
```bash
# Copy environment template
cp parenting_ai/.env.example parenting_ai/.env

# Edit with your Twilio credentials
nano parenting_ai/.env
```

### 3. Update Phone Whitelist
```yaml
# Edit parenting_ai/config.yaml
whitelist:
  phones:
    - "+1234567890"  # Your phone
    - "+1234567891"  # Wife's phone
```

### 4. Start SMS Server
```bash
./scripts/start_sms_server.sh
```

### 5. Configure Twilio Webhook
- URL: `http://your-server-ip:5000/sms/incoming`
- Method: POST
- Authentication: Twilio signature validation

## 📱 Usage Examples

### Basic Queries
```
You: "What did we talk about yesterday?"
AI: "Based on yesterday's conversations, I noticed you discussed bedtime routines with the kids. You mentioned that Sarah has been having trouble falling asleep around 8:30 PM..."

You: "more"
AI: "The specific challenges you mentioned were: 1) Sarah asking for water multiple times, 2) Getting out of bed to check on noises, 3) Requesting extra stories..."
```

### Time Range Queries
```
You: "Tell me about homework struggles last week"
AI: "I found several conversations about homework from last week. On Tuesday, you mentioned that Alex was frustrated with math problems..."

You: "What patterns do you see in our conversations?"
AI: "I've noticed a few patterns in your recent conversations: 1) Bedtime routines come up frequently, 2) Homework discussions peak around 4-6 PM..."
```

### Specific Situations
```
You: "When we talked about the playground incident"
AI: "I found your conversation about the playground incident from Tuesday afternoon. You mentioned that Emma had a disagreement with another child about sharing the swing..."
```

## 🔧 Manual Operation Guide

### Starting Services
```bash
# Start SMS server
./scripts/start_sms_server.sh

# Index new transcripts
./scripts/index_transcripts.sh

# Check system status
curl http://localhost:5000/status
```

### Monitoring System
```bash
# Resource monitoring
watch -n 2 'free -h && ps aux | grep -E "ollama|python|whisper"'

# Check logs
tail -f parenting_ai.log

# Optional: Web dashboard
python3 -m parenting_ai.monitoring
```

### Troubleshooting
```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test RAG engine
python3 -c "
from parenting_ai.rag_engine import RAGEngine
from parenting_ai.utils import load_config
config = load_config()
rag = RAGEngine(config)
print(rag.test_search())
"

# Test LLM service
python3 -c "
from parenting_ai.llm_service import LLMService
from parenting_ai.utils import load_config
config = load_config()
llm = LLMService(config)
print(llm.test_connection())
"
```

## 📊 System Requirements

### Memory Usage (Jetson Orin Nano Super)
- **Whisper TRT**: ~2-3GB (peak during transcription)
- **Ollama (3B model)**: ~2GB (always running)
- **llama-index + embeddings**: ~0.8GB (loaded on demand)
- **System overhead**: ~1.9GB base
- **Total peak**: ~7GB RAM + 1-2GB swap during concurrent operations

### Storage Requirements
- **Transcripts**: ~1-5MB per hour of audio
- **Vector Index**: ~100-500MB (depends on transcript volume)
- **Models**: ~2GB (Ollama model)

## 🔒 Privacy & Security

### Local Processing Only
- All LLM inference stays on your device
- Transcripts never leave the Jetson
- Only SMS relay goes through Twilio

### Security Measures
- Phone number whitelist (only your phones)
- Twilio webhook signature validation
- No cloud storage of conversations
- Environment variables for credentials

### Data Flow
```
Your Phone → Twilio → Your Jetson → Local Processing → Response
```

## 🛠️ Configuration

### Environment Variables (.env)
```bash
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_WEBHOOK_URL=https://your-domain.com/sms/incoming
```

### System Settings (config.yaml)
```yaml
# LLM settings
llm:
  model: "llama3.2:3b-instruct"
  temperature: 0.7
  max_tokens: 500

# RAG settings
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 5
  similarity_threshold: 0.7

# SMS formatting
sms:
  max_length: 160
  continuation_keyword: "more"
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Check logs
tail -f /tmp/ollama.log
```

#### 2. Memory Issues
```bash
# Check memory usage
free -h

# Monitor swap usage
swapon --show

# Reduce model size if needed
ollama pull llama3.2:1b-instruct  # Smaller model
```

#### 3. RAG Index Issues
```bash
# Rebuild index
python3 -m parenting_ai.transcript_indexer --rebuild --verbose

# Check index stats
python3 -c "
from parenting_ai.rag_engine import RAGEngine
from parenting_ai.utils import load_config
config = load_config()
rag = RAGEngine(config)
print(rag.get_index_stats())
"
```

#### 4. SMS Not Working
```bash
# Check webhook URL
curl http://localhost:5000/status

# Verify Twilio configuration
# Check .env file has correct credentials
# Verify webhook URL in Twilio console
```

### Log Files
- **Main logs**: `parenting_ai.log`
- **Ollama logs**: `/tmp/ollama.log`
- **SMS server logs**: Console output

### Performance Monitoring
```bash
# Real-time monitoring
watch -n 2 'free -h && ps aux | grep -E "ollama|python"'

# Check system load
htop

# Monitor disk usage
df -h
```

## 🔮 Future Enhancements

- **Time-based filtering**: "Show me conversations from last month"
- **Metadata filtering**: "What did we say in the living room?"
- **Background indexing**: Automatic transcript processing
- **Voice responses**: Twilio voice integration
- **Web dashboard**: Browse transcripts and insights
- **Pattern analysis**: Proactive parenting insights
- **Export reports**: Weekly family conversation summaries

## 📚 Additional Resources

### Related Documentation
- [Whisper TRT README](README.md) - Audio transcription system
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers) - Container framework
- [Ollama Documentation](https://ollama.ai/docs) - LLM service
- [Twilio SMS API](https://www.twilio.com/docs/sms) - SMS integration

### Support
- Check logs first: `tail -f parenting_ai.log`
- Test components individually
- Monitor system resources
- Verify network connectivity

---

**Privacy Note**: This system processes all data locally on your Jetson device. No conversations are sent to external services except for SMS delivery through Twilio.
