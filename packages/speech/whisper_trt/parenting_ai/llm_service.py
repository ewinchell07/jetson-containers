"""
LLM Service for Parenting AI SMS System
Handles Ollama integration with parenting-focused prompts
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

class LLMService:
    """Ollama LLM service wrapper with parenting context"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config['llm']
        self.logger = logging.getLogger("parenting_ai.llm")
        
        # Ollama connection settings
        self.ollama_url = "http://localhost:11434"
        self.model = self.llm_config['model']
        self.temperature = self.llm_config['temperature']
        self.max_tokens = self.llm_config['max_tokens']
        self.timeout = self.llm_config['timeout']
        
        # System prompt for parenting context
        self.system_prompt = self._create_system_prompt()
        
        # Verify Ollama is running (don't raise error, just set flag)
        self.available = self._verify_ollama()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for parenting advice context"""
        return """You are a helpful parenting coach and family advisor. You have access to transcripts of family conversations and can provide evidence-based parenting advice.

Your role:
- Provide supportive, non-judgmental parenting guidance
- Focus on practical, actionable advice
- Recognize patterns in family conversations
- Offer positive reinforcement and encouragement
- Be specific and contextual based on the conversation data provided

Guidelines:
- Always be supportive and understanding
- Provide evidence-based suggestions
- Focus on the child's development and family well-being
- Be specific about what you observed in the conversations
- Offer concrete next steps when appropriate
- Keep responses concise but helpful

Context: You're helping a family with their parenting journey by analyzing their daily conversations and providing insights and advice."""
    
    def _verify_ollama(self) -> bool:
        """Verify Ollama is running and model is loaded. Returns True if loaded, False otherwise."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if our model is available (downloaded)
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                self.logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                self.logger.info("You may need to run: ollama pull llama3.2:3b")
                return False
            
            # Check if model is actually loaded/running
            try:
                ps_response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
                ps_response.raise_for_status()
                running_models = ps_response.json().get('models', [])
                running_model_names = [m.get('name', '') for m in running_models]
                
                if self.model in running_model_names:
                    self.logger.info(f"Ollama verified with model: {self.model} (loaded and ready)")
                    return True
                else:
                    self.logger.warning(f"Model {self.model} exists but is not loaded. Available models: {model_names}")
                    self.logger.info(f"To load the model, run: ollama run {self.model}")
                    self.logger.info("AI service will not use Ollama until model is loaded")
                    return False
            except requests.exceptions.RequestException:
                # If /api/ps fails, assume model needs to be loaded
                self.logger.warning(f"Could not check if model {self.model} is loaded. Assuming not loaded.")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Ollama is not available: {e}")
            self.logger.info("AI service will respond with standard messages. Start Ollama with: ollama serve")
            return False
    
    def generate_response(self, query: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate parenting advice response"""
        # Check if service is available
        if not self.available:
            self.logger.warning("Ollama service is not available - returning standard message")
            return {
                'text': "I'm sorry, but the AI service is not currently running. Please try again later.",
                'error': 'service_unavailable',
                'processing_time': 0,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Format context for prompt
            context_text = self._format_context(context) if context else ""
            
            # Create user prompt
            user_prompt = self._create_user_prompt(query, context_text)
            
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stop": ["Human:", "Assistant:", "\n\n\n"]
                }
            }
            
            # Make request to Ollama
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            response_text = result.get('response', '').strip()
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Generated response in {processing_time:.2f}s")
            
            return {
                'text': response_text,
                'processing_time': processing_time,
                'model': self.model,
                'tokens_generated': len(response_text.split()),
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama request failed: {e}")
            return {
                'text': "I'm having trouble processing your request right now. Please try again in a moment.",
                'error': str(e),
                'processing_time': 0,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM service: {e}")
            return {
                'text': "I encountered an unexpected error. Please try again.",
                'error': str(e),
                'processing_time': 0,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create user prompt with system context"""
        if context:
            return f"""System: {self.system_prompt}

Context from recent family conversations:
{context}

Human: {query}

Assistant:"""
        else:
            return f"""System: {self.system_prompt}

Human: {query}

Assistant:"""
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format conversation context for the prompt"""
        if not context:
            return ""
        
        formatted_context = []
        for item in context:
            timestamp = item.get('timestamp', 'Unknown time')
            speaker = item.get('speaker', 'Unknown')
            text = item.get('text', '')
            location = item.get('location', 'Unknown location')
            
            formatted_context.append(
                f"[{timestamp}] {speaker} in {location}: {text}"
            )
        
        return "\n".join(formatted_context)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test LLM service connection and response"""
        try:
            test_query = "Hello, can you help me with parenting advice?"
            result = self.generate_response(test_query)
            
            return {
                'status': 'success',
                'model': self.model,
                'response_time': result.get('processing_time', 0),
                'test_response': result.get('text', '')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', ''),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            current_model = next((m for m in models if m['name'] == self.model), None)
            
            if current_model:
                return {
                    'name': current_model['name'],
                    'size': current_model.get('size', 0),
                    'modified_at': current_model.get('modified_at', ''),
                    'available': True
                }
            else:
                return {
                    'name': self.model,
                    'available': False,
                    'error': 'Model not found'
                }
        except Exception as e:
            return {
                'name': self.model,
                'available': False,
                'error': str(e)
            }



