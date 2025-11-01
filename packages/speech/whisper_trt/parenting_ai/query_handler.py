"""
Query Handler for Parenting AI SMS System
Orchestrates RAG + LLM for processing user queries
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .utils import extract_time_range, format_sms_response
from .rag_engine import RAGEngine
from .llm_service import LLMService

class QueryHandler:
    """Handles user queries by combining RAG and LLM"""
    
    def __init__(self, config: Dict[str, Any], rag_engine: RAGEngine, llm_service: LLMService):
        self.config = config
        self.rag_engine = rag_engine
        self.llm_service = llm_service
        self.logger = logging.getLogger("parenting_ai.query_handler")
        
        # Session management (simple in-memory)
        self.sessions = {}
        self.session_ttl = timedelta(minutes=config['session']['ttl_minutes'])
    
    def process_query(self, query: str, phone_number: str) -> Dict[str, Any]:
        """Process a user query and return formatted response"""
        try:
            # Check for session continuation
            if self._is_continuation_query(query, phone_number):
                return self._handle_continuation(query, phone_number)
            
            # Extract time range from query
            time_filter = extract_time_range(query)
            
            # Search for relevant context
            context = self._search_context(query, time_filter)
            
            # Generate response
            response = self._generate_response(query, context, phone_number)
            
            # Store session for potential follow-up
            self._store_session(phone_number, query, response, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return {
                'text': "I'm having trouble processing your request right now. Please try again.",
                'error': str(e),
                'formatted': format_sms_response(
                    "I'm having trouble processing your request right now. Please try again.",
                    self.config
                )
            }
    
    def _is_continuation_query(self, query: str, phone_number: str) -> bool:
        """Check if this is a continuation query (like 'more')"""
        continuation_keyword = self.config['sms']['continuation_keyword']
        return query.lower().strip() == continuation_keyword.lower()
    
    def _handle_continuation(self, query: str, phone_number: str) -> Dict[str, Any]:
        """Handle continuation queries like 'more'"""
        session = self.sessions.get(phone_number)
        
        if not session:
            return {
                'text': "I don't have a previous query to continue. Please ask a new question.",
                'formatted': format_sms_response(
                    "I don't have a previous query to continue. Please ask a new question.",
                    self.config
                )
            }
        
        # Check if session is still valid
        if datetime.now() - session['timestamp'] > self.session_ttl:
            del self.sessions[phone_number]
            return {
                'text': "Your previous query has expired. Please ask a new question.",
                'formatted': format_sms_response(
                    "Your previous query has expired. Please ask a new question.",
                    self.config
                )
            }
        
        # Return the stored response (could be enhanced to provide more detail)
        return session['response']
    
    def _search_context(self, query: str, time_filter: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search for relevant context using RAG"""
        # Check if RAG engine is available
        if self.rag_engine is None:
            self.logger.debug("RAG engine not available - returning empty context")
            return []
        
        try:
            # Perform semantic search
            results = self.rag_engine.search(query, time_filter)
            
            # Filter by similarity threshold
            threshold = self.config['rag']['similarity_threshold']
            filtered_results = [r for r in results if r.get('score', 0) >= threshold]
            
            self.logger.info(f"Found {len(filtered_results)} relevant segments")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Context search failed: {e}")
            return []
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]], phone_number: str) -> Dict[str, Any]:
        """Generate response using LLM with context"""
        # Check if LLM service is available
        if self.llm_service is None:
            self.logger.warning("LLM service not available - returning standard message")
            return {
                'text': "I'm sorry, but the AI service is not currently running. Please try again later.",
                'formatted': format_sms_response(
                    "I'm sorry, but the AI service is not currently running. Please try again later.",
                    self.config
                ),
                'error': 'service_unavailable',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Format context for LLM
            formatted_context = self._format_context_for_llm(context)
            
            # Generate response
            llm_response = self.llm_service.generate_response(query, formatted_context)
            
            # Check if LLM returned an error (service unavailable)
            if llm_response.get('error') == 'service_unavailable':
                return {
                    'text': llm_response['text'],
                    'formatted': format_sms_response(llm_response['text'], self.config),
                    'error': 'service_unavailable',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Format for SMS
            formatted_response = format_sms_response(llm_response['text'], self.config)
            
            return {
                'text': llm_response['text'],
                'formatted': formatted_response,
                'context_count': len(context),
                'processing_time': llm_response.get('processing_time', 0),
                'model': llm_response.get('model', ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return {
                'text': "I'm having trouble generating a response. Please try again.",
                'formatted': format_sms_response(
                    "I'm having trouble generating a response. Please try again.",
                    self.config
                ),
                'error': str(e)
            }
    
    def _format_context_for_llm(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format context for LLM consumption"""
        formatted_context = []
        
        for item in context:
            formatted_item = {
                'text': item['text'],
                'speaker': item.get('speaker', 'Unknown'),
                'location': item.get('location', 'Unknown'),
                'timestamp': item.get('timestamp', ''),
                'start_time': item.get('start_time', 0),
                'end_time': item.get('end_time', 0),
                'score': item.get('score', 0)
            }
            formatted_context.append(formatted_item)
        
        return formatted_context
    
    def _store_session(self, phone_number: str, query: str, response: Dict[str, Any], context: List[Dict[str, Any]]) -> None:
        """Store session data for potential follow-ups"""
        self.sessions[phone_number] = {
            'query': query,
            'response': response,
            'context': context,
            'timestamp': datetime.now()
        }
        
        # Clean up old sessions
        self._cleanup_sessions()
    
    def _cleanup_sessions(self) -> None:
        """Clean up expired sessions"""
        now = datetime.now()
        expired_phones = []
        
        for phone, session in self.sessions.items():
            if now - session['timestamp'] > self.session_ttl:
                expired_phones.append(phone)
        
        for phone in expired_phones:
            del self.sessions[phone]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'active_sessions': len(self.sessions),
            'session_ttl_minutes': self.config['session']['ttl_minutes'],
            'max_queries_per_session': self.config['session']['max_queries_per_session'],
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_sessions(self) -> None:
        """Clear all sessions"""
        self.sessions.clear()
        self.logger.info("All sessions cleared")



