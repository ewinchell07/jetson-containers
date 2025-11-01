"""
SMS Server for Parenting AI SMS System
Flask webhook server for Twilio SMS integration
"""

import os
import logging
import sys
import threading
from pathlib import Path
from flask import Flask, request, jsonify
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client as TwilioClient

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parenting_ai.utils import load_config, setup_logging, is_phone_whitelisted
from parenting_ai.llm_service import LLMService
from parenting_ai.rag_engine import RAGEngine
from parenting_ai.query_handler import QueryHandler

class SMSServer:
    """SMS webhook server for Twilio integration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.urandom(24)
        
        # Initialize services
        self._initialize_services()
        
        # Setup routes
        self._setup_routes()
        
        # Initialize Twilio REST client for active outbound messaging
        account_sid = self.config['twilio'].get('account_sid', '')
        auth_token = self.config['twilio'].get('auth_token', '')
        if account_sid and auth_token:
            try:
                self.twilio_client = TwilioClient(account_sid, auth_token)
                self.twilio_phone_number = self.config['twilio'].get('phone_number', '')
                self.use_active_messaging = True
                self.logger.info("Twilio REST client initialized - outbound messaging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Twilio REST client: {e}")
                self.twilio_client = None
                self.use_active_messaging = False
        else:
            self.twilio_client = None
            self.use_active_messaging = False
            self.logger.warning("Twilio credentials not configured - active outbound messaging disabled")
        
        # Twilio validator (only if validation is enabled)
        validate_requests = self.config['twilio'].get('validate_requests', True)
        if validate_requests and auth_token:
            self.validator = RequestValidator(auth_token)
            self.validate_requests = True
        else:
            self.validator = None
            self.validate_requests = False
            if not validate_requests:
                self.logger.warning("Twilio request validation is DISABLED (testing mode)")
            else:
                self.logger.warning("Twilio auth_token not set - request validation disabled")
    
    def _initialize_services(self):
        """Initialize all required services (Ollama and RAG are optional)"""
        try:
            self.logger.info("Initializing services...")
            
            # Initialize RAG engine (optional)
            try:
                self.rag_engine = RAGEngine(self.config)
                self.logger.info("RAG engine initialized")
            except Exception as e:
                self.logger.warning(f"RAG engine not available: {e}")
                self.logger.info("SMS server will run without RAG functionality")
                self.rag_engine = None
            
            # Initialize LLM service (optional)
            try:
                self.llm_service = LLMService(self.config)
                self.logger.info("LLM service initialized")
            except Exception as e:
                self.logger.warning(f"LLM service not available: {e}")
                self.logger.info("SMS server will run without LLM functionality")
                self.llm_service = None
            
            # Initialize query handler (works with or without RAG/LLM)
            try:
                self.query_handler = QueryHandler(self.config, self.rag_engine, self.llm_service)
                self.logger.info("Query handler initialized")
            except Exception as e:
                self.logger.warning(f"Query handler initialization failed: {e}")
                self.query_handler = None
            
            if self.llm_service or self.rag_engine:
                self.logger.info("Core services initialized (some features may be limited)")
            else:
                self.logger.warning("Running in minimal mode - no AI features available")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            # Don't raise - allow server to start in minimal mode
            self.rag_engine = None
            self.llm_service = None
            self.query_handler = None
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/sms/incoming', methods=['POST'])
        def handle_sms():
            """Handle incoming SMS from Twilio"""
            try:
                # Validate Twilio request (always validate for security)
                if self.validate_requests:
                    if not self._validate_twilio_request(request):
                        # Validation details are logged in _validate_twilio_request
                        return "Unauthorized", 401
                else:
                    self.logger.warning("WARNING: Twilio request validation is DISABLED - not secure!")
                
                # Extract message data
                from_number = request.form.get('From', '')
                message_body = request.form.get('Body', '').strip()
                
                self.logger.info(f"Received SMS from {from_number}: {message_body[:50]}...")
                
                # Check phone whitelist
                if not is_phone_whitelisted(from_number, self.config):
                    self.logger.warning(f"Unauthorized phone number: {from_number}")
                    # Still return quickly, but don't process unauthorized numbers
                    return self._create_response("Sorry, this number is not authorized to use this service.")
                
                # Return immediate response to Twilio (to avoid timeout)
                # Process message asynchronously and send via REST API
                self.logger.info(f"Returning immediate response to Twilio, processing message asynchronously...")
                
                # Start background thread to process and send message
                thread = threading.Thread(
                    target=self._process_and_send_async,
                    args=(message_body, from_number),
                    daemon=True
                )
                thread.start()
                
                # Return empty TwiML immediately (message will be sent via REST API)
                return self._create_response("")
                
            except Exception as e:
                self.logger.error(f"SMS handling error: {e}")
                return self._create_response("I'm having trouble processing your request. Please try again.")
        
        @self.app.route('/status', methods=['GET'])
        def status():
            """Health check endpoint"""
            try:
                # Check service status
                services_status = {
                    'rag_engine': self._check_rag_engine(),
                    'llm_service': self._check_llm_service(),
                    'query_handler': self._check_query_handler()
                }
                
                return jsonify({
                    'status': 'healthy' if all(services_status.values()) else 'degraded',
                    'services': services_status,
                    'timestamp': self._get_timestamp()
                })
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': self._get_timestamp()
                }), 500
        
        @self.app.route('/test', methods=['GET'])
        def test():
            """Test endpoint"""
            return jsonify({
                'message': 'Parenting AI SMS Server is running',
                'timestamp': self._get_timestamp()
            })
    
    def _validate_twilio_request(self, request) -> bool:
        """Validate Twilio webhook request"""
        try:
            if not self.validator:
                self.logger.error("No Twilio validator available - auth_token may be missing")
                return False
            
            # Get the signature from headers
            signature = request.headers.get('X-Twilio-Signature', '')
            
            if not signature:
                self.logger.warning("Missing X-Twilio-Signature header - request may not be from Twilio")
                self.logger.debug(f"Request headers: {dict(request.headers)}")
                return False
            
            # Reconstruct URL exactly as Twilio expects it
            # Twilio signs the URL without query parameters, using the Host header
            # Try different URL formats to match what Twilio configured
            configured_webhook_url = self.config['twilio'].get('webhook_url', '')
            
            # Option 1: Use request.url (full URL with query params)
            url_full = request.url
            
            # Option 2: Use URL without query parameters (what Twilio typically signs)
            url_without_query = f"{request.scheme}://{request.host}{request.path}"
            
            # Option 3: Use configured webhook URL if available
            url_configured = configured_webhook_url if configured_webhook_url else url_without_query
            
            # Log validation details (DEBUG level to reduce verbosity)
            self.logger.debug(f"Validating Twilio request:")
            self.logger.debug(f"  request.url: {url_full}")
            self.logger.debug(f"  request.host: {request.host}")
            self.logger.debug(f"  request.path: {request.path}")
            self.logger.debug(f"  request.scheme: {request.scheme}")
            self.logger.debug(f"  URL without query: {url_without_query}")
            if configured_webhook_url:
                self.logger.debug(f"  Configured webhook URL: {configured_webhook_url}")
            self.logger.debug(f"  Signature header present: Yes")
            
            # Try validation with URL without query parameters first (most common)
            # Twilio typically signs the URL without query parameters
            is_valid = self.validator.validate(url_without_query, request.form, signature)
            
            # If that fails, try with full URL
            if not is_valid:
                self.logger.debug(f"  Trying validation with full URL (with query params)...")
                is_valid = self.validator.validate(url_full, request.form, signature)
            
            # If that fails and we have a configured webhook URL, try that
            if not is_valid and configured_webhook_url:
                self.logger.debug(f"  Trying validation with configured webhook URL...")
                is_valid = self.validator.validate(configured_webhook_url, request.form, signature)
            
            if not is_valid:
                self.logger.warning("Twilio signature validation FAILED - tried all URL formats")
                self.logger.debug(f"  Tried URL without query: {url_without_query}")
                self.logger.debug(f"  Tried full URL: {url_full}")
                if configured_webhook_url:
                    self.logger.debug(f"  Tried configured URL: {configured_webhook_url}")
                self.logger.debug(f"  Request method: {request.method}")
                self.logger.debug(f"  Signature header: {signature[:20]}... (truncated)")
                self.logger.debug(f"  Auth token configured: {'Yes' if self.config['twilio'].get('auth_token') else 'No'}")
                
                # Check if URL might be the issue
                self.logger.debug(f"  Hint: Ensure webhook URL in Twilio console matches one of the URLs above")
                self.logger.debug(f"  Hint: Check that TWILIO_AUTH_TOKEN in .env matches your Twilio account")
                self.logger.debug(f"  Hint: With ngrok, ensure the ngrok URL matches what's in Twilio webhook config")
            else:
                self.logger.debug("Twilio signature validation SUCCEEDED")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Twilio validation error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _process_message(self, message: str, phone_number: str) -> dict:
        """Process incoming message"""
        try:
            # Check if query handler is available
            if self.query_handler is None:
                return {
                    'text': "AI services are not currently available. Please try again later.",
                    'error': 'query_handler_not_available'
                }
            
            # Use query handler to process the message
            result = self.query_handler.process_query(message, phone_number)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            return {
                'text': "I'm having trouble processing your request. Please try again.",
                'error': str(e)
            }
    
    def _process_and_send_async(self, message_body: str, from_number: str):
        """Process message and send SMS asynchronously via REST API"""
        try:
            self.logger.info(f"Processing message asynchronously for {from_number}...")
            
            # Process the message (this may take a while with Ollama)
            response = self._process_message(message_body, from_number)
            response_text = response.get('text', '')
            
            # Send via REST API
            if not self.use_active_messaging or not self.twilio_client or not self.twilio_phone_number:
                self.logger.error("Cannot send SMS: Twilio REST API not configured")
                return
            
            try:
                message = self.twilio_client.messages.create(
                    body=response_text,
                    from_=self.twilio_phone_number,
                    to=from_number
                )
                self.logger.info(f"Successfully sent response to {from_number} via REST API (SID: {message.sid})")
                self.logger.info(f"Full message text: {response_text}")
            except Exception as e:
                self.logger.error(f"Failed to send SMS via REST API to {from_number}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error in async message processing for {from_number}: {e}")
            # Try to send error message via REST API if possible
            if self.use_active_messaging and self.twilio_client and self.twilio_phone_number:
                try:
                    error_msg = "I'm having trouble processing your request. Please try again."
                    self.twilio_client.messages.create(
                        body=error_msg,
                        from_=self.twilio_phone_number,
                        to=from_number
                    )
                    self.logger.info(f"Sent error notification to {from_number}")
                except Exception as send_error:
                    self.logger.error(f"Failed to send error notification: {send_error}")
    
    def _create_response(self, text: str) -> str:
        """Create TwiML response for Twilio"""
        response = MessagingResponse()
        response.message(text)
        return str(response)
    
    def _check_rag_engine(self) -> bool:
        """Check if RAG engine is working"""
        if self.rag_engine is None:
            return False
        try:
            stats = self.rag_engine.get_index_stats()
            return 'error' not in stats
        except Exception:
            return False
    
    def _check_llm_service(self) -> bool:
        """Check if LLM service is working"""
        if self.llm_service is None:
            return False
        try:
            test_result = self.llm_service.test_connection()
            return test_result.get('status') == 'success'
        except Exception:
            return False
    
    def _check_query_handler(self) -> bool:
        """Check if query handler is working"""
        if self.query_handler is None:
            return False
        try:
            # Simple test query
            test_result = self.query_handler.process_query("test", "+1234567890")
            return 'error' not in test_result
        except Exception:
            return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the SMS server"""
        self.logger.info(f"Starting SMS server on {host}:{port}")
        self.logger.info(f"Webhook URL: http://{host}:{port}/sms/incoming")
        self.logger.info(f"Status URL: http://{host}:{port}/status")
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function for running SMS server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parenting AI SMS Server")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        # Create and run server
        server = SMSServer(args.config)
        server.run(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        print("\nShutting down SMS server...")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



