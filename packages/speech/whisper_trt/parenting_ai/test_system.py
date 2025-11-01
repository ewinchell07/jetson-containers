#!/usr/bin/env python3
"""
Basic functional tests for Parenting AI SMS System
"""

import sys
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parenting_ai.utils import load_config, setup_logging
from parenting_ai.llm_service import LLMService
from parenting_ai.rag_engine import RAGEngine
from parenting_ai.query_handler import QueryHandler

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    try:
        config = load_config()
        assert 'whitelist' in config
        assert 'llm' in config
        assert 'rag' in config
        print("✅ Configuration loading: PASSED")
        return True
    except Exception as e:
        print(f"❌ Configuration loading: FAILED - {e}")
        return False

def test_llm_service():
    """Test LLM service connection"""
    print("Testing LLM service...")
    try:
        config = load_config()
        llm_service = LLMService(config)
        
        # Test connection
        result = llm_service.test_connection()
        if result['status'] == 'success':
            print("✅ LLM service: PASSED")
            return True
        else:
            print(f"❌ LLM service: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ LLM service: FAILED - {e}")
        return False

def test_rag_engine():
    """Test RAG engine"""
    print("Testing RAG engine...")
    try:
        config = load_config()
        rag_engine = RAGEngine(config)
        
        # Test search
        result = rag_engine.test_search()
        if result['status'] == 'success':
            print("✅ RAG engine: PASSED")
            return True
        else:
            print(f"❌ RAG engine: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ RAG engine: FAILED - {e}")
        return False

def test_query_handler():
    """Test query handler"""
    print("Testing query handler...")
    try:
        config = load_config()
        
        # Initialize components
        rag_engine = RAGEngine(config)
        llm_service = LLMService(config)
        query_handler = QueryHandler(config, rag_engine, llm_service)
        
        # Test query processing
        result = query_handler.process_query("test query", "+1234567890")
        
        if 'error' not in result:
            print("✅ Query handler: PASSED")
            return True
        else:
            print(f"❌ Query handler: FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Query handler: FAILED - {e}")
        return False

def test_sms_formatting():
    """Test SMS response formatting"""
    print("Testing SMS formatting...")
    try:
        from parenting_ai.utils import format_sms_response
        
        config = load_config()
        
        # Test short message
        short_text = "This is a short message."
        result = format_sms_response(short_text, config)
        assert result['length'] == 'brief'
        assert result['parts'] == 1
        
        # Test long message
        long_text = "This is a very long message that should be truncated for SMS delivery. " * 10
        result = format_sms_response(long_text, config)
        assert result['length'] in ['normal', 'detailed', 'truncated']
        assert result['parts'] >= 2
        
        print("✅ SMS formatting: PASSED")
        return True
    except Exception as e:
        print(f"❌ SMS formatting: FAILED - {e}")
        return False

def test_phone_validation():
    """Test phone number validation"""
    print("Testing phone validation...")
    try:
        from parenting_ai.utils import is_phone_whitelisted, format_phone_number
        
        config = load_config()
        
        # Test phone formatting
        formatted = format_phone_number("1234567890")
        assert formatted == "+11234567890"
        
        # Test whitelist check (will fail if not configured)
        # This is expected to fail in test environment
        whitelisted = is_phone_whitelisted("+11234567890", config)
        print(f"Phone whitelist check: {whitelisted} (expected: False in test)")
        
        print("✅ Phone validation: PASSED")
        return True
    except Exception as e:
        print(f"❌ Phone validation: FAILED - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Parenting AI System Tests")
    print("=" * 40)
    
    # Setup logging
    try:
        config = load_config()
        setup_logging(config)
    except Exception as e:
        print(f"Warning: Could not setup logging: {e}")
    
    tests = [
        test_config_loading,
        test_llm_service,
        test_rag_engine,
        test_query_handler,
        test_sms_formatting,
        test_phone_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAILED - {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



