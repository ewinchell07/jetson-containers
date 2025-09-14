#!/usr/bin/env python3
"""
Test script for the new Resemblyzer-based diarization implementation.
This script tests the basic functionality without requiring actual audio files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        from transcriber import DIARIZATION_AVAILABLE, DiarizationManager
        print(f"✓ DIARIZATION_AVAILABLE: {DIARIZATION_AVAILABLE}")
        
        if DIARIZATION_AVAILABLE:
            print("✓ Resemblyzer imports successful")
            return True
        else:
            print("✗ Resemblyzer not available")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_diarization_manager():
    """Test DiarizationManager initialization"""
    print("\nTesting DiarizationManager initialization...")
    
    try:
        from transcriber import DiarizationManager, DIARIZATION_AVAILABLE
        
        if not DIARIZATION_AVAILABLE:
            print("✗ Resemblyzer not available, skipping test")
            return False
            
        # Test initialization
        manager = DiarizationManager()
        print("✓ DiarizationManager initialized successfully")
        
        # Test configuration access
        from transcriber import DIARIZATION_CONFIG
        print(f"✓ Configuration loaded: {DIARIZATION_CONFIG}")
        
        return True
        
    except Exception as e:
        print(f"✗ DiarizationManager test failed: {e}")
        return False

def test_transcriber_initialization():
    """Test Transcriber initialization with diarization"""
    print("\nTesting Transcriber initialization...")
    
    try:
        from transcriber import Transcriber, DIARIZATION_AVAILABLE
        
        # Test with diarization enabled
        transcriber = Transcriber("base.en", enable_diarization=True)
        print("✓ Transcriber initialized with diarization enabled")
        
        # Test with diarization disabled
        transcriber_no_diar = Transcriber("base.en", enable_diarization=False)
        print("✓ Transcriber initialized with diarization disabled")
        
        return True
        
    except Exception as e:
        print(f"✗ Transcriber test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Resemblyzer Implementation Test ===\n")
    
    tests = [
        test_imports,
        test_diarization_manager,
        test_transcriber_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All tests passed! Resemblyzer implementation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
