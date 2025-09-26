#!/usr/bin/env python3
"""
Test script for FamilyGuide
"""

import subprocess
import sys
import os

def test_family_guide():
    """Test the FamilyGuide script with sample questions."""
    
    transcript_folder = "/home/ethan/jetson-containers/packages/speech/whisper_trt/transcriptions/transcriptions_dev"
    
    if not os.path.exists(transcript_folder):
        print(f"❌ Transcript folder not found: {transcript_folder}")
        return False
    
    # Test questions
    test_questions = [
        "What patterns do you notice in our family conversations?",
        "How can I help my child express their emotions better?",
        "What routines seem to be working well for our family?",
        "Are there any concerning patterns in our recent interactions?"
    ]
    
    print("🧪 Testing FamilyGuide with sample questions...")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 40)
        
        try:
            # Run the family guide script
            cmd = [
                sys.executable, 
                "family_guide.py", 
                transcript_folder, 
                question,
                "--days", "3",  # Only look at last 3 days for testing
                "--max-segments", "10"  # Limit segments for testing
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ Success!")
                print(result.stdout)
            else:
                print("❌ Error:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - this is expected if nano_llm isn't available")
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("\n" + "="*60)
    
    return True

if __name__ == "__main__":
    test_family_guide()
