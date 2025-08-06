#!/usr/bin/env python3
"""
Simple nano_llm test to debug the command syntax
"""
import subprocess
import sys

def test_nano_llm():
    """Test nano_llm with a simple prompt"""
    
    simple_prompt = "Hello, please respond with 'Family AI is working!' and explain what you are."
    
    # Test different nano_llm command formats
    test_commands = [
        # Format 1: Direct prompt
        ["./run.sh", "nano_llm", "--prompt", simple_prompt],
        
        # Format 2: With model specification
        ["./run.sh", "nano_llm", "--model", "microsoft/DialoGPT-medium", "--prompt", simple_prompt],
        
        # Format 3: Interactive mode check
        ["./run.sh", "nano_llm", "--help"],
        
        # Format 4: List available models
        ["./run.sh", "nano_llm", "--list-models"]
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {' '.join(cmd[:4])}...")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd="/Users/ethan/jetson-containers",
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Return code: {result.returncode}")
            print(f"STDOUT ({len(result.stdout)} chars):")
            print(result.stdout[:1000] + ("..." if len(result.stdout) > 1000 else ""))
            print(f"\nSTDERR ({len(result.stderr)} chars):")
            print(result.stderr[:500] + ("..." if len(result.stderr) > 500 else ""))
            
            # If this test worked, we found the right format
            if result.returncode == 0 and len(result.stdout) > 200 and "family ai" in result.stdout.lower():
                print(f"\n🎉 SUCCESS! Test {i} worked!")
                return cmd[:3]  # Return the working command format
                
        except subprocess.TimeoutExpired:
            print("⏱️ Command timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return None

if __name__ == "__main__":
    print("Testing nano_llm command formats...")
    working_format = test_nano_llm()
    
    if working_format:
        print(f"\n✅ Working nano_llm format found: {working_format}")
    else:
        print(f"\n❌ No working nano_llm format found")
        print("Possible issues:")
        print("1. nano_llm container not properly built")
        print("2. Model not available")  
        print("3. Incorrect command syntax")
        print("4. Container startup issues")