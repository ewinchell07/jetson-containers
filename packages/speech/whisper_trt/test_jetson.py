#!/usr/bin/env python3
"""
Jetson Nano specific tests for the whisper_trt transcriber.
These tests are designed to work in the Jetson Nano environment with proper dependencies.
"""

import os
import sys
import tempfile
import json
import numpy as np
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work on Jetson"""
    print("Testing imports on Jetson Nano...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
            print(f"✓ CUDA device name: {torch.cuda.get_device_name()}")
    except ImportError:
        print("✗ PyTorch not available")
        return False
    
    try:
        import whisper
        print(f"✓ Whisper available")
    except ImportError:
        print("✗ Whisper not available")
        return False
    
    try:
        from transcriber import (
            GPUManager, ModelManager, DiarizationManager, Transcriber,
            get_transcription_config, get_diarization_config_local, get_audio_config_local,
            DIARIZATION_AVAILABLE
        )
        print("✓ All transcriber imports successful")
        print(f"✓ Diarization available: {DIARIZATION_AVAILABLE}")
    except ImportError as e:
        print(f"✗ Transcriber import failed: {e}")
        return False
    
    return True

def test_gpu_memory():
    """Test GPU memory management on Jetson"""
    print("\nTesting GPU memory management...")
    
    try:
        from transcriber import GPUManager
        import torch
        
        if torch.cuda.is_available():
            # Test memory cleanup
            GPUManager.cleanup_memory()
            print("✓ GPU memory cleanup successful")
            
            # Test memory usage reporting
            usage = GPUManager.get_memory_usage()
            print(f"✓ GPU memory usage: {usage:.2f} GB")
            
            # Test memory allocation and cleanup
            device = torch.device("cuda:0")
            test_tensor = torch.randn(1000, 1000, device=device)
            print(f"✓ GPU tensor allocation successful")
            
            del test_tensor
            GPUManager.cleanup_memory()
            print("✓ GPU memory cleanup after allocation successful")
            
            return True
        else:
            print("⚠ CUDA not available, skipping GPU tests")
            return True
    except Exception as e:
        print(f"✗ GPU memory test failed: {e}")
        return False

def test_model_loading():
    """Test model loading on Jetson"""
    print("\nTesting model loading...")
    
    try:
        from transcriber import ModelManager
        
        # Test with tiny model (fastest to load)
        print("Loading tiny.en model...")
        manager = ModelManager("tiny.en")
        print("✓ Model loaded successfully")
        
        # Test model properties
        print(f"✓ Model name: {manager.model_name}")
        print(f"✓ Device: {manager.device}")
        print(f"✓ Model type: {type(manager.model)}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing functionality"""
    print("\nTesting audio processing...")
    
    try:
        from transcriber import ModelManager
        import soundfile as sf
        
        # Create a test audio file
        sample_rate = 16000
        duration = 2  # 2 seconds
        frequency = 440  # A4 note
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            try:
                # Test audio loading
                manager = ModelManager("tiny.en")
                result = manager.transcribe_audio(tmp_file.name)
                
                print(f"✓ Audio processing successful")
                print(f"✓ Transcription result: {result.get('text', 'No text')}")
                print(f"✓ Segments: {len(result.get('segments', []))}")
                
                return True
            finally:
                os.unlink(tmp_file.name)
                
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        return False

def test_diarization():
    """Test speaker diarization if available"""
    print("\nTesting speaker diarization...")
    
    try:
        from transcriber import DiarizationManager, DIARIZATION_AVAILABLE
        
        if not DIARIZATION_AVAILABLE:
            print("⚠ Diarization not available (Resemblyzer not installed)")
            return True
        
        # Test diarization manager initialization
        manager = DiarizationManager()
        print("✓ DiarizationManager initialized")
        
        # Create a test audio file with multiple frequencies (simulating different speakers)
        import soundfile as sf
        sample_rate = 16000
        duration = 5
        
        # Create audio with two different frequencies
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.concatenate([
            0.3 * np.sin(2 * np.pi * 440 * t[:len(t)//2]),  # First half: 440Hz
            0.3 * np.sin(2 * np.pi * 880 * t[len(t)//2:])   # Second half: 880Hz
        ]).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            try:
                # Test diarization processing
                speaker_segments = manager.process_audio(tmp_file.name, num_speakers=2)
                
                print(f"✓ Diarization processing successful")
                print(f"✓ Found {len(speaker_segments)} speaker segments")
                
                for i, segment in enumerate(speaker_segments[:3]):  # Show first 3
                    print(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s, {segment['speaker']}")
                
                return True
            finally:
                os.unlink(tmp_file.name)
                
    except Exception as e:
        print(f"✗ Diarization test failed: {e}")
        return False

def test_full_transcription():
    """Test full transcription pipeline"""
    print("\nTesting full transcription pipeline...")
    
    try:
        from transcriber import Transcriber
        import soundfile as sf
        
        # Create a more realistic test audio file
        sample_rate = 16000
        duration = 3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create audio with speech-like characteristics
        audio_data = (
            0.2 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
            0.1 * np.sin(2 * np.pi * 400 * t) +  # First harmonic
            0.05 * np.sin(2 * np.pi * 600 * t)   # Second harmonic
        ).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                try:
                    # Test full transcription
                    transcriber = Transcriber("tiny.en", enable_diarization=False)
                    result = transcriber.process_file(tmp_file.name, output_dir)
                    
                    print(f"✓ Full transcription successful")
                    print(f"✓ Output file created: {result.get('audio_file', 'Unknown')}")
                    print(f"✓ Model used: {result.get('model', 'Unknown')}")
                    print(f"✓ GPU memory used: {result.get('gpu_memory_used_gb', 0):.2f} GB")
                    print(f"✓ Transcription text: {result.get('transcription', {}).get('text', 'No text')}")
                    
                    # Check if output file was created
                    output_files = list(Path(output_dir).glob('*.json'))
                    if output_files:
                        print(f"✓ Output JSON file created: {output_files[0].name}")
                        
                        # Verify JSON content
                        with open(output_files[0], 'r') as f:
                            saved_data = json.load(f)
                        print(f"✓ JSON file contains valid data")
                    
                    return True
                finally:
                    os.unlink(tmp_file.name)
                    
    except Exception as e:
        print(f"✗ Full transcription test failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality"""
    print("\nTesting batch processing...")
    
    try:
        from transcriber import Transcriber
        import soundfile as sf
        
        # Create temporary directory with multiple audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create 3 test audio files
            for i in range(3):
                sample_rate = 16000
                duration = 1
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio_data = 0.2 * np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32)
                
                audio_file = temp_path / f"test_audio_{i}.wav"
                sf.write(str(audio_file), audio_data, sample_rate)
            
            # Create output directory
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # Test batch processing
            transcriber = Transcriber("tiny.en", enable_diarization=False)
            results = transcriber.process_batch(str(temp_path), str(output_dir), "*.wav")
            
            print(f"✓ Batch processing successful")
            print(f"✓ Processed {len(results)} files")
            
            # Check output files
            output_files = list(output_dir.glob('*.json'))
            print(f"✓ Created {len(output_files)} output files")
            
            return True
            
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False

def main():
    """Run all Jetson-specific tests"""
    print("Jetson Nano Whisper-TRT Transcriber Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Memory Test", test_gpu_memory),
        ("Model Loading Test", test_model_loading),
        ("Audio Processing Test", test_audio_processing),
        ("Diarization Test", test_diarization),
        ("Full Transcription Test", test_full_transcription),
        ("Batch Processing Test", test_batch_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The transcriber is working correctly on Jetson Nano.")
        return 0
    else:
        print(f"✗ {total - passed} tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
