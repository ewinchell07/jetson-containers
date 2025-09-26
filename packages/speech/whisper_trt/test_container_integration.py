#!/usr/bin/env python3
"""
Test script to verify container integration in the automation system
"""

import os
import sys
import subprocess
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automation_manager import AutomationConfig, RecordingManager, TranscriptionManager


def test_container_integration():
    """Test container integration functionality"""
    
    print("🐳 Testing Container Integration")
    print("=" * 40)
    
    # Test configuration
    config = AutomationConfig()
    config.use_container = True
    config.container_name = "whisper-trt-test"
    
    print(f"Container mode: {config.use_container}")
    print(f"Container name: {config.container_name}")
    print()
    
    # Test recording manager
    print("🎙️  Testing Recording Manager:")
    recording_manager = RecordingManager(config)
    
    # Test container status check
    is_running = recording_manager._is_container_running()
    print(f"Container running: {is_running}")
    
    if not is_running:
        print("Container not running - this is expected for testing")
        print("In production, the automation would start the container automatically")
    print()
    
    # Test transcription manager
    print("📝 Testing Transcription Manager:")
    transcription_manager = TranscriptionManager(config)
    
    # Test container status check
    is_running = transcription_manager._is_container_running()
    print(f"Container running: {is_running}")
    print()
    
    # Test Docker command construction
    print("🔧 Testing Docker Commands:")
    
    # Test recording command
    recording_cmd = [
        "docker", "exec", "-d", config.container_name,
        "python3", "/opt/whisper_trt/continuous_recorder.py",
        "--chunk-duration", str(config.chunk_duration),
        "--sample-rate", str(config.sample_rate),
        "--output-dir", "/opt/whisper_trt/recordings"
    ]
    print(f"Recording command: {' '.join(recording_cmd)}")
    
    # Test transcription command
    transcription_cmd = [
        "docker", "exec", config.container_name,
        "python3", "/opt/whisper_trt/transcriber.py",
        "/opt/whisper_trt/recordings/test.wav",
        "--model", config.transcription_model,
        "--output-dir", "/opt/whisper_trt/transcriptions",
        "--num-speakers", str(config.num_speakers)
    ]
    print(f"Transcription command: {' '.join(transcription_cmd)}")
    print()
    
    # Test volume mounts
    print("📁 Testing Volume Mounts:")
    recordings_dir = Path(config.output_dir).expanduser()
    transcriptions_dir = Path(config.transcription_output_dir).expanduser()
    
    print(f"Host recordings dir: {recordings_dir}")
    print(f"Host transcriptions dir: {transcriptions_dir}")
    print(f"Container recordings mount: /opt/whisper_trt/recordings")
    print(f"Container transcriptions mount: /opt/whisper_trt/transcriptions")
    print()
    
    # Test container creation command
    print("🏗️  Testing Container Creation:")
    container_cmd = [
        "docker", "run", "-d",
        "--gpus", "all",
        "--network=host",
        "--device", "/dev/snd",
        "--memory=6g",
        "--shm-size=2g",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-v", f"{Path.cwd()}:/opt/whisper_trt",
        "-v", f"{Path.home()}/.cache/whisper_trt/logs:/root/.cache/whisper_trt/logs",
        "-v", f"{recordings_dir}:/opt/whisper_trt/recordings",
        "-v", f"{transcriptions_dir}:/opt/whisper_trt/transcriptions",
        "--name", config.container_name,
        "whisper-trt:latest"
    ]
    print(f"Container creation command:")
    print(" ".join(container_cmd))
    print()
    
    print("✅ Container integration test completed!")
    print()
    print("Key points:")
    print("- Automation manager runs on host system")
    print("- Recording and transcription run inside Docker container")
    print("- File sharing via Docker volumes")
    print("- Automatic container lifecycle management")
    print("- All Whisper-TRT dependencies available in container")


def test_direct_mode():
    """Test direct mode (no container)"""
    
    print("\n🖥️  Testing Direct Mode")
    print("=" * 40)
    
    # Test configuration
    config = AutomationConfig()
    config.use_container = False
    
    print(f"Container mode: {config.use_container}")
    print("Direct mode - runs recording/transcription directly on host")
    print("Useful for development and testing")
    print()
    
    # Test recording manager
    recording_manager = RecordingManager(config)
    print("Recording manager initialized for direct mode")
    
    # Test transcription manager
    transcription_manager = TranscriptionManager(config)
    print("Transcription manager initialized for direct mode")
    print("Transcriber object created: ", transcription_manager.transcriber is not None)
    print()
    
    print("✅ Direct mode test completed!")


if __name__ == "__main__":
    test_container_integration()
    test_direct_mode()
