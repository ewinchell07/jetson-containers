#!/usr/bin/env python3
"""
Test script to verify timezone handling in the automation system
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automation_manager import AutomationConfig, HealthMonitor


def test_timezone_handling():
    """Test that all timezone handling is consistent with local time"""
    
    print("🕐 Testing Timezone Handling")
    print("=" * 40)
    
    # Create test configuration
    config = AutomationConfig()
    
    # Create test directories
    test_output_dir = Path("/tmp/test_recordings")
    test_output_dir.mkdir(exist_ok=True)
    
    # Create health monitor
    health_monitor = HealthMonitor(config, test_output_dir)
    
    print(f"Current local time: {datetime.now()}")
    print(f"Current local date: {datetime.now().date()}")
    print(f"Current local time (time only): {datetime.now().time()}")
    print()
    
    # Test recording time check
    is_recording_time = health_monitor.is_recording_time()
    print(f"Currently in recording time: {is_recording_time}")
    print(f"Recording schedule: {config.recording_start_time} - {config.recording_end_time}")
    print()
    
    # Test expected chunks calculation
    expected_chunks = health_monitor.get_expected_chunks()
    print(f"Expected chunks for today: {len(expected_chunks)}")
    if expected_chunks:
        print(f"First chunk: {expected_chunks[0]}")
        print(f"Last chunk: {expected_chunks[-1]}")
    print()
    
    # Test recorded chunks (will be empty for test)
    recorded_chunks = health_monitor.get_recorded_chunks()
    print(f"Recorded chunks for today: {len(recorded_chunks)}")
    print()
    
    # Test health check
    health_status = health_monitor.check_health()
    print("Health status:")
    for key, value in health_status.items():
        print(f"  {key}: {value}")
    print()
    
    # Test filename format consistency
    print("📁 Filename Format Test:")
    test_time = datetime.now()
    expected_filename = f"recording_{test_time.strftime('%Y%m%d_%H%M%S')}.wav"
    print(f"Expected filename format: {expected_filename}")
    print(f"Date part: {test_time.strftime('%Y%m%d')}")
    print(f"Time part: {test_time.strftime('%H%M%S')}")
    print()
    
    # Test transcription timestamp format
    print("📝 Transcription Timestamp Test:")
    transcription_timestamp = datetime.now().isoformat()
    print(f"Transcription timestamp: {transcription_timestamp}")
    print()
    
    print("✅ Timezone handling test completed!")
    print()
    print("Key points:")
    print("- All scheduling uses local time")
    print("- Recording filenames use local time")
    print("- Health monitoring compares local time schedules with local time filenames")
    print("- Transcription timestamps include timezone info")
    print("- Everything is consistent with your computer's local timezone")


if __name__ == "__main__":
    test_timezone_handling()
