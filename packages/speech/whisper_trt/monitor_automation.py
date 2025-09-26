#!/usr/bin/env python3
"""
Simple monitoring script for the Whisper-TRT Automation Manager

This script provides a simple way to check the status of the automation system
and view recent activity.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from automation_manager import AutomationManager, AutomationConfig, load_config


def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def show_status(config_file=None):
    """Show current status"""
    config = load_config(config_file)
    manager = AutomationManager(config)
    
    status = manager.get_status()
    
    print("🤖 Whisper-TRT Automation Status")
    print("=" * 40)
    
    # Basic status
    running_status = "🟢 Running" if status["running"] else "🔴 Stopped"
    recording_status = "🔴 Recording" if status["recording"] else "⚪ Idle"
    
    print(f"Manager: {running_status}")
    print(f"Recording: {recording_status}")
    print()
    
    # Health status
    health = status["health"]
    if health["status"] == "healthy":
        health_icon = "🟢"
    elif health["status"] == "degraded":
        health_icon = "🟡"
    else:
        health_icon = "🔴"
    
    print(f"Health: {health_icon} {health['status'].title()}")
    if health["status"] != "idle":
        print(f"  Expected chunks: {health['expected_chunks']}")
        print(f"  Recorded chunks: {health['recorded_chunks']}")
        print(f"  Missing chunks: {health['missing_chunks']}")
        if health["missing_chunk_times"]:
            print(f"  Missing times: {', '.join(health['missing_chunk_times'])}")
    print()
    
    # Transcription status
    unprocessed = status["unprocessed_recordings"]
    if unprocessed > 0:
        print(f"📝 Unprocessed recordings: {unprocessed}")
    else:
        print("📝 All recordings processed")
    print()
    
    # Configuration summary
    config = status["config"]
    print("⚙️  Configuration:")
    print(f"  Recording: {config['recording_start_time']} - {config['recording_end_time']}")
    print(f"  Chunk duration: {config['chunk_duration']}s ({config['chunk_duration']//60}min)")
    print(f"  Transcription model: {config['transcription_model']}")
    print(f"  Speakers: {config['num_speakers']}")
    print(f"  Transcription time: {config['transcription_start_time']}")


def show_recent_activity(config_file=None, hours=24):
    """Show recent activity"""
    config = load_config(config_file)
    recordings_dir = Path(config.output_dir).expanduser()
    transcriptions_dir = Path(config.transcription_output_dir).expanduser()
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    print(f"📊 Recent Activity (last {hours}h)")
    print("=" * 40)
    
    # Recent recordings
    recording_files = []
    for file_path in recordings_dir.glob("recording_*.wav"):
        try:
            timestamp_str = file_path.stem.replace("recording_", "").replace("_partial", "")
            file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            if file_time >= cutoff_time:
                recording_files.append((file_time, file_path))
        except ValueError:
            continue
    
    recording_files.sort(reverse=True)
    
    print(f"🎙️  Recent Recordings ({len(recording_files)}):")
    for file_time, file_path in recording_files[:10]:  # Show last 10
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {file_time.strftime('%Y-%m-%d %H:%M:%S')} - {file_path.name} ({size_mb:.1f}MB)")
    
    if len(recording_files) > 10:
        print(f"  ... and {len(recording_files) - 10} more")
    print()
    
    # Recent transcriptions
    transcription_files = []
    for file_path in transcriptions_dir.glob("transcript_*.json"):
        try:
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time >= cutoff_time:
                transcription_files.append((file_time, file_path))
        except Exception:
            continue
    
    transcription_files.sort(reverse=True)
    
    print(f"📝 Recent Transcriptions ({len(transcription_files)}):")
    for file_time, file_path in transcription_files[:10]:  # Show last 10
        print(f"  {file_time.strftime('%Y-%m-%d %H:%M:%S')} - {file_path.name}")
    
    if len(transcription_files) > 10:
        print(f"  ... and {len(transcription_files) - 10} more")


def show_logs(config_file=None, lines=50):
    """Show recent logs"""
    log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
    
    # Find the most recent log file
    log_files = list(log_dir.glob("automation_*.log"))
    if not log_files:
        print("No log files found")
        return
    
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    
    print(f"📋 Recent Logs ({latest_log.name})")
    print("=" * 40)
    
    try:
        with open(latest_log, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading log file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Monitor Whisper-TRT Automation")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--activity", type=int, default=24, 
                       help="Show activity for last N hours (default: 24)")
    parser.add_argument("--logs", type=int, default=50,
                       help="Show last N log lines (default: 50)")
    parser.add_argument("--json", action="store_true", help="Output status as JSON")
    
    args = parser.parse_args()
    
    if args.json:
        # JSON output for programmatic use
        config = load_config(args.config)
        manager = AutomationManager(config)
        status = manager.get_status()
        print(json.dumps(status, indent=2))
    else:
        # Human-readable output
        show_status(args.config)
        print()
        show_recent_activity(args.config, args.activity)
        print()
        show_logs(args.config, args.logs)


if __name__ == "__main__":
    main()

