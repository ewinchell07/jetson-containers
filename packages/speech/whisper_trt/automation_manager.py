#!/usr/bin/env python3
"""
Automated Recording and Transcription Manager

This script manages the complete workflow:
1. Records 10-minute audio chunks from 6:30am to 8:00pm daily
2. Monitors recording health and restarts if needed
3. Transcribes all unprocessed recordings at 8:00pm using small model with 3 speakers
4. Provides comprehensive logging and status monitoring

Usage:
    python3 automation_manager.py [--config config.yaml] [--daemon]
"""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
import argparse

# Optional imports - only needed for direct mode
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logging.warning("PyYAML not available - will use default configuration")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available - limited system monitoring")

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    logging.warning("schedule not available - will use basic time-based scheduling")

# Import our existing modules (only for direct mode)
try:
    from continuous_recorder import ContinuousRecorder, AudioConfig
    from transcriber import Transcriber
    from config import get_transcribe_model, select_model_with_fallback, get_allow_swap
    HAS_WHISPER_MODULES = True
except ImportError as e:
    HAS_WHISPER_MODULES = False
    logging.warning(f"Whisper modules not available: {e}")
    logging.info("This is expected when running in container mode")


@dataclass
class AutomationConfig:
    """Configuration for the automation manager"""
    # Recording schedule (all times in LOCAL timezone)
    recording_start_time: str = "06:30"  # 6:30 AM local time
    recording_end_time: str = "20:00"    # 8:00 PM local time
    chunk_duration: int = 600            # 10 minutes in seconds
    
    # Recording settings
    output_dir: str = "~/recordings"     # Base directory for recordings
    sample_rate: int = 16000
    buffer_size: int = 4096
    latency: str = "high"
    amplify: bool = True
    gain: float = 1.5
    normalize: bool = True
    
    # Multi-channel recording settings
    channels: int = 4                    # Number of audio channels (4 for Focusrite 4i4)
    save_combined: bool = False         # Save combined multi-channel file
    save_individual: bool = True         # Save individual channel files (default)
    channel_names: List[str] = None      # Names for each channel
    apply_gain_to_channels: List[int] = None  # Channel indices to apply gain to (0-indexed)
    format: str = "auto"                # Audio format (auto, float32, int16, S32_LE)
    
    # Daily directory settings
    use_daily_directories: bool = True   # Create new directory each day
    daily_dir_format: str = "%Y-%m-%d"   # Date format for daily directories
    
    # Transcription settings
    transcription_model: str = "small"   # Use small model for efficiency
    num_speakers: int = 3               # 3 speakers as specified
    transcription_output_dir: str = "~/transcriptions"  # Base directory for transcriptions
    transcription_start_time: str = "20:00"  # 8:00 PM local time
    
    # Adaptive transcription settings
    use_adaptive_transcription: bool = True  # Enable adaptive quality-based model selection
    quality_threshold: float = 0.3       # Quality threshold for adaptive retry
    max_quality_retries: int = 2        # Maximum retries with larger models
    enable_large_models: bool = False   # Allow large models for adaptive transcription
    
    # Recording date filter
    transcribe_from_date: Optional[str] = None  # Only transcribe recordings from this date onwards (YYYY-MM-DD)
    
    # Health monitoring
    health_check_interval: int = 300     # 5 minutes
    max_missing_chunks: int = 2         # Restart if 2+ chunks missing
    restart_delay: int = 30             # Wait 30 seconds before restart
    
    # Logging
    log_level: str = "INFO"
    log_retention_days: int = 30
    
    # Container settings (if running in container)
    container_name: str = "whisper-trt"
    use_container: bool = False


class HealthMonitor:
    """Monitors recording health and detects issues"""
    
    def __init__(self, config: AutomationConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir).expanduser()
        self.last_check_time = None
        self.expected_chunks = []
        self.missing_chunks_count = 0
        
    def is_recording_time(self) -> bool:
        """Check if current time is within recording hours (using local time)"""
        now = datetime.now().time()  # Local time
        start_time = datetime.strptime(self.config.recording_start_time, "%H:%M").time()
        end_time = datetime.strptime(self.config.recording_end_time, "%H:%M").time()
        return start_time <= now <= end_time
    
    def get_expected_chunks(self) -> List[datetime]:
        """Calculate expected recording chunks for today (using local time)"""
        today = datetime.now().date()  # Local date
        start_datetime = datetime.combine(today, datetime.strptime(self.config.recording_start_time, "%H:%M").time())
        end_datetime = datetime.combine(today, datetime.strptime(self.config.recording_end_time, "%H:%M").time())
        
        chunks = []
        current = start_datetime
        while current < end_datetime:
            chunks.append(current)
            current += timedelta(seconds=self.config.chunk_duration)
        
        return chunks
    
    def get_recorded_chunks(self) -> List[datetime]:
        """Get list of actually recorded chunks for today (using local time)"""
        today = datetime.now().strftime("%Y%m%d")  # Local date
        
        # Look for both combined files and individual channel files
        patterns = [
            f"recording_{today}_*.wav",  # Combined multi-channel files
            f"recording_{today}_*_ch*.wav",  # Individual channel files
            f"recording_{today}_*_*.wav"  # Any recording files with channel names
        ]
        
        recorded_chunks = []
        seen_timestamps = set()
        
        for pattern in patterns:
            for file_path in self.output_dir.glob(pattern):
                try:
                    # Extract timestamp from filename
                    filename = file_path.stem
                    # Remove channel suffixes like _ch1, _ch2, etc.
                    timestamp_str = filename.replace("recording_", "").replace("_partial", "")
                    
                    # Remove channel suffixes (e.g., _ch1, _ch2, _ch3, _ch4)
                    for suffix in ["_ch1", "_ch2", "_ch3", "_ch4", "_ch5", "_ch6", "_ch7", "_ch8"]:
                        if timestamp_str.endswith(suffix):
                            timestamp_str = timestamp_str[:-len(suffix)]
                            break
                    
                    # Parse timestamp
                    chunk_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Only add if we haven't seen this timestamp before
                    if chunk_time not in seen_timestamps:
                        recorded_chunks.append(chunk_time)
                        seen_timestamps.add(chunk_time)
                        
                except ValueError:
                    continue
        
        return sorted(recorded_chunks)
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        if not self.is_recording_time():
            return {"status": "idle", "message": "Outside recording hours"}
        
        expected = self.get_expected_chunks()
        recorded = self.get_recorded_chunks()
        
        # Find missing chunks (allow some tolerance for timing)
        tolerance = timedelta(minutes=2)
        missing_chunks = []
        
        for expected_chunk in expected:
            found = False
            for recorded_chunk in recorded:
                if abs(expected_chunk - recorded_chunk) <= tolerance:
                    found = True
                    break
            if not found and expected_chunk < datetime.now() - tolerance:
                missing_chunks.append(expected_chunk)
        
        self.missing_chunks_count = len(missing_chunks)
        
        status = {
            "status": "healthy" if len(missing_chunks) == 0 else "degraded",
            "expected_chunks": len(expected),
            "recorded_chunks": len(recorded),
            "missing_chunks": len(missing_chunks),
            "missing_chunk_times": [chunk.strftime("%H:%M:%S") for chunk in missing_chunks],
            "needs_restart": len(missing_chunks) >= self.config.max_missing_chunks
        }
        
        return status


class RecordingManager:
    """Manages the recording process"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.recorder = None
        self.recording_thread = None
        self.is_recording = False
        self.base_output_dir = Path(config.output_dir).expanduser()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self._get_daily_output_dir()
        
        # Setup audio config (only for direct mode)
        if HAS_WHISPER_MODULES:
            self.audio_config = AudioConfig(
                target_sample_rate=config.sample_rate,
                chunk_duration=config.chunk_duration,
                blocksize=config.buffer_size,
                latency=config.latency,
                enable_amplification=config.amplify,
                normalize_audio=config.normalize,
                gain_boost=config.gain,
                channels=config.channels,
                save_combined=config.save_combined,
                save_individual=config.save_individual,
                channel_names=config.channel_names,
                apply_gain_to_channels=config.apply_gain_to_channels
            )
        else:
            self.audio_config = None
        
        # Container management
        self.container_name = config.container_name
        self.use_container = config.use_container
    
    def _get_daily_output_dir(self) -> Path:
        """Get the daily output directory for recordings"""
        if self.config.use_daily_directories:
            today = datetime.now().strftime(self.config.daily_dir_format)
            daily_dir = self.base_output_dir / today
            daily_dir.mkdir(parents=True, exist_ok=True)
            return daily_dir
        else:
            return self.base_output_dir
        
    def start_recording(self) -> bool:
        """Start recording in a separate thread"""
        if self.is_recording:
            logging.warning("Recording already in progress")
            return True
        
        try:
            logging.info("Starting recording...")
            
            if self.use_container:
                # Start recording inside Docker container
                return self._start_container_recording()
            else:
                # Start recording directly (for testing/development)
                return self._start_direct_recording()
                
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            self.is_recording = False
            return False
    
    def _start_direct_recording(self) -> bool:
        """Start recording directly (for testing/development)"""
        if not HAS_WHISPER_MODULES:
            logging.error("Whisper modules not available for direct recording")
            return False
        
        self.recorder = ContinuousRecorder(self.audio_config, str(self.output_dir))
        
        def recording_worker():
            try:
                self.recorder.start_recording()
            except Exception as e:
                logging.error(f"Recording thread error: {e}")
                self.is_recording = False
        
        self.recording_thread = threading.Thread(target=recording_worker, daemon=True)
        self.recording_thread.start()
        self.is_recording = True
        
        # Give it a moment to start
        time.sleep(2)
        
        if self.recorder and self.recorder.recording:
            logging.info("Recording started successfully (direct mode)")
            return True
        else:
            logging.error("Failed to start recording (direct mode)")
            self.is_recording = False
            return False
    
    def _start_container_recording(self) -> bool:
        """Start recording inside Docker container"""
        try:
            # Check if container is running
            if not self._is_container_running():
                logging.info("Starting Docker container...")
                self._start_container()
                time.sleep(5)  # Give container time to start
            
            # Get daily output directory for container
            daily_output_dir = self._get_daily_output_dir()
            container_output_dir = f"/opt/whisper_trt/recordings/{daily_output_dir.name}" if self.config.use_daily_directories else "/opt/whisper_trt/recordings"
            
            # Start recording inside container
            cmd = [
                "docker", "exec", "-d", self.container_name,
                "python3", "/opt/whisper_trt/continuous_recorder.py",
                "--chunk-duration", str(self.config.chunk_duration),
                "--sample-rate", str(self.config.sample_rate),
                "--output-dir", container_output_dir,
                "--buffer-size", str(self.config.buffer_size),
                "--latency", self.config.latency,
                "--gain", str(self.config.gain),
                "--channels", str(self.config.channels)
            ]
            
            # Add multi-channel specific options
            if self.config.save_combined:
                cmd.append("--save-combined")
            if not self.config.save_individual:
                cmd.append("--no-save-individual")
            if self.config.channel_names:
                cmd.extend(["--channel-names"] + self.config.channel_names)
            if self.config.apply_gain_to_channels:
                cmd.extend(["--apply-gain-to"] + [str(ch) for ch in self.config.apply_gain_to_channels])
            if self.config.format != "auto":
                cmd.extend(["--format", self.config.format])
            
            # Add amplification options
            if not self.config.amplify:
                cmd.append("--no-amplify")
            if not self.config.normalize:
                cmd.append("--no-normalize")
            
            logging.info(f"Starting container recording: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.is_recording = True
                logging.info("Recording started successfully (container mode)")
                return True
            else:
                logging.error(f"Failed to start container recording: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error starting container recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording"""
        if not self.is_recording:
            return
        
        logging.info("Stopping recording...")
        self.is_recording = False
        
        if self.use_container:
            self._stop_container_recording()
        else:
            if self.recorder:
                self.recorder.stop_recording()
            
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=10)
        
        logging.info("Recording stopped")
    
    def _stop_container_recording(self):
        """Stop recording inside Docker container"""
        try:
            # Kill the recording process inside container
            cmd = ["docker", "exec", self.container_name, "pkill", "-f", "continuous_recorder.py"]
            subprocess.run(cmd, capture_output=True, text=True)
            logging.info("Stopped container recording")
        except Exception as e:
            logging.error(f"Error stopping container recording: {e}")
    
    def _is_container_running(self) -> bool:
        """Check if Docker container is running"""
        try:
            cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return self.container_name in result.stdout
        except Exception as e:
            logging.error(f"Error checking container status: {e}")
            return False
    
    def _start_container(self):
        """Start the Docker container with proper configuration from README"""
        try:
            # Check if container exists and is running
            cmd = ["docker", "ps", "-q", "--filter", f"name={self.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                logging.info(f"Container {self.container_name} is already running")
                return True
            
            # Check if container exists but is stopped
            cmd = ["docker", "ps", "-a", "-q", "--filter", f"name={self.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                # Remove existing container first
                logging.info(f"Removing existing container {self.container_name}")
                subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
            
            # Create new container with proper configuration from README
            logging.info(f"Creating new container {self.container_name}")
            cmd = [
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
                "-v", f"{self.base_output_dir}:/opt/whisper_trt/recordings",
                "--name", self.container_name,
                "whisper-trt:latest",
                "bash", "-c", "sleep infinity"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to create container: {result.stderr}")
                return False
            
            # Wait for container to start
            time.sleep(3)
            
            # Verify container is running
            cmd = ["docker", "ps", "-q", "--filter", f"name={self.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                logging.info(f"Container {self.container_name} started successfully")
                return True
            else:
                logging.error(f"Container {self.container_name} failed to start")
                return False
                
        except Exception as e:
            logging.error(f"Error starting container: {e}")
            return False
    
    def restart_recording(self) -> bool:
        """Restart recording (stop and start)"""
        logging.info("Restarting recording...")
        self.stop_recording()
        time.sleep(self.config.restart_delay)
        return self.start_recording()


class TranscriptionManager:
    """Manages the transcription process"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.base_recordings_dir = Path(config.output_dir).expanduser()
        self.base_transcriptions_dir = Path(config.transcription_output_dir).expanduser()
        self.base_transcriptions_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir = self._get_daily_recordings_dir()
        self.transcriptions_dir = self._get_daily_transcriptions_dir()
        
        # Container management
        self.container_name = config.container_name
        self.use_container = config.use_container
    
    def _get_daily_recordings_dir(self) -> Path:
        """Get the daily recordings directory"""
        if self.config.use_daily_directories:
            today = datetime.now().strftime(self.config.daily_dir_format)
            daily_dir = self.base_recordings_dir / today
            try:
                daily_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # If we can't create the directory, try to use the base directory
                logging.warning(f"Permission denied creating {daily_dir}, using base directory")
                return self.base_recordings_dir
            return daily_dir
        else:
            return self.base_recordings_dir
    
    def _get_daily_transcriptions_dir(self) -> Path:
        """Get the daily transcriptions directory"""
        if self.config.use_daily_directories:
            today = datetime.now().strftime(self.config.daily_dir_format)
            daily_dir = self.base_transcriptions_dir / today
            try:
                daily_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # If we can't create the directory, try to use the base directory
                logging.warning(f"Permission denied creating {daily_dir}, using base directory")
                return self.base_transcriptions_dir
            return daily_dir
        else:
            return self.base_transcriptions_dir
        
        # Initialize transcriber with adaptive quality if enabled
        if not self.use_container and HAS_WHISPER_MODULES:
            model = select_model_with_fallback(config.transcription_model, get_allow_swap())
            self.transcriber = Transcriber(model, enable_diarization=True, 
                                         use_adaptive_quality=config.use_adaptive_transcription)
        else:
            self.transcriber = None
        
    def get_unprocessed_recordings(self) -> List[Path]:
        """Get list of recordings that haven't been transcribed yet"""
        unprocessed = []
        
        # Parse the date filter if specified
        filter_date = None
        if self.config.transcribe_from_date:
            try:
                filter_date = datetime.strptime(self.config.transcribe_from_date, "%Y-%m-%d").date()
                logging.info(f"Only transcribing recordings from {filter_date} onwards")
            except ValueError:
                logging.warning(f"Invalid date format in transcribe_from_date: {self.config.transcribe_from_date}")
        
        # Get recording files from all daily directories from filter_date onwards
        recording_files = self._get_all_recordings_from_date(filter_date)
        
        for recording_file in recording_files:
            # Check if transcription already exists
            base_name = recording_file.stem
            
            # For multi-channel files, we need to handle both combined and individual files
            # For individual channel files (e.g., recording_20250127_143022_ch1.wav),
            # we want to transcribe the combined file if it exists, or the first channel
            if "_ch" in base_name:
                # This is an individual channel file - check if we should transcribe it
                # Only transcribe if it's the first channel (ch1) or if no combined file exists
                if not base_name.endswith("_ch1"):
                    # Skip non-first channels - we'll transcribe the combined file or ch1
                    continue
                
                # Check if combined file exists (without _ch1 suffix)
                combined_base = base_name.replace("_ch1", "")
                combined_file = recording_file.parent / f"{combined_base}.wav"
                if combined_file.exists():
                    # Combined file exists, skip individual channel files
                    continue
                
                # Use the individual channel file
                transcription_base = base_name
            else:
                # This is a combined file or regular file
                transcription_base = base_name
            
            transcription_file = self.transcriptions_dir / f"transcript_{transcription_base}.json"
            
            if not transcription_file.exists():
                unprocessed.append(recording_file)
        
        logging.info(f"Found {len(unprocessed)} unprocessed recordings to transcribe")
        return sorted(unprocessed)
    
    def _get_all_recordings_from_date(self, filter_date: Optional[datetime.date]) -> List[Path]:
        """Get all recording files from daily directories from filter_date onwards"""
        all_recordings = []
        
        if self.config.use_daily_directories:
            # Scan all daily directories from filter_date onwards
            base_dir = self.base_recordings_dir
            
            # Get all subdirectories that look like dates
            for subdir in base_dir.iterdir():
                if not subdir.is_dir():
                    continue
                
                # Check if this directory should be included based on date filter
                if filter_date is not None:
                    if not self._is_directory_after_date(subdir, filter_date):
                        continue
                
                # Look for recording files in this directory
                # Include both combined files and individual channel files
                recordings_in_dir = []
                recordings_in_dir.extend(subdir.glob("recording_*.wav"))  # Combined files
                recordings_in_dir.extend(subdir.glob("recording_*_ch*.wav"))  # Individual channel files
                all_recordings.extend(recordings_in_dir)
                if recordings_in_dir:
                    logging.debug(f"Found {len(recordings_in_dir)} recordings in {subdir.name}")
        else:
            # Single directory mode - get all recording files
            all_recordings = []
            all_recordings.extend(self.recordings_dir.glob("recording_*.wav"))  # Combined files
            all_recordings.extend(self.recordings_dir.glob("recording_*_ch*.wav"))  # Individual channel files
        
        return all_recordings
    
    def _is_directory_after_date(self, directory: Path, filter_date: datetime.date) -> bool:
        """Check if a directory represents a date on or after the filter date"""
        dir_name = directory.name
        
        # Try different date formats
        date_formats = [
            "%Y-%m-%d",      # 2025-09-29
            "%Y%m%d",         # 20250929
            "%Y-%m-%d_*",     # 2025-09-29_suffix
        ]
        
        for date_format in date_formats:
            try:
                # Extract the date part (before any suffix)
                date_part = dir_name.split('_')[0] if '_' in dir_name else dir_name
                dir_date = datetime.strptime(date_part, date_format).date()
                return dir_date >= filter_date
            except ValueError:
                continue
        
        # If we can't parse the date, include the directory (safer to include than exclude)
        logging.debug(f"Could not parse date from directory name: {dir_name}")
        return True
    
    def _is_recording_after_date(self, recording_file: Path, filter_date: datetime.date) -> bool:
        """Check if a recording file is from or after the specified date"""
        try:
            # Extract date from filename (format: recording_YYYYMMDD_HHMMSS_*.wav)
            filename = recording_file.stem
            if filename.startswith("recording_"):
                # Remove "recording_" prefix
                date_part = filename[10:]  # Skip "recording_"
                # Extract date part (first 8 characters: YYYYMMDD)
                if len(date_part) >= 8:
                    date_str = date_part[:8]
                    file_date = datetime.strptime(date_str, "%Y%m%d").date()
                    return file_date >= filter_date
        except (ValueError, IndexError) as e:
            logging.debug(f"Could not parse date from {recording_file.name}: {e}")
        
        # If we can't parse the date, include the file (safer to include than exclude)
        return True
    
    def transcribe_recordings(self) -> Dict[str, Any]:
        """Transcribe all unprocessed recordings"""
        unprocessed = self.get_unprocessed_recordings()
        
        if not unprocessed:
            logging.info("No unprocessed recordings found")
            return {"status": "complete", "processed": 0, "errors": 0}
        
        logging.info(f"Found {len(unprocessed)} unprocessed recordings")
        
        if self.use_container:
            return self._transcribe_in_container(unprocessed)
        else:
            return self._transcribe_direct(unprocessed)
    
    def _transcribe_direct(self, unprocessed: List[Path]) -> Dict[str, Any]:
        """Transcribe recordings directly (for testing/development)"""
        if not HAS_WHISPER_MODULES:
            logging.error("Whisper modules not available for direct transcription")
            return {"status": "error", "processed": 0, "errors": len(unprocessed)}
        
        processed = 0
        errors = 0
        
        for i, recording_file in enumerate(unprocessed, 1):
            try:
                logging.info(f"Transcribing {i}/{len(unprocessed)}: {recording_file.name}")
                
                result = self.transcriber.process_file(
                    str(recording_file),
                    str(self.transcriptions_dir),
                    num_speakers=self.config.num_speakers
                )
                
                processed += 1
                logging.info(f"Transcribed: {result.text[:100]}...")
                
            except Exception as e:
                logging.error(f"Error transcribing {recording_file}: {e}")
                errors += 1
                continue
        
        logging.info(f"Transcription complete: {processed} processed, {errors} errors")
        return {"status": "complete", "processed": processed, "errors": errors}
    
    def _transcribe_in_container(self, unprocessed: List[Path]) -> Dict[str, Any]:
        """Transcribe recordings inside Docker container"""
        try:
            # Ensure container is running
            if not self._is_container_running():
                logging.info("Starting Docker container for transcription...")
                self._start_container()
                time.sleep(5)
            
            processed = 0
            errors = 0
            
            for i, recording_file in enumerate(unprocessed, 1):
                try:
                    logging.info(f"Transcribing {i}/{len(unprocessed)}: {recording_file.name}")
                    
                    # Get the actual directory where the recording file is located
                    recording_file_path = recording_file.relative_to(self.base_recordings_dir)
                    
                    # Get daily transcriptions directory for output
                    daily_transcriptions_dir = self._get_daily_transcriptions_dir()
                    
                    # Run transcription inside container with adaptive settings
                    cmd = [
                        "docker", "exec", self.container_name,
                        "python3", "/opt/whisper_trt/transcriber.py",
                        f"/opt/whisper_trt/recordings/{recording_file_path}",
                        "--output-dir", f"/opt/whisper_trt/transcriptions/{daily_transcriptions_dir.name}" if self.config.use_daily_directories else "/opt/whisper_trt/transcriptions",
                        "--num-speakers", str(self.config.num_speakers)
                    ]
                    
                    # Add adaptive transcription environment variables
                    env_vars = {
                        "ENABLE_QUALITY_RETRY": str(self.config.use_adaptive_transcription).lower(),
                        "QUALITY_THRESHOLD": str(self.config.quality_threshold),
                        "MAX_QUALITY_RETRIES": str(self.config.max_quality_retries),
                        "ENABLE_LARGE_MODELS": str(self.config.enable_large_models).lower()
                    }
                    
                    # Set environment variables in container
                    for key, value in env_vars.items():
                        subprocess.run([
                            "docker", "exec", self.container_name,
                            "sh", "-c", f"export {key}={value}"
                        ], capture_output=True, text=True)
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Forward container output to systemd logs
                    if result.stdout:
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                logging.info(f"[TRANSCRIBER] {line}")
                    
                    if result.stderr:
                        for line in result.stderr.strip().split('\n'):
                            if line.strip():
                                logging.error(f"[TRANSCRIBER] {line}")
                    
                    if result.returncode == 0:
                        processed += 1
                        logging.info(f"Transcribed: {recording_file.name}")
                    else:
                        logging.error(f"Error transcribing {recording_file.name} (exit code: {result.returncode})")
                        errors += 1
                        
                except Exception as e:
                    logging.error(f"Error transcribing {recording_file}: {e}")
                    errors += 1
                    continue
            
            logging.info(f"Transcription complete: {processed} processed, {errors} errors")
            return {"status": "complete", "processed": processed, "errors": errors}
            
        except Exception as e:
            logging.error(f"Error in container transcription: {e}")
            return {"status": "error", "processed": 0, "errors": len(unprocessed)}
    
    def _is_container_running(self) -> bool:
        """Check if Docker container is running"""
        try:
            cmd = ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return self.container_name in result.stdout
        except Exception as e:
            logging.error(f"Error checking container status: {e}")
            return False
    
    def _start_container(self):
        """Start the Docker container with proper configuration from README"""
        try:
            # Check if container exists and is running
            cmd = ["docker", "ps", "-q", "--filter", f"name={self.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                logging.info(f"Container {self.container_name} is already running")
                return True
            
            # Check if container exists but is stopped
            cmd = ["docker", "ps", "-a", "-q", "--filter", f"name={self.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                # Remove existing container first
                logging.info(f"Removing existing container {self.container_name}")
                subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
            
            # Create new container with proper configuration from README
            logging.info(f"Creating new container {self.container_name}")
            cmd = [
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
                "-v", f"{self.base_recordings_dir}:/opt/whisper_trt/recordings",
                "-v", f"{self.base_transcriptions_dir}:/opt/whisper_trt/transcriptions",
                "--name", self.container_name,
                "whisper-trt:latest",
                "bash", "-c", "sleep infinity"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to create container: {result.stderr}")
                return False
            
            # Wait for container to start
            time.sleep(3)
            
            # Verify container is running
            cmd = ["docker", "ps", "-q", "--filter", f"name={self.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                logging.info(f"Container {self.container_name} started successfully")
                return True
            else:
                logging.error(f"Container {self.container_name} failed to start")
                return False
                
        except Exception as e:
            logging.error(f"Error starting container: {e}")
            return False


class AutomationManager:
    """Main automation manager"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.health_monitor = HealthMonitor(config, config.output_dir)
        self.recording_manager = RecordingManager(config)
        self.transcription_manager = TranscriptionManager(config)
        self.running = False
        self.daemon_mode = False
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup schedule
        self._setup_schedule()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
        
        # Try to create log directory with proper permissions
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            # Ensure the directory is writable
            if not os.access(log_dir, os.W_OK):
                raise PermissionError(f"Cannot write to {log_dir}")
        except (PermissionError, OSError) as e:
            # Fallback to a writable directory
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.warning(f"Using fallback log directory: {log_dir}")
        
        # Clean old logs
        self._cleanup_old_logs(log_dir)
        
        log_file = log_dir / f"automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Setup logging with error handling
        handlers = []
        
        # Add file handler if we can write to the log file
        try:
            file_handler = logging.FileHandler(log_file)
            handlers.append(file_handler)
        except (PermissionError, OSError) as e:
            logging.warning(f"Cannot create log file {log_file}: {e}")
        
        # Add console handler if not in daemon mode
        if not self.daemon_mode:
            handlers.append(logging.StreamHandler())
        
        # If no handlers available, use NullHandler
        if not handlers:
            handlers.append(logging.NullHandler())
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=handlers
        )
        
        logging.info(f"Automation manager logging to: {log_file}")
        logging.info(f"Configuration: {asdict(self.config)}")
    
    def _cleanup_old_logs(self, log_dir: Path):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=self.config.log_retention_days)
        
        for log_file in log_dir.glob("automation_*.log"):
            try:
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
                    logging.debug(f"Removed old log: {log_file}")
            except Exception as e:
                logging.warning(f"Error removing old log {log_file}: {e}")
    
    def _setup_schedule(self):
        """Setup scheduled tasks"""
        if HAS_SCHEDULE:
            # Recording schedule
            schedule.every().day.at(self.config.recording_start_time).do(self._start_recording_schedule)
            schedule.every().day.at(self.config.recording_end_time).do(self._stop_recording_schedule)
            
            # Transcription schedule
            schedule.every().day.at(self.config.transcription_start_time).do(self._transcription_schedule)
            
            # Health monitoring
            schedule.every(self.config.health_check_interval).seconds.do(self._health_check_schedule)
            
            logging.info("Scheduled tasks configured")
        else:
            logging.warning("Schedule module not available - using basic time-based scheduling")
    
    def _start_recording_schedule(self):
        """Scheduled task to start recording"""
        logging.info("Scheduled recording start")
        if not self.recording_manager.is_recording:
            self.recording_manager.start_recording()
    
    def _stop_recording_schedule(self):
        """Scheduled task to stop recording"""
        logging.info("Scheduled recording stop")
        if self.recording_manager.is_recording:
            self.recording_manager.stop_recording()
    
    def _transcription_schedule(self):
        """Scheduled task to run transcription"""
        logging.info("Scheduled transcription start")
        
        # Forward recent container logs
        self._forward_container_logs()
        
        # Ensure container is running before transcription
        if not self._ensure_container_running():
            logging.error("Failed to start container for transcription")
            return
        
        result = self.transcription_manager.transcribe_recordings()
        logging.info(f"Transcription result: {result}")
    
    def _ensure_container_running(self) -> bool:
        """Ensure the Docker container is running, restart if needed"""
        try:
            # Check if container is running
            cmd = ["docker", "ps", "-q", "--filter", f"name={self.config.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                logging.debug(f"Container {self.config.container_name} is running")
                return True
            
            # Container is not running, try to start it
            logging.info(f"Container {self.config.container_name} is not running, attempting to start...")
            
            # Check if container exists but is stopped
            cmd = ["docker", "ps", "-a", "-q", "--filter", f"name={self.config.container_name}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout.strip():
                # Start existing container
                logging.info(f"Starting existing container {self.config.container_name}")
                cmd = ["docker", "start", self.config.container_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    time.sleep(3)  # Wait for container to start
                    return True
                else:
                    logging.error(f"Failed to start existing container: {result.stderr}")
            
            # Container doesn't exist or failed to start, create new one
            logging.info(f"Creating new container {self.config.container_name}")
            return self.transcription_manager._start_container()
            
        except Exception as e:
            logging.error(f"Error ensuring container is running: {e}")
            return False
    
    def _forward_container_logs(self):
        """Forward recent container logs to systemd logs"""
        try:
            # Get recent logs from container
            cmd = ["docker", "logs", "--tail", "10", self.config.container_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logging.info(f"[CONTAINER] {line}")
            
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logging.error(f"[CONTAINER] {line}")
                        
        except Exception as e:
            logging.debug(f"Could not forward container logs: {e}")
    
    def _health_check_schedule(self):
        """Scheduled health check"""
        if not self.health_monitor.is_recording_time():
            return
        
        # Forward recent container logs
        self._forward_container_logs()
        
        health_status = self.health_monitor.check_health()
        
        if health_status["needs_restart"]:
            logging.warning(f"Health check failed: {health_status}")
            logging.info("Restarting recording...")
            self.recording_manager.restart_recording()
        else:
            logging.debug(f"Health check passed: {health_status}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the automation manager"""
        logging.info("Starting automation manager...")
        self.running = True
        
        # Start recording if we're in recording hours
        if self.health_monitor.is_recording_time():
            logging.info("Current time is within recording hours, starting recording...")
            self.recording_manager.start_recording()
        
        # Main loop
        try:
            while self.running:
                if HAS_SCHEDULE:
                    schedule.run_pending()
                else:
                    # Basic time-based scheduling without schedule module
                    self._basic_time_scheduling()
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.stop()
    
    def _basic_time_scheduling(self):
        """Basic time-based scheduling when schedule module is not available"""
        now = datetime.now().time()
        
        # Check recording start time
        start_time = datetime.strptime(self.config.recording_start_time, "%H:%M").time()
        end_time = datetime.strptime(self.config.recording_end_time, "%H:%M").time()
        transcription_time = datetime.strptime(self.config.transcription_start_time, "%H:%M").time()
        
        # Start recording if it's time and not already recording
        if start_time <= now <= end_time and not self.recording_manager.is_recording:
            self._start_recording_schedule()
        
        # Stop recording if it's past end time and still recording
        if now > end_time and self.recording_manager.is_recording:
            self._stop_recording_schedule()
        
        # Run transcription if it's time
        if now >= transcription_time:
            # Only run once per day
            today = datetime.now().date()
            if not hasattr(self, '_last_transcription_date') or self._last_transcription_date != today:
                self._transcription_schedule()
                self._last_transcription_date = today
        
        # Health check every 5 minutes
        if not hasattr(self, '_last_health_check') or (datetime.now() - self._last_health_check).seconds >= self.config.health_check_interval:
            self._health_check_schedule()
            self._last_health_check = datetime.now()
    
    def stop(self):
        """Stop the automation manager"""
        logging.info("Stopping automation manager...")
        self.running = False
        self.recording_manager.stop_recording()
        logging.info("Automation manager stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        health_status = self.health_monitor.check_health()
        
        return {
            "running": self.running,
            "recording": self.recording_manager.is_recording,
            "health": health_status,
            "unprocessed_recordings": len(self.transcription_manager.get_unprocessed_recordings()),
            "config": asdict(self.config)
        }


def load_config(config_file: Optional[str] = None) -> AutomationConfig:
    """Load configuration from file or use defaults"""
    # If no config file specified, look for automation_config.yaml in current directory
    if config_file is None:
        default_config = "automation_config.yaml"
        if Path(default_config).exists():
            config_file = default_config
            logging.info(f"Using default config file: {config_file}")
    
    if config_file and Path(config_file).exists() and HAS_YAML:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            logging.info(f"Loaded configuration from: {config_file}")
            return AutomationConfig(**config_data)
        except Exception as e:
            logging.warning(f"Error loading config file: {e}, using defaults")
            return AutomationConfig()
    else:
        if config_file:
            logging.warning(f"Config file not found: {config_file}, using defaults")
        else:
            logging.info("No config file specified, using defaults")
        return AutomationConfig()


def save_config(config: AutomationConfig, config_file: str):
    """Save configuration to file"""
    if HAS_YAML:
        with open(config_file, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
    else:
        # Fallback to JSON if YAML not available
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Automated Recording and Transcription Manager")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode (no console output)")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--transcribe-now", action="store_true", help="Run transcription now and exit")
    parser.add_argument("--create-config", help="Create default configuration file")
    
    args = parser.parse_args()
    
    # Create default config file if requested
    if args.create_config:
        config = AutomationConfig()
        save_config(config, args.create_config)
        print(f"Default configuration saved to: {args.create_config}")
        return 0
    
    # Load configuration
    config = load_config(args.config)
    config.daemon_mode = args.daemon
    
    # Create automation manager
    manager = AutomationManager(config)
    
    # Handle special modes
    if args.status:
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        return 0
    
    if args.transcribe_now:
        logging.info("Running transcription now...")
        result = manager.transcription_manager.transcribe_recordings()
        print(f"Transcription result: {result}")
        return 0
    
    # Start the automation manager
    try:
        manager.start()
    except Exception as e:
        logging.error(f"Automation manager failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

