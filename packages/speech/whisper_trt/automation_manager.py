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
    output_dir: str = "~/recordings"
    sample_rate: int = 16000
    buffer_size: int = 4096
    latency: str = "high"
    amplify: bool = True
    gain: float = 1.5
    normalize: bool = True
    
    # Transcription settings
    transcription_model: str = "small"   # Use small model for efficiency
    num_speakers: int = 3               # 3 speakers as specified
    transcription_output_dir: str = "~/transcriptions"
    transcription_start_time: str = "20:00"  # 8:00 PM local time
    
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
        pattern = f"recording_{today}_*.wav"
        
        recorded_chunks = []
        for file_path in self.output_dir.glob(pattern):
            try:
                # Extract timestamp from filename
                filename = file_path.stem
                timestamp_str = filename.replace("recording_", "").replace("_partial", "")
                chunk_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                recorded_chunks.append(chunk_time)
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
        self.output_dir = Path(config.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup audio config (only for direct mode)
        if HAS_WHISPER_MODULES:
            self.audio_config = AudioConfig(
                target_sample_rate=config.sample_rate,
                chunk_duration=config.chunk_duration,
                blocksize=config.buffer_size,
                latency=config.latency,
                enable_amplification=config.amplify,
                normalize_audio=config.normalize,
                gain_boost=config.gain
            )
        else:
            self.audio_config = None
        
        # Container management
        self.container_name = config.container_name
        self.use_container = config.use_container
        
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
            
            # Start recording inside container
            cmd = [
                "docker", "exec", "-d", self.container_name,
                "python3", "/opt/whisper_trt/continuous_recorder.py",
                "--chunk-duration", str(self.config.chunk_duration),
                "--sample-rate", str(self.config.sample_rate),
                "--output-dir", "/opt/whisper_trt/recordings",
                "--buffer-size", str(self.config.buffer_size),
                "--latency", self.config.latency,
                "--gain", str(self.config.gain)
            ]
            
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
        """Start the Docker container"""
        try:
            # Check if container exists
            cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if self.container_name in result.stdout:
                # Container exists, start it
                cmd = ["docker", "start", self.container_name]
                subprocess.run(cmd, capture_output=True, text=True)
                logging.info(f"Started existing container: {self.container_name}")
            else:
                # Container doesn't exist, create and run it
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
                    "-v", f"{self.output_dir}:/opt/whisper_trt/recordings",
                    "--name", self.container_name,
                    "whisper-trt:latest"
                ]
                subprocess.run(cmd, capture_output=True, text=True)
                logging.info(f"Created and started new container: {self.container_name}")
                
        except Exception as e:
            logging.error(f"Error starting container: {e}")
            raise
    
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
        self.recordings_dir = Path(config.output_dir).expanduser()
        self.transcriptions_dir = Path(config.transcription_output_dir).expanduser()
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)
        
        # Container management
        self.container_name = config.container_name
        self.use_container = config.use_container
        
        # Initialize transcriber with small model (only for direct mode)
        if not self.use_container and HAS_WHISPER_MODULES:
            model = select_model_with_fallback(config.transcription_model, get_allow_swap())
            self.transcriber = Transcriber(model, enable_diarization=True)
        else:
            self.transcriber = None
        
    def get_unprocessed_recordings(self) -> List[Path]:
        """Get list of recordings that haven't been transcribed yet"""
        unprocessed = []
        
        # Get all recording files
        recording_files = list(self.recordings_dir.glob("recording_*.wav"))
        
        for recording_file in recording_files:
            # Check if transcription already exists
            base_name = recording_file.stem
            transcription_file = self.transcriptions_dir / f"transcript_{base_name}.json"
            
            if not transcription_file.exists():
                unprocessed.append(recording_file)
        
        return sorted(unprocessed)
    
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
                    
                    # Run transcription inside container
                    cmd = [
                        "docker", "exec", self.container_name,
                        "python3", "/opt/whisper_trt/transcriber.py",
                        f"/opt/whisper_trt/recordings/{recording_file.name}",
                        "--model", self.config.transcription_model,
                        "--output-dir", "/opt/whisper_trt/transcriptions",
                        "--num-speakers", str(self.config.num_speakers)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        processed += 1
                        logging.info(f"Transcribed: {recording_file.name}")
                    else:
                        logging.error(f"Error transcribing {recording_file.name}: {result.stderr}")
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
        """Start the Docker container"""
        try:
            # Check if container exists
            cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if self.container_name in result.stdout:
                # Container exists, start it
                cmd = ["docker", "start", self.container_name]
                subprocess.run(cmd, capture_output=True, text=True)
                logging.info(f"Started existing container: {self.container_name}")
            else:
                # Container doesn't exist, create and run it
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
                    "-v", f"{self.recordings_dir}:/opt/whisper_trt/recordings",
                    "-v", f"{self.transcriptions_dir}:/opt/whisper_trt/transcriptions",
                    "--name", self.container_name,
                    "whisper-trt:latest"
                ]
                subprocess.run(cmd, capture_output=True, text=True)
                logging.info(f"Created and started new container: {self.container_name}")
                
        except Exception as e:
            logging.error(f"Error starting container: {e}")
            raise


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
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old logs
        self._cleanup_old_logs(log_dir)
        
        log_file = log_dir / f"automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if not self.daemon_mode else logging.NullHandler()
            ]
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
        result = self.transcription_manager.transcribe_recordings()
        logging.info(f"Transcription result: {result}")
    
    def _health_check_schedule(self):
        """Scheduled health check"""
        if not self.health_monitor.is_recording_time():
            return
        
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
    if config_file and Path(config_file).exists() and HAS_YAML:
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            return AutomationConfig(**config_data)
        except Exception as e:
            logging.warning(f"Error loading config file: {e}, using defaults")
            return AutomationConfig()
    else:
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

