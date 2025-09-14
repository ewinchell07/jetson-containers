#!/usr/bin/env python3
"""
Simple continuous audio recorder without transcription complexity.
Designed to be lightweight and memory-efficient for long-term recording.
"""
import os
import numpy as np
import sounddevice as sd
import wave
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import argparse
import time
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not available - resampling will use basic interpolation")


@dataclass
class AudioConfig:
    """Audio configuration with native and target sample rates"""
    native_sample_rate: int = 48000  # Device native rate (will be auto-detected)
    target_sample_rate: int = 16000  # Target rate for transcription
    channels: int = 1
    dtype: type = np.float32  # Use float32 for better device compatibility
    chunk_duration: int = 600  # 10 minutes in seconds
    blocksize: int = 4096  # Increased buffer size to prevent overflow
    latency: str = 'high'  # Use high latency for more stable recording
    
    # Audio amplification settings
    enable_amplification: bool = True  # Enable automatic amplification
    normalize_audio: bool = True  # Normalize to prevent clipping
    gain_boost: float = 1.5  # Amplification factor (1.0 = no change, 2.0 = double volume)


class ContinuousRecorder:
    """Lightweight continuous audio recorder"""
    
    def __init__(self, config: AudioConfig, output_dir: str = "recordings"):
        self.config = config
        self.recording = False
        self.current_chunk = []
        self.chunk_size = 0
        self.max_chunk_size = self.config.chunk_duration * self.config.native_sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stream = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup audio device
        self._setup_audio_device()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup basic logging"""
        log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"continuous_recorder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_file}")

    def _get_supported_formats(self, device_id: int, sample_rate: int) -> List[np.dtype]:
        """Query supported audio formats for a device"""
        common_formats = [np.float32, np.int32, np.int16]
        supported_formats = []
        
        for fmt in common_formats:
            try:
                sd.check_input_settings(device=device_id, samplerate=sample_rate, 
                                       channels=self.config.channels, dtype=fmt)
                supported_formats.append(fmt)
            except sd.PortAudioError:
                continue
        
        return supported_formats
    
    def _get_supported_sample_rates(self, device_id: int) -> List[int]:
        """Query supported sample rates for a device"""
        common_rates = [8000, 16000, 22050, 44100, 48000, 88200, 96000]
        supported_rates = []
        
        for rate in common_rates:
            try:
                sd.check_input_settings(device=device_id, samplerate=rate, 
                                       channels=self.config.channels, dtype=self.config.dtype)
                supported_rates.append(rate)
            except sd.PortAudioError:
                continue
        
        return supported_rates
    
    def _choose_best_sample_rate(self, supported_rates: List[int]) -> int:
        """Choose the best sample rate from supported rates"""
        # Preference order: 48000 > 44100 > others
        preferred_order = [48000, 44100, 96000, 88200, 22050, 16000, 8000]
        
        for rate in preferred_order:
            if rate in supported_rates:
                return rate
        
        # Fallback to highest available rate
        return max(supported_rates) if supported_rates else 48000

    def _setup_audio_device(self):
        """Setup audio input device with automatic sample rate detection"""
        logging.info("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            logging.info(f"  {i}: {device['name']} - Channels: {device['max_input_channels']}")
        
        # Find best input device
        self.input_device = None
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                self.input_device = i
                break
        
        if self.input_device is None:
            self.input_device = sd.default.device[0]  # Use default input
            
        device_info = sd.query_devices(self.input_device)
        logging.info(f"Using input device: {device_info['name']}")
        
        # First, try to find best format for device default sample rate
        default_rate = int(device_info.get('default_samplerate', 48000))
        supported_formats = self._get_supported_formats(self.input_device, default_rate)
        logging.info(f"Supported formats at {default_rate}Hz: {[fmt.__name__ for fmt in supported_formats]}")
        
        # Choose best format (prefer float32, fallback to int16)
        if np.float32 in supported_formats:
            self.config.dtype = np.float32
            logging.info("Using float32 format for optimal device compatibility")
        elif np.int16 in supported_formats:
            self.config.dtype = np.int16
            logging.info("Using int16 format (float32 not supported)")
        else:
            logging.warning("No standard formats supported, using float32 anyway")
        
        # Query supported sample rates with chosen format
        supported_rates = self._get_supported_sample_rates(self.input_device)
        logging.info(f"Supported sample rates: {supported_rates}")
        
        if not supported_rates:
            logging.warning("No supported sample rates found, using device default")
            self.config.native_sample_rate = default_rate
        else:
            self.config.native_sample_rate = self._choose_best_sample_rate(supported_rates)
        
        # Update max chunk size with new sample rate
        self.max_chunk_size = self.config.chunk_duration * self.config.native_sample_rate
        
        logging.info(f"Selected format: {self.config.dtype.__name__}")
        logging.info(f"Selected sample rate: {self.config.native_sample_rate} Hz, Channels: {self.config.channels}")
        logging.info(f"Target sample rate for transcription: {self.config.target_sample_rate} Hz")

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback for recording with improved overflow handling"""
        if status:
            if status.input_overflow:
                # Only log overflow occasionally to avoid spam
                if not hasattr(self, '_overflow_count'):
                    self._overflow_count = 0
                    self._last_overflow_log = 0
                
                self._overflow_count += 1
                current_time = time.time()
                
                # Log every 10 overflows or every 5 seconds
                if (self._overflow_count % 10 == 0 or 
                    current_time - self._last_overflow_log > 5.0):
                    logging.warning(f"Input overflow detected (count: {self._overflow_count}) - consider reducing sample rate or increasing buffer size")
                    self._last_overflow_log = current_time
            else:
                logging.warning(f"Audio status: {status}")
        
        if self.recording:
            try:
                # Process audio data more efficiently
                if indata.ndim > 1:
                    # Convert stereo to mono more efficiently
                    audio_chunk = np.mean(indata, axis=1, dtype=indata.dtype)
                else:
                    audio_chunk = indata.copy()  # Make a copy to avoid memory issues
                
                # Extend chunk more efficiently
                self.current_chunk.extend(audio_chunk)
                self.chunk_size += len(audio_chunk)
                
                # Save chunk when it reaches the target duration
                if self.chunk_size >= self.max_chunk_size:
                    self._save_current_chunk()
                    
            except Exception as e:
                logging.error(f"Error in audio callback: {e}")
                self._reset_chunk()

    def _amplify_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Amplify and normalize audio data"""
        if not self.config.enable_amplification:
            return audio_data
        
        try:
            # Convert to float for processing
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Normalize first to prevent clipping
            if self.config.normalize_audio:
                max_val = np.max(np.abs(audio_float))
                if max_val > 0:
                    audio_float = audio_float / max_val
            
            # Apply gain boost
            audio_float = audio_float * self.config.gain_boost
            
            # Prevent clipping after amplification
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            # Convert back to original dtype
            if self.config.dtype == np.int16:
                return (audio_float * 32767.0).astype(np.int16)
            else:
                return audio_float.astype(self.config.dtype)
                
        except Exception as e:
            logging.error(f"Audio amplification failed: {e}")
            return audio_data  # Return original if amplification fails
    
    def _resample_audio(self, audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio data to target sample rate"""
        if original_sr == target_sr:
            return audio_data
        
        try:
            if HAS_LIBROSA:
                # Convert to float for processing if needed
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float32)
                
                # Use librosa for high-quality resampling
                resampled = librosa.resample(audio_float, orig_sr=original_sr, target_sr=target_sr)
                
                # Convert back to original dtype
                if self.config.dtype == np.int16:
                    return (resampled * 32768.0).astype(np.int16)
                else:
                    return resampled.astype(self.config.dtype)
            else:
                # Basic linear interpolation fallback
                ratio = target_sr / original_sr
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                return np.interp(indices, np.arange(len(audio_data)), audio_data).astype(self.config.dtype)
        except Exception as e:
            logging.error(f"Resampling failed: {e}")
            return audio_data  # Return original if resampling fails

    def _save_current_chunk(self, partial: bool = False):
        """Save the current audio chunk to file at target sample rate"""
        if not self.current_chunk:
            return
            
        try:
            # Convert to numpy array
            audio_data = np.array(self.current_chunk, dtype=self.config.dtype)
            
            # Apply amplification if enabled
            if self.config.enable_amplification:
                audio_data = self._amplify_audio(audio_data)
                logging.debug(f"Applied amplification: normalize={self.config.normalize_audio}, gain={self.config.gain_boost}x")
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_partial" if partial else ""
            filename = f"recording_{timestamp}{suffix}.wav"
            filepath = self.output_dir / filename
            
            # Resample to target sample rate if needed
            if self.config.native_sample_rate != self.config.target_sample_rate:
                audio_data = self._resample_audio(audio_data, self.config.native_sample_rate, self.config.target_sample_rate)
                sample_rate = self.config.target_sample_rate
            else:
                sample_rate = self.config.native_sample_rate
                logging.debug("No resampling needed - native rate matches target rate")
            
            # Use soundfile for better format support
            try:
                import soundfile as sf
                sf.write(str(filepath), audio_data, sample_rate)
                logging.info(f"Saved using soundfile (better quality)")
            except ImportError:
                # Fallback to wave module
                with wave.open(str(filepath), 'wb') as wf:
                    wf.setnchannels(self.config.channels)
                    if audio_data.dtype == np.float32:
                        # Convert float32 to int16 for wave module
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_int16.tobytes())
                    else:
                        wf.setsampwidth(2)  # 2 bytes for int16
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_data.tobytes())
            
            duration = len(audio_data) / sample_rate
            logging.info(f"Saved {duration:.1f}s audio ({sample_rate}Hz) to: {filename}")
            
            # Clean up
            del audio_data
            self._reset_chunk()
            
        except Exception as e:
            logging.error(f"Error saving audio chunk: {e}")
            self._reset_chunk()

    def _reset_chunk(self):
        """Reset the current chunk"""
        self.current_chunk = []
        self.chunk_size = 0

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.stop_recording()
        sys.exit(0)

    def start_recording(self):
        """Start continuous recording"""
        if self.recording:
            logging.warning("Already recording")
            return
            
        logging.info(f"Starting continuous recording...")
        logging.info(f"Chunk duration: {self.config.chunk_duration} seconds")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info("Press Ctrl+C to stop recording")
        
        self.recording = True
        self._reset_chunk()
        
        try:
            # Reset overflow counter
            self._overflow_count = 0
            self._last_overflow_log = 0
            
            # Use only basic, universally supported PortAudio settings
            self.stream = sd.InputStream(
                samplerate=self.config.native_sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self._audio_callback,
                blocksize=self.config.blocksize,
                latency=self.config.latency,
                device=self.input_device
            )
            
            with self.stream:
                logging.info("Recording started successfully")
                # Keep the main thread alive
                while self.recording:
                    time.sleep(1)
                    
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            self.recording = False
            raise

    def stop_recording(self):
        """Stop recording and save any remaining audio"""
        if not self.recording:
            return
            
        logging.info("Stopping recording...")
        self.recording = False
        
        # Save any remaining audio as partial chunk
        if self.current_chunk:
            self._save_current_chunk(partial=True)
        
        if self.stream:
            self.stream.close()
            
        logging.info("Recording stopped")


def main():
    parser = argparse.ArgumentParser(description="Continuous Audio Recorder")
    parser.add_argument("--chunk-duration", type=int, default=600, 
                       help="Duration of each audio chunk in seconds (default: 600)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Audio sample rate (default: 16000)")
    parser.add_argument("--output-dir", type=str, default="~/recordings",
                       help="Output directory for recordings (default: ~/recordings)")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio input device ID (default: auto-detect)")
    parser.add_argument("--buffer-size", type=int, default=4096,
                       help="Audio buffer size (default: 4096, increase if getting overflows)")
    parser.add_argument("--latency", choices=['low', 'high'], default='high',
                       help="Audio latency mode (default: high for stability)")
    parser.add_argument("--format", choices=['float32', 'int16'], default='auto',
                       help="Audio format (default: auto-detect, prefer float32)")
    parser.add_argument("--amplify", action='store_true', default=True,
                       help="Enable audio amplification (default: enabled)")
    parser.add_argument("--no-amplify", action='store_false', dest='amplify',
                       help="Disable audio amplification")
    parser.add_argument("--gain", type=float, default=1.5,
                       help="Audio gain boost multiplier (default: 1.5, range: 1.0-3.0)")
    parser.add_argument("--normalize", action='store_true', default=True,
                       help="Normalize audio to prevent clipping (default: enabled)")
    parser.add_argument("--no-normalize", action='store_false', dest='normalize',
                       help="Disable audio normalization")
    
    args = parser.parse_args()
    
    # Validate gain parameter
    if args.gain < 1.0 or args.gain > 3.0:
        print(f"Warning: Gain {args.gain} is outside recommended range 1.0-3.0")
    
    # Create configuration
    config = AudioConfig(
        target_sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        blocksize=args.buffer_size,
        latency=args.latency,
        enable_amplification=args.amplify,
        normalize_audio=args.normalize,
        gain_boost=args.gain
    )
    
    # Override format if specified
    if args.format == 'float32':
        config.dtype = np.float32
    elif args.format == 'int16':
        config.dtype = np.int16
    # 'auto' will be handled by device detection
    
    # Expand home directory path
    output_dir = os.path.expanduser(args.output_dir)
    
    # Create recorder
    recorder = ContinuousRecorder(config, output_dir)
    
    # Override device if specified
    if args.device is not None:
        recorder.input_device = args.device
        device_info = sd.query_devices(args.device)
        logging.info(f"Using specified device: {device_info['name']}")
    
    try:
        # Start recording
        recorder.start_recording()
    except KeyboardInterrupt:
        logging.info("Recording interrupted by user")
    except Exception as e:
        logging.error(f"Recording failed: {e}")
    finally:
        recorder.stop_recording()


if __name__ == "__main__":
    main()