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
    channels: int = 4  # 4 channels for Focusrite 4i4
    dtype: type = np.int16  # Use int16 for better compatibility (int32 for S32_LE compatibility)
    chunk_duration: int = 600  # 10 minutes in seconds
    blocksize: int = 8192  # Increased buffer size to prevent overflow
    latency: str = 'high'  # Use high latency for more stable recording
    
    # Audio amplification settings
    enable_amplification: bool = True  # Enable automatic amplification
    normalize_audio: bool = True  # Normalize to prevent clipping
    gain_boost: float = 1.5  # Amplification factor (1.0 = no change, 2.0 = double volume)
    
    # Audio filtering settings
    enable_noise_filtering: bool = True  # Enable noise filtering
    high_pass_freq: float = 200.0  # High-pass filter frequency (Hz) - increased to remove more low-frequency artifacts
    low_pass_freq: float = 8000.0  # Low-pass filter frequency (Hz)
    notch_freq: float = 60.0  # Notch filter frequency for power line hum (Hz)
    notch_q: float = 30.0  # Notch filter Q factor
    
    # Gain noise filtering settings (500Hz and harmonics)
    enable_gain_noise_filtering: bool = True  # Enable 500Hz gain noise filtering
    gain_noise_base_freq: float = 500.0  # Base gain noise frequency (Hz)
    gain_noise_max_freq: float = 6000.0  # Maximum frequency to filter harmonics up to (Hz)
    gain_noise_q: float = 30.0  # Q factor for gain noise notch filters
    
    # Multi-channel recording settings
    save_combined: bool = False  # Save combined multi-channel file
    save_individual: bool = True  # Save individual channel files (default)
    channel_names: List[str] = None  # Names for each channel (will be auto-generated)
    default_channel_names: List[str] = None  # Default channel names in config
    
    # Channel-specific gain settings
    channel_gains: List[float] = None  # Individual gain for each channel
    apply_gain_to_channels: List[int] = None  # Which channels to apply gain to (default: 3,4)


class ContinuousRecorder:
    """Lightweight continuous audio recorder"""
    
    def __init__(self, config: AudioConfig, output_dir: str = "recordings"):
        self.config = config
        self.recording = False
        self.current_chunks = [[] for _ in range(self.config.channels)]  # Multi-channel storage
        self.chunk_size = 0
        self.max_chunk_size = self.config.chunk_duration * self.config.native_sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.stream = None
        
        # Initialize channel names
        self._setup_channel_names()
        
        # Initialize channel gains
        self._setup_channel_gains()
        
        # Setup logging
        self._setup_logging()
        
        # Setup audio device
        self._setup_audio_device()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Setup basic logging"""
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Try to create log directory
        try:
            log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"continuous_recorder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        except Exception as e:
            # Fallback to current directory
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"continuous_recorder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            print(f"Warning: Could not create log directory in home, using: {log_dir}")
        
        # Setup handlers
        handlers = []
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)
        
        # Try to add file handler
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=handlers,
            force=True  # Force reconfiguration
        )
        
        # Test logging
        print("=" * 60)
        print("🎤 CONTINUOUS RECORDER STARTING")
        print("=" * 60)
        logging.info(f"Logging to: {log_file}")
        logging.info("Logging system initialized successfully")

    def _setup_channel_names(self):
        """Setup channel names from config or use defaults"""
        if self.config.channel_names:
            self.channel_names = self.config.channel_names
        elif self.config.default_channel_names:
            self.channel_names = self.config.default_channel_names
        else:
            # Default channel names for 4-channel setup
            if self.config.channels == 4:
                self.channel_names = ['TV Living Room', 'Dining Room', 'Penn Bedroom', 'Rowe Bedroom']
            else:
                self.channel_names = [f'ch{i+1}' for i in range(self.config.channels)]
        
        # Ensure we have the right number of channel names
        if len(self.channel_names) != self.config.channels:
            logging.warning(f"Channel names count ({len(self.channel_names)}) doesn't match channels ({self.config.channels})")
            self.channel_names = [f'ch{i+1}' for i in range(self.config.channels)]
        
        logging.info(f"Channel names: {self.channel_names}")

    def _setup_channel_gains(self):
        """Setup channel-specific gain settings"""
        if self.config.channel_gains:
            self.channel_gains = self.config.channel_gains
        else:
            # Default: no gain for channels 1-2, apply gain to channels 3-4
            self.channel_gains = [1.0] * self.config.channels
            if self.config.apply_gain_to_channels:
                for ch in self.config.apply_gain_to_channels:
                    if 0 <= ch < self.config.channels:
                        self.channel_gains[ch] = self.config.gain_boost
            else:
                # Default: apply gain to channels 3 and 4 (0-indexed: 2, 3)
                if self.config.channels >= 4:
                    self.channel_gains[2] = self.config.gain_boost  # Channel 3
                    self.channel_gains[3] = self.config.gain_boost  # Channel 4
        
        logging.info(f"Channel gains: {self.channel_gains}")

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
        logging.info("🔍 Scanning for audio devices...")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            logging.info(f"  {i}: {device['name']} - Channels: {device['max_input_channels']}")
        
        # Find Focusrite 4i4 or best multi-channel input device
        self.input_device = None
        focusrite_device = None
        
        for i, device in enumerate(devices):
            device_name = device['name'].lower()
            if ('focusrite' in device_name or 'scarlett' in device_name) and '4i4' in device_name:
                focusrite_device = i
                print(f"🎯 Found Focusrite 4i4/Scarlett at device {i}: {device['name']}")
                break
            elif device['max_input_channels'] >= self.config.channels:
                if self.input_device is None:
                    self.input_device = i
                    logging.info(f"Found multi-channel device: {device['name']} with {device['max_input_channels']} channels")
        
        # Prefer Focusrite 4i4 if found, otherwise use best multi-channel device
        if focusrite_device is not None:
            self.input_device = focusrite_device
            logging.info("✅ Using Focusrite 4i4/Scarlett device - ensure you're using plughw:0,0 for ALSA compatibility")
        elif self.input_device is None:
            self.input_device = sd.default.device[0]  # Use default input
            logging.warning("⚠️  No suitable multi-channel device found, using default")
            
        device_info = sd.query_devices(self.input_device)
        logging.info(f"🎧 Selected device: {device_info['name']}")
        
        # Check if device supports required number of channels
        if device_info['max_input_channels'] < self.config.channels:
            logging.warning(f"Device only supports {device_info['max_input_channels']} channels, but {self.config.channels} requested")
            self.config.channels = min(self.config.channels, device_info['max_input_channels'])
            logging.info(f"Adjusted to {self.config.channels} channels")
        
        # First, try to find best format for device default sample rate
        default_rate = int(device_info.get('default_samplerate', 48000))
        supported_formats = self._get_supported_formats(self.input_device, default_rate)
        logging.info(f"Supported formats at {default_rate}Hz: {[fmt.__name__ for fmt in supported_formats]}")
        
        # Choose best format (prefer int32 for S32_LE compatibility, fallback to float32, then int16)
        if np.int32 in supported_formats:
            self.config.dtype = np.int16  # Prefer int16 for better compatibility
            logging.info("Using int16 format for better compatibility")
        elif np.float32 in supported_formats:
            self.config.dtype = np.float32
            logging.info("Using float32 format for optimal device compatibility")
        elif np.int16 in supported_formats:
            self.config.dtype = np.int16
            logging.info("Using int16 format (int32/float32 not supported)")
        else:
            logging.warning("No standard formats supported, using int32 anyway")
        
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
        
        logging.info(f"📊 Selected format: {self.config.dtype.__name__}")
        logging.info(f"📊 Selected sample rate: {self.config.native_sample_rate} Hz, Channels: {self.config.channels}")
        logging.info(f"📊 Target sample rate for transcription: {self.config.target_sample_rate} Hz")

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
                
                # Log every 5 overflows or every 3 seconds
                if (self._overflow_count % 5 == 0 or 
                    current_time - self._last_overflow_log > 3.0):
                    logging.warning(f"⚠️  Input overflow detected (count: {self._overflow_count}) - consider reducing sample rate or increasing buffer size")
                    logging.warning(f"💡 Try: --buffer-size 16384 or --sample-rate 44100")
                    self._last_overflow_log = current_time
            else:
                logging.warning(f"Audio status: {status}")
        
        if self.recording:
            try:
                # Process multi-channel audio data
                if indata.ndim > 1 and indata.shape[1] > 1:
                    # Multi-channel audio - store each channel separately
                    # Ensure we have the right number of channels
                    if len(self.current_chunks) != indata.shape[1]:
                        logging.warning(f"Channel count mismatch: expected {len(self.current_chunks)}, got {indata.shape[1]}")
                        self.current_chunks = [[] for _ in range(indata.shape[1])]
                    
                    # Store each channel separately
                    for ch in range(min(indata.shape[1], len(self.current_chunks))):
                        # Convert to list to ensure compatibility
                        channel_data = indata[:, ch].tolist()
                        self.current_chunks[ch].extend(channel_data)
                else:
                    # Mono audio - convert to mono if needed
                    if indata.ndim > 1:
                        audio_chunk = np.mean(indata, axis=1, dtype=indata.dtype)
                    else:
                        audio_chunk = indata.copy()
                    
                    # Store in first channel
                    if len(self.current_chunks) > 0:
                        # Convert to list to ensure compatibility
                        audio_list = audio_chunk.tolist()
                        self.current_chunks[0].extend(audio_list)
                
                # Update chunk size (use first channel for size tracking)
                if self.current_chunks and len(self.current_chunks[0]) > 0:
                    self.chunk_size = len(self.current_chunks[0])
                
                # Save chunk when it reaches the target duration
                if self.chunk_size >= self.max_chunk_size:
                    print(f"📦 Chunk complete ({self.chunk_size} samples), saving...")
                    self._save_current_chunk()
                    
            except Exception as e:
                logging.error(f"Error in audio callback: {e}")
                logging.error(f"indata shape: {indata.shape if hasattr(indata, 'shape') else 'N/A'}")
                logging.error(f"current_chunks length: {len(self.current_chunks) if hasattr(self, 'current_chunks') else 'N/A'}")
                self._reset_chunk()

    def _amplify_audio(self, audio_data: np.ndarray, channel_idx: int = 0) -> np.ndarray:
        """Amplify and normalize audio data with channel-specific gain"""
        if not self.config.enable_amplification:
            return audio_data
        
        # Validate input audio data
        if audio_data is None or len(audio_data) == 0:
            logging.warning(f"Empty audio data for channel {channel_idx}")
            return audio_data
        
        # Check for NaN or infinite values in input
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            logging.warning(f"NaN or infinite values in input audio for channel {channel_idx}, skipping amplification")
            return audio_data
        
        try:
            # Get channel-specific gain
            gain = self.channel_gains[channel_idx] if channel_idx < len(self.channel_gains) else 1.0
            
            # Convert to float for processing
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_float = audio_data.astype(np.float32) / 2147483648.0  # 2^31
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Normalize first to prevent clipping
            if self.config.normalize_audio:
                max_val = np.max(np.abs(audio_float))
                if max_val > 0:
                    audio_float = audio_float / max_val
            
            # Apply channel-specific gain
            audio_float = audio_float * gain
            
            # Prevent clipping after amplification
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(audio_float)) or np.any(np.isinf(audio_float)):
                logging.warning(f"NaN or infinite values detected in channel {channel_idx}, using original audio")
                return audio_data
            
            # Convert back to original dtype
            if self.config.dtype == np.int16:
                scaled_audio = audio_float * 32767.0
                # Ensure values are within int16 range
                scaled_audio = np.clip(scaled_audio, -32768, 32767)
                return scaled_audio.astype(np.int16)
            elif self.config.dtype == np.int32:
                scaled_audio = audio_float * 2147483647.0
                # Ensure values are within int32 range
                scaled_audio = np.clip(scaled_audio, -2147483648, 2147483647)
                return scaled_audio.astype(np.int32)
            else:
                return audio_float.astype(self.config.dtype)
                
        except Exception as e:
            logging.error(f"Audio amplification failed for channel {channel_idx}: {e}")
            return audio_data  # Return original if amplification fails
    
    def _filter_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise filtering to remove gain whine and other artifacts"""
        if not self.config.enable_noise_filtering:
            return audio_data
            
        try:
            # Convert to float for processing
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_float = audio_data.astype(np.float32) / 2147483647.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Apply high-pass filter to remove low-frequency rumble
            if self.config.high_pass_freq > 0:
                from scipy import signal
                nyquist = sample_rate / 2
                high_pass_cutoff = min(self.config.high_pass_freq / nyquist, 0.99)
                b, a = signal.butter(2, high_pass_cutoff, btype='high')
                audio_float = signal.filtfilt(b, a, audio_float)
            
            # Apply low-pass filter to remove high-frequency noise (gain whine)
            if self.config.low_pass_freq > 0:
                from scipy import signal
                nyquist = sample_rate / 2
                low_pass_cutoff = min(self.config.low_pass_freq / nyquist, 0.99)
                b, a = signal.butter(4, low_pass_cutoff, btype='low')
                audio_float = signal.filtfilt(b, a, audio_float)
            
            # Apply notch filter to remove power line hum (60Hz)
            if self.config.notch_freq > 0:
                from scipy import signal
                notch_freq_norm = self.config.notch_freq / (sample_rate / 2)
                if notch_freq_norm < 0.99:
                    b, a = signal.iirnotch(notch_freq_norm, self.config.notch_q)
                    audio_float = signal.filtfilt(b, a, audio_float)
            
            # Apply multiple notch filters to remove 500Hz gain noise and its harmonics
            if self.config.enable_gain_noise_filtering:
                from scipy import signal
                nyquist = sample_rate / 2
                
                # Calculate all harmonic frequencies up to the maximum
                harmonic_freqs = []
                freq = self.config.gain_noise_base_freq
                while freq <= self.config.gain_noise_max_freq and freq < nyquist * 0.99:
                    harmonic_freqs.append(freq)
                    freq += self.config.gain_noise_base_freq
                
                logging.debug(f"Applying gain noise filters at frequencies: {harmonic_freqs}")
                
                # Apply notch filter for each harmonic frequency
                for harmonic_freq in harmonic_freqs:
                    freq_norm = harmonic_freq / nyquist
                    if freq_norm < 0.99:  # Ensure we don't exceed Nyquist frequency
                        b, a = signal.iirnotch(freq_norm, self.config.gain_noise_q)
                        audio_float = signal.filtfilt(b, a, audio_float)
                        logging.debug(f"Applied notch filter at {harmonic_freq}Hz")
            
            # Convert back to original dtype
            if audio_data.dtype == np.int16:
                scaled_audio = audio_float * 32768.0
                scaled_audio = np.clip(scaled_audio, -32768, 32767)
                return scaled_audio.astype(np.int16)
            elif audio_data.dtype == np.int32:
                scaled_audio = audio_float * 2147483647.0
                scaled_audio = np.clip(scaled_audio, -2147483648, 2147483647)
                return scaled_audio.astype(np.int32)
            else:
                return audio_float.astype(audio_data.dtype)
                
        except Exception as e:
            logging.warning(f"Audio filtering failed: {e}, returning original audio")
            return audio_data
    
    def _resample_audio(self, audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio data to target sample rate"""
        if original_sr == target_sr:
            return audio_data
        
        # Validate input audio data
        if audio_data is None or len(audio_data) == 0:
            logging.warning("Empty audio data for resampling")
            return audio_data
        
        # Check for NaN or infinite values in input
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            logging.warning("NaN or infinite values in input audio for resampling, returning original")
            return audio_data
        
        try:
            if HAS_LIBROSA:
                # Convert to float for processing if needed
                if audio_data.dtype == np.int16:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_float = audio_data.astype(np.float32) / 2147483647.0
                else:
                    audio_float = audio_data.astype(np.float32)
                
                # Use librosa for high-quality resampling (without kaiser_best to avoid resampy dependency)
                try:
                    resampled = librosa.resample(
                        audio_float, 
                        orig_sr=original_sr, 
                        target_sr=target_sr
                    )
                except Exception as e:
                    logging.warning(f"librosa resampling failed: {e}, using scipy fallback")
                    # Fallback to scipy
                    from scipy import signal
                    resampled = signal.resample(audio_float, int(len(audio_float) * target_sr / original_sr))
                
                # Check for NaN or infinite values in resampled audio
                if np.any(np.isnan(resampled)) or np.any(np.isinf(resampled)):
                    logging.warning(f"NaN or infinite values in resampled audio, using original")
                    return audio_data
                
                # Skip smoothing to avoid artifacts
                
                # Convert back to original dtype with conservative scaling
                if self.config.dtype == np.int16:
                    # Use standard 16-bit scaling
                    scaled_audio = resampled * 32768.0
                    scaled_audio = np.clip(scaled_audio, -32768, 32767)
                    return scaled_audio.astype(np.int16)
                elif self.config.dtype == np.int32:
                    # Use standard 32-bit scaling
                    scaled_audio = resampled * 2147483647.0
                    scaled_audio = np.clip(scaled_audio, -2147483648, 2147483647)
                    return scaled_audio.astype(np.int32)
                else:
                    return resampled.astype(self.config.dtype)
            else:
                # High-quality fallback using scipy
                try:
                    from scipy import signal
                    # Convert to float for processing
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_float = audio_data.astype(np.float32) / 2147483647.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    # Use scipy's high-quality resampling
                    resampled = signal.resample(audio_float, int(len(audio_float) * target_sr / original_sr))
                    
                    # Convert back to original dtype
                    if self.config.dtype == np.int16:
                        scaled_audio = resampled * 32768.0
                        scaled_audio = np.clip(scaled_audio, -32768, 32767)
                        return scaled_audio.astype(np.int16)
                    elif self.config.dtype == np.int32:
                        scaled_audio = resampled * 2147483647.0
                        scaled_audio = np.clip(scaled_audio, -2147483648, 2147483647)
                        return scaled_audio.astype(np.int32)
                    else:
                        return resampled.astype(self.config.dtype)
                        
                except ImportError:
                    # Basic linear interpolation as last resort
                    ratio = target_sr / original_sr
                    new_length = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
                    
                    # Check for NaN or infinite values
                    if np.any(np.isnan(resampled)) or np.any(np.isinf(resampled)):
                        logging.warning(f"NaN or infinite values in basic resampling, using original")
                        return audio_data
                    
                    return resampled.astype(self.config.dtype)
        except Exception as e:
            logging.error(f"Resampling failed: {e}")
            return audio_data  # Return original if resampling fails

    def _save_current_chunk(self, partial: bool = False):
        """Save the current audio chunk to file at target sample rate"""
        if not hasattr(self, 'current_chunks') or not self.current_chunks:
            return
            
        try:
            # Generate base filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_partial" if partial else ""
            
            # Save combined multi-channel file if enabled
            if self.config.save_combined:
                self._save_combined_file(timestamp, suffix)
            
            # Save individual channel files if enabled
            if self.config.save_individual:
                self._save_individual_files(timestamp, suffix)
            
            # Clean up
            self._reset_chunk()
            print("✅ Chunk saved successfully")
            
        except Exception as e:
            logging.error(f"❌ Error saving audio chunk: {e}")
            self._reset_chunk()

    def _save_combined_file(self, timestamp: str, suffix: str):
        """Save combined multi-channel file"""
        logging.info("💾 Saving combined multi-channel file...")
        try:
            # Prepare multi-channel audio data
            if len(self.current_chunks) == 1:
                # Mono audio
                combined_audio = np.array(self.current_chunks[0], dtype=self.config.dtype)
                channels = 1
            else:
                # Multi-channel audio - combine channels with proper length handling
                channel_lengths = [len(ch) for ch in self.current_chunks if ch]
                max_len = max(channel_lengths) if channel_lengths else 0
                
                if max_len == 0:
                    logging.warning("No audio data in any channel")
                    return
                
                # Create combined array
                combined_audio = np.zeros((max_len, len(self.current_chunks)), dtype=self.config.dtype)
                for ch_idx, channel_data in enumerate(self.current_chunks):
                    if channel_data:
                        min_len = min(len(channel_data), max_len)
                        combined_audio[:min_len, ch_idx] = channel_data[:min_len]
                
                channels = len(self.current_chunks)
            
            # Apply amplification to combined audio if enabled
            if self.config.enable_amplification:
                if combined_audio.ndim == 1:
                    combined_audio = self._amplify_audio(combined_audio, 0)
                else:
                    # Apply amplification to each channel
                    for ch in range(combined_audio.shape[1]):
                        combined_audio[:, ch] = self._amplify_audio(combined_audio[:, ch], ch)
                logging.debug(f"Applied amplification to combined audio")
            
            # Resample combined audio to target sample rate if needed
            if self.config.native_sample_rate != self.config.target_sample_rate:
                if combined_audio.ndim == 1:
                    combined_audio = self._resample_audio(combined_audio, self.config.native_sample_rate, self.config.target_sample_rate)
                else:
                    # Resample each channel
                    for ch in range(combined_audio.shape[1]):
                        combined_audio[:, ch] = self._resample_audio(combined_audio[:, ch], self.config.native_sample_rate, self.config.target_sample_rate)
                sample_rate = self.config.target_sample_rate
            else:
                sample_rate = self.config.native_sample_rate
                logging.debug("No resampling needed - native rate matches target rate")
            
            # Save combined multi-channel file
            combined_filename = f"recording_{timestamp}{suffix}.wav"
            combined_filepath = self.output_dir / combined_filename
            self._save_audio_file(combined_audio, combined_filepath, sample_rate, channels=channels)
            
            duration = len(combined_audio) / sample_rate
            logging.info(f"✅ Saved combined {duration:.1f}s audio ({sample_rate}Hz, {channels} channels) to: {combined_filename}")
            
        except Exception as e:
            logging.error(f"Failed to save combined audio file: {e}")

    def _save_individual_files(self, timestamp: str, suffix: str):
        """Save individual channel files"""
        logging.info("💾 Saving individual channel files...")
        
        for ch_idx, channel_data in enumerate(self.current_chunks):
            if not channel_data:
                logging.warning(f"Channel {ch_idx}: no data, skipping")
                continue
            
            try:
                # Convert to numpy array
                channel_audio = np.array(channel_data, dtype=self.config.dtype)
                logging.debug(f"Channel {ch_idx}: converted to numpy array with shape {channel_audio.shape}")
                
                # Apply noise filtering first to remove gain whine
                if self.config.enable_noise_filtering:
                    channel_audio = self._filter_audio(channel_audio, self.config.native_sample_rate)
                    logging.debug(f"Channel {ch_idx}: applied noise filtering")
                
                # Apply channel-specific amplification if enabled
                if self.config.enable_amplification:
                    channel_audio = self._amplify_audio(channel_audio, ch_idx)
                    logging.debug(f"Channel {ch_idx}: applied amplification (gain: {self.channel_gains[ch_idx]})")
                
                # Resample if needed
                if self.config.native_sample_rate != self.config.target_sample_rate:
                    channel_audio = self._resample_audio(channel_audio, self.config.native_sample_rate, self.config.target_sample_rate)
                    sample_rate = self.config.target_sample_rate
                    logging.debug(f"Channel {ch_idx}: resampled to {sample_rate}Hz")
                else:
                    sample_rate = self.config.native_sample_rate
                    logging.debug(f"Channel {ch_idx}: using native sample rate {sample_rate}Hz")
                
                # Generate filename for this channel
                channel_name = self.channel_names[ch_idx] if ch_idx < len(self.channel_names) else f"ch{ch_idx+1}"
                channel_filename = f"recording_{timestamp}_{channel_name}{suffix}.wav"
                channel_filepath = self.output_dir / channel_filename
                
                # Save individual channel audio
                self._save_audio_file(channel_audio, channel_filepath, sample_rate, channels=1)
                
                duration = len(channel_audio) / sample_rate
                print(f"✅ Saved channel {ch_idx+1} ({channel_name}): {duration:.1f}s audio ({sample_rate}Hz) to: {channel_filename}")
                
            except Exception as e:
                logging.error(f"❌ Failed to save channel {ch_idx+1}: {e}")
                logging.error(f"Channel data type: {type(channel_data)}, length: {len(channel_data)}")

    def _save_audio_file(self, audio_data: np.ndarray, filepath: Path, sample_rate: int, channels: int):
        """Save audio data to file using the best available method"""
        try:
            import soundfile as sf
            sf.write(str(filepath), audio_data, sample_rate)
            logging.debug(f"Saved using soundfile (better quality)")
        except ImportError:
            # Fallback to wave module
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(channels)
                if audio_data.dtype == np.float32:
                    # Convert float32 to int16 for wave module
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    if audio_data.ndim == 1:
                        wf.writeframes(audio_int16.tobytes())
                    else:
                        wf.writeframes(audio_int16.tobytes())
                elif audio_data.dtype == np.int32:
                    # Convert int32 to int16 for wave module
                    audio_int16 = (audio_data / 65536).astype(np.int16)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    if audio_data.ndim == 1:
                        wf.writeframes(audio_int16.tobytes())
                    else:
                        wf.writeframes(audio_int16.tobytes())
                else:
                    wf.setsampwidth(2)  # 2 bytes for int16
                    wf.setframerate(sample_rate)
                    if audio_data.ndim == 1:
                        wf.writeframes(audio_data.tobytes())
                    else:
                        wf.writeframes(audio_data.tobytes())

    def _reset_chunk(self):
        """Reset the current chunk"""
        if hasattr(self, 'current_chunks'):
            self.current_chunks = [[] for _ in range(self.config.channels)]
        else:
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
            
        # Clean console output
        print("=" * 60)
        print("🎤 STARTING MULTI-CHANNEL RECORDING")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏱️  Chunk duration: {self.config.chunk_duration} seconds")
        print(f"🔊 Channels: {self.config.channels}")
        print(f"📊 Sample rate: {self.config.native_sample_rate} Hz → {self.config.target_sample_rate} Hz")
        print(f"💾 Save combined: {self.config.save_combined}")
        print(f"💾 Save individual: {self.config.save_individual}")
        print(f"🎛️  Channel names: {self.channel_names}")
        print(f"🔧 Channel gains: {self.channel_gains}")
        if self.config.enable_noise_filtering:
            print(f"🔇 Noise filtering: HP={self.config.high_pass_freq}Hz, LP={self.config.low_pass_freq}Hz, Notch={self.config.notch_freq}Hz")
        else:
            print("🔇 Noise filtering: Disabled")
        
        if self.config.enable_gain_noise_filtering:
            # Calculate harmonic frequencies for display
            harmonic_freqs = []
            freq = self.config.gain_noise_base_freq
            while freq <= self.config.gain_noise_max_freq:
                harmonic_freqs.append(freq)
                freq += self.config.gain_noise_base_freq
            print(f"🔇 Gain noise filtering: {self.config.gain_noise_base_freq}Hz harmonics up to {self.config.gain_noise_max_freq}Hz: {harmonic_freqs}")
        else:
            print("🔇 Gain noise filtering: Disabled")
        print("Press Ctrl+C to stop recording")
        print("=" * 60)
        
        self.recording = True
        self._reset_chunk()
        
        # Initialize multi-channel structure
        self.current_chunks = [[] for _ in range(self.config.channels)]
        
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
                # Clean console output
                print("✅ RECORDING STARTED SUCCESSFULLY")
                print(f"🎧 Device: {sd.query_devices(self.input_device)['name']}")
                print(f"📊 Format: {self.config.dtype.__name__}, {self.config.channels} channels")
                print("🔴 Recording in progress...")
                
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
            
        logging.info("🛑 Stopping recording...")
        self.recording = False
        
        # Save any remaining audio as partial chunk
        if hasattr(self, 'current_chunks') and any(self.current_chunks):
            logging.info("💾 Saving partial chunk...")
            self._save_current_chunk(partial=True)
        
        if self.stream:
            self.stream.close()
            
        logging.info("✅ Recording stopped successfully")


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
    parser.add_argument("--buffer-size", type=int, default=8192,
                       help="Audio buffer size (default: 8192, increase if getting overflows)")
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
    parser.add_argument("--channels", type=int, default=4,
                       help="Number of audio channels to record (default: 4 for Focusrite 4i4)")
    parser.add_argument("--save-combined", action='store_true', default=False,
                       help="Save combined multi-channel file in addition to individual files")
    parser.add_argument("--save-individual", action='store_true', default=True,
                       help="Save individual channel files (default: enabled)")
    parser.add_argument("--no-save-individual", action='store_false', dest='save_individual',
                       help="Disable saving individual channel files")
    parser.add_argument("--channel-names", nargs='+', default=None,
                       help="Names for each channel (e.g., --channel-names mic1 mic2 mic3 mic4)")
    parser.add_argument("--apply-gain-to", nargs='+', type=int, default=[2, 3],
                       help="Channel indices to apply gain to (0-indexed, default: 2 3 for channels 3 and 4)")
    parser.add_argument("--audio-format", choices=["int16", "int32", "float32"], default="int16",
                       help="Audio format: int16 (best compatibility), int32 (S32_LE), float32 (highest quality)")
    parser.add_argument("--no-noise-filter", action="store_true",
                       help="Disable noise filtering (gain whine removal)")
    parser.add_argument("--high-pass", type=float, default=200.0,
                       help="High-pass filter frequency in Hz (default: 200)")
    parser.add_argument("--low-pass", type=float, default=8000.0,
                       help="Low-pass filter frequency in Hz (default: 8000)")
    parser.add_argument("--notch-freq", type=float, default=60.0,
                       help="Notch filter frequency for power line hum in Hz (default: 60)")
    parser.add_argument("--no-gain-noise-filter", action="store_true",
                       help="Disable 500Hz gain noise filtering")
    parser.add_argument("--gain-noise-base-freq", type=float, default=500.0,
                       help="Base gain noise frequency in Hz (default: 500)")
    parser.add_argument("--gain-noise-max-freq", type=float, default=6000.0,
                       help="Maximum frequency to filter gain noise harmonics up to in Hz (default: 6000)")
    parser.add_argument("--gain-noise-q", type=float, default=30.0,
                       help="Q factor for gain noise notch filters (default: 30)")
    
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
        gain_boost=args.gain,
        channels=args.channels,
        save_combined=args.save_combined,
        save_individual=args.save_individual,
        channel_names=args.channel_names,
        apply_gain_to_channels=args.apply_gain_to,
        enable_noise_filtering=not args.no_noise_filter,
        high_pass_freq=args.high_pass,
        low_pass_freq=args.low_pass,
        notch_freq=args.notch_freq,
        enable_gain_noise_filtering=not args.no_gain_noise_filter,
        gain_noise_base_freq=args.gain_noise_base_freq,
        gain_noise_max_freq=args.gain_noise_max_freq,
        gain_noise_q=args.gain_noise_q
    )
    
    # Override format if specified (this will override device detection)
    if args.audio_format == 'float32':
        config.dtype = np.float32
        print(f"🎛️  Using float32 format as requested")
    elif args.audio_format == 'int16':
        config.dtype = np.int16
        print(f"🎛️  Using int16 format as requested")
    elif args.audio_format == 'int32':
        config.dtype = np.int32
        print(f"🎛️  Using int32 format as requested")
    
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