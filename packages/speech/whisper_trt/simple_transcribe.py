#!/usr/bin/env python3
import os
import torch
import json
import numpy as np
import soundfile as sf
import warnings
import logging
from datetime import datetime
import gc
import whisper
import platform
import sounddevice as sd
import wave
import threading
import time
import queue
import argparse
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

@dataclass
class AudioConfig:
    """Configuration for audio processing"""
    sample_rate: int = 16000
    channels: int = 1
    dtype: type = np.int16
    chunk_duration: int = 600  # 10 minutes in seconds
    blocksize: int = 1024  # Reduced from 2048 to handle data faster
    latency: str = 'low'  # Changed from 'high' to 'low' for better real-time performance

@dataclass
class TranscriptionConfig:
    """Configuration for transcription
    
    Parameters that affect transcription sensitivity and accuracy:
    - temperature: Controls randomness in sampling (0.0 = deterministic, higher = more random)
    - no_speech_threshold: Threshold for considering a segment as silence (higher = more aggressive silence detection)
    - logprob_threshold: Minimum log probability for a token to be considered valid (lower = more lenient)
    - compression_ratio_threshold: Maximum compression ratio for a segment (higher = more aggressive compression)
    - condition_on_previous_text: Whether to condition on previous text (True = more context-aware)
    - initial_prompt: Optional initial prompt to guide transcription
    - word_timestamps: Whether to include word-level timestamps
    - prepend_punctuations: Punctuations to prepend to next word
    - append_punctuations: Punctuations to append to previous word
    """
    model_name: str = "base.en"  # Using base.en for better accuracy while maintaining reasonable speed
    language: str = "en"  # Language code
    task: str = "transcribe"  # Task: transcribe or translate
    temperature: float = 0.0  # Keep deterministic for consistency
    no_speech_threshold: float = 0.3  # Lower threshold to catch quiet speech
    logprob_threshold: float = -0.7  # More lenient token acceptance
    compression_ratio_threshold: float = 1.8  # Less aggressive compression
    condition_on_previous_text: bool = True  # Use context for better accuracy
    initial_prompt: Optional[str] = "This is a conversation between parents and children in a home environment. Sometimes there is a TV playing in the background, you can ignore transcribing that."  # Context prompt
    word_timestamps: bool = True  # Include word-level timestamps
    prepend_punctuations: str = '"\'¿([{-'  # Punctuations to prepend
    append_punctuations: str = '"\'.,!?;:")]}、'  # Punctuations to append

@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization"""
    use_auth_token: str = "hf_xxx"  # Replace with your HuggingFace token
    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 5

class AudioLogger:
    """Handles logging configuration and setup"""
    def __init__(self):
        self.log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"whisper_trt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

class GPUManager:
    """Manages GPU resources and memory"""
    _initialized = False
    _memory_threshold = 0.8  # 80% memory threshold

    @staticmethod
    def initialize_cuda():
        """Safely initialize CUDA environment"""
        if GPUManager._initialized:
            return

        try:
            logging.info("Initializing CUDA environment...")
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logging.warning("CUDA is not available")
                return

            # Check if CUDA is already initialized
            if not torch.cuda.is_initialized():
                logging.info("CUDA not initialized, initializing...")
                torch.cuda.init()
            
            # Get device info
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            logging.info(f"CUDA Device: {props.name}")
            logging.info(f"CUDA Capability: {props.major}.{props.minor}")
            logging.info(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Set more conservative memory settings
            torch.cuda.set_per_process_memory_fraction(0.4)  # Reduced from 0.6 to 0.4
            
            # Configure cuDNN
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Clear any existing memory
            torch.cuda.empty_cache()
            
            GPUManager._initialized = True
            logging.info("CUDA environment initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing CUDA: {str(e)}")
            raise

    @staticmethod
    def get_gpu_memory_info() -> Optional[Dict[str, float]]:
        if not GPUManager._initialized:
            return None
            
        try:
            if not torch.cuda.is_available():
                return None
                
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - reserved_memory
            
            return {
                'total_memory_gb': total_memory / 1024**3,
                'reserved_memory_gb': reserved_memory / 1024**3,
                'allocated_memory_gb': allocated_memory / 1024**3,
                'free_memory_gb': free_memory / 1024**3
            }
        except Exception as e:
            logging.warning(f"Error getting GPU memory info: {str(e)}")
            return None

    @staticmethod
    def log_gpu_memory():
        if not GPUManager._initialized:
            return
            
        try:
            memory_info = GPUManager.get_gpu_memory_info()
            if memory_info:
                logging.info("GPU Memory Usage:")
                for key, value in memory_info.items():
                    logging.info(f"  {key}: {value:.2f} GB")
            else:
                logging.info("No GPU available or error getting memory info")
        except Exception as e:
            logging.warning(f"Error logging GPU memory: {str(e)}")

    @staticmethod
    def cleanup_memory():
        """Safely clean up GPU memory"""
        if not GPUManager._initialized:
            return
            
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            logging.warning(f"Error during memory cleanup: {str(e)}")

    @staticmethod
    def check_cuda_compatibility():
        """Check CUDA compatibility and set up GPU environment"""
        try:
            logging.info("Checking CUDA compatibility...")
            is_jetson = os.path.exists('/etc/nv_tegra_release')
            logging.info(f"Running on Jetson: {is_jetson}")
            
            if torch.cuda.is_available():
                try:
                    # Initialize CUDA environment
                    GPUManager.initialize_cuda()
                    
                    # Log CUDA version and device info
                    cuda_version = torch.version.cuda
                    device = torch.cuda.current_device()
                    props = torch.cuda.get_device_properties(device)
                    
                    logging.info(f"CUDA Version: {cuda_version}")
                    logging.info(f"GPU Device: {props.name}")
                    logging.info(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
                    
                    # Only enable TF32 if available and on a compatible device
                    tf32_available = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                    if tf32_available:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        logging.info("TF32 enabled for better performance")
                    
                    if is_jetson:
                        logging.info("Jetson-specific CUDA settings applied")
                        
                except Exception as e:
                    logging.warning(f"Error setting up CUDA: {str(e)}")
            else:
                logging.warning("CUDA is not available, running on CPU")
        except Exception as e:
            logging.error(f"Error checking CUDA compatibility: {str(e)}")
            logging.warning("Falling back to CPU mode")

    @staticmethod
    def check_memory_usage():
        """Check if memory usage is above threshold"""
        try:
            if not torch.cuda.is_available():
                return False
                
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)
            
            memory_usage = allocated_memory / total_memory
            if memory_usage > GPUManager._memory_threshold:
                logging.warning(f"High memory usage detected: {memory_usage:.2%}")
                return True
            return False
        except Exception as e:
            logging.warning(f"Error checking memory usage: {str(e)}")
            return False

class AudioValidator:
    """Handles audio file validation and preprocessing"""
    @staticmethod
    def validate_audio_file(audio_file: str) -> Tuple[np.ndarray, int]:
        """Validate audio file and return processed audio data and sample rate.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Tuple of (audio_data, samplerate)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is empty or invalid
            RuntimeError: If audio processing fails
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        try:
            # Check file size first
            file_size = os.path.getsize(audio_file)
            if file_size == 0:
                raise ValueError("Audio file is empty (0 bytes)")
            
            # Try to read the audio file
            try:
                data, samplerate = sf.read(audio_file)
            except Exception as e:
                raise RuntimeError(f"Failed to read audio file: {str(e)}")
            
            # Validate audio data
            if data is None or len(data) == 0:
                raise ValueError("Audio file contains no data")
            
            duration = len(data) / samplerate
            if duration == 0:
                raise ValueError("Audio file has zero duration")
            
            # Check if audio is too short (less than 0.1 seconds)
            if duration < 0.1:
                raise ValueError(f"Audio file is too short ({duration:.2f} seconds)")
            
            # Convert to float32 and ensure correct shape
            try:
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
            except Exception as e:
                raise RuntimeError(f"Failed to process audio data: {str(e)}")
            
            # Check for silent audio
            max_val = np.max(np.abs(data))
            if max_val == 0:
                raise ValueError("Audio file contains only silence")
            
            # Normalize audio
            try:
                if max_val > 0:
                    data = data / max_val
            except Exception as e:
                raise RuntimeError(f"Failed to normalize audio: {str(e)}")
            
            # Resample to 16kHz if needed
            if samplerate != 16000:
                try:
                    logging.info(f"Resampling audio from {samplerate}Hz to 16000Hz")
                    import librosa
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                    samplerate = 16000
                except Exception as e:
                    raise RuntimeError(f"Failed to resample audio: {str(e)}")
            
            # Final validation checks
            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Audio data contains NaN or Inf values")
            
            if len(data) < 1600:  # Less than 0.1 seconds at 16kHz
                raise ValueError("Audio file is too short after processing")
            
            # Log audio statistics
            logging.info(f"Audio file validated: {duration:.2f} seconds, {len(data)} samples, {samplerate} Hz")
            logging.info(f"Audio data min: {np.min(data):.3f}, max: {np.max(data):.3f}, mean: {np.mean(data):.3f}")
            logging.info(f"Audio data shape: {data.shape}, dtype: {data.dtype}")
            
            # Check if audio is too quiet
            if np.max(np.abs(data)) < 0.01:
                logging.warning("Audio signal appears to be very quiet (max amplitude < 0.01)")
            
            return data, samplerate
            
        except Exception as e:
            logging.error(f"Error validating audio file: {str(e)}")
            raise

    @staticmethod
    def validate_audio_data(audio_data: np.ndarray, samplerate: int) -> Tuple[np.ndarray, int]:
        """Validate and preprocess audio data.
        
        Args:
            audio_data: Raw audio data as numpy array
            samplerate: Sample rate of the audio data
            
        Returns:
            Tuple of (processed_audio_data, samplerate)
            
        Raises:
            ValueError: If audio data is invalid
            RuntimeError: If audio processing fails
        """
        try:
            # Check if data is empty
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("Audio data is empty")
            
            # Convert to float32 and ensure correct shape
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Check for silent audio
            max_val = np.max(np.abs(audio_data))
            if max_val == 0:
                raise ValueError("Audio data contains only silence")
            
            # Normalize audio
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Resample if needed
            if samplerate != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
                samplerate = 16000
            
            # Final validation
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                raise ValueError("Audio data contains NaN or Inf values")
            
            if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
                raise ValueError("Audio data is too short after processing")
            
            return audio_data, samplerate
            
        except Exception as e:
            logging.error(f"Error validating audio data: {str(e)}")
            raise

class ModelManager:
    """Manages model loading and inference"""
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Initialize CUDA before loading model
        if torch.cuda.is_available():
            GPUManager.initialize_cuda()
            
        # Initialize model after CUDA setup
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with proper error handling and memory management"""
        try:
            # Clean up any existing model
            if self.model is not None:
                try:
                    del self.model
                except Exception as e:
                    logging.warning(f"Error cleaning up existing model: {str(e)}")
            
            # Only perform memory cleanup if CUDA is available and initialized
            if torch.cuda.is_available() and GPUManager._initialized:
                try:
                    GPUManager.cleanup_memory()
                except Exception as e:
                    logging.warning(f"Error during memory cleanup: {str(e)}")
            
            logging.info(f"Loading Whisper model ({self.config.model_name})...")
            try:
                # Try regular Whisper model first
                self.model = whisper.load_model(self.config.model_name)
                if torch.cuda.is_available() and GPUManager._initialized:
                    self.model = self.model.to(self.device)
                self.model.eval()
                logging.info("Successfully loaded regular Whisper model")
            except Exception as e:
                logging.error(f"Error loading regular Whisper model: {str(e)}")
                logging.info("Falling back to Whisper-TRT model...")
                try:
                    from whisper_trt import load_trt_model
                    self.model = load_trt_model(self.config.model_name, verbose=True)
                    logging.info("Successfully loaded Whisper-TRT model")
                except Exception as e:
                    logging.error(f"Error loading Whisper-TRT model: {str(e)}")
                    raise
        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            raise

    def transcribe(self, audio_tensor: torch.Tensor) -> Dict[str, Any]:
        """Transcribe audio with proper error handling and memory management"""
        try:
            # Ensure model is initialized
            if self.model is None:
                self._initialize_model()
            
            # Validate input tensor
            if audio_tensor is None or audio_tensor.numel() == 0:
                raise ValueError("Empty audio tensor provided")
            
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                raise ValueError("Audio tensor contains NaN or Inf values")
            
            # Ensure tensor is on the correct device
            if audio_tensor.device != self.device:
                audio_tensor = audio_tensor.to(self.device)
            
            # Clean up memory before transcription
            GPUManager.cleanup_memory()
            
            # Perform transcription with proper error handling
            try:
                if isinstance(self.model, whisper.Whisper):
                    with torch.cuda.amp.autocast(enabled=True):
                        with torch.no_grad():
                            result = self.model.transcribe(
                                audio_tensor,
                                language=self.config.language,
                                task=self.config.task,
                                fp16=torch.cuda.is_available(),
                                verbose=True,
                                temperature=self.config.temperature,
                                no_speech_threshold=self.config.no_speech_threshold,
                                logprob_threshold=self.config.logprob_threshold,
                                compression_ratio_threshold=self.config.compression_ratio_threshold
                            )
                else:
                    result = self.model.transcribe(audio_tensor)
                
                # Validate result
                if not result or not isinstance(result, dict):
                    raise ValueError("Invalid transcription result")
                
                if not result.get('text'):
                    logging.warning("Transcription returned empty text")
                
                return result
                
            except Exception as e:
                logging.error(f"Error during transcription: {str(e)}")
                # Try to reinitialize model and retry once
                logging.info("Attempting to reinitialize model and retry...")
                self._initialize_model()
                if isinstance(self.model, whisper.Whisper):
                    with torch.cuda.amp.autocast(enabled=True):
                        with torch.no_grad():
                            return self.model.transcribe(
                                audio_tensor,
                                language=self.config.language,
                                task=self.config.task,
                                fp16=torch.cuda.is_available(),
                                verbose=True,
                                temperature=self.config.temperature,
                                no_speech_threshold=self.config.no_speech_threshold,
                                logprob_threshold=self.config.logprob_threshold,
                                compression_ratio_threshold=self.config.compression_ratio_threshold
                            )
                else:
                    return self.model.transcribe(audio_tensor)
                    
        except Exception as e:
            logging.error(f"Fatal error during transcription: {str(e)}")
            raise
        finally:
            # Clean up memory after transcription
            GPUManager.cleanup_memory()

    def __del__(self):
        """Cleanup when the model manager is destroyed"""
        try:
            if self.model is not None:
                del self.model
            GPUManager.cleanup_memory()
        except Exception as e:
            logging.error(f"Error during model cleanup: {str(e)}")

class AudioRecorder:
    """Handles audio recording and processing"""
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recording = False
        self.audio_queue = queue.Queue(maxsize=100)  # Reduced maxsize to prevent memory buildup
        self.current_chunk = []
        self.overflow_count = 0
        self.last_overflow_time = time.time()
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        self.chunk_size = 0
        self.max_chunk_size = self.config.chunk_duration * self.config.sample_rate
        
        self._setup_audio_device()

    def _setup_audio_device(self):
        """Setup audio device with optimized settings"""
        logging.info("\nAvailable audio devices:")
        logging.info(sd.query_devices())
        
        # Try to find the best input device
        devices = sd.query_devices()
        self.input_device = 0
        
        # Look for a device with good buffer settings
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # It's an input device
                if device.get('default_samplerate', 0) >= self.config.sample_rate:
                    self.input_device = i
                    break
        
        device_info = sd.query_devices(self.input_device)
        logging.info(f"\nUsing input device: {device_info['name']}")
        logging.info(f"Device settings: {device_info}")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording with improved overflow handling"""
        if status:
            current_time = time.time()
            if status.input_overflow:
                self.overflow_count += 1
                if current_time - self.last_overflow_time > 5:
                    logging.warning(f"Input buffer overflow detected. This may cause audio loss. (Overflow count: {self.overflow_count})")
                    self.last_overflow_time = current_time
                    
                    # More aggressive queue clearing on overflow
                    if self.audio_queue.qsize() > 50:  # Reduced threshold
                        try:
                            # Clear more data to prevent overflow
                            for _ in range(min(50, self.audio_queue.qsize())):
                                self.audio_queue.get_nowait()
                            logging.info("Cleared audio queue to prevent overflow")
                        except queue.Empty:
                            pass
            else:
                logging.warning(f"Status: {status}")
                
        if self.recording:
            try:
                # Check queue size and warn if getting too large
                if self.audio_queue.qsize() > 50:  # Reduced threshold
                    logging.warning(f"Audio queue is getting large ({self.audio_queue.qsize()} chunks). Processing may be falling behind.")
                    # Clear some data to prevent memory buildup
                    try:
                        for _ in range(min(25, self.audio_queue.qsize())):
                            self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Add data to current chunk
                self.current_chunk.extend(indata)
                self.chunk_size += len(indata)
                
                # If we've reached the chunk size, add to queue and reset
                if self.chunk_size >= self.max_chunk_size:
                    try:
                        # Convert to numpy array and copy once
                        audio_data = np.array(self.current_chunk, dtype=self.config.dtype)
                        # Use non-blocking put with timeout
                        self.audio_queue.put(audio_data, timeout=0.1)
                        # Reset chunk
                        self.current_chunk = []
                        self.chunk_size = 0
                    except queue.Full:
                        logging.warning("Audio queue is full, dropping chunk")
                        # Reset chunk even if queue is full
                        self.current_chunk = []
                        self.chunk_size = 0
                    
            except Exception as e:
                logging.error(f"Error in audio callback: {str(e)}")
                # Reset chunk on error
                self.current_chunk = []
                self.chunk_size = 0

    def save_audio_chunk(self, audio_data: np.ndarray, filename: str):
        """Save audio chunk to file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data.tobytes())
        except Exception as e:
            logging.error(f"Error saving audio chunk: {str(e)}")
            raise

    def start_recording(self):
        """Start audio recording with optimized settings"""
        self.recording = True
        self.current_chunk = []
        self.chunk_size = 0
        self.overflow_count = 0
        self.last_overflow_time = time.time()
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self.audio_callback,
                blocksize=self.config.blocksize,
                latency=self.config.latency,
                device=self.input_device
            )
            self.stream.start()
            logging.info("\nAudio stream started successfully")
            logging.info(f"Stream settings: blocksize={self.config.blocksize}, latency={self.config.latency}")
        except Exception as e:
            logging.error(f"Error starting audio stream: {str(e)}")
            raise

    def stop_recording(self):
        """Stop audio recording and cleanup"""
        self.recording = False
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
                del self.stream
            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.current_chunk = []
            self.chunk_size = 0
            logging.info("Recording stopped.")
        except Exception as e:
            logging.error(f"Error stopping recording: {str(e)}")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            self.stop_recording()
        except Exception as e:
            logging.warning(f"Error during audio recorder cleanup: {str(e)}")

class DiarizationManager:
    """Manages speaker diarization using pyannote.audio"""
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the pyannote.audio pipeline"""
        try:
            logging.info("Initializing speaker diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.use_auth_token
            )
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            logging.info("Speaker diarization pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing diarization pipeline: {str(e)}")
            raise

    def process_audio(self, audio_file: str) -> List[Dict[str, Any]]:
        """Process audio file and return speaker segments
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            List of dictionaries containing speaker segments with start time, end time, and speaker ID
        """
        try:
            logging.info(f"Processing audio file for diarization: {audio_file}")
            diarization = self.pipeline(
                audio_file,
                num_speakers=self.config.num_speakers,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
            
            # Convert diarization results to a list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            logging.info(f"Found {len(segments)} speaker segments")
            return segments
        except Exception as e:
            logging.error(f"Error during diarization: {str(e)}")
            return []

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if self.pipeline is not None:
                del self.pipeline
                torch.cuda.empty_cache()
        except Exception as e:
            logging.warning(f"Error during diarization cleanup: {str(e)}")

class TranscriptionManager:
    """Manages the transcription process"""
    def __init__(self, model_config: TranscriptionConfig, audio_config: AudioConfig, diarization_config: Optional[DiarizationConfig] = None, record_only: bool = False):
        self.model_config = model_config
        self.audio_config = audio_config
        self.record_only = record_only
        if not record_only:
            self.model_manager = ModelManager(model_config)
            self.diarization_manager = DiarizationManager(diarization_config) if diarization_config else None
        self.audio_recorder = AudioRecorder(audio_config)
        self.process_thread = None
        self.chunk_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.processing_thread = None
        self._running = False
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 60  # Cleanup every 60 seconds

    def process_audio_chunks(self):
        """Collect audio chunks and add them to the processing queue"""
        while self._running:
            try:
                # Get data from audio recorder queue
                try:
                    audio_data = self.audio_recorder.audio_queue.get(timeout=1)
                    
                    # Check if we need to do periodic cleanup
                    current_time = time.time()
                    if current_time - self.last_cleanup_time > self.cleanup_interval:
                        self._perform_cleanup()
                        self.last_cleanup_time = current_time
                    
                    # Add to processing queue with timeout
                    try:
                        self.chunk_queue.put(audio_data, timeout=0.1)
                        logging.info(f"Added new audio chunk to processing queue (size: {len(audio_data)} samples)")
                    except queue.Full:
                        logging.warning("Processing queue is full, dropping chunk")
                        # Force cleanup if queue is full
                        self._perform_cleanup()
                        
                except queue.Empty:
                    if not self._running:
                        break
                    continue
                    
            except Exception as e:
                logging.error(f"Error in audio chunk collection: {str(e)}")
                if not self._running:
                    break
                continue

    def _perform_cleanup(self):
        """Perform periodic cleanup of resources"""
        try:
            # Clear processing queue if it's getting too large
            while not self.chunk_queue.empty():
                try:
                    self.chunk_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info("Performed periodic cleanup")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def process_chunks(self):
        """Process audio chunks from the queue and transcribe them"""
        while self._running or not self.chunk_queue.empty():
            try:
                # Get the next chunk from the queue
                audio_data = self.chunk_queue.get(timeout=1)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = self.audio_recorder.recordings_dir / f"recording_{timestamp}.wav"
                
                # Save the audio file
                self.audio_recorder.save_audio_chunk(audio_data, str(audio_file))
                logging.info(f"\nSaved audio chunk to {audio_file}")
                
                if not self.record_only:
                    # Transcribe immediately
                    self._transcribe_and_save(audio_file, timestamp)
                    
                    # Clean up after each chunk
                    if os.path.exists(audio_file):
                        os.remove(audio_file)  # Remove audio file after processing
                else:
                    logging.info(f"Record-only mode: Saved audio chunk {timestamp}")
                
                # Mark the chunk as processed
                self.chunk_queue.task_done()
                
                # Clean up audio data
                del audio_data
                gc.collect()
                
            except queue.Empty:
                if not self._running:
                    break
                continue
            except Exception as e:
                logging.error(f"Error processing audio chunk: {str(e)}")
                continue

    def _transcribe_and_save(self, audio_file: Path, timestamp: str):
        """Transcribe audio file and save results with speaker diarization if enabled"""
        try:
            # Get transcription
            audio_data, samplerate = AudioValidator.validate_audio_file(str(audio_file))
            audio_tensor = torch.from_numpy(audio_data).float()
            
            # Clear audio data after creating tensor
            del audio_data
            gc.collect()
            
            result = self.model_manager.transcribe(audio_tensor)
            
            # Clear tensor after transcription
            del audio_tensor
            gc.collect()
            
            # Get diarization if enabled
            diarization_segments = []
            if self.diarization_manager:
                diarization_segments = self.diarization_manager.process_audio(str(audio_file))
            
            # Combine transcription with diarization
            output = {
                "timestamp": timestamp,
                "audio_file": str(audio_file),
                "transcription": result,
                "diarization": diarization_segments
            }
            
            # Save results
            output_file = audio_file.parent / f"transcript_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            logging.info(f"Saved transcription and diarization to {output_file}")
            
            # Clean up
            del result
            del diarization_segments
            del output
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error in transcription and diarization: {str(e)}")

    def start_continuous_transcription(self):
        """Start recording and processing audio chunks"""
        self._running = True
        self.last_cleanup_time = time.time()
        self.audio_recorder.start_recording()
        
        # Start the collection thread
        self.process_thread = threading.Thread(target=self.process_audio_chunks)
        self.process_thread.start()
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self.process_chunks)
        self.processing_thread.start()
        
        logging.info("Recording started... Press Ctrl+C to stop.")

    def stop_continuous_transcription(self):
        """Stop recording and wait for all chunks to be processed"""
        logging.info("Stopping recording...")
        self._running = False
        
        # Stop the audio recorder first
        if hasattr(self, 'audio_recorder') and self.audio_recorder:
            self.audio_recorder.stop_recording()
        
        # Wait for the collection thread to finish
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
        
        # Wait for all chunks to be processed
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Clear the chunk queue
        if hasattr(self, 'chunk_queue'):
            while not self.chunk_queue.empty():
                try:
                    self.chunk_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Clean up resources
        try:
            if not self.record_only:
                if hasattr(self, 'model_manager') and self.model_manager:
                    del self.model_manager
                if hasattr(self, 'diarization_manager') and self.diarization_manager:
                    del self.diarization_manager
            if hasattr(self, 'audio_recorder') and self.audio_recorder:
                del self.audio_recorder
            GPUManager.cleanup_memory()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
        
        logging.info("Recording stopped.")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if hasattr(self, '_running') and self._running:
                self.stop_continuous_transcription()
        except Exception as e:
            logging.warning(f"Error during transcription manager cleanup: {str(e)}")

class SimpleAudioRecorder:
    """Simplified audio recorder for record-only mode - no queues, direct file saving"""
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recording = False
        self.current_chunk = []
        self.chunk_size = 0
        self.max_chunk_size = self.config.chunk_duration * self.config.sample_rate
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        self._setup_audio_device()

    def _setup_audio_device(self):
        """Setup audio device with optimized settings"""
        logging.info("\nAvailable audio devices:")
        logging.info(sd.query_devices())
        
        # Try to find the best input device
        devices = sd.query_devices()
        self.input_device = 0
        
        # Look for a device with good buffer settings
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # It's an input device
                if device.get('default_samplerate', 0) >= self.config.sample_rate:
                    self.input_device = i
                    break
        
        device_info = sd.query_devices(self.input_device)
        logging.info(f"\nUsing input device: {device_info['name']}")
        logging.info(f"Device settings: {device_info}")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording - direct file saving"""
        if status:
            if status.input_overflow:
                logging.warning("Input buffer overflow detected. This may cause audio loss.")
            else:
                logging.warning(f"Status: {status}")
                
        if self.recording:
            try:
                # Add data to current chunk
                self.current_chunk.extend(indata)
                self.chunk_size += len(indata)
                
                # If we've reached the chunk size, save directly to file
                if self.chunk_size >= self.max_chunk_size:
                    # Convert to numpy array
                    audio_data = np.array(self.current_chunk, dtype=self.config.dtype)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_file = self.recordings_dir / f"recording_{timestamp}.wav"
                    
                    # Save the audio file directly
                    self.save_audio_chunk(audio_data, str(audio_file))
                    logging.info(f"Saved audio chunk to {audio_file}")
                    
                    # Reset chunk
                    self.current_chunk = []
                    self.chunk_size = 0
                    
                    # Clean up audio data immediately
                    del audio_data
                    
            except Exception as e:
                logging.error(f"Error in audio callback: {str(e)}")
                # Reset chunk on error
                self.current_chunk = []
                self.chunk_size = 0

    def save_audio_chunk(self, audio_data: np.ndarray, filename: str):
        """Save audio chunk to file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data.tobytes())
        except Exception as e:
            logging.error(f"Error saving audio chunk: {str(e)}")
            raise

    def start_recording(self):
        """Start audio recording with optimized settings"""
        self.recording = True
        self.current_chunk = []
        self.chunk_size = 0
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self.audio_callback,
                blocksize=self.config.blocksize,
                latency=self.config.latency,
                device=self.input_device
            )
            self.stream.start()
            logging.info("\nAudio stream started successfully")
            logging.info(f"Stream settings: blocksize={self.config.blocksize}, latency={self.config.latency}")
        except Exception as e:
            logging.error(f"Error starting audio stream: {str(e)}")
            raise

    def stop_recording(self):
        """Stop audio recording and cleanup"""
        self.recording = False
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
                del self.stream
            
            # Save any remaining audio data
            if self.current_chunk and self.chunk_size > 0:
                audio_data = np.array(self.current_chunk, dtype=self.config.dtype)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = self.recordings_dir / f"recording_{timestamp}_partial.wav"
                self.save_audio_chunk(audio_data, str(audio_file))
                logging.info(f"Saved partial audio chunk to {audio_file}")
                del audio_data
            
            self.current_chunk = []
            self.chunk_size = 0
            logging.info("Recording stopped.")
        except Exception as e:
            logging.error(f"Error stopping recording: {str(e)}")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            self.stop_recording()
        except Exception as e:
            logging.warning(f"Error during simple audio recorder cleanup: {str(e)}")

def process_recordings_in_range(start_time: datetime, end_time: datetime, model_config: TranscriptionConfig, diarization_config: Optional[DiarizationConfig] = None):
    """Process all recordings within a specified time range"""
    try:
        # Initialize model manager
        model_manager = ModelManager(model_config)
        
        # Get recordings directory
        recordings_dir = Path("recordings")
        if not recordings_dir.exists():
            logging.error("Recordings directory not found")
            return
        
        # Find all WAV files in the directory
        wav_files = list(recordings_dir.glob("recording_*.wav"))
        if not wav_files:
            logging.error("No recording files found")
            return
        
        # Filter files by date range
        files_to_process = []
        for wav_file in wav_files:
            try:
                # Extract timestamp from filename (format: recording_YYYYMMDD_HHMMSS.wav)
                timestamp_str = wav_file.stem.split('_')[1] + '_' + wav_file.stem.split('_')[2]
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if start_time <= file_time <= end_time:
                    files_to_process.append(wav_file)
            except Exception as e:
                logging.warning(f"Error parsing timestamp from {wav_file}: {str(e)}")
                continue
        
        if not files_to_process:
            logging.error(f"No recordings found between {start_time} and {end_time}")
            return
        
        logging.info(f"Found {len(files_to_process)} recordings to process")
        
        # Process each file
        for wav_file in sorted(files_to_process):
            try:
                logging.info(f"\nProcessing {wav_file.name}...")
                
                # Check if transcript already exists
                transcript_file = wav_file.parent / f"transcript_{wav_file.stem.split('_')[1]}_{wav_file.stem.split('_')[2]}.json"
                if transcript_file.exists():
                    logging.info(f"Transcript already exists for {wav_file.name}, skipping...")
                    continue
                
                # Validate and load audio
                audio_data, _ = AudioValidator.validate_audio_file(str(wav_file))
                audio_tensor = torch.from_numpy(audio_data).float().to(model_manager.device)
                
                # Transcribe
                result = model_manager.transcribe(audio_tensor)
                
                # Get diarization if enabled
                diarization_segments = []
                if diarization_config:
                    diarization_manager = DiarizationManager(diarization_config)
                    diarization_segments = diarization_manager.process_audio(str(wav_file))
                
                # Save results
                output = {
                    "audio_file": str(wav_file),
                    "transcription": result,
                    "diarization": diarization_segments
                }
                
                with open(transcript_file, 'w') as f:
                    json.dump(output, f, indent=2)
                
                logging.info(f"Saved transcription to {transcript_file}")
                
                # Clean up
                del audio_data
                del audio_tensor
                del result
                del diarization_segments
                del output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error processing {wav_file}: {str(e)}")
                continue
        
        logging.info("\nBatch processing completed")
        
    except Exception as e:
        logging.error(f"Fatal error during batch processing: {str(e)}")
    finally:
        # Clean up
        if 'model_manager' in locals():
            del model_manager
        GPUManager.cleanup_memory()

def main():
    parser = argparse.ArgumentParser(description='Whisper Transcription Tool')
    parser.add_argument('--mode', choices=['file', 'continuous', 'batch'], default='file',
                      help='Transcription mode: "file" for single file, "continuous" for continuous recording, "batch" for processing multiple recordings')
    parser.add_argument('--model', default='base.en',
                      help='Model to use (tiny.en, base.en, small.en, medium.en, large)')
    parser.add_argument('--audio_file', help='Path to audio file (required for file mode)')
    parser.add_argument('--chunk_duration', type=int, default=300,
                      help='Duration of audio chunks in seconds for continuous mode (default: 300)')
    parser.add_argument('--language', default='en', help='Language for transcription')
    parser.add_argument('--hf_token', help='HuggingFace token for speaker diarization')
    parser.add_argument('--num_speakers', type=int, default=2,
                      help='Number of speakers (optional)')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Sampling temperature (0.0 = deterministic, higher = more random)')
    parser.add_argument('--no_speech_threshold', type=float, default=0.3,
                      help='Threshold for silence detection (0.0-1.0)')
    parser.add_argument('--logprob_threshold', type=float, default=-0.7,
                      help='Minimum log probability for tokens')
    parser.add_argument('--compression_ratio', type=float, default=1.8,
                      help='Maximum compression ratio for segments')
    parser.add_argument('--initial_prompt', 
                      default="This is a conversation between parents and children in a home environment.",
                      help='Optional initial prompt to guide transcription')
    parser.add_argument('--record_only', action='store_true',
                      help='Only record audio without transcription (continuous mode only)')
    parser.add_argument('--start_time', 
                      help='Start time for batch processing (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time',
                      help='End time for batch processing (format: YYYY-MM-DD HH:MM:SS)')
    args = parser.parse_args()

    # Initialize logging
    AudioLogger()
    
    # Check CUDA compatibility only if not in record-only mode
    if not args.record_only:
        GPUManager.check_cuda_compatibility()
    
    if args.mode == 'batch':
        if not args.start_time or not args.end_time:
            logging.error("Error: --start_time and --end_time are required in batch mode")
            return
            
        try:
            start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
            
            model_config = TranscriptionConfig(
                model_name=args.model,
                language=args.language,
                temperature=args.temperature,
                no_speech_threshold=args.no_speech_threshold,
                logprob_threshold=args.logprob_threshold,
                compression_ratio_threshold=args.compression_ratio,
                initial_prompt=args.initial_prompt
            )
            
            diarization_config = None
            if args.hf_token:
                diarization_config = DiarizationConfig(
                    use_auth_token=args.hf_token,
                    num_speakers=args.num_speakers
                )
            
            process_recordings_in_range(start_time, end_time, model_config, diarization_config)
            
        except ValueError as e:
            logging.error(f"Error parsing date/time: {str(e)}")
            return
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            return
    elif args.mode == 'file':
        if not args.audio_file:
            logging.error("Error: --audio_file is required in file mode")
            return
            
        try:
            model_config = TranscriptionConfig(
                model_name=args.model,
                language=args.language,
                temperature=args.temperature,
                no_speech_threshold=args.no_speech_threshold,
                logprob_threshold=args.logprob_threshold,
                compression_ratio_threshold=args.compression_ratio,
                initial_prompt=args.initial_prompt
            )
            model_manager = ModelManager(model_config)
            
            audio_data, _ = AudioValidator.validate_audio_file(args.audio_file)
            audio_tensor = torch.from_numpy(audio_data).float().to(model_manager.device)
            
            result = model_manager.transcribe(audio_tensor)
            
            # Get diarization if enabled
            diarization_segments = []
            if args.hf_token:
                diarization_config = DiarizationConfig(
                    use_auth_token=args.hf_token,
                    num_speakers=args.num_speakers
                )
                diarization_manager = DiarizationManager(diarization_config)
                diarization_segments = diarization_manager.process_audio(args.audio_file)
            
            # Combine transcription with diarization
            output = {
                "audio_file": args.audio_file,
                "transcription": result,
                "diarization": diarization_segments
            }
            
            output_file = os.path.splitext(args.audio_file)[0] + "_transcript.json"
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            logging.info(f"\nTranscription saved to {output_file}")
            
            logging.info("\nTranscription:")
            for segment in result["segments"]:
                text = segment["text"]
                logging.info(f"{text}")
                
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            return
    else:  # continuous mode
        try:
            model_config = TranscriptionConfig(
                model_name=args.model,
                language=args.language,
                temperature=args.temperature,
                no_speech_threshold=args.no_speech_threshold,
                logprob_threshold=args.logprob_threshold,
                compression_ratio_threshold=args.compression_ratio,
                initial_prompt=args.initial_prompt
            )
            audio_config = AudioConfig(chunk_duration=args.chunk_duration)
            
            diarization_config = None
            if args.hf_token and not args.record_only:
                diarization_config = DiarizationConfig(
                    use_auth_token=args.hf_token,
                    num_speakers=args.num_speakers
                )
            
            if args.record_only:
                record_only_manager = SimpleAudioRecorder(audio_config)
                record_only_manager.start_recording()
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logging.info("\nStopping recording...")
                    record_only_manager.stop_recording()
            else:
                transcription_manager = TranscriptionManager(
                    model_config, 
                    audio_config, 
                    diarization_config,
                    record_only=args.record_only
                )
                transcription_manager.start_continuous_transcription()
                
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logging.info("\nStopping recording...")
                    transcription_manager.stop_continuous_transcription()
        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            return

if __name__ == "__main__":
    main() 