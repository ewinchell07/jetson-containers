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
    blocksize: int = 2048
    latency: str = 'high'

@dataclass
class TranscriptionConfig:
    """Configuration for transcription"""
    model_name: str = "tiny.en"
    language: str = "en"
    task: str = "transcribe"
    temperature: float = 0.0
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

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
            
            # Set conservative memory settings
            torch.cuda.set_per_process_memory_fraction(0.6)  # Even more conservative
            
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
        self.audio_queue = queue.Queue()
        self.current_chunk = []
        self.overflow_count = 0
        self.last_overflow_time = time.time()
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        self._setup_audio_device()

    def _setup_audio_device(self):
        logging.info("\nAvailable audio devices:")
        logging.info(sd.query_devices())
        self.input_device = 0
        logging.info(f"\nUsing input device: {sd.query_devices(self.input_device)['name']}")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio recording"""
        if status:
            current_time = time.time()
            if status.input_overflow:
                self.overflow_count += 1
                if current_time - self.last_overflow_time > 5:
                    logging.warning(f"Input buffer overflow detected. This may cause audio loss. (Overflow count: {self.overflow_count})")
                    self.last_overflow_time = current_time
            else:
                logging.warning(f"Status: {status}")
                
        if self.recording:
            if self.audio_queue.qsize() > 100:
                logging.warning(f"Audio queue is getting large ({self.audio_queue.qsize()} chunks). Processing may be falling behind.")
            
            self.current_chunk.extend(indata.copy())
            self.audio_queue.put(indata.copy())

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
        """Start audio recording"""
        self.recording = True
        self.current_chunk = []
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
    def __init__(self, model_config: TranscriptionConfig, audio_config: AudioConfig, diarization_config: Optional[DiarizationConfig] = None):
        self.model_config = model_config
        self.audio_config = audio_config
        self.model_manager = ModelManager(model_config)
        self.diarization_manager = DiarizationManager(diarization_config) if diarization_config else None
        self.audio_recorder = AudioRecorder(audio_config)
        self.process_thread = None
        self.chunk_queue = queue.Queue()
        self.processing_thread = None
        self._running = False

    def process_audio_chunks(self):
        """Collect audio chunks and add them to the processing queue"""
        while self._running:
            try:
                samples_per_chunk = self.audio_recorder.config.chunk_duration * self.audio_recorder.config.sample_rate
                audio_chunk = []
                
                logging.info(f"\nCollecting {self.audio_recorder.config.chunk_duration} seconds of audio...")
                while len(audio_chunk) < samples_per_chunk and self._running:
                    try:
                        data = self.audio_recorder.audio_queue.get(timeout=1)
                        audio_chunk.extend(data)
                    except queue.Empty:
                        if not self._running:
                            break
                        logging.info("No audio data received in the last second")
                        continue

                if audio_chunk and self._running:
                    audio_data = np.array(audio_chunk, dtype=self.audio_recorder.config.dtype)
                    
                    if np.max(np.abs(audio_data)) < 0.01:
                        logging.info("\nNo audio detected in this chunk, skipping...")
                        continue
                    
                    # Add chunk to processing queue
                    self.chunk_queue.put(audio_data)
                    logging.info(f"Added new audio chunk to processing queue (size: {len(audio_data)} samples)")
            except Exception as e:
                logging.error(f"Error in audio chunk collection: {str(e)}")
                if not self._running:
                    break
                continue

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
                
                # Transcribe immediately
                self._transcribe_and_save(audio_file, timestamp)
                
                # Mark the chunk as processed
                self.chunk_queue.task_done()
                
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
            result = self.model_manager.transcribe(audio_tensor)
            
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
            
        except Exception as e:
            logging.error(f"Error in transcription and diarization: {str(e)}")

    def start_continuous_transcription(self):
        """Start recording and processing audio chunks"""
        self._running = True
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
        if self.audio_recorder:
            self.audio_recorder.stop_recording()
        
        # Wait for the collection thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
        
        # Wait for all chunks to be processed
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Clear the chunk queue
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clean up resources
        try:
            if self.model_manager:
                del self.model_manager
            if self.diarization_manager:
                del self.diarization_manager
            if self.audio_recorder:
                del self.audio_recorder
            GPUManager.cleanup_memory()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
        
        logging.info("Transcription stopped.")

    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            self.stop_continuous_transcription()
        except Exception as e:
            logging.warning(f"Error during transcription manager cleanup: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Whisper Transcription Tool')
    parser.add_argument('--mode', choices=['file', 'continuous'], default='file',
                      help='Transcription mode: "file" for single file, "continuous" for continuous recording')
    parser.add_argument('--model', default='tiny.en',
                      help='Model to use (tiny.en, base.en, etc.)')
    parser.add_argument('--audio_file', help='Path to audio file (required for file mode)')
    parser.add_argument('--chunk_duration', type=int, default=600,
                      help='Duration of audio chunks in seconds for continuous mode (default: 600)')
    parser.add_argument('--language', default='en', help='Language for transcription')
    parser.add_argument('--hf_token', help='HuggingFace token for speaker diarization')
    parser.add_argument('--num_speakers', type=int, help='Number of speakers (optional)')
    args = parser.parse_args()

    # Initialize logging
    AudioLogger()
    
    # Check CUDA compatibility
    GPUManager.check_cuda_compatibility()
    
    if args.mode == 'file':
        if not args.audio_file:
            logging.error("Error: --audio_file is required in file mode")
            return
            
        try:
            model_config = TranscriptionConfig(
                model_name=args.model,
                language=args.language
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
                language=args.language
            )
            audio_config = AudioConfig(chunk_duration=args.chunk_duration)
            
            diarization_config = None
            if args.hf_token:
                diarization_config = DiarizationConfig(
                    use_auth_token=args.hf_token,
                    num_speakers=args.num_speakers
                )
            
            transcription_manager = TranscriptionManager(model_config, audio_config, diarization_config)
            transcription_manager.start_continuous_transcription()
            
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