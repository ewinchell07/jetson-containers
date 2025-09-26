#!/usr/bin/env python3
"""
Refactored Whisper-TRT Transcriber

A clean, well-structured transcription system for Jetson Nano with:
- GPU memory management
- Model loading with fallback (Whisper -> Whisper-TRT)
- Speaker diarization using Resemblyzer
- Batch and single file processing
- Comprehensive error handling and logging
- Configurable transcription parameters
"""

import os
import torch
import json
import numpy as np
import soundfile as sf
import warnings
import logging
import gc
import whisper
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Union
import argparse
from contextlib import contextmanager

from config import (
    get_transcribe_model, get_transcribe_prompt, select_model_with_fallback,
    get_allow_swap, log_system_info, get_diarization_config, get_audio_config,
    get_quality_config
)

# Optional imports for diarization
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.hparams import sampling_rate
    import scipy.spatial.distance
    DIARIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diarization not available: {e}")
    DIARIZATION_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class TranscriptionConfig:
    """Configuration for transcription parameters"""
    temperature: float = 0.0
    beam_size: int = 5
    patience: float = 1.0
    no_speech_threshold: float = 0.7
    logprob_threshold: float = -1.0
    compression_ratio: float = 2.0
    word_timestamps: bool = True
    noise_reduction: bool = True
    chunk_length: int = 60
    chunk_overlap: int = 5
    vad_filter: bool = True
    language: str = "en"
    condition_on_previous_text: bool = False
    initial_prompt: str = ""
    task: str = "transcribe"


@dataclass
class TranscriptionResult:
    """Result of transcription process"""
    text: str
    segments: List[Dict[str, Any]]
    speaker_segments: List[Dict[str, Any]]
    merged_segments: List[Dict[str, Any]]
    model_name: str
    processing_time: float
    gpu_memory_used: float
    timestamp: str
    audio_file: str


class GPUManager:
    """GPU memory management utilities"""
    
    @staticmethod
    def cleanup_memory() -> None:
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logging.warning(f"GPU cleanup error: {e}")

    @staticmethod
    def get_memory_usage() -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    @staticmethod
    def get_device() -> torch.device:
        """Get the appropriate device (CUDA if available, else CPU)"""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @contextmanager
    def memory_context(self):
        """Context manager for GPU memory cleanup"""
        try:
            yield
        finally:
            self.cleanup_memory()


class AudioProcessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def load_audio(self, audio_file: str) -> tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            audio_data, sample_rate = sf.read(audio_file)
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                logging.info(f"Resampling from {sample_rate} to {self.target_sample_rate} Hz")
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.target_sample_rate
                )
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data, self.target_sample_rate
            
        except Exception as e:
            logging.error(f"Audio loading error: {e}")
            raise

    def audio_to_tensor(self, audio_data: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert audio data to PyTorch tensor"""
        audio_tensor = torch.from_numpy(audio_data)
        if device.type == 'cuda':
            audio_tensor = audio_tensor.to(device)
        return audio_tensor

    def chunk_audio(self, audio_data: np.ndarray, sample_rate: int, 
                   chunk_length: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Break audio into overlapping chunks for processing"""
        chunk_samples = chunk_length * sample_rate
        overlap_samples = chunk_overlap * sample_rate
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        for start_sample in range(0, len(audio_data), step_samples):
            end_sample = min(start_sample + chunk_samples, len(audio_data))
            chunk_audio = audio_data[start_sample:end_sample]
            
            # Skip very short chunks (less than 1 second)
            if len(chunk_audio) < sample_rate:
                continue
                
            chunks.append({
                "audio": chunk_audio,
                "start_time": start_sample / sample_rate,
                "end_time": end_sample / sample_rate,
                "start_sample": start_sample,
                "end_sample": end_sample
            })
        
        return chunks


class ModelManager:
    """Handles model loading and transcription"""
    
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.model_name = model_name
        self.device = device or GPUManager.get_device()
        self.model = None
        self.audio_processor = AudioProcessor()
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model with fallback to Whisper-TRT"""
        try:
            GPUManager.cleanup_memory()
            logging.info(f"Loading Whisper model: {self.model_name}")
            
            # Try regular Whisper first
            try:
                self.model = whisper.load_model(self.model_name)
                if self.device.type == 'cuda':
                    self.model = self.model.to(self.device)
                self.model.eval()
                logging.info("Loaded regular Whisper model")
            except Exception as e:
                logging.warning(f"Regular Whisper failed: {e}")
                logging.info("Trying Whisper-TRT...")
                from whisper_trt import load_trt_model
                self.model = load_trt_model(self.model_name, verbose=True)
                logging.info("Loaded Whisper-TRT model")
                
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def transcribe_audio(self, audio_file: str, config: TranscriptionConfig) -> Dict[str, Any]:
        """Transcribe audio file with given configuration using chunking"""
        try:
            # Load and preprocess audio
            logging.info(f"Loading audio: {audio_file}")
            audio_data, sample_rate = self.audio_processor.load_audio(audio_file)
            
            # Calculate audio duration
            audio_duration = len(audio_data) / sample_rate
            logging.info(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Clean memory before transcription
            GPUManager.cleanup_memory()
            
            # Transcribe
            logging.info("Starting transcription...")
            start_time = datetime.now()
            
            # Check if we need to chunk the audio
            if audio_duration <= config.chunk_length:
                # Process as single chunk
                logging.info("Processing as single chunk")
                audio_tensor = self.audio_processor.audio_to_tensor(audio_data, self.device)
                
                if isinstance(self.model, whisper.Whisper):
                    result = self._transcribe_with_whisper(audio_tensor, config)
                else:
                    # TRT model
                    result = self.model.transcribe(audio_tensor)
            else:
                # Process in chunks
                logging.info(f"Processing in chunks of {config.chunk_length}s with {config.chunk_overlap}s overlap")
                chunks = self.audio_processor.chunk_audio(audio_data, sample_rate, 
                                                        config.chunk_length, config.chunk_overlap)
                logging.info(f"Created {len(chunks)} chunks for processing")
                
                chunk_results = []
                for i, chunk in enumerate(chunks, 1):
                    logging.info(f"Processing chunk {i}/{len(chunks)} ({chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s)")
                    
                    # Convert chunk to tensor
                    chunk_tensor = self.audio_processor.audio_to_tensor(chunk["audio"], self.device)
                    
                    # Transcribe chunk
                    if isinstance(self.model, whisper.Whisper):
                        chunk_result = self._transcribe_with_whisper(chunk_tensor, config)
                    else:
                        # TRT model
                        chunk_result = self.model.transcribe(chunk_tensor)
                    
                    if chunk_result and chunk_result.get('text'):
                        chunk_results.append(chunk_result)
                        logging.info(f"Chunk {i} transcribed: {len(chunk_result['text'])} characters")
                    else:
                        logging.warning(f"Chunk {i} produced empty result")
                    
                    # Clean up memory between chunks
                    GPUManager.cleanup_memory()
                
                # Merge chunk results
                logging.info("Merging chunk results...")
                result = self._merge_chunk_results(chunk_results, chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not result or not result.get('text'):
                logging.warning("Empty transcription result")
                return {"text": "", "segments": [], "processing_time": processing_time}
            
            result["processing_time"] = processing_time
            logging.info(f"Transcription completed in {processing_time:.2f}s. Text length: {len(result['text'])}")
            return result
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            raise
        finally:
            GPUManager.cleanup_memory()

    def _transcribe_with_whisper(self, audio_tensor: torch.Tensor, config: TranscriptionConfig) -> Dict[str, Any]:
        """Transcribe using regular Whisper model"""
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                return self.model.transcribe(
                    audio_tensor,
                    language=config.language,
                    task=config.task,
                    fp16=self.device.type == 'cuda',
                    verbose=True,
                    temperature=config.temperature,
                    no_speech_threshold=config.no_speech_threshold,
                    logprob_threshold=config.logprob_threshold,
                    compression_ratio_threshold=config.compression_ratio,
                    condition_on_previous_text=config.condition_on_previous_text,
                    initial_prompt=config.initial_prompt,
                    word_timestamps=config.word_timestamps
                )

    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]], 
                           chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge transcription results from multiple chunks"""
        if not chunk_results:
            return {"text": "", "segments": []}
        
        # Merge text
        full_text = " ".join([result.get("text", "") for result in chunk_results if result.get("text")])
        
        # Merge segments with proper time offsets
        all_segments = []
        for i, (result, chunk) in enumerate(zip(chunk_results, chunks)):
            if not result.get("segments"):
                continue
                
            chunk_start_time = chunk["start_time"]
            for segment in result["segments"]:
                # Adjust segment timestamps to global time
                adjusted_segment = segment.copy()
                adjusted_segment["start"] = segment["start"] + chunk_start_time
                adjusted_segment["end"] = segment["end"] + chunk_start_time
                
                # Adjust word timestamps if they exist
                if "words" in adjusted_segment:
                    for word in adjusted_segment["words"]:
                        word["start"] = word["start"] + chunk_start_time
                        word["end"] = word["end"] + chunk_start_time
                
                all_segments.append(adjusted_segment)
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start"])
        
        return {
            "text": full_text,
            "segments": all_segments
        }

    def __del__(self):
        """Cleanup model resources"""
        try:
            if self.model:
                del self.model
            GPUManager.cleanup_memory()
        except Exception as e:
            logging.warning(f"Model cleanup error: {e}")


class DiarizationManager:
    """Handles speaker diarization using Resemblyzer"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.encoder = None
        self._load_encoder()

    def _load_encoder(self) -> None:
        """Load Resemblyzer voice encoder"""
        try:
            logging.info("Loading Resemblyzer voice encoder...")
            self.encoder = VoiceEncoder()
            logging.info("Resemblyzer encoder loaded")
        except Exception as e:
            logging.error(f"Failed to load Resemblyzer encoder: {e}")
            raise

    def _segment_audio(self, wav: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Segment audio into overlapping chunks for speaker analysis"""
        diarization_config = get_diarization_config()
        segment_duration = diarization_config["segment_duration"]
        overlap_duration = diarization_config["overlap_duration"]
        
        segment_samples = int(segment_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_samples = segment_samples - overlap_samples
        
        segments = []
        for start_sample in range(0, len(wav) - segment_samples + 1, step_samples):
            end_sample = start_sample + segment_samples
            segment_wav = wav[start_sample:end_sample]
            
            segments.append({
                "start": start_sample / sample_rate,
                "end": end_sample / sample_rate,
                "wav": segment_wav
            })
        
        return segments

    def _cluster_speakers(self, embeddings: np.ndarray, num_speakers: Optional[int] = None) -> List[int]:
        """Cluster speaker embeddings to identify different speakers"""
        from sklearn.cluster import AgglomerativeClustering
        
        diarization_config = get_diarization_config()
        if num_speakers is None:
            num_speakers = min(max(2, len(embeddings) // 10), diarization_config["max_speakers"])
        
        num_speakers = max(diarization_config["min_speakers"], 
                          min(num_speakers, diarization_config["max_speakers"]))
        
        clustering = AgglomerativeClustering(n_clusters=num_speakers, linkage='ward')
        speaker_labels = clustering.fit_predict(embeddings)
        
        return speaker_labels

    def process_audio(self, audio_file: str, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process audio for speaker diarization"""
        try:
            logging.info(f"Processing diarization for: {audio_file}")
            
            # Load and preprocess audio
            wav = preprocess_wav(audio_file)
            sample_rate = sampling_rate
            
            # Segment audio
            segments = self._segment_audio(wav, sample_rate)
            logging.info(f"Created {len(segments)} segments for analysis")
            
            # Extract speaker embeddings
            embeddings = self._extract_embeddings(segments)
            
            if not embeddings:
                logging.warning("No embeddings extracted")
                return []
            
            # Cluster speakers
            speaker_labels = self._cluster_speakers(embeddings, num_speakers)
            
            # Create speaker segments
            speaker_segments = self._create_speaker_segments(segments, speaker_labels)
            
            logging.info(f"Found {len(set(speaker_labels))} speakers in {len(speaker_segments)} segments")
            return speaker_segments
            
        except Exception as e:
            logging.error(f"Diarization error: {e}")
            return []

    def _extract_embeddings(self, segments: List[Dict[str, Any]]) -> np.ndarray:
        """Extract embeddings for all segments"""
        embeddings = []
        for segment in segments:
            try:
                embedding = self.encoder.embed_utterance(segment["wav"])
                embeddings.append(embedding)
            except Exception as e:
                logging.warning(f"Failed to extract embedding for segment {segment['start']:.2f}-{segment['end']:.2f}: {e}")
                # Use zero embedding as fallback
                embeddings.append(np.zeros(256))
        
        return np.array(embeddings) if embeddings else np.array([])

    def _create_speaker_segments(self, segments: List[Dict[str, Any]], speaker_labels: List[int]) -> List[Dict[str, Any]]:
        """Create speaker segments from segments and labels"""
        speaker_segments = []
        for segment, speaker_id in zip(segments, speaker_labels):
            speaker_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": f"SPEAKER_{speaker_id:02d}"
            })
        return speaker_segments

    def __del__(self):
        """Cleanup encoder resources"""
        try:
            if self.encoder:
                del self.encoder
            GPUManager.cleanup_memory()
        except Exception as e:
            logging.warning(f"Diarization cleanup error: {e}")


class SpeakerMerger:
    """Handles merging transcription with speaker diarization"""
    
    @staticmethod
    def merge_transcription_with_speakers(transcription: Dict[str, Any], 
                                        speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge Whisper transcription segments with speaker diarization"""
        if not speaker_segments or 'segments' not in transcription:
            return transcription.get('segments', [])
        
        def get_speaker_at_time(timestamp: float) -> Optional[str]:
            for seg in speaker_segments:
                if seg['start'] <= timestamp <= seg['end']:
                    return seg['speaker']
            return None
        
        merged_segments = []
        for seg in transcription['segments']:
            mid_time = (seg['start'] + seg['end']) / 2
            speaker = get_speaker_at_time(mid_time)
            
            enhanced_seg = seg.copy()
            enhanced_seg['speaker'] = speaker
            enhanced_seg['speaker_confidence'] = SpeakerMerger._calculate_speaker_confidence(
                seg['start'], seg['end'], speaker_segments, speaker
            )
            
            merged_segments.append(enhanced_seg)
        
        return merged_segments
    
    @staticmethod
    def _calculate_speaker_confidence(start: float, end: float, 
                                    speaker_segments: List[Dict[str, Any]], 
                                    assigned_speaker: str) -> float:
        """Calculate confidence that the assigned speaker is correct"""
        if not assigned_speaker:
            return 0.0
        
        total_overlap = 0.0
        segment_duration = end - start
        
        for seg in speaker_segments:
            if seg['speaker'] == assigned_speaker:
                overlap_start = max(start, seg['start'])
                overlap_end = min(end, seg['end'])
                if overlap_start < overlap_end:
                    total_overlap += overlap_end - overlap_start
        
        return total_overlap / segment_duration if segment_duration > 0 else 0.0
    
    @staticmethod
    def deduplicate_speaker_segments(speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping speaker segments and merge adjacent ones"""
        if not speaker_segments:
            return []
        
        sorted_segments = sorted(speaker_segments, key=lambda x: x['start'])
        deduplicated = []
        
        for seg in sorted_segments:
            if not deduplicated:
                deduplicated.append(seg)
                continue
            
            last_seg = deduplicated[-1]
            
            if (seg['speaker'] == last_seg['speaker'] and 
                seg['start'] <= last_seg['end'] + 0.1):  # 0.1s tolerance
                last_seg['end'] = max(last_seg['end'], seg['end'])
            else:
                deduplicated.append(seg)
        
        return deduplicated


class Transcriber:
    """Main transcription processor with clean architecture"""
    
    def __init__(self, model_name: str, hf_token: Optional[str] = None, enable_diarization: bool = True,
                 use_adaptive_quality: bool = None):
        self.model_manager = ModelManager(model_name)
        self.diarization_manager = None
        self.speaker_merger = SpeakerMerger()
        
        # Load quality configuration
        self.quality_config = get_quality_config()
        
        # Determine if adaptive quality should be used
        if use_adaptive_quality is None:
            self.use_adaptive_quality = self.quality_config['enable_quality_retry']
        else:
            self.use_adaptive_quality = use_adaptive_quality
        
        if enable_diarization and DIARIZATION_AVAILABLE:
            try:
                self.diarization_manager = DiarizationManager(hf_token)
                logging.info("Diarization enabled with Resemblyzer")
            except Exception as e:
                logging.warning(f"Failed to initialize diarization: {e}")
                self.diarization_manager = None
        elif enable_diarization and not DIARIZATION_AVAILABLE:
            logging.warning("Diarization requested but Resemblyzer is not available")
        
        # Initialize adaptive transcriber if enabled
        if self.use_adaptive_quality:
            try:
                from adaptive_transcriber import AdaptiveTranscriber, QualityThresholds
                thresholds = QualityThresholds(
                    quality_threshold=self.quality_config['quality_threshold'],
                    avg_logprob_threshold=self.quality_config['avg_logprob_threshold'],
                    no_speech_prob_threshold=self.quality_config['no_speech_prob_threshold'],
                    compression_ratio_threshold=self.quality_config['compression_ratio_threshold']
                )
                self.adaptive_transcriber = AdaptiveTranscriber(thresholds)
                logging.info("Adaptive quality transcriber initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize adaptive transcriber: {e}")
                self.use_adaptive_quality = False
        else:
            self.adaptive_transcriber = None
        
        logging.info(f"Adaptive quality enabled: {self.use_adaptive_quality}")
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"transcriber_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_file}")

    def process_file(self, audio_file: str, output_dir: str = "transcriptions", 
                    num_speakers: Optional[int] = None) -> TranscriptionResult:
        """Process single audio file and return structured result"""
        try:
            audio_path = Path(audio_file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = audio_path.stem
            output_file = output_path / f"transcript_{base_name}_{timestamp}.json"
            
            logging.info(f"Processing: {audio_file}")
            logging.info(f"Output will be saved to: {output_file}")
            
            # Create transcription configuration
            config = TranscriptionConfig(
                initial_prompt=get_transcribe_prompt()
            )
            
            # Transcribe with adaptive quality if enabled
            if self.use_adaptive_quality and self.adaptive_transcriber:
                logging.info("Using adaptive quality transcription")
                result = self.adaptive_transcriber.transcribe_with_quality_check(audio_file, self.model_manager.model_name)
                # Convert TranscriptionResult to dict format for diarization
                result_dict = {
                    'text': result.text,
                    'segments': result.segments,
                    'processing_time': result.processing_time,
                    'gpu_memory_used': result.gpu_memory_used
                }
            else:
                logging.info("Using standard transcription")
                result = self.model_manager.transcribe_audio(audio_file, config)
                result_dict = result
            
            # Add diarization if available
            speaker_segments = []
            merged_segments = []
            if self.diarization_manager:
                raw_speaker_segments = self.diarization_manager.process_audio(audio_file, num_speakers)
                speaker_segments = self.speaker_merger.deduplicate_speaker_segments(raw_speaker_segments)
                merged_segments = self.speaker_merger.merge_transcription_with_speakers(result_dict, speaker_segments)
                logging.info(f"Merged {len(result_dict.get('segments', []))} transcription segments with {len(speaker_segments)} speaker segments")
            
            # Create structured result
            if self.use_adaptive_quality and self.adaptive_transcriber:
                # Use the TranscriptionResult from adaptive transcriber
                transcription_result = result
                # Update with diarization data
                transcription_result.speaker_segments = speaker_segments
                transcription_result.merged_segments = merged_segments
            else:
                # Create new TranscriptionResult from dict
                transcription_result = TranscriptionResult(
                    text=result_dict.get('text', ''),
                    segments=result_dict.get('segments', []),
                    speaker_segments=speaker_segments,
                    merged_segments=merged_segments,
                    model_name=self.model_manager.model_name,
                    processing_time=result_dict.get('processing_time', 0.0),
                    gpu_memory_used=GPUManager.get_memory_usage(),
                    timestamp=datetime.now().isoformat(),
                    audio_file=str(audio_path.absolute())
                )
            
            # Save results
            self._save_results(transcription_result, output_file)
            
            logging.info(f"Transcription saved to: {output_file}")
            logging.info(f"Transcribed text: {transcription_result.text[:200]}...")
            
            return transcription_result
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            raise

    def _save_results(self, result: TranscriptionResult, output_file: Path) -> None:
        """Save transcription results to JSON file"""
        output_data = {
            "timestamp": result.timestamp,
            "audio_file": result.audio_file,
            "model": result.model_name,
            "transcription": {
                "text": result.text,
                "segments": result.segments
            },
            "speaker_segments": result.speaker_segments,
            "merged_segments": result.merged_segments,
            "gpu_memory_used_gb": result.gpu_memory_used,
            "processing_time_seconds": result.processing_time,
            "config": {
                "temperature": 0.0,
                "language": "en",
                "task": "transcribe"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def process_batch(self, audio_dir: str, output_dir: str = "transcriptions", 
                     pattern: str = "*.wav", num_speakers: Optional[int] = None) -> List[TranscriptionResult]:
        """Process multiple audio files"""
        audio_path = Path(audio_dir)
        audio_files = list(audio_path.glob(pattern))
        
        if not audio_files:
            logging.warning(f"No audio files found in {audio_dir} matching {pattern}")
            return []
        
        logging.info(f"Found {len(audio_files)} audio files to process")
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                logging.info(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
                result = self.process_file(str(audio_file), output_dir, num_speakers)
                results.append(result)
                
                # Clean up between files
                GPUManager.cleanup_memory()
                
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
                continue
        
        logging.info(f"Batch processing completed. Processed {len(results)}/{len(audio_files)} files")
        return results


def get_transcription_config() -> Dict[str, Any]:
    """Get transcription configuration (legacy compatibility)"""
    config = TranscriptionConfig(initial_prompt=get_transcribe_prompt())
    return {
        "temperature": config.temperature,
        "beam_size": config.beam_size,
        "patience": config.patience,
        "no_speech_threshold": config.no_speech_threshold,
        "logprob_threshold": config.logprob_threshold,
        "compression_ratio": config.compression_ratio,
        "word_timestamps": config.word_timestamps,
        "chunk_length": config.chunk_length,
        "chunk_overlap": config.chunk_overlap,
        "vad_filter": config.vad_filter,
        "language": config.language,
        "condition_on_previous_text": config.condition_on_previous_text,
        "initial_prompt": config.initial_prompt,
        "task": config.task
    }


def get_diarization_config_local() -> Dict[str, Any]:
    """Get diarization configuration (legacy compatibility)"""
    return get_diarization_config()


def get_audio_config_local() -> Dict[str, Any]:
    """Get audio configuration (legacy compatibility)"""
    return get_audio_config()


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Audio Transcription Tool")
    parser.add_argument("audio_file", help="Audio file to transcribe (or directory for batch mode)")
    parser.add_argument("--model", default=None, 
                       help="Whisper model (tiny.en, base.en, small.en, medium.en, large). If not specified, uses config defaults.")
    parser.add_argument("--output-dir", default="transcriptions",
                       help="Output directory for transcriptions")
    parser.add_argument("--hf-token", help="HuggingFace token (kept for compatibility, not used with Resemblyzer)")
    parser.add_argument("--num-speakers", type=int, help="Number of speakers for diarization")
    parser.add_argument("--no-diarization", action="store_true", 
                       help="Disable speaker diarization")
    parser.add_argument("--batch", action="store_true", 
                       help="Process all audio files in the specified directory")
    parser.add_argument("--pattern", default="*.wav",
                       help="File pattern for batch processing")
    
    args = parser.parse_args()
    
    # Log system information
    log_system_info()
    
    # Determine model to use
    if args.model:
        requested_model = args.model
    else:
        requested_model = get_transcribe_model()
    
    # Use config module to select the best model
    selected_model = select_model_with_fallback(requested_model, get_allow_swap())
    logging.info(f"Using model: {selected_model}")
    
    # Create transcriber
    enable_diarization = not args.no_diarization
    transcriber = Transcriber(selected_model, args.hf_token, enable_diarization)
    
    try:
        if args.batch:
            # Batch processing
            results = transcriber.process_batch(
                args.audio_file, 
                args.output_dir, 
                args.pattern, 
                args.num_speakers
            )
            print(f"Processed {len(results)} files")
        else:
            # Single file processing
            result = transcriber.process_file(
                args.audio_file, 
                args.output_dir, 
                args.num_speakers
            )
            print(f"Transcription: {result.text}")
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
