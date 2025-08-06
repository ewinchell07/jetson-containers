#!/usr/bin/env python3
"""
Simplified transcription-only script for processing audio files.
Extracts essential transcription functionality without the complexity and memory overhead
of the full simple_transcribe.py system.
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
from typing import Optional, Dict, List, Any
import argparse
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration constants - edit these as needed
TRANSCRIPTION_CONFIG = {
    "temperature": 0.0,              # Deterministic transcription
    "no_speech_threshold": 0.3,      # Silence detection threshold
    "logprob_threshold": -0.7,       # Token validity threshold
    "compression_ratio": 1.8,        # Compression threshold
    "condition_on_previous_text": True,  # Use context
    "initial_prompt": "This is a conversation between parents and children in a home environment. Sometimes there is a TV playing in the background, you can ignore transcribing that.",
    "word_timestamps": True,         # Include word-level timestamps
    "language": "en",               # Default language
    "task": "transcribe"            # Task type
}

DIARIZATION_CONFIG = {
    "min_speakers": 1,
    "max_speakers": 5,
    "pipeline_model": "pyannote/speaker-diarization-3.1"
}

AUDIO_CONFIG = {
    "target_sample_rate": 16000     # Whisper expects 16kHz
}


class GPUManager:
    """Simple GPU memory management"""
    @staticmethod
    def cleanup_memory():
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logging.warning(f"GPU cleanup error: {e}")

    @staticmethod
    def get_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0


class ModelManager:
    """Handles model loading and transcription"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model with fallback"""
        try:
            GPUManager.cleanup_memory()
            logging.info(f"Loading Whisper model: {self.model_name}")
            
            # Try regular Whisper first
            try:
                self.model = whisper.load_model(self.model_name)
                if torch.cuda.is_available():
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

    def transcribe_audio(self, audio_file: str) -> Dict[str, Any]:
        """Transcribe audio file"""
        try:
            # Load and preprocess audio
            logging.info(f"Loading audio: {audio_file}")
            audio_data, sample_rate = sf.read(audio_file)
            
            # Ensure mono
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if needed
            if sample_rate != AUDIO_CONFIG["target_sample_rate"]:
                logging.info(f"Resampling from {sample_rate} to {AUDIO_CONFIG['target_sample_rate']} Hz")
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=AUDIO_CONFIG["target_sample_rate"]
                )
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data)
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.to(self.device)
            
            # Clean memory before transcription
            GPUManager.cleanup_memory()
            
            # Transcribe
            logging.info("Starting transcription...")
            if isinstance(self.model, whisper.Whisper):
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    with torch.no_grad():
                        result = self.model.transcribe(
                            audio_tensor,
                            language=TRANSCRIPTION_CONFIG["language"],
                            task=TRANSCRIPTION_CONFIG["task"],
                            fp16=torch.cuda.is_available(),
                            verbose=True,
                            temperature=TRANSCRIPTION_CONFIG["temperature"],
                            no_speech_threshold=TRANSCRIPTION_CONFIG["no_speech_threshold"],
                            logprob_threshold=TRANSCRIPTION_CONFIG["logprob_threshold"],
                            compression_ratio_threshold=TRANSCRIPTION_CONFIG["compression_ratio"],
                            condition_on_previous_text=TRANSCRIPTION_CONFIG["condition_on_previous_text"],
                            initial_prompt=TRANSCRIPTION_CONFIG["initial_prompt"],
                            word_timestamps=TRANSCRIPTION_CONFIG["word_timestamps"]
                        )
            else:
                # TRT model
                result = self.model.transcribe(audio_tensor)
            
            if not result or not result.get('text'):
                logging.warning("Empty transcription result")
                return {"text": "", "segments": []}
            
            logging.info(f"Transcription completed. Text length: {len(result['text'])}")
            return result
            
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            raise
        finally:
            GPUManager.cleanup_memory()

    def __del__(self):
        """Cleanup"""
        try:
            if self.model:
                del self.model
            GPUManager.cleanup_memory()
        except Exception as e:
            logging.warning(f"Model cleanup error: {e}")


class DiarizationManager:
    """Handles speaker diarization"""
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load diarization pipeline"""
        try:
            logging.info("Loading speaker diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                DIARIZATION_CONFIG["pipeline_model"],
                use_auth_token=self.hf_token
            )
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            logging.info("Diarization pipeline loaded")
        except Exception as e:
            logging.error(f"Failed to load diarization pipeline: {e}")
            raise

    def process_audio(self, audio_file: str, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process audio for speaker diarization"""
        try:
            logging.info(f"Processing diarization for: {audio_file}")
            diarization = self.pipeline(
                audio_file,
                num_speakers=num_speakers,
                min_speakers=DIARIZATION_CONFIG["min_speakers"],
                max_speakers=DIARIZATION_CONFIG["max_speakers"]
            )
            
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
            logging.error(f"Diarization error: {e}")
            return []

    def __del__(self):
        """Cleanup"""
        try:
            if self.pipeline:
                del self.pipeline
            GPUManager.cleanup_memory()
        except Exception as e:
            logging.warning(f"Diarization cleanup error: {e}")


class Transcriber:
    """Main transcription processor"""
    def __init__(self, model_name: str, hf_token: Optional[str] = None):
        self.model_manager = ModelManager(model_name)
        self.diarization_manager = None
        if hf_token:
            self.diarization_manager = DiarizationManager(hf_token)
        
        self._setup_logging()

    def _merge_transcription_with_speakers(self, transcription: Dict[str, Any], 
                                         speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge Whisper transcription segments with speaker diarization"""
        if not speaker_segments or 'segments' not in transcription:
            return transcription.get('segments', [])
        
        # Create speaker lookup by time
        def get_speaker_at_time(timestamp: float) -> Optional[str]:
            for seg in speaker_segments:
                if seg['start'] <= timestamp <= seg['end']:
                    return seg['speaker']
            return None
        
        # Process each transcription segment
        merged_segments = []
        for seg in transcription['segments']:
            # Get speaker for the middle of the segment
            mid_time = (seg['start'] + seg['end']) / 2
            speaker = get_speaker_at_time(mid_time)
            
            # Add speaker info to segment
            enhanced_seg = seg.copy()
            enhanced_seg['speaker'] = speaker
            enhanced_seg['speaker_confidence'] = self._calculate_speaker_confidence(
                seg['start'], seg['end'], speaker_segments, speaker
            )
            
            merged_segments.append(enhanced_seg)
        
        return merged_segments
    
    def _calculate_speaker_confidence(self, start: float, end: float, 
                                    speaker_segments: List[Dict[str, Any]], 
                                    assigned_speaker: str) -> float:
        """Calculate confidence that the assigned speaker is correct for this segment"""
        if not assigned_speaker:
            return 0.0
        
        # Find overlap with speaker segments
        total_overlap = 0.0
        segment_duration = end - start
        
        for seg in speaker_segments:
            if seg['speaker'] == assigned_speaker:
                # Calculate overlap
                overlap_start = max(start, seg['start'])
                overlap_end = min(end, seg['end'])
                if overlap_start < overlap_end:
                    total_overlap += overlap_end - overlap_start
        
        return total_overlap / segment_duration if segment_duration > 0 else 0.0
    
    def _deduplicate_speaker_segments(self, speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping speaker segments and merge adjacent ones"""
        if not speaker_segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(speaker_segments, key=lambda x: x['start'])
        deduplicated = []
        
        for seg in sorted_segments:
            if not deduplicated:
                deduplicated.append(seg)
                continue
            
            last_seg = deduplicated[-1]
            
            # Check for overlap or adjacency with same speaker
            if (seg['speaker'] == last_seg['speaker'] and 
                seg['start'] <= last_seg['end'] + 0.1):  # 0.1s tolerance
                # Merge segments
                last_seg['end'] = max(last_seg['end'], seg['end'])
            else:
                # Add as new segment
                deduplicated.append(seg)
        
        return deduplicated

    def _setup_logging(self):
        """Setup logging"""
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
                    num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """Process single audio file"""
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
            
            # Transcribe
            result = self.model_manager.transcribe_audio(audio_file)
            
            # Add diarization if available
            speaker_segments = []
            merged_segments = []
            if self.diarization_manager:
                raw_speaker_segments = self.diarization_manager.process_audio(audio_file, num_speakers)
                speaker_segments = self._deduplicate_speaker_segments(raw_speaker_segments)
                merged_segments = self._merge_transcription_with_speakers(result, speaker_segments)
                logging.info(f"Merged {len(result.get('segments', []))} transcription segments with {len(speaker_segments)} speaker segments")
            
            # Combine results
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "audio_file": str(audio_path.absolute()),
                "model": self.model_manager.model_name,
                "transcription": result,
                "speaker_segments": speaker_segments,
                "merged_segments": merged_segments,  # New: segments with speaker info
                "gpu_memory_used_gb": GPUManager.get_memory_usage(),
                "config": TRANSCRIPTION_CONFIG
            }
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Transcription saved to: {output_file}")
            logging.info(f"Transcribed text: {result.get('text', '')[:200]}...")
            
            return output_data
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            raise

    def process_batch(self, audio_dir: str, output_dir: str = "transcriptions", 
                     pattern: str = "*.wav", num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
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


def main():
    parser = argparse.ArgumentParser(description="Audio Transcription Tool")
    parser.add_argument("audio_file", help="Audio file to transcribe (or directory for batch mode)")
    parser.add_argument("--model", default="base.en", 
                       help="Whisper model (tiny.en, base.en, small.en, medium.en, large)")
    parser.add_argument("--output-dir", default="transcriptions",
                       help="Output directory for transcriptions")
    parser.add_argument("--hf-token", help="HuggingFace token for speaker diarization")
    parser.add_argument("--num-speakers", type=int, help="Number of speakers for diarization")
    parser.add_argument("--batch", action="store_true", 
                       help="Process all audio files in the specified directory")
    parser.add_argument("--pattern", default="*.wav",
                       help="File pattern for batch processing")
    
    args = parser.parse_args()
    
    # Create transcriber
    transcriber = Transcriber(args.model, args.hf_token)
    
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
            print(f"Transcription: {result['transcription']['text']}")
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())