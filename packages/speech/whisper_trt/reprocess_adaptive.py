#!/usr/bin/env python3
"""
Reprocess existing transcripts with adaptive quality-based model selection.

This script analyzes existing transcript files and reprocesses them with larger models
if the quality is insufficient based on configurable thresholds.
"""

import os
import json
import logging
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np

from adaptive_transcriber import AdaptiveTranscriber, QualityThresholds
from config import get_quality_config


class TranscriptReprocessor:
    """Reprocess existing transcripts with adaptive quality"""
    
    def __init__(self, quality_threshold: float = 0.3, max_retries: int = 2):
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        
        # Create adaptive transcriber
        thresholds = QualityThresholds(quality_threshold=quality_threshold)
        self.adaptive_transcriber = AdaptiveTranscriber(thresholds)
        self.adaptive_transcriber.max_retries = max_retries
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'reprocessed': 0,
            'quality_improved': 0,
            'quality_degraded': 0,
            'errors': 0
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path.home() / ".cache" / "whisper_trt" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"reprocess_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_file}")
    
    def analyze_transcript_quality(self, transcript_file: str) -> Dict[str, Any]:
        """Analyze the quality of an existing transcript"""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract segments
            segments = data.get('transcription', {}).get('segments', [])
            if not segments:
                return {"error": "No segments found", "quality_score": 1.0}
            
            # Create a mock TranscriptionResult for analysis
            from transcriber import TranscriptionResult
            result = TranscriptionResult(
                text=data.get('transcription', {}).get('text', ''),
                segments=segments,
                speaker_segments=[],
                merged_segments=[],
                model_name=data.get('model', 'unknown'),
                processing_time=0.0,
                gpu_memory_used=0.0,
                timestamp=data.get('timestamp', ''),
                audio_file=data.get('audio_file', '')
            )
            
            # Get quality report
            quality_report = self.adaptive_transcriber.get_quality_report(result)
            return quality_report
            
        except Exception as e:
            logging.error(f"Error analyzing {transcript_file}: {e}")
            return {"error": str(e), "quality_score": 1.0}
    
    def find_audio_file(self, transcript_file: str) -> str:
        """Find the corresponding audio file for a transcript"""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            audio_file = data.get('audio_file', '')
            if audio_file and os.path.exists(audio_file):
                return audio_file
            
            # Try to find audio file in recordings directory
            base_name = Path(transcript_file).stem
            # Remove transcript prefix if present
            if base_name.startswith('transcript_'):
                base_name = base_name[10:]  # Remove 'transcript_' prefix
            
            # Look for audio files with similar names
            recordings_dir = Path("/opt/whisper_trt/recordings")
            if recordings_dir.exists():
                for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                    audio_file = recordings_dir / f"{base_name}{ext}"
                    if audio_file.exists():
                        return str(audio_file)
            
            return ""
            
        except Exception as e:
            logging.error(f"Error finding audio file for {transcript_file}: {e}")
            return ""
    
    def reprocess_transcript(self, transcript_file: str, audio_file: str = None) -> Tuple[bool, Dict[str, Any]]:
        """Reprocess a transcript with adaptive quality"""
        try:
            # Find audio file if not provided
            if not audio_file:
                audio_file = self.find_audio_file(transcript_file)
                if not audio_file:
                    return False, {"error": "Audio file not found"}
            
            if not os.path.exists(audio_file):
                return False, {"error": f"Audio file not found: {audio_file}"}
            
            logging.info(f"Reprocessing: {transcript_file} -> {audio_file}")
            
            # Get original quality
            original_quality = self.analyze_transcript_quality(transcript_file)
            original_score = original_quality.get('overall_quality_score', 1.0)
            
            # Reprocess with adaptive quality
            result = self.adaptive_transcriber.transcribe_with_quality_check(audio_file)
            
            # Get new quality
            new_quality = self.adaptive_transcriber.get_quality_report(result)
            new_score = new_quality.get('overall_quality_score', 1.0)
            
            # Determine if quality improved
            quality_improved = new_score < original_score
            improvement = original_score - new_score
            
            logging.info(f"Quality: {original_score:.3f} -> {new_score:.3f} (improvement: {improvement:+.3f})")
            
            return True, {
                "success": True,
                "original_quality": original_score,
                "new_quality": new_score,
                "improvement": improvement,
                "quality_improved": quality_improved,
                "result": result,
                "quality_report": new_quality
            }
            
        except Exception as e:
            logging.error(f"Error reprocessing {transcript_file}: {e}")
            return False, {"error": str(e)}
    
    def reprocess_directory(self, transcript_dir: str, output_dir: str = None, 
                          quality_threshold: float = 0.3, 
                          min_improvement: float = 0.1) -> Dict[str, Any]:
        """Reprocess all transcripts in a directory"""
        transcript_path = Path(transcript_dir)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
        
        # Find all transcript files
        transcript_files = list(transcript_path.glob("*.json"))
        if not transcript_files:
            logging.warning(f"No transcript files found in {transcript_dir}")
            return {"error": "No transcript files found"}
        
        logging.info(f"Found {len(transcript_files)} transcript files")
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = transcript_path / "reprocessed"
            output_path.mkdir(exist_ok=True)
        
        results = []
        
        for transcript_file in transcript_files:
            try:
                self.stats['total_processed'] += 1
                
                # Analyze original quality
                quality_analysis = self.analyze_transcript_quality(str(transcript_file))
                original_score = quality_analysis.get('overall_quality_score', 1.0)
                
                logging.info(f"Processing {transcript_file.name}: quality={original_score:.3f}")
                
                # Only reprocess if quality is below threshold
                if original_score > quality_threshold:
                    logging.info(f"Quality below threshold, reprocessing...")
                    
                    success, reprocess_result = self.reprocess_transcript(str(transcript_file))
                    
                    if success:
                        self.stats['reprocessed'] += 1
                        
                        improvement = reprocess_result.get('improvement', 0.0)
                        if improvement >= min_improvement:
                            self.stats['quality_improved'] += 1
                            
                            # Save improved result
                            output_file = output_path / f"improved_{transcript_file.name}"
                            self._save_improved_transcript(reprocess_result['result'], output_file)
                            
                            logging.info(f"Quality improved by {improvement:.3f}, saved to {output_file}")
                        else:
                            self.stats['quality_degraded'] += 1
                            logging.info(f"Quality did not improve significantly ({improvement:.3f})")
                    else:
                        self.stats['errors'] += 1
                        logging.error(f"Reprocessing failed: {reprocess_result.get('error', 'Unknown error')}")
                else:
                    logging.info(f"Quality acceptable ({original_score:.3f}), skipping")
                
                results.append({
                    "file": str(transcript_file),
                    "original_quality": original_score,
                    "reprocessed": original_score > quality_threshold,
                    "success": success if original_score > quality_threshold else True,
                    "improvement": reprocess_result.get('improvement', 0.0) if original_score > quality_threshold else 0.0
                })
                
            except Exception as e:
                self.stats['errors'] += 1
                logging.error(f"Error processing {transcript_file}: {e}")
                results.append({
                    "file": str(transcript_file),
                    "error": str(e)
                })
        
        # Generate summary
        summary = {
            "total_processed": self.stats['total_processed'],
            "reprocessed": self.stats['reprocessed'],
            "quality_improved": self.stats['quality_improved'],
            "quality_degraded": self.stats['quality_degraded'],
            "errors": self.stats['errors'],
            "output_directory": str(output_path),
            "results": results
        }
        
        # Save summary
        summary_file = output_path / "reprocessing_summary.json"
        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            summary_serializable = convert_numpy_types(summary)
            json.dump(summary_serializable, f, indent=2)
        
        logging.info(f"Reprocessing complete. Summary saved to {summary_file}")
        return summary
    
    def _save_improved_transcript(self, result, output_file: Path):
        """Save improved transcript result"""
        try:
            # Convert TranscriptionResult to dict
            result_dict = {
                "timestamp": result.timestamp,
                "audio_file": result.audio_file,
                "model": result.model_name,
                "transcription": {
                    "text": result.text,
                    "segments": result.segments
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving improved transcript to {output_file}: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Reprocess transcripts with adaptive quality")
    parser.add_argument("transcript_dir", help="Directory containing transcript files")
    parser.add_argument("--output-dir", help="Output directory for improved transcripts")
    parser.add_argument("--quality-threshold", type=float, default=0.3, 
                       help="Quality threshold for reprocessing (0.0=perfect, 1.0=terrible)")
    parser.add_argument("--min-improvement", type=float, default=0.1,
                       help="Minimum improvement required to save new transcript")
    parser.add_argument("--max-retries", type=int, default=2,
                       help="Maximum retries with larger models")
    parser.add_argument("--single-file", help="Process a single transcript file")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze quality, don't reprocess")
    
    args = parser.parse_args()
    
    # Create reprocessor
    reprocessor = TranscriptReprocessor(
        quality_threshold=args.quality_threshold,
        max_retries=args.max_retries
    )
    
    if args.single_file:
        # Process single file
        if args.analyze_only:
            quality = reprocessor.analyze_transcript_quality(args.single_file)
            print(json.dumps(quality, indent=2))
        else:
            success, result = reprocessor.reprocess_transcript(args.single_file)
            print(json.dumps(result, indent=2))
    else:
        # Process directory
        summary = reprocessor.reprocess_directory(
            args.transcript_dir,
            args.output_dir,
            args.quality_threshold,
            args.min_improvement
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


