#!/usr/bin/env python3
"""
Standalone quality analysis for existing transcripts.
This script can run outside the Docker environment.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QualityThresholds:
    """Quality assessment thresholds and weights"""
    avg_logprob_threshold: float = -1.2
    no_speech_prob_threshold: float = 0.4
    compression_ratio_threshold: float = 3.0
    min_segments_per_minute: float = 2.0
    max_segments_per_minute: float = 20.0
    logprob_weight: float = 0.4
    no_speech_weight: float = 0.3
    compression_weight: float = 0.2
    fragmentation_weight: float = 0.1
    quality_threshold: float = 0.3


class QualityAnalyzer:
    """Standalone quality analyzer for existing transcripts"""
    
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()
    
    def analyze_transcript(self, transcript_file: str) -> Dict[str, Any]:
        """Analyze the quality of a transcript file"""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract segments
            segments = data.get('transcription', {}).get('segments', [])
            if not segments:
                return {"error": "No segments found", "quality_score": 1.0}
            
            # Extract metrics
            avg_logprobs = [s.get('avg_logprob', -3.0) for s in segments]
            no_speech_probs = [s.get('no_speech_prob', 1.0) for s in segments]
            compression_ratios = [s.get('compression_ratio', 10.0) for s in segments]
            
            # Calculate individual scores
            logprob_score = self._score_logprob(avg_logprobs)
            no_speech_score = self._score_no_speech(no_speech_probs)
            compression_score = self._score_compression(compression_ratios)
            fragmentation_score = self._score_fragmentation(segments)
            
            # Overall quality score
            overall_score = (
                logprob_score * self.thresholds.logprob_weight +
                no_speech_score * self.thresholds.no_speech_weight +
                compression_score * self.thresholds.compression_weight +
                fragmentation_score * self.thresholds.fragmentation_weight
            )
            
            return {
                "overall_quality_score": overall_score,
                "is_acceptable": overall_score <= self.thresholds.quality_threshold,
                "metrics": {
                    "avg_logprob": np.mean(avg_logprobs),
                    "avg_no_speech_prob": np.mean(no_speech_probs),
                    "avg_compression_ratio": np.mean(compression_ratios),
                    "segments_per_minute": len(segments) / (self._get_duration(segments) / 60.0) if self._get_duration(segments) > 0 else 0
                },
                "scores": {
                    "logprob_score": logprob_score,
                    "no_speech_score": no_speech_score,
                    "compression_score": compression_score,
                    "fragmentation_score": fragmentation_score
                },
                "thresholds": {
                    "quality_threshold": self.thresholds.quality_threshold,
                    "logprob_threshold": self.thresholds.avg_logprob_threshold,
                    "no_speech_threshold": self.thresholds.no_speech_prob_threshold,
                    "compression_threshold": self.thresholds.compression_ratio_threshold
                }
            }
            
        except Exception as e:
            return {"error": str(e), "quality_score": 1.0}
    
    def _score_logprob(self, logprobs: List[float]) -> float:
        """Score based on average log probability (0.0 = perfect, 1.0 = terrible)"""
        if not logprobs:
            return 1.0
        
        avg_logprob = np.mean(logprobs)
        return max(0.0, min(1.0, (-avg_logprob - 0.5) / 2.5))
    
    def _score_no_speech(self, no_speech_probs: List[float]) -> float:
        """Score based on no-speech probability (0.0 = perfect, 1.0 = terrible)"""
        if not no_speech_probs:
            return 1.0
        
        avg_no_speech = np.mean(no_speech_probs)
        return max(0.0, min(1.0, (avg_no_speech - 0.1) / 0.7))
    
    def _score_compression(self, compression_ratios: List[float]) -> float:
        """Score based on compression ratio (0.0 = perfect, 1.0 = terrible)"""
        if not compression_ratios:
            return 1.0
        
        avg_compression = np.mean(compression_ratios)
        return max(0.0, min(1.0, (avg_compression - 1.0) / 4.0))
    
    def _score_fragmentation(self, segments: List[Dict]) -> float:
        """Score based on segment fragmentation (0.0 = perfect, 1.0 = terrible)"""
        if not segments:
            return 1.0
        
        duration = self._get_duration(segments)
        if duration <= 0:
            return 1.0
        
        segments_per_minute = len(segments) / (duration / 60.0)
        
        if 3.0 <= segments_per_minute <= 8.0:
            return 0.0
        elif segments_per_minute < 3.0:
            return (3.0 - segments_per_minute) / 3.0
        else:
            return min(1.0, (segments_per_minute - 8.0) / 12.0)
    
    def _get_duration(self, segments: List[Dict]) -> float:
        """Get total duration from segments"""
        if not segments:
            return 0.0
        
        max_end = max(segment.get('end', 0) for segment in segments)
        return max_end


def analyze_transcript_file(transcript_file: str, quality_threshold: float = 0.3):
    """Analyze a single transcript file"""
    print(f"Analyzing: {transcript_file}")
    
    # Create analyzer
    thresholds = QualityThresholds(quality_threshold=quality_threshold)
    analyzer = QualityAnalyzer(thresholds)
    
    # Analyze transcript
    result = analyzer.analyze_transcript(transcript_file)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return result
    
    print(f"\n=== QUALITY ANALYSIS ===")
    print(f"Overall quality score: {result['overall_quality_score']:.3f}")
    print(f"Is acceptable: {result['is_acceptable']}")
    
    print(f"\n=== METRICS ===")
    metrics = result['metrics']
    print(f"Average log probability: {metrics['avg_logprob']:.3f}")
    print(f"Average no-speech probability: {metrics['avg_no_speech_prob']:.3f}")
    print(f"Average compression ratio: {metrics['avg_compression_ratio']:.3f}")
    print(f"Segments per minute: {metrics['segments_per_minute']:.1f}")
    
    print(f"\n=== SCORES ===")
    scores = result['scores']
    print(f"Log probability score: {scores['logprob_score']:.3f}")
    print(f"No-speech score: {scores['no_speech_score']:.3f}")
    print(f"Compression score: {scores['compression_score']:.3f}")
    print(f"Fragmentation score: {scores['fragmentation_score']:.3f}")
    
    print(f"\n=== THRESHOLDS ===")
    thresholds_info = result['thresholds']
    print(f"Quality threshold: {thresholds_info['quality_threshold']}")
    print(f"Log probability threshold: {thresholds_info['logprob_threshold']}")
    print(f"No-speech threshold: {thresholds_info['no_speech_threshold']}")
    print(f"Compression threshold: {thresholds_info['compression_threshold']}")
    
    return result


def analyze_directory(transcript_dir: str, quality_threshold: float = 0.3):
    """Analyze all transcripts in a directory"""
    transcript_path = Path(transcript_dir)
    if not transcript_path.exists():
        print(f"Error: Directory not found: {transcript_dir}")
        return
    
    # Find all transcript files
    transcript_files = list(transcript_path.glob("*.json"))
    if not transcript_files:
        print(f"No transcript files found in {transcript_dir}")
        return
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Create analyzer
    thresholds = QualityThresholds(quality_threshold=quality_threshold)
    analyzer = QualityAnalyzer(thresholds)
    
    results = []
    low_quality_count = 0
    
    for transcript_file in transcript_files:
        print(f"\n{'='*60}")
        result = analyzer.analyze_transcript(str(transcript_file))
        
        if 'error' not in result:
            quality_score = result['overall_quality_score']
            is_acceptable = result['is_acceptable']
            
            print(f"File: {transcript_file.name}")
            print(f"Quality: {quality_score:.3f} ({'✅ Acceptable' if is_acceptable else '❌ Needs reprocessing'})")
            
            if not is_acceptable:
                low_quality_count += 1
                print(f"  - Log prob: {result['metrics']['avg_logprob']:.3f}")
                print(f"  - No speech: {result['metrics']['avg_no_speech_prob']:.3f}")
                print(f"  - Compression: {result['metrics']['avg_compression_ratio']:.3f}")
            
            results.append({
                "file": str(transcript_file),
                "quality_score": quality_score,
                "is_acceptable": is_acceptable,
                "metrics": result['metrics']
            })
        else:
            print(f"Error analyzing {transcript_file.name}: {result['error']}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"Total files: {len(transcript_files)}")
    print(f"Low quality files: {low_quality_count}")
    print(f"Acceptable files: {len(transcript_files) - low_quality_count}")
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze transcript quality")
    parser.add_argument("path", help="Transcript file or directory")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                       help="Quality threshold (0.0=perfect, 1.0=terrible)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        analyze_transcript_file(str(path), args.quality_threshold)
    elif path.is_dir():
        analyze_directory(str(path), args.quality_threshold)
    else:
        print(f"Error: Path not found: {path}")


if __name__ == "__main__":
    main()



