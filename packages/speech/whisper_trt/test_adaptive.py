#!/usr/bin/env python3
"""
Test script for adaptive quality-based transcription.

This script tests the adaptive transcriber with sample files and provides
quality analysis.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

from adaptive_transcriber import AdaptiveTranscriber, QualityThresholds
from reprocess_adaptive import TranscriptReprocessor


def test_adaptive_transcription(audio_file: str, quality_threshold: float = 0.3):
    """Test adaptive transcription on a single file"""
    print(f"Testing adaptive transcription on: {audio_file}")
    
    # Create adaptive transcriber
    thresholds = QualityThresholds(quality_threshold=quality_threshold)
    transcriber = AdaptiveTranscriber(thresholds)
    
    try:
        # Transcribe with adaptive quality
        result = transcriber.transcribe_with_quality_check(audio_file)
        
        # Get quality report
        quality_report = transcriber.get_quality_report(result)
        
        print(f"\n=== TRANSCRIPTION RESULT ===")
        print(f"Model used: {result.model_name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Text length: {len(result.text)} characters")
        print(f"Number of segments: {len(result.segments)}")
        
        print(f"\n=== QUALITY ANALYSIS ===")
        print(f"Overall quality score: {quality_report['overall_quality_score']:.3f}")
        print(f"Is acceptable: {quality_report['is_acceptable']}")
        
        print(f"\n=== METRICS ===")
        metrics = quality_report['metrics']
        print(f"Average log probability: {metrics['avg_logprob']:.3f}")
        print(f"Average no-speech probability: {metrics['avg_no_speech_prob']:.3f}")
        print(f"Average compression ratio: {metrics['avg_compression_ratio']:.3f}")
        print(f"Segments per minute: {metrics['segments_per_minute']:.1f}")
        
        print(f"\n=== SCORES ===")
        scores = quality_report['scores']
        print(f"Log probability score: {scores['logprob_score']:.3f}")
        print(f"No-speech score: {scores['no_speech_score']:.3f}")
        print(f"Compression score: {scores['compression_score']:.3f}")
        print(f"Fragmentation score: {scores['fragmentation_score']:.3f}")
        
        print(f"\n=== THRESHOLDS ===")
        thresholds_info = quality_report['thresholds']
        print(f"Quality threshold: {thresholds_info['quality_threshold']}")
        print(f"Log probability threshold: {thresholds_info['logprob_threshold']}")
        print(f"No-speech threshold: {thresholds_info['no_speech_threshold']}")
        print(f"Compression threshold: {thresholds_info['compression_threshold']}")
        
        return result, quality_report
        
    except Exception as e:
        print(f"Error during adaptive transcription: {e}")
        return None, None


def test_quality_analysis(transcript_file: str):
    """Test quality analysis on an existing transcript"""
    print(f"Analyzing quality of: {transcript_file}")
    
    reprocessor = TranscriptReprocessor()
    quality_analysis = reprocessor.analyze_transcript_quality(transcript_file)
    
    print(f"\n=== QUALITY ANALYSIS ===")
    if 'error' in quality_analysis:
        print(f"Error: {quality_analysis['error']}")
        return
    
    print(f"Overall quality score: {quality_analysis['overall_quality_score']:.3f}")
    print(f"Is acceptable: {quality_analysis['is_acceptable']}")
    
    print(f"\n=== METRICS ===")
    metrics = quality_analysis['metrics']
    print(f"Average log probability: {metrics['avg_logprob']:.3f}")
    print(f"Average no-speech probability: {metrics['avg_no_speech_prob']:.3f}")
    print(f"Average compression ratio: {metrics['avg_compression_ratio']:.3f}")
    print(f"Segments per minute: {metrics['segments_per_minute']:.1f}")
    
    print(f"\n=== SCORES ===")
    scores = quality_analysis['scores']
    print(f"Log probability score: {scores['logprob_score']:.3f}")
    print(f"No-speech score: {scores['no_speech_score']:.3f}")
    print(f"Compression score: {scores['compression_score']:.3f}")
    print(f"Fragmentation score: {scores['fragmentation_score']:.3f}")
    
    return quality_analysis


def compare_transcripts(original_file: str, improved_file: str):
    """Compare original and improved transcripts"""
    print(f"Comparing transcripts:")
    print(f"Original: {original_file}")
    print(f"Improved: {improved_file}")
    
    # Analyze both files
    reprocessor = TranscriptReprocessor()
    
    original_quality = reprocessor.analyze_transcript_quality(original_file)
    improved_quality = reprocessor.analyze_transcript_quality(improved_file)
    
    print(f"\n=== COMPARISON ===")
    print(f"Original quality: {original_quality.get('overall_quality_score', 1.0):.3f}")
    print(f"Improved quality: {improved_quality.get('overall_quality_score', 1.0):.3f}")
    
    improvement = original_quality.get('overall_quality_score', 1.0) - improved_quality.get('overall_quality_score', 1.0)
    print(f"Improvement: {improvement:+.3f}")
    
    if improvement > 0:
        print("✅ Quality improved!")
    elif improvement < 0:
        print("❌ Quality degraded")
    else:
        print("➖ Quality unchanged")
    
    return improvement


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Test adaptive quality transcription")
    parser.add_argument("--audio-file", help="Audio file to transcribe")
    parser.add_argument("--transcript-file", help="Existing transcript file to analyze")
    parser.add_argument("--compare", nargs=2, metavar=("ORIGINAL", "IMPROVED"),
                       help="Compare two transcript files")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                       help="Quality threshold for adaptive processing")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if args.audio_file:
        # Test adaptive transcription
        if not os.path.exists(args.audio_file):
            print(f"Error: Audio file not found: {args.audio_file}")
            return
        
        result, quality_report = test_adaptive_transcription(
            args.audio_file, args.quality_threshold
        )
        
        if result:
            print(f"\n=== TRANSCRIPT PREVIEW ===")
            print(f"Text: {result.text[:200]}...")
    
    elif args.transcript_file:
        # Test quality analysis
        if not os.path.exists(args.transcript_file):
            print(f"Error: Transcript file not found: {args.transcript_file}")
            return
        
        test_quality_analysis(args.transcript_file)
    
    elif args.compare:
        # Compare transcripts
        original, improved = args.compare
        if not os.path.exists(original):
            print(f"Error: Original file not found: {original}")
            return
        if not os.path.exists(improved):
            print(f"Error: Improved file not found: {improved}")
            return
        
        compare_transcripts(original, improved)
    
    else:
        print("Please specify --audio-file, --transcript-file, or --compare")
        parser.print_help()


if __name__ == "__main__":
    main()



