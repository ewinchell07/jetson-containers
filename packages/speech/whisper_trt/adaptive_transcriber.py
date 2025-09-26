#!/usr/bin/env python3
"""
Adaptive Quality-Based Model Selection for Whisper Transcription

This module implements intelligent model selection based on transcription quality metrics.
It automatically retries with larger models when quality is insufficient.
"""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import time
from datetime import datetime

from transcriber import Transcriber, TranscriptionResult, TranscriptionConfig
from config import get_transcribe_model, get_allow_swap


@dataclass
class QualityThresholds:
    """Quality assessment thresholds and weights"""
    # Primary quality indicators
    avg_logprob_threshold: float = -1.2      # Below this = low confidence
    no_speech_prob_threshold: float = 0.4    # Above this = too much false speech
    compression_ratio_threshold: float = 3.0 # Above this = too repetitive
    
    # Secondary indicators
    min_segments_per_minute: float = 2.0      # Below this = too fragmented
    max_segments_per_minute: float = 20.0    # Above this = over-segmented
    
    # Quality scoring weights
    logprob_weight: float = 0.4              # 40% weight on confidence
    no_speech_weight: float = 0.3            # 30% weight on false speech
    compression_weight: float = 0.2           # 20% weight on repetition
    fragmentation_weight: float = 0.1        # 10% weight on fragmentation
    
    # Overall quality threshold (0.0 = perfect, 1.0 = terrible)
    quality_threshold: float = 0.3          # Below this = acceptable quality


@dataclass
class ModelTier:
    """Model tier configuration"""
    name: str
    models: List[str]
    description: str


class AdaptiveTranscriber:
    """Adaptive transcription with quality-based model selection"""
    
    def __init__(self, quality_thresholds: QualityThresholds = None):
        self.quality_thresholds = quality_thresholds or QualityThresholds()
        
        # Model tiers in order of increasing quality and resource usage
        # Allow large models only if explicitly enabled
        enable_large_models = os.getenv('ENABLE_LARGE_MODELS', 'false').lower() == 'true'
        
        self.model_tiers = [
            ModelTier('fast', ['tiny', 'base'], 'Fast but less accurate'),
            ModelTier('balanced', ['small', 'small.en'], 'Good balance of speed and accuracy'),
            ModelTier('accurate', ['medium', 'medium.en'], 'More accurate, slower')
        ]
        
        if enable_large_models:
            self.model_tiers.append(
                ModelTier('premium', ['large-v2', 'large-v3'], 'Most accurate, slowest')
            )
        
        # Maximum retries to prevent infinite loops
        self.max_retries = int(os.getenv('MAX_QUALITY_RETRIES', '2'))
        
        # Enable/disable adaptive quality
        self.enabled = os.getenv('ENABLE_QUALITY_RETRY', 'true').lower() == 'true'
        
        logging.info(f"AdaptiveTranscriber initialized: enabled={self.enabled}, max_retries={self.max_retries}")
    
    def transcribe_with_quality_check(self, audio_file: str, 
                                     initial_model: str = None) -> TranscriptionResult:
        """Transcribe with adaptive model selection based on quality"""
        
        if not self.enabled:
            # Fall back to standard transcription
            transcriber = Transcriber(initial_model or get_transcribe_model())
            config = TranscriptionConfig()
            result = transcriber.model_manager.transcribe_audio(audio_file, config)
            return TranscriptionResult(
                text=result.get('text', ''),
                segments=result.get('segments', []),
                speaker_segments=[],
                merged_segments=[],
                model_name=transcriber.model_manager.model_name,
                processing_time=result.get('processing_time', 0.0),
                gpu_memory_used=0.0,
                timestamp=datetime.now().isoformat(),
                audio_file=audio_file
            )
        
        if initial_model is None:
            initial_model = get_transcribe_model()
        
        current_model = initial_model
        retry_count = 0
        results = []
        
        logging.info(f"Starting adaptive transcription with {current_model}")
        
        while retry_count <= self.max_retries:
            try:
                # Transcribe with current model
                transcriber = Transcriber(current_model)
                config = TranscriptionConfig()
                result = transcriber.model_manager.transcribe_audio(audio_file, config)
                results.append((current_model, result))
                
                # Convert result to TranscriptionResult for quality assessment
                transcription_result = TranscriptionResult(
                    text=result.get('text', ''),
                    segments=result.get('segments', []),
                    speaker_segments=[],
                    merged_segments=[],
                    model_name=current_model,
                    processing_time=result.get('processing_time', 0.0),
                    gpu_memory_used=0.0,
                    timestamp=datetime.now().isoformat(),
                    audio_file=audio_file
                )
                
                # Assess quality
                quality_score = self._assess_quality(transcription_result)
                logging.info(f"Quality assessment for {current_model}: {quality_score:.3f}")
                
                # Check if quality is acceptable
                if self._is_quality_acceptable(quality_score):
                    logging.info(f"Quality acceptable with {current_model} (score: {quality_score:.3f})")
                    return transcription_result
                
                # Select next model tier
                next_model = self._select_next_model(current_model, quality_score)
                if not next_model:
                    logging.warning(f"No higher tier available, using {current_model}")
                    return transcription_result
                
                logging.info(f"Quality insufficient ({quality_score:.3f}), retrying with {next_model}")
                current_model = next_model
                retry_count += 1
                
            except Exception as e:
                logging.error(f"Error during transcription with {current_model}: {e}")
                if retry_count < self.max_retries:
                    next_model = self._select_next_model(current_model, 1.0)  # Assume worst quality
                    if next_model:
                        current_model = next_model
                        retry_count += 1
                        continue
                raise
        
        # Return the best result if all retries failed
        if results:
            # Convert the best result to TranscriptionResult
            best_model, best_result = min(results, key=lambda x: self._assess_quality(
                TranscriptionResult(
                    text=x[1].get('text', ''),
                    segments=x[1].get('segments', []),
                    speaker_segments=[],
                    merged_segments=[],
                    model_name=x[0],
                    processing_time=x[1].get('processing_time', 0.0),
                    gpu_memory_used=0.0,
                    timestamp=datetime.now().isoformat(),
                    audio_file=audio_file
                )
            ))
            logging.warning(f"Using best available result from {best_model}")
            return TranscriptionResult(
                text=best_result.get('text', ''),
                segments=best_result.get('segments', []),
                speaker_segments=[],
                merged_segments=[],
                model_name=best_model,
                processing_time=best_result.get('processing_time', 0.0),
                gpu_memory_used=0.0,
                timestamp=datetime.now().isoformat(),
                audio_file=audio_file
            )
        
        raise RuntimeError("All transcription attempts failed")
    
    def _assess_quality(self, result: TranscriptionResult) -> float:
        """Calculate overall quality score (0.0 = perfect, 1.0 = terrible)"""
        segments = result.segments
        
        if not segments:
            return 1.0  # No segments = terrible quality
        
        # Extract metrics
        avg_logprobs = [s.get('avg_logprob', -3.0) for s in segments]
        no_speech_probs = [s.get('no_speech_prob', 1.0) for s in segments]
        compression_ratios = [s.get('compression_ratio', 10.0) for s in segments]
        
        # Calculate quality components
        logprob_score = self._score_logprob(avg_logprobs)
        no_speech_score = self._score_no_speech(no_speech_probs)
        compression_score = self._score_compression(compression_ratios)
        fragmentation_score = self._score_fragmentation(segments, result.processing_time)
        
        # Weighted combination
        quality_score = (
            logprob_score * self.quality_thresholds.logprob_weight +
            no_speech_score * self.quality_thresholds.no_speech_weight +
            compression_score * self.quality_thresholds.compression_weight +
            fragmentation_score * self.quality_thresholds.fragmentation_weight
        )
        
        return quality_score
    
    def _score_logprob(self, logprobs: List[float]) -> float:
        """Score based on average log probability (0.0 = perfect, 1.0 = terrible)"""
        if not logprobs:
            return 1.0
        
        avg_logprob = np.mean(logprobs)
        # Convert to 0-1 scale where -0.5 = 0.0 (perfect) and -3.0 = 1.0 (terrible)
        return max(0.0, min(1.0, (-avg_logprob - 0.5) / 2.5))
    
    def _score_no_speech(self, no_speech_probs: List[float]) -> float:
        """Score based on no-speech probability (0.0 = perfect, 1.0 = terrible)"""
        if not no_speech_probs:
            return 1.0
        
        avg_no_speech = np.mean(no_speech_probs)
        # Convert to 0-1 scale where 0.1 = 0.0 (perfect) and 0.8 = 1.0 (terrible)
        return max(0.0, min(1.0, (avg_no_speech - 0.1) / 0.7))
    
    def _score_compression(self, compression_ratios: List[float]) -> float:
        """Score based on compression ratio (0.0 = perfect, 1.0 = terrible)"""
        if not compression_ratios:
            return 1.0
        
        avg_compression = np.mean(compression_ratios)
        # Convert to 0-1 scale where 1.0 = 0.0 (perfect) and 5.0 = 1.0 (terrible)
        return max(0.0, min(1.0, (avg_compression - 1.0) / 4.0))
    
    def _score_fragmentation(self, segments: List[Dict], duration: float) -> float:
        """Score based on segment fragmentation (0.0 = perfect, 1.0 = terrible)"""
        if not segments or duration <= 0:
            return 1.0
        
        segments_per_minute = len(segments) / (duration / 60.0)
        
        # Optimal: 3-8 segments per minute
        if 3.0 <= segments_per_minute <= 8.0:
            return 0.0
        elif segments_per_minute < 3.0:
            return (3.0 - segments_per_minute) / 3.0
        else:
            return min(1.0, (segments_per_minute - 8.0) / 12.0)
    
    def _is_quality_acceptable(self, quality_score: float) -> bool:
        """Determine if quality is acceptable"""
        return quality_score <= self.quality_thresholds.quality_threshold
    
    def _select_next_model(self, current_model: str, quality_score: float) -> Optional[str]:
        """Select next model tier for retry"""
        current_tier_index = self._get_model_tier_index(current_model)
        
        if current_tier_index is None:
            return None
        
        # Move to next tier
        next_tier_index = current_tier_index + 1
        if next_tier_index >= len(self.model_tiers):
            return None
        
        # Select best model from next tier
        next_tier = self.model_tiers[next_tier_index]
        
        # Prefer .en models for English content
        en_models = [m for m in next_tier.models if m.endswith('.en')]
        if en_models:
            return en_models[0]
        
        return next_tier.models[0]
    
    def _get_model_tier_index(self, model: str) -> Optional[int]:
        """Get the tier index for a model"""
        for i, tier in enumerate(self.model_tiers):
            if model in tier.models:
                return i
        return None
    
    def get_quality_report(self, result: TranscriptionResult) -> Dict[str, Any]:
        """Generate detailed quality report"""
        segments = result.segments
        
        if not segments:
            return {"error": "No segments to analyze"}
        
        # Extract metrics
        avg_logprobs = [s.get('avg_logprob', -3.0) for s in segments]
        no_speech_probs = [s.get('no_speech_prob', 1.0) for s in segments]
        compression_ratios = [s.get('compression_ratio', 10.0) for s in segments]
        
        # Calculate individual scores
        logprob_score = self._score_logprob(avg_logprobs)
        no_speech_score = self._score_no_speech(no_speech_probs)
        compression_score = self._score_compression(compression_ratios)
        fragmentation_score = self._score_fragmentation(segments, result.processing_time)
        
        # Overall quality score
        overall_score = (
            logprob_score * self.quality_thresholds.logprob_weight +
            no_speech_score * self.quality_thresholds.no_speech_weight +
            compression_score * self.quality_thresholds.compression_weight +
            fragmentation_score * self.quality_thresholds.fragmentation_weight
        )
        
        return {
            "overall_quality_score": overall_score,
            "is_acceptable": self._is_quality_acceptable(overall_score),
            "metrics": {
                "avg_logprob": np.mean(avg_logprobs),
                "avg_no_speech_prob": np.mean(no_speech_probs),
                "avg_compression_ratio": np.mean(compression_ratios),
                "segments_per_minute": len(segments) / (result.processing_time / 60.0) if result.processing_time > 0 else 0
            },
            "scores": {
                "logprob_score": logprob_score,
                "no_speech_score": no_speech_score,
                "compression_score": compression_score,
                "fragmentation_score": fragmentation_score
            },
            "thresholds": {
                "quality_threshold": self.quality_thresholds.quality_threshold,
                "logprob_threshold": self.quality_thresholds.avg_logprob_threshold,
                "no_speech_threshold": self.quality_thresholds.no_speech_prob_threshold,
                "compression_threshold": self.quality_thresholds.compression_ratio_threshold
            }
        }


def create_adaptive_transcriber(quality_threshold: float = 0.3, 
                                max_retries: int = 2) -> AdaptiveTranscriber:
    """Create an adaptive transcriber with custom settings"""
    thresholds = QualityThresholds(quality_threshold=quality_threshold)
    transcriber = AdaptiveTranscriber(thresholds)
    transcriber.max_retries = max_retries
    return transcriber


if __name__ == "__main__":
    # Test the adaptive transcriber
    import argparse
    
    parser = argparse.ArgumentParser(description="Test adaptive transcription")
    parser.add_argument("audio_file", help="Audio file to transcribe")
    parser.add_argument("--model", default=None, help="Initial model to use")
    parser.add_argument("--quality-threshold", type=float, default=0.3, help="Quality threshold")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retries")
    
    args = parser.parse_args()
    
    # Create adaptive transcriber
    transcriber = create_adaptive_transcriber(args.quality_threshold, args.max_retries)
    
    # Transcribe with quality check
    result = transcriber.transcribe_with_quality_check(args.audio_file, args.model)
    
    # Print quality report
    quality_report = transcriber.get_quality_report(result)
    print(json.dumps(quality_report, indent=2))


