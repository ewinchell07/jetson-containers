#!/usr/bin/env python3
"""
Configuration module for Whisper-TRT transcription system.
Handles environment variables and system-specific settings.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_TRANSCRIBE_MODEL = "small"
DEFAULT_ALLOW_SWAP = True
DEFAULT_MAX_DURATION_MIN = None
DEFAULT_TRANSCRIBE_PROMPT = ("")

# Diarization configuration defaults
DEFAULT_DIARIZATION_CONFIG = {
    "min_speakers": 1,
    "max_speakers": 5,
    "similarity_threshold": 0.6,  # Threshold for speaker similarity
    "segment_duration": 1.5,      # Duration of segments for speaker analysis (seconds)
    "overlap_duration": 0.5       # Overlap between segments (seconds)
}

# Audio configuration defaults
DEFAULT_AUDIO_CONFIG = {
    "target_sample_rate": 16000     # Whisper expects 16kHz
}

# Adaptive quality configuration defaults
# Optimized for normal audio conditions (20250916 baseline)
DEFAULT_QUALITY_CONFIG = {
    "enable_quality_retry": True,
    "quality_threshold": 0.35,          # 0.0 = perfect, 1.0 = terrible (more lenient for normal audio)
    "max_quality_retries": 2,           # Maximum retries with larger models
    "avg_logprob_threshold": -1.0,      # Below this = low confidence (stricter confidence requirement)
    "no_speech_prob_threshold": 0.35,   # Above this = too much false speech (more lenient for normal audio)
    "compression_ratio_threshold": 4.0, # Above this = too repetitive (more lenient for normal audio)
    "min_segments_per_minute": 2.0,     # Below this = too fragmented
    "max_segments_per_minute": 20.0,    # Above this = over-segmented
    "logprob_weight": 0.4,              # Weight for log probability scoring
    "no_speech_weight": 0.3,            # Weight for no-speech scoring
    "compression_weight": 0.2,          # Weight for compression scoring
    "fragmentation_weight": 0.1         # Weight for fragmentation scoring
}

# Model requirements (minimum swap size in GB)
MODEL_SWAP_REQUIREMENTS = {
    "tiny": 0,
    "tiny.en": 0,
    "base": 0,
    "base.en": 0,
    "small": 0,
    "small.en": 0,
    "medium": 8,
    "medium.en": 8,
    "large": 16,
    "large-v2": 16,
    "large-v3": 16
}

# Allowed model names
ALLOWED_MODELS = list(MODEL_SWAP_REQUIREMENTS.keys())


def get_transcribe_model() -> str:
    """Get the transcription model from environment or default."""
    model = os.getenv('TRANSCRIBE_MODEL', DEFAULT_TRANSCRIBE_MODEL).lower()
    if model not in ALLOWED_MODELS:
        logger.warning(f"Invalid model '{model}', using default '{DEFAULT_TRANSCRIBE_MODEL}'")
        return DEFAULT_TRANSCRIBE_MODEL
    return model


def get_allow_swap() -> bool:
    """Get whether swap is allowed from environment or default."""
    allow_swap = os.getenv('ALLOW_SWAP', str(DEFAULT_ALLOW_SWAP)).lower()
    return allow_swap in ('true', '1', 'yes', 'on')


def get_max_duration_min() -> Optional[int]:
    """Get maximum duration in minutes from environment or None."""
    max_duration = os.getenv('MAX_DURATION_MIN')
    if max_duration:
        try:
            return int(max_duration)
        except ValueError:
            logger.warning(f"Invalid MAX_DURATION_MIN value: {max_duration}")
    return None


def get_transcribe_prompt() -> str:
    """Get the transcription prompt from environment or default."""
    return os.getenv('TRANSCRIBE_PROMPT', DEFAULT_TRANSCRIBE_PROMPT)


def get_diarization_config() -> dict:
    """Get diarization configuration from environment or defaults."""
    config = DEFAULT_DIARIZATION_CONFIG.copy()
    
    # Allow environment variable overrides
    if os.getenv('DIARIZATION_MIN_SPEAKERS'):
        try:
            config['min_speakers'] = int(os.getenv('DIARIZATION_MIN_SPEAKERS'))
        except ValueError:
            logger.warning(f"Invalid DIARIZATION_MIN_SPEAKERS value: {os.getenv('DIARIZATION_MIN_SPEAKERS')}")
    
    if os.getenv('DIARIZATION_MAX_SPEAKERS'):
        try:
            config['max_speakers'] = int(os.getenv('DIARIZATION_MAX_SPEAKERS'))
        except ValueError:
            logger.warning(f"Invalid DIARIZATION_MAX_SPEAKERS value: {os.getenv('DIARIZATION_MAX_SPEAKERS')}")
    
    if os.getenv('DIARIZATION_SIMILARITY_THRESHOLD'):
        try:
            config['similarity_threshold'] = float(os.getenv('DIARIZATION_SIMILARITY_THRESHOLD'))
        except ValueError:
            logger.warning(f"Invalid DIARIZATION_SIMILARITY_THRESHOLD value: {os.getenv('DIARIZATION_SIMILARITY_THRESHOLD')}")
    
    if os.getenv('DIARIZATION_SEGMENT_DURATION'):
        try:
            config['segment_duration'] = float(os.getenv('DIARIZATION_SEGMENT_DURATION'))
        except ValueError:
            logger.warning(f"Invalid DIARIZATION_SEGMENT_DURATION value: {os.getenv('DIARIZATION_SEGMENT_DURATION')}")
    
    if os.getenv('DIARIZATION_OVERLAP_DURATION'):
        try:
            config['overlap_duration'] = float(os.getenv('DIARIZATION_OVERLAP_DURATION'))
        except ValueError:
            logger.warning(f"Invalid DIARIZATION_OVERLAP_DURATION value: {os.getenv('DIARIZATION_OVERLAP_DURATION')}")
    
    return config


def get_audio_config() -> dict:
    """Get audio configuration from environment or defaults."""
    config = DEFAULT_AUDIO_CONFIG.copy()
    
    # Allow environment variable overrides
    if os.getenv('AUDIO_TARGET_SAMPLE_RATE'):
        try:
            config['target_sample_rate'] = int(os.getenv('AUDIO_TARGET_SAMPLE_RATE'))
        except ValueError:
            logger.warning(f"Invalid AUDIO_TARGET_SAMPLE_RATE value: {os.getenv('AUDIO_TARGET_SAMPLE_RATE')}")
    
    return config


def is_jetson_nano() -> bool:
    """
    Detect if running on a Jetson Nano.
    
    Returns:
        True if running on Jetson Nano, False otherwise
    """
    try:
        # Check device tree model
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            return 'NVIDIA Jetson Nano' in model
    except Exception:
        pass
    
    try:
        # Check environment variable
        return os.getenv('JETSON_NANO', '').lower() in ('true', '1', 'yes', 'on')
    except Exception:
        pass
    
    return False


def get_swap_size_gb() -> float:
    """
    Get total swap size in GB.
    
    Returns:
        Total swap size in GB, or 0.0 if no swap or error
    """
    try:
        total_kb = 0
        with open('/proc/swaps', 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        total_kb += int(parts[2])
                    except ValueError:
                        continue
        return total_kb / (1024 * 1024)  # Convert KB to GB
    except Exception as e:
        logger.warning(f"Could not determine swap size: {e}")
        return 0.0


def select_model_with_fallback(requested_model: str, allow_swap: bool) -> str:
    """
    Select the best available model based on system capabilities.
    
    Args:
        requested_model: The requested model name
        allow_swap: Whether swap is allowed
        
    Returns:
        The selected model name
    """
    requested_model = requested_model.lower()
    on_nano = is_jetson_nano()
    
    logger.info(f"Model selection: requested={requested_model}, allow_swap={allow_swap}, on_nano={on_nano}")
    
    # If not on Nano, just use the requested model (let CUDA handle VRAM)
    if not on_nano:
        logger.info(f"Not on Jetson Nano, using requested model: {requested_model}")
        return requested_model
    
    # On Nano, check swap requirements
    if requested_model not in MODEL_SWAP_REQUIREMENTS:
        logger.warning(f"Unknown model {requested_model}, falling back to small")
        return "small"
    
    required_swap = MODEL_SWAP_REQUIREMENTS[requested_model]
    
    if required_swap == 0:
        # No swap required
        logger.info(f"Model {requested_model} requires no swap, using it")
        return requested_model
    
    if not allow_swap:
        logger.warning(f"Model {requested_model} requires {required_swap}GB swap but ALLOW_SWAP=false")
        return _fallback_to_smaller_model(requested_model)
    
    # Check available swap
    available_swap = get_swap_size_gb()
    logger.info(f"Available swap: {available_swap:.1f}GB, required: {required_swap}GB")
    
    if available_swap >= required_swap:
        logger.info(f"Sufficient swap available, using {requested_model}")
        return requested_model
    else:
        logger.warning(f"Insufficient swap: {available_swap:.1f}GB < {required_swap}GB")
        return _fallback_to_smaller_model(requested_model)


def _fallback_to_smaller_model(requested_model: str) -> str:
    """
    Fall back to a smaller model when swap is insufficient.
    
    Args:
        requested_model: The originally requested model
        
    Returns:
        A smaller model that should work
    """
    fallback_chain = {
        "large-v3": "large-v2",
        "large-v2": "large",
        "large": "medium",
        "medium": "small",
        "medium.en": "small.en"
    }
    
    current = requested_model
    while current in fallback_chain:
        current = fallback_chain[current]
        required_swap = MODEL_SWAP_REQUIREMENTS[current]
        available_swap = get_swap_size_gb()
        
        if required_swap == 0 or available_swap >= required_swap:
            logger.info(f"Falling back to {current} (requires {required_swap}GB swap)")
            return current
    
    # Final fallback to small
    logger.info("Final fallback to small model")
    return "small"


def get_model_swap_requirements(model: str) -> int:
    """
    Get the swap requirements for a model.
    
    Args:
        model: Model name
        
    Returns:
        Required swap size in GB
    """
    return MODEL_SWAP_REQUIREMENTS.get(model.lower(), 0)


def get_quality_config() -> dict:
    """
    Get adaptive quality configuration from environment variables.
    
    Returns:
        Dictionary with quality configuration settings
    """
    config = DEFAULT_QUALITY_CONFIG.copy()
    
    # Override with environment variables
    if os.getenv('ENABLE_QUALITY_RETRY') is not None:
        config['enable_quality_retry'] = os.getenv('ENABLE_QUALITY_RETRY', 'true').lower() == 'true'
    
    if os.getenv('QUALITY_THRESHOLD') is not None:
        config['quality_threshold'] = float(os.getenv('QUALITY_THRESHOLD', '0.3'))
    
    if os.getenv('MAX_QUALITY_RETRIES') is not None:
        config['max_quality_retries'] = int(os.getenv('MAX_QUALITY_RETRIES', '2'))
    
    if os.getenv('AVG_LOGPROB_THRESHOLD') is not None:
        config['avg_logprob_threshold'] = float(os.getenv('AVG_LOGPROB_THRESHOLD', '-1.2'))
    
    if os.getenv('NO_SPEECH_PROB_THRESHOLD') is not None:
        config['no_speech_prob_threshold'] = float(os.getenv('NO_SPEECH_PROB_THRESHOLD', '0.4'))
    
    if os.getenv('COMPRESSION_RATIO_THRESHOLD') is not None:
        config['compression_ratio_threshold'] = float(os.getenv('COMPRESSION_RATIO_THRESHOLD', '3.0'))
    
    return config


def log_system_info():
    """Log system information for debugging."""
    logger.info("=== System Information ===")
    logger.info(f"Jetson Nano: {is_jetson_nano()}")
    logger.info(f"Swap size: {get_swap_size_gb():.1f}GB")
    
    # Log quality configuration
    quality_config = get_quality_config()
    logger.info("=== Quality Configuration ===")
    logger.info(f"Quality retry enabled: {quality_config['enable_quality_retry']}")
    logger.info(f"Quality threshold: {quality_config['quality_threshold']}")
    logger.info(f"Max quality retries: {quality_config['max_quality_retries']}")
    logger.info(f"Allow swap: {get_allow_swap()}")
    logger.info(f"Requested model: {get_transcribe_model()}")
    logger.info(f"Selected model: {select_model_with_fallback(get_transcribe_model(), get_allow_swap())}")
    logger.info("=========================")


if __name__ == "__main__":
    # Test configuration
    logging.basicConfig(level=logging.INFO)
    log_system_info()
