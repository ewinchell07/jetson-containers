#!/usr/bin/env python3
"""
Configuration module for Whisper-TRT transcription system.
Handles environment variables and system-specific settings.
"""

import os
import logging
import re
from pathlib import Path
from typing import Optional
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_TRANSCRIBE_MODEL = "small.en"  # Use English-only model by default
DEFAULT_ALLOW_SWAP = True
DEFAULT_MAX_DURATION_MIN = None
DEFAULT_TRANSCRIBE_PROMPT = ""  # Empty by default, will be generated dynamically

# Timezone configuration
UTC_TZ = pytz.UTC
PT_TZ = pytz.timezone('America/Los_Angeles')  # Pacific Time (handles PST/PDT automatically)

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


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract UTC timestamp from recording filename.
    Expected format: recording_YYYYMMDD_HHMMSS_Channel Name.wav
    or: recording_YYYYMMDD_HHMMSS.wav
    
    Args:
        filename: Recording filename
        
    Returns:
        Datetime object in UTC, or None if parsing fails
    """
    try:
        # Remove directory path if present
        base_name = Path(filename).stem
        
        # Match pattern: recording_YYYYMMDD_HHMMSS_*
        match = re.search(r'recording_(\d{8})_(\d{6})', base_name)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            
            # Parse as UTC
            utc_dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            # Localize to UTC
            utc_dt = UTC_TZ.localize(utc_dt)
            
            return utc_dt
        else:
            logger.warning(f"Could not extract timestamp from filename: {filename}")
            return None
    except Exception as e:
        logger.warning(f"Error extracting timestamp from filename {filename}: {e}")
        return None


def convert_utc_to_pt(utc_dt: datetime) -> datetime:
    """
    Convert UTC datetime to Pacific Time (handles PST/PDT automatically).
    
    Args:
        utc_dt: UTC datetime (must be timezone-aware)
        
    Returns:
        Pacific Time datetime
    """
    if utc_dt.tzinfo is None:
        # If naive, assume UTC
        utc_dt = UTC_TZ.localize(utc_dt)
    
    # Convert to PT
    pt_dt = utc_dt.astimezone(PT_TZ)
    return pt_dt


def generate_context_prompt(room_name: str, recording_time_utc: Optional[datetime] = None) -> str:
    """
    Generate an optimized context-aware prompt for family interaction transcription.
    
    The prompt is designed to:
    - Provide context (time and location)
    - Bias Whisper toward family/parenting vocabulary
    - Help with child speech recognition (common mispronunciations, developing language)
    - Include common parenting phrases and terminology
    
    Args:
        room_name: Name of the room/channel
        recording_time_utc: UTC datetime of recording (will be converted to PT)
        
    Returns:
        Optimized prompt string with family interaction vocabulary
    """
    # Build base context (time and location)
    base_context = ""
    if recording_time_utc:
        pt_time = convert_utc_to_pt(recording_time_utc)
        
        # Format: "Saturday, October 26, 2025 at 5:56 PM" in PT
        day_name = pt_time.strftime("%A")
        month_name = pt_time.strftime("%B")
        day = pt_time.day
        year = pt_time.year
        hour = pt_time.hour
        minute = pt_time.minute
        
        # Convert to 12-hour format
        if hour == 0:
            hour_12 = 12
            am_pm = "AM"
        elif hour < 12:
            hour_12 = hour
            am_pm = "AM"
        elif hour == 12:
            hour_12 = 12
            am_pm = "PM"
        else:
            hour_12 = hour - 12
            am_pm = "PM"
        
        # Format time: "5:56 PM" or "5 PM" (no minutes if zero, no leading zero for hour)
        if minute == 0:
            time_str = f"{hour_12} {am_pm}"
        else:
            time_str = f"{hour_12}:{minute:02d} {am_pm}"
        
        base_context = f"Family conversation in the {room_name} on {day_name}, {month_name} {day}, {year} at {time_str}. "
    else:
        base_context = f"Family conversation in the {room_name}. "
    
    # Family and parenting vocabulary to bias the model
    # Natural language format that demonstrates vocabulary in context
    # This helps Whisper understand pronunciation variations and common phrases
    family_vocab = (
        "Common terms include mommy, daddy, baby, child, kid, parent, family. "
        "Food words like waffle, milk, water, candy, croissant, snack, breakfast, lunch, dinner. "
        "Activities such as play, toy, game, sleep, nap, bed, bedtime, potty, diaper, pee, poop, tinkle. "
        "Phrases: I love you, thank you, please, all done, good job, let's try, that's okay, you're okay. "
        "Common words: yes, no, okay, oh no, what, why, where, help, want, need, can't, don't, won't. "
        "Child speech patterns: wawa for water, nana for banana, paci for pacifier, potty training terminology."
    )
    
    # Combine context with vocabulary bias
    # The prompt helps Whisper understand the domain and recognize child speech patterns
    prompt = base_context + family_vocab
    
    return prompt


def extract_room_name_from_filename(filename: str) -> str:
    """
    Extract room name from recording filename.
    Expected format: recording_YYYYMMDD_HHMMSS_Channel Name.wav
    
    Args:
        filename: Recording filename
        
    Returns:
        Room/channel name, or "Unknown" if not found
    """
    try:
        base_name = Path(filename).stem
        
        # Try to extract room name after timestamp
        match = re.search(r'recording_\d{8}_\d{6}_(.+)', base_name)
        if match:
            room_name = match.group(1)
            # Clean up any remaining suffixes
            room_name = re.sub(r'(_ch\d+|_partial)$', '', room_name)
            return room_name
        
        # If no room name found, try to infer from pattern
        # Check for common channel indicators
        if '_ch1' in base_name or 'TV Living Room' in base_name:
            return 'TV Living Room'
        elif '_ch2' in base_name or 'Dining Room' in base_name:
            return 'Dining Room'
        elif '_ch3' in base_name or 'Penn Bedroom' in base_name:
            return 'Penn Bedroom'
        elif '_ch4' in base_name or 'Rowe Bedroom' in base_name:
            return 'Rowe Bedroom'
        
        return "Unknown Location"
    except Exception as e:
        logger.warning(f"Error extracting room name from filename {filename}: {e}")
        return "Unknown Location"


def get_transcribe_prompt(audio_file: Optional[str] = None) -> str:
    """
    Get the transcription prompt from environment or generate dynamically.
    
    If audio_file is provided, extracts timestamp and room name from filename
    to generate a context-aware prompt with timezone conversion (UTC to PT).
    
    Args:
        audio_file: Optional path to audio file for dynamic prompt generation
        
    Returns:
        Transcription prompt string
    """
    # Check for explicit environment variable override
    env_prompt = os.getenv('TRANSCRIBE_PROMPT')
    if env_prompt:
        return env_prompt
    
    # If audio file provided, generate dynamic prompt
    if audio_file:
        try:
            room_name = extract_room_name_from_filename(audio_file)
            recording_time = extract_timestamp_from_filename(audio_file)
            prompt = generate_context_prompt(room_name, recording_time)
            logger.info(f"Generated context prompt: {prompt[:80]}...")
            return prompt
        except Exception as e:
            logger.warning(f"Error generating dynamic prompt, using default: {e}")
    
    # Fallback to default
    return DEFAULT_TRANSCRIBE_PROMPT


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
    Prefers English-only models to prevent language detection issues.
    
    Args:
        requested_model: The originally requested model
        
    Returns:
        A smaller model that should work
    """
    fallback_chain = {
        "large-v3": "large-v2",
        "large-v2": "large",
        "large": "medium.en",  # Prefer English-only models
        "medium": "medium.en",  # Convert to English-only
        "medium.en": "small.en",
        "small": "small.en"  # Convert to English-only
    }
    
    current = requested_model
    while current in fallback_chain:
        current = fallback_chain[current]
        required_swap = MODEL_SWAP_REQUIREMENTS[current]
        available_swap = get_swap_size_gb()
        
        if required_swap == 0 or available_swap >= required_swap:
            logger.info(f"Falling back to {current} (requires {required_swap}GB swap)")
            return current
    
    # Final fallback to small.en (English-only)
    logger.info("Final fallback to small.en model (English-only)")
    return "small.en"


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
