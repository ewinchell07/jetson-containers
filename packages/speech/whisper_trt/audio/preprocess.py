#!/usr/bin/env python3
"""
Audio pre-processing module for improving transcription quality.
Applies noise suppression, filtering, and normalization before Whisper transcription.
"""

import os
import subprocess
import shlex
import logging
from pathlib import Path
from typing import Optional

# Configuration constants
FFMPEG_FILTER = 'highpass=f=100,arnndn=m=rnnoise,loudnorm=I=-23:TP=-2:LRA=11'
FALLBACK_FILTER = 'highpass=f=100,afftdn=nf=-25:nt=w,aloudnorm=I=-23:TP=-2:LRA=11'

# Audio processing settings
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = 'pcm_s16le'

logger = logging.getLogger(__name__)


def preprocess_audio(input_path: Path) -> Path:
    """
    Pre-process audio file for better transcription quality.
    
    Applies:
    - High-pass filter (100Hz) to reduce rumble
    - RNNoise noise suppression (with fallback to afftdn)
    - Loudness normalization (ITU BS.1770)
    - Conversion to mono, 16kHz, 16-bit PCM WAV
    
    Args:
        input_path: Path to input audio file
        
    Returns:
        Path to preprocessed audio file, or original file if preprocessing fails
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return input_path
    
    # Create output filename: {stem}.clean.wav
    out_path = input_path.with_suffix('').with_name(input_path.stem + '.clean.wav')
    
    # Try with RNNoise first
    success = _run_ffmpeg_preprocessing(input_path, out_path, FFMPEG_FILTER)
    
    if not success:
        logger.warning("RNNoise preprocessing failed, trying fallback filter")
        # Fallback to afftdn if RNNoise fails
        success = _run_ffmpeg_preprocessing(input_path, out_path, FALLBACK_FILTER)
    
    if not success:
        logger.warning(f"Audio preprocessing failed for {input_path}, using original file")
        return input_path
    
    logger.info(f"Audio preprocessed successfully: {out_path}")
    return out_path


def _run_ffmpeg_preprocessing(input_path: Path, output_path: Path, filter_chain: str) -> bool:
    """
    Run ffmpeg preprocessing with the specified filter chain.
    
    Args:
        input_path: Input audio file
        output_path: Output audio file
        filter_chain: FFmpeg filter chain
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-i', str(input_path),
            '-af', filter_chain,
            '-ar', str(TARGET_SAMPLE_RATE),
            '-ac', str(TARGET_CHANNELS),
            '-c:a', TARGET_FORMAT,
            str(output_path)
        ]
        
        logger.debug(f"Running ffmpeg command: {' '.join(shlex.quote(str(arg)) for arg in cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Verify output file was created and has reasonable size
            if output_path.exists() and output_path.stat().st_size > 1000:  # At least 1KB
                return True
            else:
                logger.error(f"Output file creation failed or file too small: {output_path}")
                return False
        else:
            logger.error(f"FFmpeg failed with return code {result.returncode}")
            logger.error(f"FFmpeg stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg preprocessing timed out for {input_path}")
        return False
    except Exception as e:
        logger.error(f"FFmpeg preprocessing error for {input_path}: {e}")
        return False


def check_ffmpeg_capabilities() -> dict:
    """
    Check which audio filters are available in the current ffmpeg build.
    
    Returns:
        Dictionary with available filter information
    """
    capabilities = {
        'rnnoise': False,
        'afftdn': False,
        'loudnorm': False,
        'highpass': False
    }
    
    try:
        # Check for available filters
        result = subprocess.run(
            ['ffmpeg', '-filters'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            filters_output = result.stdout.lower()
            capabilities['rnnoise'] = 'arnndn' in filters_output
            capabilities['afftdn'] = 'afftdn' in filters_output
            capabilities['loudnorm'] = 'loudnorm' in filters_output
            capabilities['highpass'] = 'highpass' in filters_output
            
            logger.info(f"FFmpeg capabilities: {capabilities}")
        else:
            logger.warning("Could not check ffmpeg capabilities")
            
    except Exception as e:
        logger.warning(f"Error checking ffmpeg capabilities: {e}")
    
    return capabilities


def get_audio_info(file_path: Path) -> Optional[dict]:
    """
    Get basic audio file information using ffprobe.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio info or None if failed
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(file_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        else:
            logger.warning(f"Could not get audio info for {file_path}")
            return None
            
    except Exception as e:
        logger.warning(f"Error getting audio info for {file_path}: {e}")
        return None


if __name__ == "__main__":
    # Test the preprocessing function
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <audio_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check capabilities
    caps = check_ffmpeg_capabilities()
    print(f"FFmpeg capabilities: {caps}")
    
    # Preprocess audio
    output_file = preprocess_audio(input_file)
    print(f"Preprocessing result: {output_file}")
    
    # Show file info
    if output_file.exists():
        info = get_audio_info(output_file)
        if info:
            print(f"Output file info: {info}")
