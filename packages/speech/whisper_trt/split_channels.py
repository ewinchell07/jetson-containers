#!/usr/bin/env python3
"""
Split multi-channel audio file into individual mono files.

This script is designed to work with files recorded using:
arecord -D plughw:0,0 -f S32_LE -r 48000 -c 4 -t wav recording.wav

The script will create 4 individual mono files:
- recording_ch1.wav (channel 1)
- recording_ch2.wav (channel 2) 
- recording_ch3.wav (channel 3)
- recording_ch4.wav (channel 4)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def load_audio_file(filepath: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return data and sample rate"""
    try:
        logging.info(f"Loading audio file: {filepath}")
        audio_data, sample_rate = sf.read(filepath, dtype='float32')
        logging.info(f"Loaded audio: {audio_data.shape} samples at {sample_rate}Hz")
        return audio_data, sample_rate
    except Exception as e:
        logging.error(f"Failed to load audio file: {e}")
        raise


def split_channels(audio_data: np.ndarray, sample_rate: int, 
                  output_dir: Path, base_name: str, 
                  channel_names: List[str] = None) -> List[str]:
    """Split multi-channel audio into individual mono files"""
    
    if audio_data.ndim != 2:
        raise ValueError(f"Expected 2D audio data (samples, channels), got shape: {audio_data.shape}")
    
    num_channels = audio_data.shape[1]
    logging.info(f"Splitting {num_channels} channels")
    
    if channel_names is None:
        channel_names = [f"ch{i+1}" for i in range(num_channels)]
    
    if len(channel_names) != num_channels:
        logging.warning(f"Channel names count ({len(channel_names)}) doesn't match audio channels ({num_channels})")
        channel_names = [f"ch{i+1}" for i in range(num_channels)]
    
    output_files = []
    
    for ch in range(num_channels):
        # Extract mono channel
        mono_audio = audio_data[:, ch]
        
        # Generate output filename
        output_filename = f"{base_name}_{channel_names[ch]}.wav"
        output_path = output_dir / output_filename
        
        try:
            # Save mono file
            sf.write(str(output_path), mono_audio, sample_rate)
            
            duration = len(mono_audio) / sample_rate
            logging.info(f"✅ Saved channel {ch+1} ({channel_names[ch]}): {duration:.2f}s to {output_filename}")
            output_files.append(str(output_path))
            
        except Exception as e:
            logging.error(f"❌ Failed to save channel {ch+1}: {e}")
    
    return output_files


def validate_audio_file(filepath: str) -> bool:
    """Validate that the audio file exists and is readable"""
    if not os.path.exists(filepath):
        logging.error(f"File does not exist: {filepath}")
        return False
    
    try:
        # Try to read the file header
        with sf.SoundFile(filepath) as f:
            logging.info(f"Audio file info:")
            logging.info(f"  - Channels: {f.channels}")
            logging.info(f"  - Sample rate: {f.samplerate}Hz")
            logging.info(f"  - Duration: {f.frames / f.samplerate:.2f}s")
            logging.info(f"  - Format: {f.format}")
            logging.info(f"  - Subtype: {f.subtype}")
            
            if f.channels < 2:
                logging.warning(f"File has only {f.channels} channel(s), splitting may not be useful")
            
            return True
            
    except Exception as e:
        logging.error(f"Cannot read audio file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Split multi-channel audio file into individual mono files")
    parser.add_argument("input_file", help="Input multi-channel audio file")
    parser.add_argument("--output-dir", "-o", default=".", 
                       help="Output directory for split files (default: current directory)")
    parser.add_argument("--channel-names", nargs='+', 
                       help="Names for each channel (e.g., --channel-names mic1 mic2 mic3 mic4)")
    parser.add_argument("--prefix", default=None,
                       help="Prefix for output filenames (default: input filename without extension)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not validate_audio_file(args.input_file):
        sys.exit(1)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename prefix
    input_path = Path(args.input_file)
    if args.prefix:
        base_name = args.prefix
    else:
        base_name = input_path.stem
    
    try:
        # Load audio file
        audio_data, sample_rate = load_audio_file(args.input_file)
        
        # Check if we have multi-channel audio
        if audio_data.ndim == 1:
            logging.warning("Input file is mono, nothing to split")
            sys.exit(0)
        elif audio_data.ndim != 2:
            logging.error(f"Unexpected audio format: {audio_data.shape}")
            sys.exit(1)
        
        # Split channels
        output_files = split_channels(
            audio_data, sample_rate, output_dir, base_name, args.channel_names
        )
        
        logging.info(f"Successfully split {len(output_files)} channels")
        logging.info("Output files:")
        for file in output_files:
            logging.info(f"  - {file}")
            
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


