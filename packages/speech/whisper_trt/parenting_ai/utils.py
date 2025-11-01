"""
Utility functions for Parenting AI SMS System
"""

import os
import re
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution"""
    config_file = Path(__file__).parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Replace environment variables
    content = os.path.expandvars(content)
    
    return yaml.safe_load(content)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("parenting_ai")
    logger.setLevel(getattr(logging, config['logging']['level']))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - handle permission errors gracefully
    log_file = Path(config['logging']['file'])
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not create log file {log_file}: {e}. Continuing with console logging only.")
    
    return logger

def extract_time_range(query: str) -> Optional[Dict[str, Any]]:
    """Extract time range hints from user query"""
    query_lower = query.lower()
    
    # Time patterns
    patterns = {
        'yesterday': {'hours': 24, 'label': 'yesterday'},
        'today': {'hours': 24, 'label': 'today'},
        'last week': {'hours': 168, 'label': 'last week'},
        'last 3 days': {'hours': 72, 'label': 'last 3 days'},
        'last 2 days': {'hours': 48, 'label': 'last 2 days'},
        'this week': {'hours': 168, 'label': 'this week'},
    }
    
    for pattern, time_info in patterns.items():
        if pattern in query_lower:
            return {
                'hours': time_info['hours'],
                'label': time_info['label'],
                'start_time': datetime.now() - timedelta(hours=time_info['hours'])
            }
    
    return None

def format_phone_number(phone: str) -> str:
    """Format phone number to standard format"""
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    
    # Add + if not present and not starting with 1
    if not phone.startswith('+'):
        if len(digits) == 10:
            digits = '1' + digits
        phone = '+' + digits
    
    return phone

def is_phone_whitelisted(phone: str, config: Dict[str, Any]) -> bool:
    """Check if phone number is in whitelist"""
    formatted_phone = format_phone_number(phone)
    whitelist = config['whitelist']['phones']
    
    return formatted_phone in whitelist

def truncate_sms(text: str, max_length: int = 160) -> str:
    """Truncate text for SMS with smart word boundaries"""
    if len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can keep 80% of the text
        return text[:last_space] + "..."
    else:
        return text[:max_length-3] + "..."

def format_sms_response(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Format response for SMS with length management"""
    sms_config = config['sms']
    
    if len(text) <= sms_config['brief_length']:
        return {
            'text': text,
            'length': 'brief',
            'parts': 1
        }
    elif len(text) <= sms_config['normal_length']:
        return {
            'text': text,
            'length': 'normal', 
            'parts': 2
        }
    elif len(text) <= sms_config['detailed_length']:
        return {
            'text': text,
            'length': 'detailed',
            'parts': 4
        }
    else:
        # Truncate and add continuation hint
        truncated = truncate_sms(text, sms_config['detailed_length'])
        return {
            'text': truncated + f"\n\nReply '{sms_config['continuation_keyword']}' for more details.",
            'length': 'truncated',
            'parts': 4,
            'has_more': True
        }

def extract_location_from_filename(filename: str) -> str:
    """Extract location from transcript filename"""
    # Pattern: recording_YYYYMMDD_HHMMSS_Location Name.wav
    match = re.search(r'recording_\d{8}_\d{6}_(.+?)(?:_partial)?\.wav', filename)
    if match:
        return match.group(1)
    
    # Pattern: transcript_recording_YYYYMMDD_HHMMSS_chN_partial_YYYYMMDD_HHMMSS.json
    match = re.search(r'transcript_recording_\d{8}_\d{6}_ch(\d+)', filename)
    if match:
        channel = int(match.group(1))
        locations = {
            1: "TV Living Room",
            2: "Dining Room", 
            3: "Rowe Bedroom",
            4: "Penn Bedroom"
        }
        return locations.get(channel, f"Channel {channel}")
    
    return "Unknown Location"

def get_system_info() -> Dict[str, Any]:
    """Get system resource information"""
    try:
        # Memory info
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1)) / 1024  # MB
        mem_available = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1)) / 1024  # MB
        
        # CPU info
        with open('/proc/loadavg', 'r') as f:
            loadavg = f.read().strip().split()
        
        return {
            'memory_total_mb': mem_total,
            'memory_available_mb': mem_available,
            'memory_used_percent': ((mem_total - mem_available) / mem_total) * 100,
            'load_1min': float(loadavg[0]),
            'load_5min': float(loadavg[1]),
            'load_15min': float(loadavg[2]),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }



