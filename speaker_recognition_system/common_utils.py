"""
Common utilities for Speaker Recognition System
Shared functions across multiple modules
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from config import Config

def validate_audio_file(file_path: str) -> bool:
    """
    Validate if an audio file is readable and has appropriate properties
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if file is valid
    """
    try:
        import librosa
        # Try to load the file
        y, sr = librosa.load(file_path, sr=None)
        
        # Check duration (should be at least minimum duration)
        duration = len(y) / sr
        if duration < Config.MIN_DURATION:
            print(f"Warning: {file_path} is too short ({duration:.2f}s)")
            return False
        
        # Check if audio has signal (not silent)
        if np.max(np.abs(y)) < 0.001:
            print(f"Warning: {file_path} appears to be silent")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating {file_path}: {str(e)}")
        return False

def get_audio_files(directory: str) -> List[str]:
    """
    Get all audio files from a directory
    
    Args:
        directory: Directory path
        
    Returns:
        List of audio file paths
    """
    if not os.path.exists(directory):
        return []
    
    audio_files = []
    for file in os.listdir(directory):
        if Config.is_audio_file(file):
            file_path = os.path.join(directory, file)
            if validate_audio_file(file_path):
                audio_files.append(file_path)
    
    return audio_files

def count_samples_per_speaker(data_dir: str) -> dict:
    """
    Count audio samples for each speaker
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dictionary with speaker names and sample counts
    """
    speaker_counts = {}
    
    if not os.path.exists(data_dir):
        return speaker_counts
    
    for speaker_name in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_name)
        if os.path.isdir(speaker_path):
            audio_files = get_audio_files(speaker_path)
            speaker_counts[speaker_name] = len(audio_files)
    
    return speaker_counts

def ensure_minimum_samples(data_dir: str, min_samples: int = Config.MIN_SAMPLES_PER_SPEAKER) -> Tuple[bool, str]:
    """
    Check if all speakers have minimum required samples
    
    Args:
        data_dir: Data directory path
        min_samples: Minimum samples per speaker
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    speaker_counts = count_samples_per_speaker(data_dir)
    
    if len(speaker_counts) < 2:
        return False, f"Need at least 2 speakers, found {len(speaker_counts)}"
    
    insufficient_speakers = []
    for speaker, count in speaker_counts.items():
        if count < min_samples:
            insufficient_speakers.append(f"{speaker}({count})")
    
    if insufficient_speakers:
        return False, f"Insufficient samples for: {', '.join(insufficient_speakers)}. Need {min_samples} minimum."
    
    return True, "All speakers have sufficient samples"

def create_speaker_directory(data_dir: str, speaker_name: str) -> str:
    """
    Create directory for a speaker
    
    Args:
        data_dir: Base data directory
        speaker_name: Name of the speaker
        
    Returns:
        Path to speaker directory
    """
    speaker_dir = os.path.join(data_dir, speaker_name)
    os.makedirs(speaker_dir, exist_ok=True)
    return speaker_dir

def safe_filename(name: str) -> str:
    """
    Convert a name to a safe filename
    
    Args:
        name: Original name
        
    Returns:
        Safe filename
    """
    import re
    # Replace spaces with underscores and remove invalid characters
    safe_name = re.sub(r'[^\w\s-]', '', name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name.strip('_')

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def calculate_dataset_stats(data_dir: str) -> dict:
    """
    Calculate comprehensive dataset statistics
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dictionary with dataset statistics
    """
    speaker_counts = count_samples_per_speaker(data_dir)
    
    stats = {
        'total_speakers': len(speaker_counts),
        'total_samples': sum(speaker_counts.values()),
        'min_samples': min(speaker_counts.values()) if speaker_counts else 0,
        'max_samples': max(speaker_counts.values()) if speaker_counts else 0,
        'avg_samples': sum(speaker_counts.values()) / len(speaker_counts) if speaker_counts else 0,
        'speaker_counts': speaker_counts
    }
    
    return stats
