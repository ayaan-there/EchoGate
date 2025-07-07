"""
Utility Functions for Speaker Recognition System
Common helper functions and data processing utilities
"""

import os
import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

def validate_audio_file(file_path: str) -> bool:
    """
    Validate if an audio file is readable and has appropriate properties
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if file is valid
    """
    try:
        # Try to load the file
        y, sr = librosa.load(file_path, sr=None)
        
        # Check duration (should be at least 0.5 seconds)
        duration = len(y) / sr
        if duration < 0.5:
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

def normalize_audio_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize audio features
    
    Args:
        features: Feature array
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        Normalized features
    """
    if method == 'standard':
        # Standard normalization (z-score)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        # Avoid division by zero
        std[std == 0] = 1
        return (features - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (features - min_val) / range_val
    
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1
        return (features - median) / iqr
    
    else:
        return features

def augment_audio_data(audio_data: np.ndarray, sr: int, 
                      augmentation_type: str = 'noise') -> np.ndarray:
    """
    Apply data augmentation to audio
    
    Args:
        audio_data: Raw audio data
        sr: Sample rate
        augmentation_type: Type of augmentation
        
    Returns:
        Augmented audio data
    """
    if augmentation_type == 'noise':
        # Add white noise
        noise_factor = 0.005
        noise = np.random.normal(0, noise_factor, len(audio_data))
        return audio_data + noise
    
    elif augmentation_type == 'pitch_shift':
        # Pitch shifting
        try:
            steps = np.random.uniform(-2, 2)  # Shift by up to 2 semitones
            return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=steps)
        except:
            return audio_data
    
    elif augmentation_type == 'time_stretch':
        # Time stretching
        try:
            rate = np.random.uniform(0.9, 1.1)  # 10% variation
            return librosa.effects.time_stretch(audio_data, rate=rate)
        except:
            return audio_data
    
    elif augmentation_type == 'volume':
        # Volume adjustment
        factor = np.random.uniform(0.8, 1.2)
        return audio_data * factor
    
    else:
        return audio_data

def extract_enhanced_features(audio_path: str, sr: Union[int, float] = 22050) -> Optional[np.ndarray]:
    """
    Extract enhanced features including MFCC, spectral features, and rhythm
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        
    Returns:
        Enhanced feature vector
    """
    try:
        # Load audio
        y, actual_sr = librosa.load(audio_path, sr=sr)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=actual_sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Delta and delta-delta MFCC
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        delta_mean = np.mean(delta_mfccs, axis=1)
        delta2_mean = np.mean(delta2_mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=actual_sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=actual_sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=actual_sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Statistical features of spectral characteristics
        spectral_features = np.array([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(zero_crossing_rate),
            np.std(zero_crossing_rate)
        ])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=actual_sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            delta_mean, delta2_mean,
            spectral_features,
            chroma_mean
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting enhanced features from {audio_path}: {str(e)}")
        return None

def split_audio_file(audio_path: str, segment_duration: float = 3.0, 
                    overlap: float = 0.5) -> List[np.ndarray]:
    """
    Split a long audio file into shorter segments
    
    Args:
        audio_path: Path to audio file
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments (0-1)
        
    Returns:
        List of audio segments
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        segment_samples = int(segment_duration * sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        start = 0
        
        while start + segment_samples <= len(y):
            segment = y[start:start + segment_samples]
            segments.append(segment)
            start += hop_samples
        
        return segments
        
    except Exception as e:
        print(f"Error splitting audio file {audio_path}: {str(e)}")
        return []

def calculate_snr(audio_data: np.ndarray, noise_duration: float = 0.5) -> float:
    """
    Calculate Signal-to-Noise Ratio of audio
    
    Args:
        audio_data: Audio signal
        noise_duration: Duration to consider as noise (from beginning)
        
    Returns:
        SNR in dB
    """
    try:
        # Assume first part is noise
        noise_samples = int(noise_duration * 22050)  # Assume 22050 Hz
        noise_samples = min(noise_samples, len(audio_data) // 4)  # Max 25% as noise
        
        if noise_samples > 0:
            noise = audio_data[:noise_samples]
            signal = audio_data[noise_samples:]
            
            noise_power = np.mean(noise ** 2)
            signal_power = np.mean(signal ** 2)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                return snr
        
        return float('inf')  # Very clean signal
        
    except:
        return 0.0

def preprocess_for_training(data_dir: str, output_dir: Optional[str] = None, 
                          augment: bool = False) -> None:
    """
    Preprocess all audio files for training
    
    Args:
        data_dir: Directory containing speaker folders
        output_dir: Output directory for processed files
        augment: Whether to apply data augmentation
    """
    if output_dir is None:
        output_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all speaker folders
    speakers = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and d != "processed"]
    
    for speaker in speakers:
        speaker_input_dir = os.path.join(data_dir, speaker)
        speaker_output_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(speaker_input_dir) 
                      if f.lower().endswith(('.mp3', '.wav', '.flac'))]
        
        for audio_file in audio_files:
            input_path = os.path.join(speaker_input_dir, audio_file)
            
            # Validate file
            if not validate_audio_file(input_path):
                continue
            
            try:
                # Load and normalize audio
                y, actual_sr = librosa.load(input_path, sr=22050)
                
                # Normalize volume
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y)) * 0.9
                
                # Save processed file
                output_filename = os.path.splitext(audio_file)[0] + "_processed.wav"
                output_path = os.path.join(speaker_output_dir, output_filename)
                
                import soundfile as sf
                sf.write(output_path, y, actual_sr)
                
                # Apply augmentation if requested
                if augment:
                    # Create augmented versions
                    augmentations = ['noise', 'pitch_shift', 'volume']
                    
                    for aug_type in augmentations:
                        augmented = augment_audio_data(y, int(actual_sr), aug_type)
                        aug_filename = f"{os.path.splitext(audio_file)[0]}_{aug_type}.wav"
                        aug_path = os.path.join(speaker_output_dir, aug_filename)
                        sf.write(aug_path, augmented, int(actual_sr))
                
                print(f"✅ Processed: {speaker}/{audio_file}")
                
            except Exception as e:
                print(f"❌ Failed to process {speaker}/{audio_file}: {str(e)}")

def analyze_dataset_quality(data_dir: str) -> Dict[str, Any]:
    """
    Analyze the quality of the dataset
    
    Args:
        data_dir: Directory containing speaker folders
        
    Returns:
        Dataset quality analysis
    """
    analysis = {
        'speakers': {},
        'total_files': 0,
        'total_duration': 0,
        'average_snr': [],
        'issues': []
    }
    
    speakers = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    for speaker in speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        audio_files = [f for f in os.listdir(speaker_dir) 
                      if f.lower().endswith(('.mp3', '.wav', '.flac'))]
        
        speaker_info = {
            'file_count': len(audio_files),
            'total_duration': 0,
            'avg_duration': 0,
            'snr_values': [],
            'valid_files': 0
        }
        
        for audio_file in audio_files:
            file_path = os.path.join(speaker_dir, audio_file)
            
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
                
                speaker_info['total_duration'] += duration
                analysis['total_duration'] += duration
                
                # Calculate SNR
                snr = calculate_snr(y)
                speaker_info['snr_values'].append(snr)
                analysis['average_snr'].append(snr)
                
                if validate_audio_file(file_path):
                    speaker_info['valid_files'] += 1
                
            except Exception as e:
                analysis['issues'].append(f"Could not load {speaker}/{audio_file}: {str(e)}")
        
        if len(audio_files) > 0:
            speaker_info['avg_duration'] = speaker_info['total_duration'] / len(audio_files)
        
        analysis['speakers'][speaker] = speaker_info
        analysis['total_files'] += len(audio_files)
    
    # Overall statistics
    if analysis['average_snr']:
        analysis['overall_snr'] = np.mean(analysis['average_snr'])
    else:
        analysis['overall_snr'] = 0
    
    # Check for issues
    if len(speakers) < 2:
        analysis['issues'].append("Need at least 2 speakers for training")
    
    for speaker, info in analysis['speakers'].items():
        if info['file_count'] < 3:
            analysis['issues'].append(f"Speaker {speaker} has only {info['file_count']} files (recommend 3+)")
    
    return analysis

def print_dataset_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print formatted dataset analysis
    
    Args:
        analysis: Analysis results from analyze_dataset_quality
    """
    print("\n📊 Dataset Quality Analysis")
    print("=" * 40)
    
    print(f"Total Speakers: {len(analysis['speakers'])}")
    print(f"Total Files: {analysis['total_files']}")
    print(f"Total Duration: {analysis['total_duration']:.2f} seconds")
    
    if analysis['average_snr']:
        print(f"Average SNR: {analysis['overall_snr']:.2f} dB")
    
    print("\n👤 Speaker Details:")
    print("-" * 20)
    
    for speaker, info in analysis['speakers'].items():
        print(f"{speaker}:")
        print(f"  Files: {info['file_count']} (Valid: {info['valid_files']})")
        print(f"  Duration: {info['total_duration']:.2f}s (Avg: {info['avg_duration']:.2f}s)")
        if info['snr_values']:
            print(f"  SNR: {np.mean(info['snr_values']):.2f} dB")
    
    if analysis['issues']:
        print("\n⚠️  Issues Found:")
        print("-" * 15)
        for issue in analysis['issues']:
            print(f"• {issue}")
    else:
        print("\n✅ No issues found!")

def main():
    """Test utility functions"""
    from config import Config
    data_dir = Config.DATA_DIR
    
    if os.path.exists(data_dir):
        print("🔍 Analyzing dataset quality...")
        analysis = analyze_dataset_quality(data_dir)
        print_dataset_analysis(analysis)
    else:
        print("❌ Data directory not found. Run main.py --mode setup first.")

if __name__ == "__main__":
    main()
