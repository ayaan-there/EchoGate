"""
Feature Extraction Module for Speaker Recognition
Extracts enhanced acoustic features from audio files for speaker recognition

Features extracted include:
1. MFCC (Mel-Frequency Cepstral Coefficients) with extended statistics
   - Mean, standard deviation, min, max, median, skewness
2. Delta and Delta-Delta coefficients with statistics
3. Spectral features (centroid, bandwidth, contrast)
4. Fundamental frequency (pitch) statistics

Also implements Voice Activity Detection (VAD) to:
- Remove silence segments
- Focus on speech-only portions
- Improve signal-to-noise ratio
"""

import os
import librosa
import numpy as np
import pandas as pd
import scipy.stats
from typing import Tuple, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=20, n_fft=2048, hop_length=512, 
                top_db=20, preemphasis=0.97):
        """
        Initialize the feature extractor with enhanced parameters
        
        Args:
            sr: Sample rate for audio loading
            n_mfcc: Number of MFCC coefficients to extract (increased from default 13)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            top_db: Threshold (in dB) for silence detection in VAD
            preemphasis: Pre-emphasis filter coefficient
        """
        self.sr = sr
        self.n_mfcc = n_mfcc  # Increased from 13 to 20 for better speaker discrimination
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.top_db = top_db
        self.preemphasis = preemphasis
    
    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """
        Clean feature array by handling NaN, inf values and outliers
        
        Args:
            features: Feature array possibly containing NaN values
            
        Returns:
            Cleaned feature array
        """
        # Check for NaN values before cleaning
        if np.isnan(features).any():
            print(f"Warning: NaN values detected in features, replacing with zeros")
        
        # Replace NaN and inf values with zeros
        cleaned_features = np.nan_to_num(features)
        
        # Double-check no NaN values remain
        assert not np.isnan(cleaned_features).any(), "NaN values still present after cleaning!"
        
        return cleaned_features
    
    def extract_mfcc_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract MFCC features from a single audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            MFCC features as numpy array with enhanced statistics
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Apply Voice Activity Detection (VAD) to remove silence
            # First, trim leading and trailing silence
            y, _ = librosa.effects.trim(y, top_db=self.top_db)
            
            # Then split into speech segments
            intervals = librosa.effects.split(y, top_db=self.top_db)
            if len(intervals) > 0:
                # Concatenate all speech segments
                y_speech = np.concatenate([y[start:end] for start, end in intervals])
            else:
                # If no intervals found, use the original audio
                y_speech = y
                
            # Apply pre-emphasis to enhance high frequencies
            y_preemph = librosa.effects.preemphasis(y_speech, coef=self.preemphasis)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=y_preemph, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Extract enhanced statistics across time
            mfcc_mean = np.mean(mfccs, axis=1)  # Central tendency
            mfcc_std = np.std(mfccs, axis=1)    # Variation
            mfcc_max = np.max(mfccs, axis=1)    # Peak values
            mfcc_min = np.min(mfccs, axis=1)    # Minimum values
            mfcc_median = np.median(mfccs, axis=1)  # Robust central tendency
            mfcc_skew = scipy.stats.skew(mfccs, axis=1)  # Distribution asymmetry
            
            # Compute deltas (velocity) and delta-deltas (acceleration)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Get statistics for deltas
            delta_mean = np.mean(delta_mfccs, axis=1)
            delta_std = np.std(delta_mfccs, axis=1)
            
            # Get statistics for delta-deltas
            delta2_mean = np.mean(delta2_mfccs, axis=1)
            delta2_std = np.std(delta2_mfccs, axis=1)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_preemph, sr=sr)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_preemph, sr=sr)[0])
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y_preemph, sr=sr), axis=1)
            
            # Fundamental frequency (pitch) statistics
            pitches, magnitudes = librosa.piptrack(y=y_preemph, sr=sr)
            pitch_stats = np.array([])
            if magnitudes.size > 0:
                # Get pitches with highest magnitudes
                pitch_indices = np.argmax(magnitudes, axis=0)
                pitches_max = pitches[pitch_indices, np.arange(magnitudes.shape[1])]
                # Filter out zero pitches (silence)
                pitches_valid = pitches_max[pitches_max > 0]
                if pitches_valid.size > 0:
                    pitch_stats = np.array([
                        np.mean(pitches_valid),
                        np.std(pitches_valid),
                        np.min(pitches_valid),
                        np.max(pitches_valid)
                    ])
            
            # If no valid pitches found, use zeros
            if pitch_stats.size == 0:
                pitch_stats = np.zeros(4)
            
            # Concatenate all features
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min, mfcc_median, mfcc_skew,
                delta_mean, delta_std, delta2_mean, delta2_std,
                spectral_contrast, [spectral_centroid, spectral_bandwidth], 
                pitch_stats
            ])
            
            # Clean features by handling NaN and inf values
            features_cleaned = self._clean_features(features)
            
            return features_cleaned
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def load_dataset_from_folders(self, data_dir: str, use_pca: bool = True, n_components: int = 50, 
                                 is_librispeech: bool = False, max_speakers: Optional[int] = None, 
                                 max_files_per_speaker: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[object]]:
        """
        Load audio files from folders and extract features
        
        Args:
            data_dir: Directory containing speaker folders
            use_pca: Whether to apply PCA for dimensionality reduction
            n_components: Number of PCA components to keep
            is_librispeech: Whether the dataset follows LibriSpeech structure
            max_speakers: Maximum number of speakers to load (None for all)
            max_files_per_speaker: Maximum files per speaker (None for all)
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
            label_names: List of speaker names
        """
        features_list = []
        labels_list = []
        label_names = []
        
        if is_librispeech:
            # LibriSpeech structure: speaker_id/chapter_id/*.flac
            speaker_folders = [f for f in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()]
        else:
            # Standard structure: user_1, user_2, etc.
            speaker_folders = [f for f in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, f))]
        
        # Limit number of speakers if specified
        if max_speakers:
            speaker_folders = speaker_folders[:max_speakers]
        
        print(f"Found {len(speaker_folders)} speaker folders: {speaker_folders[:10]}{'...' if len(speaker_folders) > 10 else ''}")
        
        for speaker_idx, speaker_folder in enumerate(speaker_folders):
            speaker_path = os.path.join(data_dir, speaker_folder)
            label_names.append(speaker_folder)
            
            audio_files = []
            
            if is_librispeech:
                # LibriSpeech: Look in chapter subdirectories for .flac files
                for chapter_folder in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter_folder)
                    if os.path.isdir(chapter_path):
                        chapter_files = [f for f in os.listdir(chapter_path) 
                                       if f.lower().endswith(('.flac', '.wav', '.mp3'))]
                        for file in chapter_files:
                            audio_files.append(os.path.join(chapter_path, file))
            else:
                # Standard structure: files directly in speaker folder
                audio_files = [os.path.join(speaker_path, f) for f in os.listdir(speaker_path) 
                              if f.lower().endswith(('.mp3', '.wav', '.flac'))]
            
            # Limit files per speaker if specified
            if max_files_per_speaker:
                audio_files = audio_files[:max_files_per_speaker]
            
            print(f"Processing {len(audio_files)} files for speaker {speaker_folder}")
            
            for file_path in audio_files:
                features = self.extract_mfcc_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(speaker_idx)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Final safety check: ensure no NaN values in the dataset
        if np.isnan(X).any():
            print("Warning: NaN values found in final dataset. Applying additional cleaning...")
            X = np.nan_to_num(X)
            
        # Apply PCA for dimensionality reduction if requested
        pca = None
        if use_pca and X.shape[0] > n_components:
            from sklearn.decomposition import PCA
            print(f"Applying PCA to reduce dimensions from {X.shape[1]} to {n_components}...")
            pca = PCA(n_components=n_components, random_state=42)
            X = pca.fit_transform(X)
            explained_variance = sum(pca.explained_variance_ratio_) * 100
            print(f"PCA explained variance: {explained_variance:.2f}%")
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(label_names)} speakers")
        
        return X, y, label_names, pca
    
    def extract_features_from_audio_data(self, audio_data: np.ndarray, sr: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Extract MFCC features from raw audio data (for real-time processing)
        
        Args:
            audio_data: Raw audio data as numpy array
            sr: Sample rate (if None, uses self.sr)
            
        Returns:
            Enhanced MFCC features as numpy array
        """
        if sr is None:
            sr = self.sr
            
        try:
            # Resample if necessary
            if sr != self.sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sr)
                
            # Apply Voice Activity Detection (VAD) to remove silence
            # First, trim leading and trailing silence
            audio_data, _ = librosa.effects.trim(audio_data, top_db=self.top_db)
            
            # Then split into speech segments
            intervals = librosa.effects.split(audio_data, top_db=self.top_db)
            if len(intervals) > 0:
                # Concatenate all speech segments
                y_speech = np.concatenate([audio_data[start:end] for start, end in intervals])
            else:
                # If no intervals found, use the original audio
                y_speech = audio_data
                
            # Apply pre-emphasis to enhance high frequencies
            y_preemph = librosa.effects.preemphasis(y_speech, coef=self.preemphasis)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=y_preemph, 
                sr=self.sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Extract enhanced statistics across time
            mfcc_mean = np.mean(mfccs, axis=1)  # Central tendency
            mfcc_std = np.std(mfccs, axis=1)    # Variation
            mfcc_max = np.max(mfccs, axis=1)    # Peak values
            mfcc_min = np.min(mfccs, axis=1)    # Minimum values
            mfcc_median = np.median(mfccs, axis=1)  # Robust central tendency
            mfcc_skew = scipy.stats.skew(mfccs, axis=1)  # Distribution asymmetry
            
            # Compute deltas (velocity) and delta-deltas (acceleration)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Get statistics for deltas
            delta_mean = np.mean(delta_mfccs, axis=1)
            delta_std = np.std(delta_mfccs, axis=1)
            
            # Get statistics for delta-deltas
            delta2_mean = np.mean(delta2_mfccs, axis=1)
            delta2_std = np.std(delta2_mfccs, axis=1)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_preemph, sr=sr)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_preemph, sr=sr)[0])
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y_preemph, sr=sr), axis=1)
            
            # Fundamental frequency (pitch) statistics
            pitches, magnitudes = librosa.piptrack(y=y_preemph, sr=sr)
            pitch_stats = np.array([])
            if magnitudes.size > 0:
                # Get pitches with highest magnitudes
                pitch_indices = np.argmax(magnitudes, axis=0)
                pitches_max = pitches[pitch_indices, np.arange(magnitudes.shape[1])]
                # Filter out zero pitches (silence)
                pitches_valid = pitches_max[pitches_max > 0]
                if pitches_valid.size > 0:
                    pitch_stats = np.array([
                        np.mean(pitches_valid),
                        np.std(pitches_valid),
                        np.min(pitches_valid),
                        np.max(pitches_valid)
                    ])
            
            # If no valid pitches found, use zeros
            if pitch_stats.size == 0:
                pitch_stats = np.zeros(4)
            
            # Concatenate all features
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min, mfcc_median, mfcc_skew,
                delta_mean, delta_std, delta2_mean, delta2_std,
                spectral_contrast, [spectral_centroid, spectral_bandwidth], 
                pitch_stats
            ])
            
            # Clean features by handling NaN and inf values
            features_cleaned = self._clean_features(features)
            
            return features_cleaned
            
        except Exception as e:
            print(f"Error processing audio data: {str(e)}")
            return None

def main():
    """
    Feature extraction system - Use the Streamlit app instead
    """
    print("Feature Extraction System")
    print("=" * 40)
    print("For feature extraction, please use the Streamlit web interface:")
    print("streamlit run streamlit_app.py")
    print()
    print("The web interface provides:")
    print("- Interactive dataset management")
    print("- Feature extraction with progress tracking")
    print("- PCA configuration and visualization")
    print("- Dataset balancing and cleaning")

if __name__ == "__main__":
    main()
