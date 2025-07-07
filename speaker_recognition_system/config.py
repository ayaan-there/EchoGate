"""
Configuration file for Speaker Recognition System
Centralizes all configuration parameters and paths
"""

import os
import warnings

# Suppress warnings globally
warnings.filterwarnings('ignore')

class Config:
    """Configuration class for the Speaker Recognition System"""
    
    # Base paths - Make these relative for portability
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    
    # LibriSpeech dataset path (adjust as needed)
    COMMON_VOICE_PATH  = os.path.join(
        os.path.dirname(BASE_DIR), 
        "LibriSpeech",
        "dev-clean"
    )
    
    # Audio processing parameters
    SAMPLE_RATE = 22050
    N_MFCC = 13
    DEFAULT_DURATION = 3.0
    MIN_DURATION = 0.5
    
    # Model training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CONFIDENCE_THRESHOLD = 0.7
    
    # Feature extraction parameters
    N_FEATURES_EXPECTED = 39  # 13 MFCC + 13 delta + 13 delta2 + spectral features
    
    # Training parameters
    MIN_SAMPLES_PER_SPEAKER = 3
    MAX_SAMPLES_PER_SPEAKER = 10
    DEFAULT_SAMPLES_PER_SPEAKER = 5
    
    # File extensions
    AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.m4a')
    
    # Model names
    MODELS = {
        'KNN': 'K-Nearest Neighbors',
        'SVM': 'Support Vector Machine', 
        'RANDOM_FOREST': 'Random Forest',
        'CNN': 'Convolutional Neural Network'
    }
    
    # CNN parameters (if TensorFlow is available)
    CNN_EPOCHS = 50
    CNN_BATCH_SIZE = 32
    CNN_PATIENCE = 10
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get full path for a model file"""
        return os.path.join(cls.MODELS_DIR, f"{model_name}.pkl")
    
    @classmethod
    def get_speaker_path(cls, speaker_name: str) -> str:
        """Get full path for a speaker directory"""
        return os.path.join(cls.DATA_DIR, speaker_name)
    
    @classmethod
    def is_audio_file(cls, filename: str) -> bool:
        """Check if file is a supported audio format"""
        return filename.lower().endswith(cls.AUDIO_EXTENSIONS)

# Initialize directories when config is imported
Config.ensure_directories()
