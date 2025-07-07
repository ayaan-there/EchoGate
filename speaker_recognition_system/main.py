"""
Main Script for Speaker Recognition System
Combines feature extraction, model training, and evaluation
"""

import os
import sys
import argparse
from config import Config

from feature_extraction import FeatureExtractor
from model_training import SpeakerRecognitionTrainer
from model_utils import ModelManager
from real_time_recognition import RealTimeRecognizer
from speaker_enrollment import SpeakerEnrollment

def setup_directories():
    """Create necessary directories"""
    Config.ensure_directories()
    
    directories = {
        'data': Config.DATA_DIR,
        'models': Config.MODELS_DIR,
        'temp': Config.TEMP_DIR
    }
    
    for name, path in directories.items():
        print(f"✅ {name.title()} directory: {path}")
    
    return directories

def create_sample_data_structure(data_dir):
    """Create sample data structure with example folders"""
    print("\n🗂️  Creating sample data structure...")
    
    # Create sample speaker folders
    sample_speakers = ['user_1', 'user_2', 'user_3']
    
    for speaker in sample_speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Create a README in each folder
        readme_path = os.path.join(speaker_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(f"Speaker: {speaker}\n")
            f.write("Add MP3 or WAV audio files here for training.\n")
            f.write("Recommended: 3-10 audio files per speaker.\n")
            f.write("Each file should be 2-5 seconds of clear speech.\n")
    
    print(f"Created sample folders: {sample_speakers}")
    print("Add audio files to these folders before training.")

def train_model_pipeline(data_dir, models_dir):
    """Complete model training pipeline"""
    print("\n🎯 Starting Model Training Pipeline")
    print("=" * 50)
    
    # Initialize components
    extractor = FeatureExtractor()
    trainer = SpeakerRecognitionTrainer()
    model_manager = ModelManager(models_dir)
    
    try:
        # Extract features
        print("1. Extracting features from audio files...")
        X, y, label_names, pca = extractor.load_dataset_from_folders(data_dir)
        
        if len(X) == 0:
            print("❌ No audio files found!")
            print("Please add MP3/WAV files to speaker folders in the data directory.")
            return False
        
        if len(label_names) < 2:
            print("❌ Need at least 2 speakers for training!")
            print(f"Found only: {label_names}")
            return False
        
        print(f"✅ Loaded {X.shape[0]} samples from {len(label_names)} speakers")
        
        # Train models
        print("\n2. Training multiple models...")
        results = trainer.train_all_models(X, y, label_names)
        
        # Save the best model
        print("\n3. Saving best model...")
        trainer.save_best_model(results, X, label_names, pca)
        
        # Print results summary
        print("\n📊 Training Results Summary:")
        print("-" * 30)
        for model_name, result in results.items():
            print(f"{model_name.upper()}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
        
        # Get best model
        best_model_name, _ = trainer.get_best_model(results)
        print(f"\n🏆 Best Model: {best_model_name.upper()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def test_recognition(models_dir):
    """Test the trained model with voice recognition"""
    print("\n🎤 Testing Voice Recognition")
    print("=" * 30)
    
    try:
        recognizer = RealTimeRecognizer(model_path=models_dir)
        
        if recognizer.model is None:
            print("❌ No trained model found!")
            return False
        
        print(f"✅ Model loaded successfully!")
        print(f"Available speakers: {recognizer.label_names}")
        
        # Test recording (without actual microphone for automated testing)
        print("\n⚠️  Voice recognition test requires a microphone.")
        print("Run 'python real_time_recognition.py' for interactive testing.")
        
        return True
        
    except Exception as e:
        print(f"❌ Recognition test failed: {str(e)}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("\n🔍 Checking Dependencies")
    print("=" * 25)
    
    dependencies = {
        'librosa': 'Audio processing',
        'sklearn': 'Machine learning',
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation',
        'matplotlib': 'Plotting',
        'joblib': 'Model persistence',
        'sounddevice': 'Audio recording (optional)',
        'streamlit': 'Web interface (optional)',
    }
    
    missing = []
    
    for package, description in dependencies.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✅ All dependencies available!")
    return True

def copy_sample_audio_files(data_dir):
    """Copy sample audio files from Common Voice dataset if available"""
    print("\n📁 Looking for sample audio files...")
    
    if os.path.exists(Config.COMMON_VOICE_PATH):
        import shutil
        import random
        
        # Get list of MP3 files
        mp3_files = [f for f in os.listdir(Config.COMMON_VOICE_PATH) if f.endswith('.mp3')]
        
        if len(mp3_files) >= 15:  # Need at least 5 files per speaker for 3 speakers
            print(f"Found {len(mp3_files)} audio files in Common Voice dataset")
            
            # Create 3 sample speakers with 5 files each for better training
            speakers = ['user_1', 'user_2', 'user_3']
            files_per_speaker = 5
            
            # Randomly select files for each speaker
            selected_files = random.sample(mp3_files, len(speakers) * files_per_speaker)
            
            for i, speaker in enumerate(speakers):
                speaker_dir = os.path.join(data_dir, speaker)
                os.makedirs(speaker_dir, exist_ok=True)
                
                # Copy files for this speaker
                start_idx = i * files_per_speaker
                end_idx = start_idx + files_per_speaker
                
                for j, file in enumerate(selected_files[start_idx:end_idx]):
                    src_path = os.path.join(Config.COMMON_VOICE_PATH, file)
                    dst_filename = f"{speaker}_sample_{j+1:02d}.mp3"
                    dst_path = os.path.join(speaker_dir, dst_filename)
                    
                    shutil.copy2(src_path, dst_path)
                    print(f"✅ Copied {dst_filename} for {speaker}")
            
            print(f"✅ Created sample dataset with {len(speakers)} speakers")
            return True
        else:
            print("❌ Not enough audio files in Common Voice dataset")
    else:
        print("❌ Common Voice dataset not found")
    
    return False

def main():
    """
    Main function - kept for compatibility
    Use the Streamlit app for the full system: streamlit run streamlit_app.py
    """
    print("🎤 Speaker Recognition System")
    print("=" * 40)
    print("For the full speaker recognition system, please use the Streamlit web interface:")
    print("streamlit run streamlit_app.py")
    print()
    print("The web interface provides:")
    print("- Voice authentication and login")
    print("- Speaker enrollment")
    print("- Model training and management")
    print("- Dataset creation and management")
    print("- Real-time voice recognition")
    print("- System configuration")

if __name__ == "__main__":
    main()
