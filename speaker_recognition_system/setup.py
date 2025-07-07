"""
Quick Setup Script for Speaker Recognition System
Automated setup and first-time configuration
"""

import os
import sys
import subprocess
import platform
import importlib.util
from pathlib import Path

def print_header():
    """Print welcome header"""
    print("🎤 Speaker Recognition System - Quick Setup")
    print("=" * 50)
    print("This script will help you set up and test the system.")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_basic_requirements():
    """Install basic requirements"""
    print("\n📦 Installing basic requirements...")
    
    basic_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "librosa>=0.10.0",
        "joblib>=1.1.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0"
    ]
    
    for package in basic_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ {package} - FAILED")
            return False
    
    return True

def install_audio_packages():
    """Install audio-related packages"""
    print("\n🎵 Installing audio packages...")
    
    audio_packages = [
        "sounddevice>=0.4.0",
        "soundfile>=0.12.0"
    ]
    
    for package in audio_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ {package} - Optional, skipped")

def install_optional_packages():
    """Install optional packages"""
    print("\n🌟 Installing optional packages...")
    
    optional_packages = [
        "streamlit>=1.28.0",
        "seaborn>=0.11.0",
        "tensorflow>=2.10.0"
    ]
    
    for package in optional_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ {package} - Optional, skipped")

def check_ffmpeg():
    """Check and provide instructions for FFmpeg with better detection"""
    print("\n🎬 Checking FFmpeg...")
    
    # Multiple ways to check for FFmpeg
    ffmpeg_commands = ['ffmpeg', 'ffmpeg.exe']
    
    for cmd in ffmpeg_commands:
        try:
            result = subprocess.run([cmd, '-version'], 
                                  capture_output=True, 
                                  check=True,
                                  text=True,
                                  timeout=10)
            print("✅ FFmpeg is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # Check if ffmpeg is in common locations
    common_paths = []
    if platform.system() == "Windows":
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe", 
            os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
            # Conda environments
            os.path.join(sys.prefix, "Scripts", "ffmpeg.exe"),
            os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe")
        ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"✅ FFmpeg found at: {path}")
            return True
    
    print("❌ FFmpeg not found in PATH or common locations")
    print("\n📥 FFmpeg Installation Instructions:")
    
    if platform.system() == "Windows":
        print("Windows (choose one):")
        print("1. conda install -c conda-forge ffmpeg  (recommended if using conda)")
        print("2. Download from https://ffmpeg.org/download.html")
        print("3. Install via chocolatey: choco install ffmpeg")
        print("4. Install via winget: winget install ffmpeg")
    elif platform.system() == "Darwin":
        print("macOS:")
        print("brew install ffmpeg")
    else:
        print("Linux:")
        print("sudo apt update && sudo apt install ffmpeg  (Ubuntu/Debian)")
        print("sudo yum install ffmpeg  (CentOS/RHEL)")
    
    print("\nNote: FFmpeg is needed for audio file processing.")
    print("The system may work with some audio formats without it.")
    
    return False

def setup_project_structure():
    """Set up project directory structure"""
    print("\n📁 Setting up project structure...")
    
    # Import config to get proper paths
    try:
        from config import Config
        directories = [
            Config.DATA_DIR,
            os.path.join(Config.DATA_DIR, "user_1"),
            os.path.join(Config.DATA_DIR, "user_2"), 
            os.path.join(Config.DATA_DIR, "user_3"),
            Config.MODELS_DIR,
            Config.TEMP_DIR
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            rel_path = os.path.relpath(dir_path, Config.BASE_DIR)
            print(f"✅ {rel_path}/")
            
    except ImportError:
        # Fallback to basic setup
        base_dir = os.path.dirname(os.path.abspath(__file__))
        directories = [
            "data", "data/user_1", "data/user_2", "data/user_3",
            "models", "temp"
        ]
        
        for directory in directories:
            dir_path = os.path.join(base_dir, directory)
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {directory}/")
    
    return True

def create_sample_data_option():
    """Offer to create sample data with better error handling"""
    print("\n🎵 Sample Data Setup")
    print("-" * 20)
    
    try:
        # Import config for proper paths
        from config import Config
        cv_path = Config.COMMON_VOICE_PATH
        
        if os.path.exists(cv_path):
            try:
                # Check files with proper encoding handling
                mp3_files = []
                for item in os.listdir(cv_path):
                    if item.lower().endswith('.mp3'):
                        mp3_files.append(item)
                
                if len(mp3_files) >= 9:
                    print(f"✅ Found Common Voice dataset with {len(mp3_files)} files")
                    
                    choice = input("Create sample data from Common Voice? (y/n): ").lower().strip()
                    if choice == 'y':
                        return create_sample_data_safe()
                else:
                    print(f"⚠️ Found only {len(mp3_files)} MP3 files, need at least 9")
                    
            except (UnicodeDecodeError, OSError) as e:
                print(f"⚠️ Error accessing Common Voice files: {e}")
                print("This might be due to file encoding issues")
        else:
            print(f"ℹ️ Common Voice dataset not found at: {cv_path}")
            
    except ImportError:
        print("⚠️ Could not import config, using fallback path detection")
    
    print("ℹ️ No sample data created")
    print("Add your own MP3/WAV files to data/user_1/, data/user_2/, etc.")
    return False


def create_sample_data_safe():
    """Safely create sample data with proper error handling"""
    try:
        import subprocess
        import sys
        
        # Run create_sample_data.py as a subprocess to handle encoding issues
        result = subprocess.run([
            sys.executable, "create_sample_data.py", "--auto"
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            print("✅ Sample data created successfully")
            return True
        else:
            print(f"❌ Failed to create sample data: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        # Fallback: try to import and run directly
        try:
            from create_sample_data import copy_sample_files_from_common_voice
            return copy_sample_files_from_common_voice()
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False

def run_basic_test():
    """Run a basic system test"""
    print("\n🧪 Running basic system test...")
    
    try:
        # Test imports
        import numpy
        import pandas  
        import sklearn
        import librosa
        print("✅ Core modules imported successfully")
        
        # Test feature extraction
        from feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        print("✅ Feature extractor initialized")
        
        # Test model training components
        from model_training import SpeakerRecognitionTrainer
        trainer = SpeakerRecognitionTrainer()
        print("✅ Model trainer initialized")
        
        print("✅ Basic system test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {str(e)}")
        return False

def show_next_steps(has_data=False):
    """Show next steps to the user"""
    print("\n🎯 Next Steps")
    print("=" * 12)
    
    if has_data:
        print("✅ You have sample data ready!")
        print("\n1. Train the model:")
        print("   python main.py --mode train")
        print("\n2. Test recognition:")
        print("   python main.py --mode test")
        print("\n3. Launch web interface:")
        print("   streamlit run streamlit_app.py")
        print("\n4. Enroll new speakers:")
        print("   python speaker_enrollment.py")
    else:
        print("📝 Manual setup required:")
        print("\n1. Add audio files:")
        print("   - Copy MP3/WAV files to data/user_1/, data/user_2/, etc.")
        print("   - Need at least 2 speakers with 3+ files each")
        print("\n2. Or create sample data:")
        print("   python create_sample_data.py")
        print("\n3. Then train the model:")
        print("   python main.py --mode train")
    
    print("\n📚 Documentation:")
    print("   - README.md: General information")
    print("   - python test_system.py: Comprehensive system test")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return
    
    # Install packages
    print("\n🚀 Installing packages...")
    
    if not install_basic_requirements():
        print("\n❌ Setup failed: Could not install basic requirements")
        return
    
    install_audio_packages()
    install_optional_packages()
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    
    # Setup project
    setup_project_structure()
    
    # Test system
    test_ok = run_basic_test()
    
    # Sample data
    has_data = create_sample_data_option()
    
    # Summary
    print("\n📊 Setup Summary")
    print("=" * 15)
    print(f"Python: ✅")
    print(f"Packages: ✅")
    print(f"FFmpeg: {'✅' if ffmpeg_ok else '⚠️'}")
    print(f"System Test: {'✅' if test_ok else '❌'}")
    print(f"Sample Data: {'✅' if has_data else '⚠️'}")
    
    if test_ok:
        print("\n🎉 Setup completed successfully!")
        show_next_steps(has_data)
    else:
        print("\n⚠️ Setup completed with issues")
        print("Check the error messages above and install missing dependencies")

if __name__ == "__main__":
    main()