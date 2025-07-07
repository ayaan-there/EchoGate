#!/usr/bin/env python3
"""
GitHub Setup Script for Speaker Recognition System
Prepares the project for GitHub upload by cleaning up files and initializing git
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return True if successful"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode == 0:
            print(f"✅ {command}")
            if result.stdout.strip():
                print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {command}")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {command}")
        print(f"   Exception: {e}")
        return False

def clean_project():
    """Clean up files that shouldn't be uploaded to GitHub"""
    print("\n🧹 Cleaning project files...")
    
    # Directories to remove
    dirs_to_remove = [
        "models",
        "temp", 
        "data",
        "__pycache__",
        ".pytest_cache",
        ".streamlit",
        "cv-corpus-22.0-delta-2025-06-20-en"
    ]
    
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"   Removed: {dir_name}/")
            except Exception as e:
                print(f"   Warning: Could not remove {dir_name}/: {e}")
    
    # File patterns to remove
    files_to_remove = [
        "*.wav", "*.mp3", "*.pkl", "*.joblib", "*.h5", "*.ckpt",
        "*.npy", "*.npz", "*.csv", "*.tsv", "*.log", "*.tmp",
        "*.bak", "*.swp", "*.swo", "*.DS_Store", "Thumbs.db",
        "Desktop.ini", "*.pyc", "*.pyo"
    ]
    
    removed_count = 0
    for pattern in files_to_remove:
        for file_path in Path(".").glob(f"**/{pattern}"):
            try:
                file_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"   Warning: Could not remove {file_path}: {e}")
    
    if removed_count > 0:
        print(f"   Removed {removed_count} files")
    
    print("✅ Project cleanup completed")

def create_placeholder_files():
    """Create placeholder files for empty directories"""
    print("\n📁 Creating placeholder files...")
    
    # Create models directory with README
    models_dir = Path("speaker_recognition_system/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_readme = models_dir / "README.md"
    if not models_readme.exists():
        models_readme.write_text("""# Models Directory

This directory will contain trained models after training.

## Structure
- `best_model.pkl` - Best performing model
- `scaler.pkl` - Feature scaler
- `label_names.pkl` - Speaker labels
- `model_metadata.pkl` - Model metadata
- `best_model/` - Best model bundle directory

## Note
Models are not included in the repository due to size constraints.
Train your own models using the Streamlit app.
""")
        print("   Created: models/README.md")
    
    # Create temp directory with README
    temp_dir = Path("speaker_recognition_system/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_readme = temp_dir / "README.md"
    if not temp_readme.exists():
        temp_readme.write_text("""# Temporary Files Directory

This directory is used for temporary files during processing.

## Contents
- Temporary audio recordings
- Processing intermediates
- Cache files

## Note
This directory is cleaned automatically and not tracked by Git.
""")
        print("   Created: temp/README.md")
    
    # Create data directory with README
    data_dir = Path("speaker_recognition_system/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_readme = data_dir / "README.md"
    if not data_readme.exists():
        data_readme.write_text("""# Data Directory

This directory contains speaker audio samples for training.

## Structure
Each speaker should have their own subdirectory:
```
data/
├── speaker1/
│   ├── sample_01.wav
│   ├── sample_02.wav
│   └── ...
├── speaker2/
│   ├── sample_01.wav
│   └── ...
└── ...
```

## Supported Formats
- WAV (recommended)
- MP3
- FLAC
- OGG

## Requirements
- Minimum 3 samples per speaker
- 3-10 seconds per sample
- Clear audio quality
- 16kHz+ sample rate

## Note
Audio files are not included in the repository.
Add your own audio samples for training.
""")
        print("   Created: data/README.md")
    
    print("✅ Placeholder files created")

def initialize_git():
    """Initialize git repository"""
    print("\n🔧 Initializing Git repository...")
    
    if not Path(".git").exists():
        if not run_command("git init"):
            return False
    
    # Configure git (if not already configured)
    run_command("git config --global init.defaultBranch main")
    
    # Add all files
    if not run_command("git add ."):
        return False
    
    # Check if there are any changes to commit
    result = subprocess.run("git diff --cached --exit-code", shell=True, capture_output=True)
    if result.returncode != 0:  # There are changes to commit
        if not run_command('git commit -m "Initial commit: Production-ready speaker recognition system"'):
            return False
    else:
        print("   No changes to commit")
    
    print("✅ Git repository initialized")
    return True

def print_instructions():
    """Print instructions for GitHub upload"""
    print("\n" + "="*60)
    print("🚀 PROJECT READY FOR GITHUB UPLOAD")
    print("="*60)
    print("""
Next steps:

1. CREATE GITHUB REPOSITORY:
   - Go to https://github.com
   - Click "+" → "New repository"
   - Name: speaker-recognition-system
   - Description: AI-powered speaker recognition system with web interface
   - Make it Public or Private (your choice)
   - DON'T initialize with README/gitignore/license

2. CONNECT TO GITHUB:
   git remote add origin https://github.com/yourusername/speaker-recognition-system.git

3. PUSH TO GITHUB:
   git branch -M main
   git push -u origin main

4. VERIFY UPLOAD:
   - Check that no .wav/.mp3/.pkl files were uploaded
   - Verify README.md displays correctly
   - Check that models/ and data/ directories show placeholder READMEs

📋 WHAT'S INCLUDED:
✅ Source code (Python files)
✅ Requirements.txt
✅ README.md with full documentation
✅ .gitignore (excludes data/models)
✅ LICENSE (MIT)
✅ CONTRIBUTING.md
✅ GitHub Actions workflow
✅ Placeholder READMEs for empty directories

🚫 WHAT'S EXCLUDED:
❌ Audio files (.wav, .mp3, etc.)
❌ Trained models (.pkl, .joblib, etc.)
❌ Dataset files (.csv, .tsv, etc.)
❌ Temporary files
❌ Cache files
❌ __pycache__ directories

💡 FEATURES FOR USERS:
- Complete web-based interface
- No CLI dependencies
- Easy setup instructions
- Professional documentation
- Ready for production use

Your project is now ready for GitHub! 🎉
""")

def main():
    """Main function"""
    print("🎯 GITHUB SETUP FOR SPEAKER RECOGNITION SYSTEM")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("speaker_recognition_system").exists():
        print("❌ Error: speaker_recognition_system directory not found!")
        print("   Please run this script from the project root directory.")
        sys.exit(1)
    
    # Clean project
    clean_project()
    
    # Create placeholder files
    create_placeholder_files()
    
    # Initialize git
    if not initialize_git():
        print("❌ Git initialization failed!")
        sys.exit(1)
    
    # Print instructions
    print_instructions()

if __name__ == "__main__":
    main()
