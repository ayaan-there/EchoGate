#!/usr/bin/env python3
"""
GitHub Setup Script for Speaker Recognition System
This script prepares the project for GitHub upload
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, check=True, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None, e.stderr

def check_git_installed():
    """Check if git is installed"""
    stdout, stderr = run_command("git --version", check=False)
    if stdout:
        print(f"✅ Git is installed: {stdout}")
        return True
    else:
        print("❌ Git is not installed. Please install Git first.")
        return False

def initialize_git_repo():
    """Initialize git repository"""
    print("🔄 Initializing Git repository...")
    
    # Check if .git directory exists
    if os.path.exists(".git"):
        print("✅ Git repository already initialized")
        return True
    
    # Initialize git repo
    stdout, stderr = run_command("git init")
    if stdout is not None:
        print("✅ Git repository initialized")
        return True
    else:
        print(f"❌ Failed to initialize git repository: {stderr}")
        return False

def create_gitignore():
    """Create or update .gitignore file"""
    print("🔄 Checking .gitignore file...")
    
    if os.path.exists(".gitignore"):
        print("✅ .gitignore file already exists")
        return True
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Streamlit
.streamlit/

# Audio files (large datasets)
*.wav
*.mp3
*.flac
*.m4a
*.ogg

# Model files (can be large)
*.pkl
*.h5
*.pb
*.onnx

# Temporary files
temp/
tmp/
*.tmp
*.temp

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Data directories (comment out if you want to include sample data)
# data/
# datasets/

# Log files
*.log
logs/

# Large dataset files
cv-corpus-*/
librispeech/
common_voice/
"""
    
    try:
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ .gitignore file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .gitignore: {e}")
        return False

def add_files_to_git():
    """Add files to git staging area"""
    print("🔄 Adding files to git...")
    
    # Add all files
    stdout, stderr = run_command("git add .")
    if stdout is not None:
        print("✅ Files added to git staging area")
        return True
    else:
        print(f"❌ Failed to add files: {stderr}")
        return False

def create_initial_commit():
    """Create initial commit"""
    print("🔄 Creating initial commit...")
    
    # Check if there are any commits
    stdout, stderr = run_command("git log --oneline -1", check=False)
    if "fatal: your current branch 'main' does not have any commits yet" in stderr or not stdout:
        # Create initial commit
        commit_message = "Initial commit: Production-ready Speaker Recognition System"
        stdout, stderr = run_command(f'git commit -m "{commit_message}"')
        if stdout is not None:
            print("✅ Initial commit created")
            return True
        else:
            print(f"❌ Failed to create commit: {stderr}")
            return False
    else:
        print("✅ Repository already has commits")
        return True

def configure_git_user():
    """Configure git user (if not already configured)"""
    print("🔄 Checking git configuration...")
    
    # Check if user name is configured
    stdout, stderr = run_command("git config user.name", check=False)
    if not stdout:
        print("⚠️  Git user name not configured")
        print("Please configure your git user name:")
        print("git config --global user.name 'Your Name'")
        return False
    
    # Check if user email is configured
    stdout, stderr = run_command("git config user.email", check=False)
    if not stdout:
        print("⚠️  Git user email not configured")
        print("Please configure your git user email:")
        print("git config --global user.email 'your.email@example.com'")
        return False
    
    print("✅ Git user configuration is complete")
    return True

def show_next_steps():
    """Show next steps for GitHub upload"""
    print("\n" + "="*60)
    print("🎉 PROJECT READY FOR GITHUB UPLOAD!")
    print("="*60)
    print("\nNext steps to upload to GitHub:")
    print("\n1. 🌐 Create a new repository on GitHub:")
    print("   - Go to https://github.com")
    print("   - Click '+' → 'New repository'")
    print("   - Name: 'speaker-recognition-system'")
    print("   - Description: 'Production-ready speaker recognition system'")
    print("   - Choose Public or Private")
    print("   - Do NOT initialize with README (you already have one)")
    print("\n2. 🔗 Connect your local repo to GitHub:")
    print("   git remote add origin https://github.com/USERNAME/speaker-recognition-system.git")
    print("   (Replace USERNAME with your GitHub username)")
    print("\n3. 🚀 Push your code:")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\n4. ✅ Your project is now live on GitHub!")
    print("\n" + "="*60)
    print("📁 Project includes:")
    print("   ✅ Professional README.md")
    print("   ✅ MIT License")
    print("   ✅ Contributing guidelines")
    print("   ✅ Changelog")
    print("   ✅ CI/CD pipeline")
    print("   ✅ Proper .gitignore")
    print("   ✅ All source code")
    print("="*60)

def main():
    """Main function to prepare project for GitHub"""
    print("🚀 Speaker Recognition System - GitHub Setup")
    print("="*50)
    
    # Check if we're in the right directory
    required_files = ['streamlit_app.py', 'requirements.txt', 'config.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Check git installation
    if not check_git_installed():
        sys.exit(1)
    
    # Configure git user
    if not configure_git_user():
        sys.exit(1)
    
    # Initialize git repo
    if not initialize_git_repo():
        sys.exit(1)
    
    # Create .gitignore
    if not create_gitignore():
        sys.exit(1)
    
    # Add files to git
    if not add_files_to_git():
        sys.exit(1)
    
    # Create initial commit
    if not create_initial_commit():
        sys.exit(1)
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
