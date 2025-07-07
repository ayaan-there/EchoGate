"""
Sample Data Creator
Copy sample audio files from Common Voice dataset for testing
"""

import os
import shutil
import random
import sys
import argparse
from typing import List
from pathlib import Path
import time
import gc

# Import config for consistent paths
try:
    from config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback if config not available
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    COMMON_VOICE_PATH = os.path.join(
        os.path.dirname(BASE_DIR), 
        "cv-corpus-22.0-delta-2025-06-20-en",
        "cv-corpus-22.0-delta-2025-06-20", 
        "en", "clips"
    )

def clear_locked_files(target_dir):
    """
    Clear any potentially locked files in the target directory
    
    Args:
        target_dir: Directory to clear
    """
    try:
        if os.path.exists(target_dir):
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Try to open the file to see if it's locked
                        with open(file_path, 'r+b') as f:
                            pass
                    except (PermissionError, IOError):
                        print(f"[WARNING] File appears locked: {file}")
                        try:
                            # Force garbage collection
                            gc.collect()
                            time.sleep(0.1)
                            
                            # Try to delete the locked file
                            os.remove(file_path)
                            print(f"[INFO] Removed locked file: {file}")
                        except:
                            print(f"[WARNING] Could not remove locked file: {file}")
    except Exception as e:
        print(f"[WARNING] Error clearing locked files: {e}")

def copy_sample_files_from_dataset(num_speakers=None, files_per_speaker=None, auto_mode=False, use_librispeech=False):
    """
    Copy sample audio files from the dataset to create test speakers
    
    Args:
        num_speakers: Number of speakers to create (defaults to interactive prompt)
        files_per_speaker: Number of files per speaker (defaults to interactive prompt)
        auto_mode: If True, use default values without prompting
        use_librispeech: If True, use LibriSpeech dataset structure
    """
    # Use config paths if available, otherwise fallback
    if CONFIG_AVAILABLE:
        source_dir = Config.COMMON_VOICE_PATH  # This now points to LibriSpeech
        target_base_dir = Config.DATA_DIR
        
        # Auto-detect if it's LibriSpeech based on directory structure
        if not use_librispeech and os.path.exists(source_dir):
            # Check if it looks like LibriSpeech (numeric speaker folders)
            try:
                contents = os.listdir(source_dir)
                numeric_folders = [f for f in contents if f.isdigit() and os.path.isdir(os.path.join(source_dir, f))]
                if len(numeric_folders) > 0:
                    print("[AUTO-DETECT] LibriSpeech structure detected!")
                    use_librispeech = True
            except:
                pass
    else:
        source_dir = COMMON_VOICE_PATH
        target_base_dir = DATA_DIR
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        if use_librispeech:
            print("Please check the path to your LibriSpeech dataset.")
        else:
            print("Please check the path to your Common Voice dataset.")
        return False
    
    # Get list of audio files based on dataset type
    audio_files = []
    available_speakers = []
    
    if use_librispeech:
        print("[INFO] Using LibriSpeech dataset structure...")
        # LibriSpeech structure: speaker_id/chapter_id/*.flac
        try:
            speaker_folders = [f for f in os.listdir(source_dir) 
                             if os.path.isdir(os.path.join(source_dir, f)) and f.isdigit()]
            
            for speaker_id in speaker_folders:
                speaker_path = os.path.join(source_dir, speaker_id)
                speaker_files = []
                
                # Look in chapter subdirectories
                for chapter_folder in os.listdir(speaker_path):
                    chapter_path = os.path.join(speaker_path, chapter_folder)
                    if os.path.isdir(chapter_path):
                        chapter_files = [f for f in os.listdir(chapter_path) 
                                       if f.lower().endswith(('.flac', '.wav', '.mp3'))]
                        for file in chapter_files:
                            speaker_files.append(os.path.join(chapter_path, file))
                
                if speaker_files:  # Only include speakers with audio files
                    available_speakers.append(speaker_id)
                    audio_files.extend([(speaker_id, file_path) for file_path in speaker_files])
                    
        except Exception as e:
            print(f"[ERROR] Error reading LibriSpeech directory: {e}")
            return False
    else:
        print("[INFO] Using Common Voice dataset structure...")
        # Common Voice structure: flat directory with .mp3 files
        try:
            all_files = os.listdir(source_dir)
            
            for file in all_files:
                try:
                    if file.lower().endswith('.mp3'):
                        file_path = os.path.join(source_dir, file)
                        if os.path.isfile(file_path):
                            # For Common Voice, we'll create artificial speaker IDs
                            audio_files.append((None, file_path))
                except (UnicodeDecodeError, OSError) as e:
                    print(f"[WARNING] Skipping file due to encoding issue: {e}")
                    continue
                    
        except (UnicodeDecodeError, OSError) as e:
            print(f"[ERROR] Error reading directory {source_dir}: {e}")
            print("This might be due to file encoding issues on Windows.")
            return False
    
    if use_librispeech:
        print(f"[SUCCESS] Found {len(available_speakers)} speakers with {len(audio_files)} total audio files")
        if len(available_speakers) == 0:
            print("[ERROR] No speakers found in LibriSpeech dataset")
            return False
    else:
        print(f"[SUCCESS] Found {len(audio_files)} audio files in Common Voice dataset")
    
    # Ask for configuration if not in auto mode and values not provided
    if not auto_mode:
        if num_speakers is None:
            max_speakers = len(available_speakers) if use_librispeech else 20
            try:
                num_speakers = int(input(f"[INPUT] Number of speakers to create (2-{max_speakers}): ").strip())
                if num_speakers < 2:
                    print("[WARNING] Need at least 2 speakers. Using 2 as minimum.")
                    num_speakers = 2
                elif use_librispeech and num_speakers > len(available_speakers):
                    print(f"[WARNING] Only {len(available_speakers)} speakers available. Using {len(available_speakers)}.")
                    num_speakers = len(available_speakers)
                elif not use_librispeech and num_speakers > 20:
                    print("[WARNING] Too many speakers for Common Voice. Using 20 as maximum.")
                    num_speakers = 20
            except ValueError:
                print("[WARNING] Invalid input. Using 5 speakers as default.")
                num_speakers = min(5, len(available_speakers) if use_librispeech else 5)
        
        if files_per_speaker is None:
            try:
                files_per_speaker = int(input("[INPUT] Number of files per speaker (5-100): ").strip())
                if files_per_speaker < 5:
                    print("[WARNING] Too few files per speaker. Using 5 as minimum.")
                    files_per_speaker = 5
                elif files_per_speaker > 100:
                    print("[WARNING] Too many files might be slow to process. Using 100 as maximum.")
                    files_per_speaker = 100
            except ValueError:
                print("[WARNING] Invalid input. Using 20 files per speaker as default.")
                files_per_speaker = 20
    else:
        # Default values for auto mode
        if num_speakers is None:
            num_speakers = min(5, len(available_speakers) if use_librispeech else 5)
        if files_per_speaker is None:
            files_per_speaker = 20
    
    # Clear any potentially locked files first
    clear_locked_files(target_base_dir)
    
    # Create speaker dataset
    if use_librispeech:
        # Use actual LibriSpeech speakers
        selected_speakers = available_speakers[:num_speakers]
        
        print(f"[INFO] Creating dataset with {num_speakers} LibriSpeech speakers...")
        
        for i, speaker_id in enumerate(selected_speakers):
            speaker_dir = os.path.join(target_base_dir, f"speaker_{speaker_id}")
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Get files for this speaker
            speaker_files = [file_path for spk_id, file_path in audio_files if spk_id == speaker_id]
            
            # Randomly select files for this speaker
            selected_files = random.sample(speaker_files, min(files_per_speaker, len(speaker_files)))
            
            for j, file_path in enumerate(selected_files):
                source_filename = os.path.basename(file_path)
                target_filename = f"speaker_{speaker_id}_sample_{j+1:02d}.flac"
                target_path = os.path.join(speaker_dir, target_filename)
                
                try:
                    # Check if target file already exists and remove it first
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                        except PermissionError:
                            # If can't remove, try with a different name
                            import time
                            timestamp = int(time.time())
                            target_filename = f"speaker_{speaker_id}_sample_{j+1:02d}_{timestamp}.flac"
                            target_path = os.path.join(speaker_dir, target_filename)
                    
                    # Ensure target directory has proper permissions
                    os.chmod(speaker_dir, 0o755)
                    
                    # Copy the file
                    shutil.copy2(file_path, target_path)
                    
                    # Set permissions on the copied file
                    os.chmod(target_path, 0o644)
                    
                    print(f"[SUCCESS] Copied {target_filename}")
                    
                except PermissionError as e:
                    print(f"[WARNING] Permission denied for {source_filename}: {str(e)}")
                    print(f"[INFO] Trying alternative copy method...")
                    
                    # Try alternative copy method
                    try:
                        import time
                        timestamp = int(time.time())
                        alt_filename = f"speaker_{speaker_id}_sample_{j+1:02d}_{timestamp}.flac"
                        alt_path = os.path.join(speaker_dir, alt_filename)
                        
                        # Use basic file read/write instead of shutil.copy2
                        with open(file_path, 'rb') as src, open(alt_path, 'wb') as dst:
                            dst.write(src.read())
                        
                        print(f"[SUCCESS] Copied {alt_filename} (alternative method)")
                        
                    except Exception as e2:
                        print(f"[ERROR] Failed to copy {source_filename} with alternative method: {str(e2)}")
                        print("[INFO] Skipping this file and continuing...")
                        continue
                        
                except Exception as e:
                    print(f"[ERROR] Failed to copy {source_filename}: {str(e)}")
                    print("[INFO] Skipping this file and continuing...")
                    continue
    else:
        # Create artificial speakers from Common Voice files
        total_files_needed = num_speakers * files_per_speaker
        if len(audio_files) < total_files_needed:
            print(f"[ERROR] Not enough audio files. Found: {len(audio_files)}, need: {total_files_needed}")
            files_per_speaker = len(audio_files) // num_speakers
            if files_per_speaker < 5:
                print(f"[ERROR] Cannot create dataset with requested parameters.")
                return False
        
        # Generate speaker names
        speakers = [f'user_{i+1}' for i in range(num_speakers)]
        
        # Randomly select files
        selected_files = random.sample([file_path for _, file_path in audio_files], 
                                     num_speakers * files_per_speaker)
        
        print(f"[INFO] Creating dataset with {num_speakers} speakers and {files_per_speaker} files per speaker...")
        
        for i, speaker in enumerate(speakers):
            speaker_dir = os.path.join(target_base_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Copy files for this speaker
            start_idx = i * files_per_speaker
            end_idx = start_idx + files_per_speaker
            
            for j, file_path in enumerate(selected_files[start_idx:end_idx]):
                filename = os.path.basename(file_path)
                target_filename = f"{speaker}_sample_{j+1:02d}.mp3"
                target_path = os.path.join(speaker_dir, target_filename)
                
                try:
                    # Check if target file already exists and remove it first
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                        except PermissionError:
                            # If can't remove, try with a different name
                            import time
                            timestamp = int(time.time())
                            target_filename = f"{speaker}_sample_{j+1:02d}_{timestamp}.mp3"
                            target_path = os.path.join(speaker_dir, target_filename)
                    
                    # Ensure target directory has proper permissions
                    os.chmod(speaker_dir, 0o755)
                    
                    # Copy the file
                    shutil.copy2(file_path, target_path)
                    
                    # Set permissions on the copied file
                    os.chmod(target_path, 0o644)
                    
                    print(f"[SUCCESS] Copied {target_filename} for {speaker}")
                    
                except PermissionError as e:
                    print(f"[WARNING] Permission denied for {filename}: {str(e)}")
                    print(f"[INFO] Trying alternative copy method...")
                    
                    # Try alternative copy method
                    try:
                        import time
                        timestamp = int(time.time())
                        alt_filename = f"{speaker}_sample_{j+1:02d}_{timestamp}.mp3"
                        alt_path = os.path.join(speaker_dir, alt_filename)
                        
                        # Use basic file read/write instead of shutil.copy2
                        with open(file_path, 'rb') as src, open(alt_path, 'wb') as dst:
                            dst.write(src.read())
                        
                        print(f"[SUCCESS] Copied {alt_filename} for {speaker} (alternative method)")
                        
                    except Exception as e2:
                        print(f"[ERROR] Failed to copy {filename} with alternative method: {str(e2)}")
                        print("[INFO] Skipping this file and continuing...")
                        continue
                        
                except Exception as e:
                    print(f"[ERROR] Failed to copy {filename}: {str(e)}")
                    print("[INFO] Skipping this file and continuing...")
                    continue
    
    print(f"\n[SUCCESS] Sample dataset created successfully!")
    print(f"[INFO] Location: {target_base_dir}")
    if use_librispeech:
        print(f"[INFO] LibriSpeech speakers: {selected_speakers}")
    else:
        print(f"[INFO] Speakers: {speakers}")
    print(f"[INFO] Files per speaker: {files_per_speaker}")
    
    print(f"\n[NEXT STEPS]:")
    print(f"1. Run: python model_training.py")
    print(f"2. Web UI: streamlit run streamlit_app.py")
    
    return True

def remove_sample_data():
    """Remove all sample data"""
    if CONFIG_AVAILABLE:
        data_dir = Config.DATA_DIR
    else:
        data_dir = DATA_DIR
    
    # Get all speaker directories
    try:
        speakers = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    except (UnicodeDecodeError, OSError) as e:
        print(f"[ERROR] Error reading data directory: {e}")
        return
    
    if not speakers:
        print("[INFO] No speaker directories found.")
        return
    
    print(f"[WARNING] This will remove all audio files from {len(speakers)} speakers:")
    print(", ".join(speakers))
    confirmation = input("Continue? (y/n): ").strip().lower()
    
    if confirmation != 'y':
        print("[INFO] Operation canceled.")
        return
    
    files_removed = 0
    for speaker in speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        if os.path.exists(speaker_dir):
            try:
                # Remove all audio files but keep README
                for file in os.listdir(speaker_dir):
                    if file.lower().endswith(('.mp3', '.wav', '.flac')):
                        file_path = os.path.join(speaker_dir, file)
                        os.remove(file_path)
                        files_removed += 1
                
                # Remove the speaker directory if it's empty
                remaining_files = os.listdir(speaker_dir)
                if not remaining_files or all(f.lower() == 'readme.txt' for f in remaining_files):
                    try:
                        if remaining_files:  # Remove README if it exists
                            for f in remaining_files:
                                os.remove(os.path.join(speaker_dir, f))
                        os.rmdir(speaker_dir)
                        print(f"[INFO] Removed empty speaker directory: {speaker}")
                    except OSError:
                        pass  # Directory not empty, leave it
                        
            except (UnicodeDecodeError, OSError) as e:
                print(f"[WARNING] Error removing files from {speaker}: {e}")
    
    print(f"[SUCCESS] Removed {files_removed} audio files from {len(speakers)} speakers")

def main():
    """
    Main function - kept for compatibility
    Use the Streamlit app for data management: streamlit run streamlit_app.py
    """
    print("Speaker Recognition - Sample Data Creator")
    print("=" * 50)
    print("For data management, please use the Streamlit web interface:")
    print("streamlit run streamlit_app.py")
    print()
    print("The web interface provides:")
    print("- Sample data creation from LibriSpeech/Common Voice")
    print("- Speaker management")
    print("- Dataset overview and statistics")
    print("- File management tools")

if __name__ == "__main__":
    main()
