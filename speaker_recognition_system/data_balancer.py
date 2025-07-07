"""
Data Balancing Utility for Speaker Recognition
Helps balance training data by cleaning duplicates and ensuring equal representation
"""

import os
import librosa
import numpy as np
from collections import defaultdict
import shutil
from typing import Dict, List, Tuple, Any
import hashlib

class DataBalancer:
    def __init__(self, data_dir: str):
        """
        Initialize the data balancer
        
        Args:
            data_dir: Path to the speaker data directory
        """
        self.data_dir = data_dir
        
    def analyze_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the current dataset for imbalances and issues
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        for speaker_dir in os.listdir(self.data_dir):
            speaker_path = os.path.join(self.data_dir, speaker_dir)
            
            if not os.path.isdir(speaker_path):
                continue
                
            # Count files
            audio_files = [f for f in os.listdir(speaker_path) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            
            # Check for duplicates
            file_hashes = []
            duplicate_groups = defaultdict(list)
            
            for audio_file in audio_files:
                file_path = os.path.join(speaker_path, audio_file)
                try:
                    # Generate hash of file content
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    file_hashes.append(file_hash)
                    duplicate_groups[file_hash].append(audio_file)
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
            
            # Find duplicates
            duplicates = {hash_val: files for hash_val, files in duplicate_groups.items() if len(files) > 1}
            
            # Calculate audio duration statistics
            durations = []
            for audio_file in audio_files[:10]:  # Sample first 10 files
                file_path = os.path.join(speaker_path, audio_file)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    duration = len(y) / sr
                    durations.append(duration)
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
            
            analysis[speaker_dir] = {
                'total_files': len(audio_files),
                'unique_files': len(set(file_hashes)),
                'duplicate_groups': len(duplicates),
                'total_duplicates': sum(len(files) - 1 for files in duplicates.values()),
                'avg_duration': np.mean(durations) if durations else 0,
                'duplicate_details': duplicates
            }
        
        return analysis
    
    def remove_duplicates(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove duplicate files from the dataset
        
        Args:
            dry_run: If True, only report what would be removed
            
        Returns:
            Dictionary with removal statistics
        """
        removal_stats = {}
        
        for speaker_dir in os.listdir(self.data_dir):
            speaker_path = os.path.join(self.data_dir, speaker_dir)
            
            if not os.path.isdir(speaker_path):
                continue
            
            audio_files = [f for f in os.listdir(speaker_path) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            
            # Group files by hash
            file_groups = defaultdict(list)
            
            for audio_file in audio_files:
                file_path = os.path.join(speaker_path, audio_file)
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    file_groups[file_hash].append(audio_file)
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
            
            # Remove duplicates (keep first file in each group)
            removed_count = 0
            for file_hash, files in file_groups.items():
                if len(files) > 1:
                    # Keep the first file, remove others
                    files_to_remove = files[1:]
                    
                    for file_to_remove in files_to_remove:
                        file_path = os.path.join(speaker_path, file_to_remove)
                        
                        if dry_run:
                            print(f"Would remove: {file_path}")
                        else:
                            try:
                                os.remove(file_path)
                                print(f"Removed: {file_path}")
                                removed_count += 1
                            except Exception as e:
                                print(f"Error removing {file_path}: {e}")
            
            removal_stats[speaker_dir] = removed_count
        
        return removal_stats
    
    def balance_dataset(self, target_files_per_speaker: int = 10, method: str = 'limit') -> Dict[str, int]:
        """
        Balance the dataset by ensuring equal representation
        
        Args:
            target_files_per_speaker: Target number of files per speaker
            method: 'limit' (remove excess) or 'augment' (add more)
            
        Returns:
            Dictionary with balancing statistics
        """
        balancing_stats = {}
        
        for speaker_dir in os.listdir(self.data_dir):
            speaker_path = os.path.join(self.data_dir, speaker_dir)
            
            if not os.path.isdir(speaker_path):
                continue
            
            audio_files = [f for f in os.listdir(speaker_path) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac'))]
            
            current_count = len(audio_files)
            
            if method == 'limit' and current_count > target_files_per_speaker:
                # Remove excess files (keep the first N files)
                files_to_remove = audio_files[target_files_per_speaker:]
                
                for file_to_remove in files_to_remove:
                    file_path = os.path.join(speaker_path, file_to_remove)
                    try:
                        # Move to backup folder instead of deleting
                        backup_dir = os.path.join(speaker_path, 'backup')
                        os.makedirs(backup_dir, exist_ok=True)
                        backup_path = os.path.join(backup_dir, file_to_remove)
                        shutil.move(file_path, backup_path)
                        print(f"Moved to backup: {file_path}")
                    except Exception as e:
                        print(f"Error moving {file_path}: {e}")
                
                final_count = min(current_count, target_files_per_speaker)
                balancing_stats[speaker_dir] = final_count
            
            elif method == 'augment' and current_count < target_files_per_speaker:
                # This would require audio augmentation techniques
                # For now, just report the current state
                balancing_stats[speaker_dir] = current_count
                print(f"Speaker {speaker_dir} has only {current_count} files, needs augmentation")
            
            else:
                balancing_stats[speaker_dir] = current_count
        
        return balancing_stats
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive dataset analysis report
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_dataset()
        
        report = "📊 Dataset Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        total_files = sum(data['total_files'] for data in analysis.values())
        total_speakers = len(analysis)
        
        report += f"📈 Summary:\n"
        report += f"  Total Speakers: {total_speakers}\n"
        report += f"  Total Files: {total_files}\n"
        report += f"  Average Files per Speaker: {total_files / total_speakers:.1f}\n\n"
        
        # Find imbalances
        file_counts = [data['total_files'] for data in analysis.values()]
        min_files = min(file_counts)
        max_files = max(file_counts)
        
        report += f"📊 Balance Analysis:\n"
        report += f"  Min Files per Speaker: {min_files}\n"
        report += f"  Max Files per Speaker: {max_files}\n"
        report += f"  Imbalance Ratio: {max_files / min_files:.2f}x\n\n"
        
        # Speaker details
        report += "👥 Speaker Details:\n"
        for speaker, data in sorted(analysis.items()):
            report += f"  {speaker}:\n"
            report += f"    Total Files: {data['total_files']}\n"
            report += f"    Unique Files: {data['unique_files']}\n"
            report += f"    Duplicates: {data['total_duplicates']}\n"
            report += f"    Avg Duration: {data['avg_duration']:.2f}s\n"
            
            if data['duplicate_groups'] > 0:
                report += f"    ⚠️ Has {data['duplicate_groups']} groups of duplicates\n"
            
            report += "\n"
        
        # Recommendations
        report += "💡 Recommendations:\n"
        if max_files / min_files > 2:
            report += "  - Dataset is imbalanced, consider balancing\n"
        
        total_duplicates = sum(data['total_duplicates'] for data in analysis.values())
        if total_duplicates > 0:
            report += f"  - Remove {total_duplicates} duplicate files\n"
        
        if min_files < 5:
            report += "  - Some speakers have very few samples (< 5)\n"
        
        if max_files > 20:
            report += "  - Some speakers have many samples (> 20), consider limiting\n"
        
        return report


def main():
    """Test the data balancer"""
    from config import Config
    
    balancer = DataBalancer(Config.DATA_DIR)
    
    # Generate and print report
    report = balancer.generate_report()
    print(report)
    
    # Remove duplicates (dry run first)
    print("\n🧹 Duplicate Analysis (Dry Run):")
    removal_stats = balancer.remove_duplicates(dry_run=True)
    for speaker, count in removal_stats.items():
        if count > 0:
            print(f"  {speaker}: {count} duplicates found")


if __name__ == "__main__":
    main()
