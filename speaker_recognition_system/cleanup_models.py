"""
Model Cleanup Script
Removes old inconsistent model files and ensures proper model loading
"""

import os
import shutil
from config import Config

def cleanup_old_model_files():
    """Remove old model files from the main models directory to prevent conflicts"""
    models_dir = Config.MODELS_DIR
    
    # Files that should be removed from main directory (they exist in best_model/)
    old_files = [
        'best_model.pkl',
        'scaler.pkl', 
        'pca.pkl',
        'label_names.pkl',
        'model_metadata.pkl'
    ]
    
    print("🧹 Cleaning up old model files...")
    
    removed_count = 0
    for filename in old_files:
        file_path = os.path.join(models_dir, filename)
        if os.path.exists(file_path):
            try:
                # Create backup first
                backup_dir = os.path.join(models_dir, 'backup_old_files')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, filename)
                
                shutil.move(file_path, backup_path)
                print(f"  ✅ Moved {filename} to backup")
                removed_count += 1
                
            except Exception as e:
                print(f"  ❌ Error moving {filename}: {e}")
    
    print(f"\n📊 Cleanup Summary:")
    print(f"  Files moved to backup: {removed_count}")
    print(f"  Backup location: {os.path.join(models_dir, 'backup_old_files')}")
    
    # Verify best_model directory is intact
    best_model_dir = os.path.join(models_dir, 'best_model')
    if os.path.exists(best_model_dir):
        files = os.listdir(best_model_dir)
        print(f"  ✅ best_model directory contains: {files}")
        
        required_files = ['model.pkl', 'scaler.pkl', 'label_names.pkl', 'metadata.pkl']
        missing_files = [f for f in required_files if f not in files]
        
        if missing_files:
            print(f"  ⚠️ Missing files in best_model: {missing_files}")
        else:
            print(f"  ✅ All required files present in best_model directory")
    else:
        print(f"  ❌ best_model directory not found!")
    
    return removed_count > 0

def main():
    """Main cleanup function"""
    print("🔧 Model File Cleanup Tool")
    print("=" * 40)
    
    try:
        success = cleanup_old_model_files()
        
        if success:
            print("\n✅ Cleanup completed successfully!")
            print("\n💡 Benefits:")
            print("  - No more version conflicts")
            print("  - Consistent model loading")
            print("  - No more PCA/scaler mismatch warnings")
            print("\n⚠️ Note: Please restart the Streamlit app to see changes")
        else:
            print("\n✅ No cleanup needed - files already organized correctly")
            
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
