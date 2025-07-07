"""
Quick Dataset Fix Script
Balances the dataset to improve speaker recognition accuracy
"""

import os
import sys
from data_balancer import DataBalancer
from config import Config

def main():
    print("🔧 Quick Dataset Fix Script")
    print("=" * 40)
    
    balancer = DataBalancer(Config.DATA_DIR)
    
    # Show current analysis
    print("\n📊 Current Dataset Analysis:")
    report = balancer.generate_report()
    print(report)
    
    # Ask for confirmation
    print("\n🔧 Proposed fixes:")
    print("1. Remove duplicate files")
    print("2. Balance dataset to 10 files per speaker")
    print("3. Move excess files to backup folders")
    
    response = input("\nProceed with fixes? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Operation cancelled")
        return
    
    try:
        # Remove duplicates
        print("\n🧹 Removing duplicates...")
        removal_stats = balancer.remove_duplicates(dry_run=False)
        
        total_removed = sum(removal_stats.values())
        if total_removed > 0:
            print(f"✅ Removed {total_removed} duplicate files")
        else:
            print("ℹ️ No duplicates found")
        
        # Balance dataset
        print("\n⚖️ Balancing dataset...")
        balance_stats = balancer.balance_dataset(target_files_per_speaker=10, method='limit')
        
        print("\n📈 Final Dataset Stats:")
        for speaker, count in balance_stats.items():
            print(f"  {speaker}: {count} files")
        
        print("\n✅ Dataset fix completed successfully!")
        print("\n⚠️ IMPORTANT:")
        print("   - Excess files moved to backup folders (not deleted)")
        print("   - Please retrain your model for best results")
        print("   - Use the Streamlit app -> Model Training page")
        print("   - Check Dataset Management page for adequacy warnings")
        
    except Exception as e:
        print(f"❌ Error during fix: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
