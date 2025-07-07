"""
Speaker Enrollment Module
Allows new users to enroll in the speaker recognition system
"""

import os
import numpy as np
import joblib
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

from real_time_recognition import RealTimeRecognizer
from feature_extraction import FeatureExtractor
from model_training import SpeakerRecognitionTrainer

class SpeakerEnrollment:
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize speaker enrollment
        
        Args:
            data_dir: Directory to store audio data
            models_dir: Directory to store trained models
        """
        self.data_dir = os.path.abspath(data_dir)
        self.models_dir = os.path.abspath(models_dir)
        self.recorder = RealTimeRecognizer()
        self.feature_extractor = FeatureExtractor()
        self.trainer = SpeakerRecognitionTrainer()
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def get_existing_speakers(self) -> List[str]:
        """
        Get list of existing speakers
        
        Returns:
            List of speaker names
        """
        speakers = []
        if os.path.exists(self.data_dir):
            speakers = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        return speakers
    
    def create_speaker_directory(self, speaker_name: str) -> str:
        """
        Create directory for new speaker
        
        Args:
            speaker_name: Name of the speaker
            
        Returns:
            Path to speaker directory
        """
        # Clean speaker name (remove special characters)
        clean_name = "".join(c for c in speaker_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_')
        
        speaker_dir = os.path.join(self.data_dir, clean_name)
        os.makedirs(speaker_dir, exist_ok=True)
        
        return speaker_dir
    
    def record_enrollment_samples(self, speaker_name: str, num_samples: int = 5, 
                                duration: float = 3.0) -> List[str]:
        """
        Record multiple audio samples for speaker enrollment
        
        Args:
            speaker_name: Name of the speaker
            num_samples: Number of audio samples to record
            duration: Duration of each recording in seconds
            
        Returns:
            List of paths to recorded files
        """
        speaker_dir = self.create_speaker_directory(speaker_name)
        recorded_files = []
        
        print(f"Enrolling speaker: {speaker_name}")
        print(f"Please record {num_samples} samples of {duration} seconds each.")
        print("Speak clearly and try to vary your tone/content slightly.")
        print("-" * 50)
        
        print("You will now record multiple samples in a continuous session.")
        print("After each recording, you'll be prompted to continue or finish.")
        input("Press Enter when ready to begin the recording session...")
        
        i = 0
        keep_recording = True
        while i < num_samples and keep_recording:
            print(f"\nRecording sample {i+1}/{num_samples}")
            print("Tips: Say your name, count numbers, or speak a short phrase")
            
            input("Press Enter to record this sample...")
            
            # Record audio
            audio_data = self.recorder.record_audio_sounddevice(duration=duration)
            
            # Save audio file
            filename = f"{speaker_name}_sample_{i+1:02d}.wav"
            filepath = os.path.join(speaker_dir, filename)
            
            # Save the recording
            try:
                import soundfile as sf
                sf.write(filepath, audio_data, self.recorder.sr)
                recorded_files.append(filepath)
                print(f"✅ Sample {i+1} saved: {filename}")
            except Exception as e:
                print(f"❌ Error saving sample {i+1}: {str(e)}")
            
            i += 1
            
            # Check if we should continue recording more samples
            if i < num_samples:
                user_choice = input(f"\nRecorded {i}/{num_samples} samples. Press Enter to continue recording or 'q' to finish: ").strip().lower()
                if user_choice == 'q':
                    keep_recording = False
                    print("Recording session finished early at user request.")
            else:
                print("\nAll required samples have been recorded!")
        
        print(f"\n✅ Enrollment complete! Recorded {len(recorded_files)} samples.")
        return recorded_files
    
    def validate_enrollment_quality(self, speaker_dir: str) -> bool:
        """
        Validate the quality of enrollment recordings
        
        Args:
            speaker_dir: Path to speaker directory
            
        Returns:
            True if enrollment quality is acceptable
        """
        audio_files = [f for f in os.listdir(speaker_dir) 
                      if f.lower().endswith(('.wav', '.mp3'))]
        
        if len(audio_files) < 3:
            print("❌ Warning: Less than 3 audio samples. Consider recording more.")
            return False
        
        # Extract features from all files to check consistency
        features_list = []
        for audio_file in audio_files:
            file_path = os.path.join(speaker_dir, audio_file)
            features = self.feature_extractor.extract_mfcc_features(file_path)
            
            if features is not None:
                features_list.append(features)
        
        if len(features_list) < len(audio_files) * 0.8:  # At least 80% should be processable
            print("❌ Warning: Many audio files could not be processed.")
            return False
        
        # Check feature consistency (low variance indicates good quality)
        if len(features_list) >= 2:
            features_array = np.array(features_list)
            feature_std = np.std(features_array, axis=0)
            avg_std = np.mean(feature_std)
            
            # This is a simple heuristic - you might want to tune this threshold
            if avg_std > 2.0:
                print("❌ Warning: High variability in voice features. Consider re-recording.")
                return False
        
        print("✅ Enrollment quality check passed!")
        return True
    
    def retrain_model(self) -> bool:
        """
        Retrain the model with all available data including new speaker
        
        Returns:
            True if retraining successful
        """
        try:
            print("Loading all speaker data...")
            
            # Load dataset with all speakers
            X, y, label_names, pca = self.feature_extractor.load_dataset_from_folders(self.data_dir)
            
            if len(X) == 0:
                print("❌ No data found for training.")
                return False
            
            if len(label_names) < 2:
                print("❌ Need at least 2 speakers for training.")
                return False
            
            print(f"Training model with {len(label_names)} speakers...")
            
            # Train all models
            results = self.trainer.train_all_models(X, y, label_names)
            
            # Get best model
            best_model_name, best_model = self.trainer.get_best_model(results)
            
            # Check if we got a valid model
            if best_model_name == "unknown" or best_model is None:
                print("⚠️ Warning: No valid model could be trained")
                return False
            
            # Save the best model and preprocessing components
            self.save_model_components(best_model, best_model_name, results, pca)
            
            print(f"✅ Model retrained successfully!")
            print(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during retraining: {str(e)}")
            return False
    
    def save_model_components(self, best_model, model_name: str, results: dict, pca=None) -> None:
        """
        Save model and preprocessing components
        
        Args:
            best_model: The best performing model
            model_name: Name of the best model
            results: Training results
            pca: PCA object for dimensionality reduction (if any)
        """
        # Save individual components for backward compatibility
        # Save best model
        model_path = os.path.join(self.models_dir, 'best_model.pkl')
        joblib.dump(best_model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        joblib.dump(self.trainer.scaler, scaler_path)
        
        # Save PCA if available
        if pca is not None:
            pca_path = os.path.join(self.models_dir, 'pca.pkl')
            joblib.dump(pca, pca_path)
        
        # Save label names
        labels_path = os.path.join(self.models_dir, 'label_names.pkl')
        joblib.dump(self.trainer.label_names, labels_path)
        
        # Save model metadata
        metadata = {
            'model_type': model_name,
            'accuracy': results[model_name]['accuracy'],
            'f1_score': results[model_name]['f1_score'],
            'speakers': self.trainer.label_names,
            'num_features': len(self.trainer.scaler.scale_) if (hasattr(self.trainer.scaler, 'scale_') and self.trainer.scaler.scale_ is not None) else 'unknown',
            'num_speakers': len(self.trainer.label_names),
            'has_pca': pca is not None,
            'pca_components': pca.n_components_ if pca is not None else None
        }
        
        metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        # Also save as bundle format for the model info page
        bundle_dir = os.path.join(self.models_dir, 'best_model')
        os.makedirs(bundle_dir, exist_ok=True)
        
        # Save bundle components
        joblib.dump(best_model, os.path.join(bundle_dir, 'model.pkl'))
        joblib.dump(self.trainer.scaler, os.path.join(bundle_dir, 'scaler.pkl'))
        joblib.dump(self.trainer.label_names, os.path.join(bundle_dir, 'label_names.pkl'))
        joblib.dump(metadata, os.path.join(bundle_dir, 'metadata.pkl'))
        
        # Save PCA to bundle if available
        if pca is not None:
            joblib.dump(pca, os.path.join(bundle_dir, 'pca.pkl'))
        
        print(f"Model components saved to: {self.models_dir}")
        if pca is not None:
            print(f"PCA components: {pca.n_components_}")
    
    def enroll_new_speaker(self, speaker_name: str, num_samples: int = 5, 
                          duration: float = 3.0, auto_retrain: bool = True) -> bool:
        """
        Complete enrollment process for a new speaker
        
        Args:
            speaker_name: Name of the new speaker
            num_samples: Number of audio samples to record
            duration: Duration of each recording
            auto_retrain: Whether to automatically retrain the model
            
        Returns:
            True if enrollment successful
        """
        # Check if speaker already exists
        existing_speakers = self.get_existing_speakers()
        if speaker_name in existing_speakers:
            print(f"Speaker '{speaker_name}' already exists!")
            overwrite = input("Do you want to overwrite existing data? (y/n): ").lower().strip()
            if overwrite != 'y':
                return False
        
        print(f"Starting enrollment for: {speaker_name}")
        
        # Record samples
        recorded_files = self.record_enrollment_samples(speaker_name, num_samples, duration)
        
        if len(recorded_files) < 3:
            print("❌ Enrollment failed: Not enough samples recorded.")
            return False
        
        # Validate quality
        speaker_dir = self.create_speaker_directory(speaker_name)
        quality_ok = self.validate_enrollment_quality(speaker_dir)
        
        if not quality_ok:
            print("⚠️ Quality check failed, but continuing with enrollment.")
            retry = input("Do you want to re-record samples? (y/n): ").lower().strip()
            if retry == 'y':
                return self.enroll_new_speaker(speaker_name, num_samples, duration, auto_retrain)
        
        # Retrain model if requested
        if auto_retrain:
            success = self.retrain_model()
            if not success:
                print("❌ Model retraining failed, but enrollment data saved.")
                return False
        
        print(f"✅ Speaker '{speaker_name}' enrolled successfully!")
        
        return True
    
    def list_enrolled_speakers(self) -> None:
        """
        List all enrolled speakers and their sample counts
        """
        speakers = self.get_existing_speakers()
        
        if not speakers:
            print("No speakers enrolled yet.")
            return
        
        print("Enrolled Speakers:")
        print("-" * 30)
        
        for speaker in speakers:
            speaker_dir = os.path.join(self.data_dir, speaker)
            audio_files = [f for f in os.listdir(speaker_dir) 
                          if f.lower().endswith(('.wav', '.mp3'))]
            print(f"{speaker}: {len(audio_files)} samples")
    
    def remove_speaker(self, speaker_name: str) -> bool:
        """
        Remove a speaker from the system
        
        Args:
            speaker_name: Name of speaker to remove
            
        Returns:
            True if removal successful
        """
        speaker_dir = os.path.join(self.data_dir, speaker_name)
        
        if not os.path.exists(speaker_dir):
            print(f"Speaker '{speaker_name}' not found.")
            return False
        
        confirm = input(f"Are you sure you want to remove '{speaker_name}'? (y/n): ").lower().strip()
        if confirm != 'y':
            return False
        
        # Remove speaker directory
        import shutil
        shutil.rmtree(speaker_dir)
        
        print(f"Speaker '{speaker_name}' removed.")
        
        # Retrain model
        remaining_speakers = self.get_existing_speakers()
        if len(remaining_speakers) >= 2:
            print("Retraining model...")
            self.retrain_model()
        else:
            print("Less than 2 speakers remaining. Model training skipped.")
        
        return True

def main():
    """
    Speaker enrollment system - Use the Streamlit app instead
    """
    print("Speaker Enrollment System")
    print("=" * 40)
    print("For speaker enrollment, please use the Streamlit web interface:")
    print("streamlit run streamlit_app.py")
    print()
    print("The web interface provides:")
    print("- Interactive speaker enrollment")
    print("- Dataset management")
    print("- Real-time voice testing")
    print("- Model training and evaluation")

if __name__ == "__main__":
    main()
