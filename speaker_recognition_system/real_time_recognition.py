"""
Real-Time Voice Recognition Module
Records voice input and predicts speaker identity
"""

import numpy as np
import sounddevice as sd
import librosa
import joblib
import os
import time
from typing import Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

class RealTimeRecognizer:
    def __init__(self, model_path: Optional[str] = None, duration: float = 3.0, sr: int = 22050):
        """
        Initialize the real-time recognizer
        
        Args:
            model_path: Path to saved model directory
            duration: Recording duration in seconds
            sr: Sample rate
        """
        self.duration = duration
        self.sr = sr
        self.model = None
        self.scaler = None
        self.pca = None  # PCA object for dimensionality reduction
        self.label_names = []
        self.feature_extractor = None
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def setup_feature_extractor(self):
        """Setup feature extractor with same parameters as training"""
        from feature_extraction import FeatureExtractor
        self.feature_extractor = FeatureExtractor(sr=self.sr)
    
    def record_audio_sounddevice(self, duration: Optional[float] = None, countdown: bool = True) -> np.ndarray:
        """
        Record audio using sounddevice
        
        Args:
            duration: Recording duration (uses self.duration if None)
            countdown: Whether to show a countdown before recording starts
            
        Returns:
            Audio data as numpy array
        """
        if duration is None:
            duration = self.duration
        
        if countdown:
            print("Preparing to record...")
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(0.7)
            
        print(f"Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sr), 
            samplerate=self.sr, 
            channels=1, 
            dtype='float64'
        )
        sd.wait()  # Wait until recording is finished
        
        print("Recording finished!")
        return audio_data.flatten()
    
    def record_audio_pyaudio(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record audio using pyaudio (alternative method)
        
        Args:
            duration: Recording duration (uses self.duration if None)
            
        Returns:
            Audio data as numpy array
        """
        if not PYAUDIO_AVAILABLE:
            raise ImportError("PyAudio not available. Use sounddevice instead.")
            
        if duration is None:
            duration = self.duration
        
        chunk = 1024
        format = pyaudio.paFloat32
        channels = 1
        
        p = pyaudio.PyAudio()
        
        print(f"Recording for {duration} seconds... Speak now!")
        
        stream = p.open(
            format=format,
            channels=channels,
            rate=self.sr,
            input=True,
            frames_per_buffer=chunk
        )
        
        frames = []
        for _ in range(0, int(self.sr / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("Recording finished!")
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        return audio_data
    
    def save_audio_temp(self, audio_data: np.ndarray, filename: str = "temp_recording.wav") -> str:
        """
        Save audio data to temporary file
        
        Args:
            audio_data: Audio data
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        from config import Config
        temp_dir = Config.TEMP_DIR
        os.makedirs(temp_dir, exist_ok=True)
        
        filepath = os.path.join(temp_dir, filename)
        
        # Save using soundfile (alternative to librosa.output.write_wav)
        import soundfile as sf
        sf.write(filepath, audio_data, self.sr)
        
        return filepath
    
    def predict_speaker(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """
        Predict speaker from audio data
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Tuple of (predicted_speaker, confidence)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
            
        if self.feature_extractor is None:
            self.setup_feature_extractor()
        
        if self.feature_extractor is None:
            raise ValueError("Feature extractor is not available")
        
        # Extract features
        features = self.feature_extractor.extract_features_from_audio_data(audio_data, self.sr)
        
        if features is None:
            raise ValueError("Failed to extract features from audio")
        
        # Apply PCA if available
        if self.pca is not None:
            features = self.pca.transform(features.reshape(1, -1))
            features = features.flatten()
        
        if self.scaler is None:
            raise ValueError("Scaler is not loaded")
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        else:
            predicted_class = self.model.predict(features_scaled)[0]
            confidence = 1.0  # Default confidence for models without probability
        
        predicted_speaker = self.label_names[predicted_class]
        
        return predicted_speaker, confidence
    
    def predict_speaker_with_confidence_analysis(self, audio_data: np.ndarray) -> Tuple[str, float, dict]:
        """
        Enhanced speaker prediction with detailed confidence analysis
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Tuple of (predicted_speaker, confidence, analysis_details)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
            
        if self.feature_extractor is None:
            self.setup_feature_extractor()
        
        if self.feature_extractor is None:
            raise ValueError("Feature extractor is not available")
        
        # Extract features
        features = self.feature_extractor.extract_features_from_audio_data(audio_data, self.sr)
        
        if features is None:
            raise ValueError("Failed to extract features from audio")
        
        # Apply PCA if available
        if self.pca is not None:
            features = self.pca.transform(features.reshape(1, -1))
            features = features.flatten()
        
        if self.scaler is None:
            raise ValueError("Scaler is not loaded")
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Calculate confidence metrics
            sorted_probs = np.sort(probabilities)[::-1]  # Sort descending
            top_prob = sorted_probs[0]
            second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
            confidence_margin = top_prob - second_prob
            
            # Enhanced decision logic
            min_confidence = 0.4  # Minimum confidence threshold
            min_margin = 0.1     # Minimum margin between top 2 predictions
            
            # Determine if prediction is reliable
            is_reliable = (confidence >= min_confidence and confidence_margin >= min_margin)
            
            analysis = {
                'all_probabilities': dict(zip(self.label_names, probabilities)),
                'top_confidence': top_prob,
                'second_confidence': second_prob,
                'confidence_margin': confidence_margin,
                'is_reliable': is_reliable,
                'min_confidence_threshold': min_confidence,
                'min_margin_threshold': min_margin
            }
            
        else:
            # For models without probability support
            predicted_class = self.model.predict(features_scaled)[0]
            confidence = 0.5  # Default confidence
            analysis = {
                'all_probabilities': {self.label_names[predicted_class]: confidence},
                'is_reliable': False,
                'note': 'Model does not support probability predictions'
            }
        
        predicted_speaker = self.label_names[predicted_class]
        
        return predicted_speaker, confidence, analysis

    def load_model(self, model_path: str) -> None:
        """
        Load saved model and preprocessing components
        
        Args:
            model_path: Path to model directory
        """
        try:
            # Check if best_model subdirectory exists (newer format)
            best_model_dir = os.path.join(model_path, 'best_model')
            if os.path.exists(best_model_dir):
                print("Loading from best_model directory (recommended format)")
                model_dir = best_model_dir
            else:
                print("Loading from main models directory")
                model_dir = model_path
            
            # Load model
            model_file = os.path.join(model_dir, 'model.pkl' if model_dir == best_model_dir else 'best_model.pkl')
            self.model = joblib.load(model_file)
            
            # Load scaler
            scaler_file = os.path.join(model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_file)
            
            # Load PCA if available
            pca_file = os.path.join(model_dir, 'pca.pkl')
            if os.path.exists(pca_file):
                self.pca = joblib.load(pca_file)
                print(f"PCA loaded with {self.pca.n_components_} components")
            else:
                self.pca = None
                print("No PCA file found - using raw features")
            
            # Load label names
            labels_file = os.path.join(model_dir, 'label_names.pkl')
            self.label_names = joblib.load(labels_file)
            
            # Load metadata if available
            metadata_file = os.path.join(model_dir, 'metadata.pkl')
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                print(f"Model type: {metadata.get('model_type', 'Unknown')}")
                print(f"Accuracy: {metadata.get('accuracy', 'Unknown'):.3f}")
                if metadata.get('has_pca', False):
                    print(f"PCA components: {metadata.get('pca_components', 'Unknown')}")
            
            print(f"Model loaded successfully from: {model_dir}")
            print(f"Available speakers: {self.label_names}")
            
            # Validate component compatibility
            if self.pca is not None and hasattr(self.scaler, 'n_features_in_'):
                expected_features = self.pca.n_components_
                scaler_features = self.scaler.n_features_in_
                if expected_features != scaler_features:
                    # This should not happen with properly saved models
                    print(f"⚠️ Warning: PCA outputs {expected_features} features but scaler expects {scaler_features}")
                    print("🔧 Tip: Retrain the model to ensure component compatibility")
                else:
                    print(f"✅ Model components are compatible: {expected_features} features")
            elif self.pca is not None:
                print(f"✅ Using PCA with {self.pca.n_components_} components")
            else:
                print("✅ Using raw features (no PCA)")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def continuous_recognition(self, threshold: float = 0.7) -> None:
        """
        Continuous speaker recognition loop
        
        Args:
            threshold: Confidence threshold for successful recognition
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        print("Starting continuous recognition. Press Ctrl+C to stop.")
        print(f"Confidence threshold: {threshold}")
        print("-" * 50)
        
        try:
            while True:
                # Get user input
                input("Press Enter to record, or Ctrl+C to quit...")
                
                # Record audio
                audio_data = self.record_audio_sounddevice()
                
                # Predict speaker
                try:
                    speaker, confidence = self.predict_speaker(audio_data)
                    
                    print(f"Predicted Speaker: {speaker}")
                    print(f"Confidence: {confidence:.3f}")
                    
                    if confidence >= threshold:
                        print("✅ LOGIN SUCCESSFUL!")
                    else:
                        print("❌ LOGIN FAILED - Low confidence")
                        
                except Exception as e:
                    print(f"Error in prediction: {str(e)}")
                
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nStopping recognition...")
    
    def authenticate_user(self, target_speaker: str, threshold: float = 0.7, 
                         max_attempts: int = 3) -> bool:
        """
        Authenticate a specific user
        
        Args:
            target_speaker: Expected speaker name
            threshold: Confidence threshold
            max_attempts: Maximum authentication attempts
            
        Returns:
            True if authentication successful, False otherwise
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        print(f"Authenticating user: {target_speaker}")
        print(f"You have {max_attempts} attempts.")
        
        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")
            
            # Record audio
            audio_data = self.record_audio_sounddevice()
            
            try:
                # Predict speaker
                predicted_speaker, confidence = self.predict_speaker(audio_data)
                
                print(f"Predicted: {predicted_speaker} (confidence: {confidence:.3f})")
                
                # Check if prediction matches target and meets threshold
                if predicted_speaker == target_speaker and confidence >= threshold:
                    print("✅ AUTHENTICATION SUCCESSFUL!")
                    return True
                else:
                    print("❌ Authentication failed")
                    if predicted_speaker != target_speaker:
                        print(f"Expected: {target_speaker}, Got: {predicted_speaker}")
                    if confidence < threshold:
                        print(f"Confidence too low: {confidence:.3f} < {threshold}")
                        
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
        
        print("❌ AUTHENTICATION FAILED - Maximum attempts exceeded")
        return False

def main():
    """
    Real-time recognition system - Use the Streamlit app instead
    """
    print("Real-Time Speaker Recognition System")
    print("=" * 40)
    print("For real-time recognition, please use the Streamlit web interface:")
    print("streamlit run streamlit_app.py")
    print()
    print("The web interface provides:")
    print("- Real-time voice recognition")
    print("- Speaker authentication")
    print("- Confidence analysis")
    print("- Interactive voice testing")

if __name__ == "__main__":
    main()