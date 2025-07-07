"""
Streamlit Web Application for Speaker Recognition System
Provides a user-friendly interface for login and enrollment
"""

import streamlit as st
import numpy as np
import os
import tempfile
import time
from typing import Optional, Tuple
from config import Config

# Configure Streamlit page
st.set_page_config(
    page_title="Speaker Recognition System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from real_time_recognition import RealTimeRecognizer
    from speaker_enrollment import SpeakerEnrollment
    from model_utils import ModelManager
    from feature_extraction import FeatureExtractor
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Error importing modules: {str(e)}")

# Audio recording functionality
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.error("Audio packages not available. Install with: pip install sounddevice soundfile")

class StreamlitSpeakerApp:
    def __init__(self):
        """Initialize the Streamlit app"""
        # Set up paths using config
        self.base_dir = Config.BASE_DIR
        self.data_dir = Config.DATA_DIR
        self.models_dir = Config.MODELS_DIR
        self.temp_dir = Config.TEMP_DIR
        
        # Ensure directories exist
        Config.ensure_directories()
        
        # Initialize components
        if MODULES_AVAILABLE:
            self.recognizer = None
            self.enrollment = SpeakerEnrollment(self.data_dir, self.models_dir)
            self.model_manager = ModelManager(self.models_dir)
            self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.recognizer = RealTimeRecognizer(model_path=self.models_dir)
            if self.recognizer.model is not None:
                st.session_state['model_loaded'] = True
                st.session_state['speakers'] = self.recognizer.label_names
                
                # Also update model info in session state for the model info page
                try:
                    model_info = self.model_manager.get_model_info("best_model")
                    st.session_state['model_info'] = model_info
                except Exception as e:
                    st.session_state['model_info'] = None
            else:
                st.session_state['model_loaded'] = False
                st.session_state['speakers'] = []
                st.session_state['model_info'] = None
        except Exception as e:
            st.session_state['model_loaded'] = False
            st.session_state['speakers'] = []
            st.session_state['model_info'] = None
    
    def record_audio_streamlit(self, duration: float = Config.DEFAULT_DURATION, sr: int = Config.SAMPLE_RATE) -> Optional[np.ndarray]:
        """
        Record audio for Streamlit app
        
        Args:
            duration: Recording duration in seconds
            sr: Sample rate
            
        Returns:
            Audio data or None if recording failed
        """
        if not AUDIO_AVAILABLE:
            st.error("Audio recording not available. Please install sounddevice and soundfile.")
            return None
        
        try:
            # Show countdown before recording
            countdown_placeholder = st.empty()
            for i in range(3, 0, -1):
                countdown_placeholder.warning(f"Recording starts in {i} seconds...")
                time.sleep(1)
            
            countdown_placeholder.info(f"🔴 RECORDING NOW - Speak for {duration} seconds!")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sr), 
                samplerate=sr, 
                channels=1, 
                dtype='float64'
            )
            sd.wait()
            
            countdown_placeholder.success("✅ Recording completed!")
            
            # Check if audio was recorded
            if audio_data is None or len(audio_data) == 0:
                st.error("No audio data recorded")
                return None
            
            return audio_data.flatten()
            
        except Exception as e:
            st.error(f"Recording failed: {str(e)}")
            st.error("Make sure your microphone is connected and permissions are granted.")
            return None
    
    def save_temp_audio(self, audio_data: np.ndarray, filename: str = "temp_audio.wav") -> str:
        """Save audio to temporary file"""
        try:
            filepath = os.path.join(self.temp_dir, filename)
            sf.write(filepath, audio_data, Config.SAMPLE_RATE)
            return filepath
        except Exception as e:
            st.error(f"Failed to save audio file: {str(e)}")
            return ""
    
    def login_page(self):
        """Login page interface"""
        st.title("🎤 Voice Authentication Login")
        st.markdown("---")
        
        # Check if model is loaded
        if not st.session_state.get('model_loaded', False):
            st.error("❌ No trained model found!")
            st.info("Please enroll speakers first using the Enrollment page.")
            return
        
        # Show available speakers
        speakers = st.session_state.get('speakers', [])
        st.info(f"**Available Speakers:** {', '.join(speakers)}")
        
        # Authentication settings
        col1, col2 = st.columns(2)
        
        with col1:
            target_speaker = st.selectbox(
                "Select Speaker to Authenticate:",
                options=speakers,
                help="Choose the speaker you want to authenticate as"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.3,
                max_value=0.95,
                value=0.8,  # Increased default threshold
                step=0.05,
                help="Higher threshold = more strict authentication. Recommended: 0.7-0.9"
            )
        
        st.markdown("---")
        
        # Recording controls
        st.subheader("🎙️ Voice Recording")
        
        duration = st.slider(
            "Recording Duration (seconds):",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        
        # Record button
        if st.button("🔴 Start Recording", type="primary", use_container_width=True):
            if not AUDIO_AVAILABLE:
                st.error("Audio recording not available!")
                return
            
            # Record audio
            audio_data = self.record_audio_streamlit(duration)
            
            if audio_data is not None:
                # Save audio for playback
                audio_file = self.save_temp_audio(audio_data, "login_recording.wav")
                
                # Show audio player
                st.audio(audio_file, format='audio/wav')
                
                # Predict speaker
                try:
                    if self.recognizer is None:
                        st.error("Recognizer not initialized. Please load a model first.")
                        return
                    
                    with st.spinner("Analyzing voice..."):
                        # Use enhanced prediction method
                        if hasattr(self.recognizer, 'predict_speaker_with_confidence_analysis'):
                            predicted_speaker, confidence, analysis = self.recognizer.predict_speaker_with_confidence_analysis(audio_data)
                            is_reliable = analysis.get('is_reliable', False)
                            confidence_margin = analysis.get('confidence_margin', 0)
                            all_probs = analysis.get('all_probabilities', {})
                        else:
                            # Fallback to original method
                            predicted_speaker, confidence = self.recognizer.predict_speaker(audio_data)
                            is_reliable = confidence >= 0.7
                            confidence_margin = 0
                            all_probs = {predicted_speaker: confidence}
                    
                    # Display results
                    st.markdown("### 📊 Authentication Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Predicted Speaker", predicted_speaker)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    
                    with col3:
                        st.metric("Margin", f"{confidence_margin:.3f}")
                    
                    with col4:
                        reliability_status = "✅ Reliable" if is_reliable else "⚠️ Uncertain"
                        st.metric("Reliability", reliability_status)
                    
                    # Show all speaker probabilities
                    st.markdown("#### 🎯 Speaker Probabilities")
                    for speaker, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                        is_target = speaker == predicted_speaker
                        emoji = "🎯" if is_target else "📊"
                        st.write(f"{emoji} **{speaker}**: {prob:.3f}")
                    
                    # Authentication decision
                    final_success = (
                        predicted_speaker == target_speaker and 
                        confidence >= confidence_threshold and
                        is_reliable
                    )
                    
                    if final_success:
                        st.success(f"🎉 Welcome back, {predicted_speaker}!")
                        st.balloons()
                    else:
                        st.error("❌ Authentication Failed")
                        if predicted_speaker != target_speaker:
                            st.warning(f"Expected: {target_speaker}, Got: {predicted_speaker}")
                        if confidence < confidence_threshold:
                            st.warning(f"Confidence too low: {confidence:.3f} < {confidence_threshold}")
                        if not is_reliable:
                            st.warning("⚠️ Prediction is not reliable - try recording again")
                
                except Exception as e:
                    st.error(f"Authentication error: {str(e)}")
    
    def enrollment_page(self):
        """Speaker enrollment page interface"""
        st.title("👤 Speaker Enrollment")
        st.markdown("---")
        
        # Show current speakers
        existing_speakers = self.enrollment.get_existing_speakers()
        
        if existing_speakers:
            st.subheader("📋 Enrolled Speakers")
            for speaker in existing_speakers:
                speaker_dir = os.path.join(self.data_dir, speaker)
                audio_files = [f for f in os.listdir(speaker_dir) 
                              if f.lower().endswith(('.wav', '.mp3'))]
                st.write(f"• **{speaker}**: {len(audio_files)} samples")
        else:
            st.info("No speakers enrolled yet.")
        
        st.markdown("---")
        
        # New speaker enrollment
        st.subheader("🆕 Enroll New Speaker")
        
        # Speaker details
        speaker_name = st.text_input(
            "Speaker Name:",
            placeholder="Enter your name (e.g., John_Doe)",
            help="Use alphanumeric characters, spaces, hyphens, or underscores"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_samples = st.number_input(
                "Number of Samples:",
                min_value=3,
                max_value=10,
                value=5,
                help="More samples = better accuracy"
            )
        
        with col2:
            duration = st.number_input(
                "Recording Duration (seconds):",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5
            )
        
        # Enrollment process - Step by step approach
        if speaker_name:
            if speaker_name in existing_speakers:
                overwrite = st.checkbox("Speaker exists. Overwrite existing data?")
                if not overwrite:
                    st.warning("Please check the overwrite option or choose a different name.")
                    return
            
            # Initialize session state for enrollment
            if 'enrollment_step' not in st.session_state:
                st.session_state.enrollment_step = 0
            if 'enrollment_speaker' not in st.session_state:
                st.session_state.enrollment_speaker = ""
            if 'recorded_samples' not in st.session_state:
                st.session_state.recorded_samples = []
            
            # Start enrollment
            if st.session_state.enrollment_speaker != speaker_name:
                st.session_state.enrollment_step = 0
                st.session_state.enrollment_speaker = speaker_name
                st.session_state.recorded_samples = []
            
            # Show progress
            progress = st.session_state.enrollment_step / num_samples
            st.progress(progress)
            st.write(f"Progress: {st.session_state.enrollment_step}/{num_samples} samples recorded")
            
            if st.session_state.enrollment_step < num_samples:
                # Current recording step
                current_step = st.session_state.enrollment_step + 1
                st.subheader(f"🎙️ Record Sample {current_step}/{num_samples}")
                st.write("**Tips:** Say your name, count numbers, or speak a short phrase clearly")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(f"🔴 Record Sample {current_step}", type="primary", use_container_width=True, key=f"record_btn_{current_step}"):
                        try:
                            with st.spinner(f"Recording sample {current_step} for {duration} seconds..."):
                                audio_data = self.record_audio_streamlit(duration)
                            
                            if audio_data is not None:
                                # Create speaker directory
                                speaker_dir = self.enrollment.create_speaker_directory(speaker_name)
                                
                                # Save the recording
                                filename = f"{speaker_name}_sample_{current_step:02d}.wav"
                                filepath = os.path.join(speaker_dir, filename)
                                
                                sf.write(filepath, audio_data, 22050)
                                st.session_state.recorded_samples.append(filepath)
                                st.session_state.enrollment_step += 1
                                
                                st.success(f"✅ Sample {current_step} recorded successfully!")
                                st.audio(filepath, format='audio/wav')
                                
                                # Auto-refresh to next step
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to record sample {current_step}")
                        
                        except Exception as e:
                            st.error(f"❌ Recording error: {str(e)}")
                
                with col2:
                    if st.button("🔄 Reset", type="secondary", use_container_width=True):
                        st.session_state.enrollment_step = 0
                        st.session_state.recorded_samples = []
                        st.rerun()
            
            else:
                # All samples recorded - finalize enrollment
                st.success(f"🎉 All {num_samples} samples recorded!")
                
                if len(st.session_state.recorded_samples) >= 3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Complete Enrollment", type="primary", use_container_width=True):
                            with st.spinner("Training model with new speaker..."):
                                try:
                                    success = self.enrollment.retrain_model()
                                    if success:
                                        st.success(f"🎉 {speaker_name} enrolled successfully!")
                                        # Reset enrollment state
                                        st.session_state.enrollment_step = 0
                                        st.session_state.enrollment_speaker = ""
                                        st.session_state.recorded_samples = []
                                        # Clear model info cache to force refresh
                                        st.session_state['model_info'] = None
                                        self.load_model()  # Reload model
                                        st.rerun()
                                    else:
                                        st.error("❌ Model training failed")
                                except Exception as e:
                                    st.error(f"❌ Training error: {str(e)}")
                    
                    with col2:
                        if st.button("🔄 Start Over", type="secondary", use_container_width=True):
                            st.session_state.enrollment_step = 0
                            st.session_state.recorded_samples = []
                            st.rerun()
                
                else:
                    st.error("❌ Not enough samples recorded. Need at least 3 samples.")
                    if st.button("🔄 Try Again", type="secondary"):
                        st.session_state.enrollment_step = 0
                        st.session_state.recorded_samples = []
                        st.rerun()
        
        # Management options
        st.markdown("---")
        st.subheader("🔧 Management Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retrain Model", use_container_width=True):
                with st.spinner("Retraining model..."):
                    success = self.enrollment.retrain_model()
                    if success:
                        st.success("✅ Model retrained successfully!")
                        self.load_model()  # This will update model_info in session state
                        # Clear cache to ensure fresh model info is loaded
                        st.session_state['model_info'] = None
                        self.load_model()  # Load again to update the model_info
                    else:
                        st.error("❌ Model training failed")
        
        with col2:
            if existing_speakers and st.button("🗑️ Remove Speaker", use_container_width=True):
                st.session_state['show_remove'] = True
        
        # Speaker removal interface
        if st.session_state.get('show_remove', False):
            st.subheader("Remove Speaker")
            speaker_to_remove = st.selectbox("Select speaker to remove:", existing_speakers)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("❌ Confirm Removal", type="secondary"):
                    success = self.enrollment.remove_speaker(speaker_to_remove)
                    if success:
                        st.success(f"Speaker {speaker_to_remove} removed!")
                        self.load_model()
                        st.session_state['show_remove'] = False
                        st.rerun()
            
            with col2:
                if st.button("↩️ Cancel"):
                    st.session_state['show_remove'] = False
                    st.rerun()
    
    def model_info_page(self):
        """Model information and management page"""
        st.title("📊 Model Information & Management")
        st.markdown("---")
        
        # Force refresh model info when this page is loaded
        if 'current_page' not in st.session_state or st.session_state.get('current_page') != 'model_info':
            st.session_state['current_page'] = 'model_info'
            st.session_state['model_info'] = None  # Clear cached info
            try:
                # Try to get fresh model info
                info = self.model_manager.get_model_info("best_model")
                st.session_state['model_info'] = info
            except Exception as e:
                st.session_state['model_info'] = None
        
        # Refresh button
        col_refresh, col_status = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Refresh Model Info", key="refresh_model_info"):
                st.session_state['model_info'] = None  # Clear cached info
                try:
                    info = self.model_manager.get_model_info("best_model")
                    st.session_state['model_info'] = info
                except Exception:
                    pass
                st.rerun()
        
        # Model status
        if st.session_state.get('model_loaded', False):
            with col_status:
                st.success("✅ Model is loaded and ready")
            
            # Get model info from session state or fetch it fresh
            info = st.session_state.get('model_info')
            
            # If no info in session state, try to fetch it now
            if info is None:
                try:
                    info = self.model_manager.get_model_info("best_model")
                    st.session_state['model_info'] = info
                except Exception as e:
                    st.warning(f"Could not load model details: {str(e)}")
            
            if info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Type", info.get('model_type', 'Unknown'))
                    st.metric("Accuracy", f"{info.get('accuracy', 0):.3f}")
                
                with col2:
                    st.metric("F1-Score", f"{info.get('f1_score', 0):.3f}")
                    st.metric("Number of Speakers", info.get('num_speakers', 0))
                
                with col3:
                    st.metric("Feature Dimension", info.get('feature_dim', 'Unknown'))
                    
                # Speaker list
                st.subheader("📋 Trained Speakers")
                speakers = info.get('speakers', [])
                for i, speaker in enumerate(speakers, 1):
                    st.write(f"{i}. {speaker}")
            else:
                st.warning("Model information is not available. Try refreshing.")
        
        else:
            with col_status:
                st.error("❌ No model loaded")
            st.info("Train a model by enrolling speakers first.")
        
        st.markdown("---")
        
        # Model management
        st.subheader("🔧 Model Management")
        
        # Available models
        available_models = self.model_manager.list_available_models()
        
        if available_models:
            st.write("**Available Models:**")
            for model in available_models:
                st.write(f"• {model}")
        
        # Export model
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Model for Deployment", use_container_width=True):
                try:
                    export_path = self.model_manager.export_model_for_deployment()
                    st.success(f"✅ Model exported to: {export_path}")
                    self.load_model()  # Reload model info after export
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        with col2:
            if st.button("🧹 Cleanup Old Models", use_container_width=True):
                try:
                    self.model_manager.cleanup_old_models()
                    st.success("✅ Old models cleaned up!")
                    self.load_model()  # Reload model info after cleanup
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {str(e)}")
    
    def dataset_management_page(self):
        """Dataset management page"""
        st.title("📊 Dataset Management")
        st.markdown("---")
        
        # Import required modules
        from create_sample_data import copy_sample_files_from_dataset, remove_sample_data
        
        # Current dataset status
        st.subheader("📈 Current Dataset Status")
        
        # Get current speakers
        try:
            speakers = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')
                       and d != "temp"]  # Exclude temp directory
            
            if speakers:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Speakers", len(speakers))
                
                # Count total files
                total_files = 0
                for speaker in speakers:
                    speaker_dir = os.path.join(self.data_dir, speaker)
                    try:
                        files = os.listdir(speaker_dir)
                        audio_files = [f for f in files 
                                      if f.lower().endswith(('.mp3', '.wav', '.flac'))]
                        total_files += len(audio_files)
                    except:
                        pass
                
                with col2:
                    st.metric("Total Audio Files", total_files)
                
                # Show speaker details
                st.subheader("👥 Speaker Details")
                for i, speaker in enumerate(speakers, 1):
                    speaker_dir = os.path.join(self.data_dir, speaker)
                    try:
                        files = os.listdir(speaker_dir)
                        audio_files = [f for f in files 
                                      if f.lower().endswith(('.mp3', '.wav', '.flac'))]
                        st.write(f"**{i}. {speaker}**: {len(audio_files)} files")
                    except:
                        st.write(f"**{i}. {speaker}**: Error reading directory")
                
                # Dataset adequacy analysis
                self._show_dataset_adequacy_warning(speakers, total_files)
            else:
                st.info("No speakers found in the dataset")
                
        except Exception as e:
            st.error(f"Error reading dataset: {str(e)}")
        
        st.markdown("---")
        
        # Dataset creation tools
        st.subheader("🛠️ Dataset Creation Tools")
        
        # Create sample data
        st.write("**Create Sample Data from LibriSpeech/Common Voice**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_speakers = st.number_input("Number of Speakers", min_value=2, max_value=20, value=5)
            files_per_speaker = st.number_input("Files per Speaker", min_value=5, max_value=100, value=20)
        
        with col2:
            auto_detect = st.checkbox("Auto-detect Dataset Format", value=True)
            
        if st.button("🎯 Create Sample Dataset", type="primary"):
            with st.spinner("Creating sample dataset..."):
                try:
                    # Clear any existing files first to avoid permission issues
                    if os.path.exists(self.data_dir):
                        import time
                        import gc
                        
                        # Force garbage collection to release any file handles
                        gc.collect()
                        time.sleep(0.5)
                    
                    success = copy_sample_files_from_dataset(
                        num_speakers=num_speakers,
                        files_per_speaker=files_per_speaker,
                        auto_mode=True
                    )
                    
                    if success:
                        st.success("✅ Sample dataset created successfully!")
                        st.balloons()
                        st.info("💡 If some files couldn't be copied due to permission issues, the system will continue with the available files.")
                        # Refresh the page
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to create sample dataset")
                        st.info("💡 Try running the application as administrator if permission issues persist.")
                        
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")
                    st.info("💡 Common solutions:")
                    st.info("- Close any audio players or file explorers")
                    st.info("- Run the application as administrator")
                    st.info("- Check that the source dataset path is accessible")
        
        st.markdown("---")
        
        # Dataset management tools
        st.subheader("🧹 Dataset Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Show Dataset Summary", use_container_width=True):
                try:
                    # Show dataset summary inline
                    data_dir = Config.DATA_DIR
                    if os.path.exists(data_dir):
                        speakers = [d for d in os.listdir(data_dir) 
                                   if os.path.isdir(os.path.join(data_dir, d))]
                        
                        total_files = 0
                        speaker_info = []
                        
                        for speaker in speakers:
                            speaker_dir = os.path.join(data_dir, speaker)
                            try:
                                files = os.listdir(speaker_dir)
                                audio_files = [f for f in files 
                                              if f.lower().endswith(('.mp3', '.wav', '.flac'))]
                                speaker_info.append(f"{speaker}: {len(audio_files)} files")
                                total_files += len(audio_files)
                            except Exception as e:
                                speaker_info.append(f"{speaker}: Error reading directory")
                        
                        st.info(f"**Dataset Summary:**\n- Total Speakers: {len(speakers)}\n- Total Files: {total_files}")
                        
                        if speaker_info:
                            st.write("**Speaker Details:**")
                            for info in speaker_info:
                                st.write(f"- {info}")
                        
                        # Show detailed adequacy warning
                        self._show_dataset_adequacy_warning(speakers, total_files)
                    else:
                        st.error("Data directory not found")
                except Exception as e:
                    st.error(f"Error showing summary: {str(e)}")
        
        with col2:
            if st.button("🗑️ Remove All Sample Data", use_container_width=True):
                st.warning("This will remove ALL audio files from speaker directories!")
                if st.checkbox("I understand this action cannot be undone"):
                    try:
                        remove_sample_data()
                        st.success("Sample data removed successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error removing data: {str(e)}")
        
        # Dataset balancing section
        st.markdown("---")
        st.subheader("⚖️ Dataset Balancing")
        
        # Import data balancer
        try:
            from data_balancer import DataBalancer
            
            balancer = DataBalancer(self.data_dir)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Analyze Dataset Balance", use_container_width=True):
                    with st.spinner("Analyzing dataset..."):
                        report = balancer.generate_report()
                        st.text_area("Dataset Analysis Report", report, height=400)
            
            with col2:
                if st.button("🧹 Clean Duplicates & Balance", use_container_width=True):
                    with st.spinner("Cleaning and balancing dataset..."):
                        try:
                            # Remove duplicates
                            st.info("Step 1: Removing duplicate files...")
                            removal_stats = balancer.remove_duplicates(dry_run=False)
                            
                            total_removed = sum(removal_stats.values())
                            if total_removed > 0:
                                st.success(f"✅ Removed {total_removed} duplicate files")
                            else:
                                st.info("No duplicates found")
                            
                            # Balance dataset
                            st.info("Step 2: Balancing dataset...")
                            balance_stats = balancer.balance_dataset(target_files_per_speaker=10, method='limit')
                            
                            st.success("✅ Dataset balanced successfully!")
                            
                            # Show final stats
                            st.subheader("📈 Final Dataset Stats")
                            for speaker, count in balance_stats.items():
                                st.write(f"**{speaker}**: {count} files")
                            

                            st.warning("⚠️ **Important**: Please retrain your model after balancing the dataset!")
                            
                        except Exception as e:
                            st.error(f"Error during balancing: {str(e)}")
        
        except ImportError:
            st.warning("Data balancer not available")

    def model_training_page(self):
        """Model training page"""
        st.title("🎯 Model Training")
        st.markdown("---")
        
        # Import training components
        from model_training import SpeakerRecognitionTrainer
        from feature_extraction import FeatureExtractor
        
        # Check if data is available and show dataset adequacy
        try:
            speakers = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')
                       and d != "temp"]  # Exclude temp directory
            
            # Count total files
            total_files = 0
            for speaker in speakers:
                speaker_dir = os.path.join(self.data_dir, speaker)
                try:
                    files = os.listdir(speaker_dir)
                    audio_files = [f for f in files 
                                  if f.lower().endswith(('.mp3', '.wav', '.flac'))]
                    total_files += len(audio_files)
                except:
                    pass
            
            # Show dataset adequacy warning
            st.subheader("📊 Dataset Status")
            self._show_dataset_adequacy_warning(speakers, total_files)
            
            # Check if training can proceed
            if len(speakers) < 2:
                st.error("❌ Training cannot proceed: Need at least 2 speakers")
                st.info("Please add speakers in the Dataset Management page first")
                return
                
            # Check minimum files per speaker
            min_files_needed = 3
            speakers_with_insufficient_files = []
            for speaker in speakers:
                speaker_dir = os.path.join(self.data_dir, speaker)
                try:
                    files = os.listdir(speaker_dir)
                    audio_files = [f for f in files 
                                  if f.lower().endswith(('.mp3', '.wav', '.flac'))]
                    if len(audio_files) < min_files_needed:
                        speakers_with_insufficient_files.append(f"{speaker} ({len(audio_files)} files)")
                except:
                    speakers_with_insufficient_files.append(f"{speaker} (error reading)")
            
            if speakers_with_insufficient_files:
                st.error("❌ Training cannot proceed: Some speakers have insufficient audio files")
                st.error(f"Speakers needing more files: {', '.join(speakers_with_insufficient_files)}")
                st.error(f"Each speaker needs at least {min_files_needed} audio files")
                return
            
        except Exception as e:
            st.error(f"Error checking dataset: {str(e)}")
            return
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_pca = st.checkbox("Use PCA Dimensionality Reduction", value=True)
            n_components = st.slider("PCA Components", min_value=20, max_value=100, value=50) if use_pca else 50
            
        with col2:
            optimize_hyperparams = st.checkbox("Hyperparameter Optimization", value=True)
            validation_size = st.slider("Validation Set Size", min_value=0.1, max_value=0.4, value=0.2)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        train_knn = st.checkbox("Train KNN", value=True)
        train_svm = st.checkbox("Train SVM", value=True)
        train_rf = st.checkbox("Train Random Forest", value=True)
        train_cnn = st.checkbox("Train CNN (requires TensorFlow)", value=False)
        
        # Current model status
        st.subheader("📊 Current Model Status")
        
        try:
            if os.path.exists(os.path.join(self.models_dir, "model_metadata.pkl")):
                import joblib
                metadata = joblib.load(os.path.join(self.models_dir, "model_metadata.pkl"))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Model", metadata.get('model_type', 'Unknown'))
                with col2:
                    st.metric("Accuracy", f"{metadata.get('accuracy', 0):.3f}")
                with col3:
                    st.metric("Speakers", metadata.get('num_speakers', 0))
                
                st.info(f"**Model Features**: {metadata.get('num_features', 'Unknown')}")
                st.info(f"**F1-Score**: {metadata.get('f1_score', 'Unknown'):.3f}")
            else:
                st.info("No trained model found")
                
        except Exception as e:
            st.warning(f"Could not load model metadata: {str(e)}")
        
        st.markdown("---")
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not any([train_knn, train_svm, train_rf, train_cnn]):
                st.error("Please select at least one model to train")
                return
            
            # Initialize components
            extractor = FeatureExtractor()
            trainer = SpeakerRecognitionTrainer()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Load dataset
                status_text.text("Loading dataset...")
                progress_bar.progress(10)
                
                X, y, label_names, pca = extractor.load_dataset_from_folders(
                    self.data_dir,
                    use_pca=use_pca,
                    n_components=n_components,
                    is_librispeech=False
                )
                
                if len(X) == 0:
                    st.error("No audio files found in dataset")
                    return
                
                status_text.text(f"Dataset loaded: {X.shape[0]} samples, {len(label_names)} speakers")
                progress_bar.progress(20)
                
                # Prepare data
                status_text.text("Preparing data...")
                X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, label_names, test_size=validation_size)
                progress_bar.progress(30)
                
                # Train models
                models_to_train = []
                if train_knn: models_to_train.append(('KNN', trainer.train_knn))
                if train_svm: models_to_train.append(('SVM', trainer.train_svm))
                if train_rf: models_to_train.append(('Random Forest', trainer.train_random_forest))
                if train_cnn: models_to_train.append(('CNN', lambda x, y: trainer.train_cnn(x, y, X_test, y_test)))
                
                progress_per_model = 40 / len(models_to_train)
                current_progress = 30
                
                for model_name, train_func in models_to_train:
                    status_text.text(f"Training {model_name}...")
                    
                    if model_name == 'CNN':
                        train_func(X_train, y_train)
                    else:
                        train_func(X_train, y_train)
                    
                    current_progress += progress_per_model
                    progress_bar.progress(int(current_progress))
                
                # Evaluate models
                status_text.text("Evaluating models...")
                progress_bar.progress(80)
                
                results = trainer.evaluate_models(X_test, y_test)
                
                # Save best model
                status_text.text("Saving best model...")
                progress_bar.progress(90)
                
                trainer.save_best_model(results, X, label_names, pca)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                # Display results
                st.success("🎉 Training completed successfully!")
                
                # Show results
                st.subheader("📈 Training Results")
                
                best_model_name, _ = trainer.get_best_model(results)
                
                for model_name, result in results.items():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        emoji = "🏆" if model_name == best_model_name else "📊"
                        st.write(f"{emoji} **{model_name.upper()}**")
                    
                    with col2:
                        st.write(f"Accuracy: {result['accuracy']:.3f} | F1: {result['f1_score']:.3f}")
                
                st.balloons()
                
                # Update model in session state
                self.load_model()
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                st.error("Please check the console for detailed error information")
    
    def system_config_page(self):
        """System configuration page"""
        st.title("⚙️ System Configuration")
        st.markdown("---")
        
        # System paths
        st.subheader("📁 System Paths")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Base Directory**")
            st.code(self.base_dir)
            
            st.write("**Data Directory**")
            st.code(self.data_dir)
            
        with col2:
            st.write("**Models Directory**")
            st.code(self.models_dir)
            
            st.write("**Temp Directory**")
            st.code(self.temp_dir)
        
        # System status
        st.subheader("📊 System Status")
        
        # Check dependencies
        dependencies = [
            ("librosa", "Audio processing"),
            ("sklearn", "Machine learning"),
            ("numpy", "Numerical computing"),
            ("pandas", "Data manipulation"),
            ("matplotlib", "Plotting"),
            ("joblib", "Model persistence"),
            ("sounddevice", "Audio recording"),
            ("soundfile", "Audio file handling"),
            ("streamlit", "Web interface"),
            ("tensorflow", "Deep learning (optional)")
        ]
        
        for package, description in dependencies:
            try:
                __import__(package)
                st.success(f"✅ {package}: {description}")
            except ImportError:
                st.error(f"❌ {package}: {description} (Not installed)")
        
        # Directory status
        st.subheader("📂 Directory Status")
        
        directories = [
            ("Data", self.data_dir),
            ("Models", self.models_dir),
            ("Temp", self.temp_dir)
        ]
        
        for name, path in directories:
            if os.path.exists(path):
                try:
                    files = os.listdir(path)
                    st.success(f"✅ {name}: {len(files)} items")
                except:
                    st.warning(f"⚠️ {name}: Exists but cannot read")
            else:
                st.error(f"❌ {name}: Directory not found")
        
        # Model deployment
        st.subheader("🚀 Model Deployment")
        
        if st.button("📦 Export Model for Deployment"):
            try:
                # Create deployment model
                model_path = os.path.join(self.models_dir, "best_model", "model.pkl")
                scaler_path = os.path.join(self.models_dir, "best_model", "scaler.pkl")
                labels_path = os.path.join(self.models_dir, "best_model", "label_names.pkl")
                metadata_path = os.path.join(self.models_dir, "best_model", "metadata.pkl")
                
                if all(os.path.exists(p) for p in [model_path, scaler_path, labels_path, metadata_path]):
                    import joblib
                    
                    # Load components
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    labels = joblib.load(labels_path)
                    metadata = joblib.load(metadata_path)
                    
                    # Create deployment bundle
                    deployment_bundle = {
                        'model': model,
                        'scaler': scaler,
                        'label_names': labels,
                        'metadata': metadata
                    }
                    
                    # Save to deployment location
                    os.makedirs(os.path.join(self.models_dir, "exported_model"), exist_ok=True)
                    deployment_path = os.path.join(self.models_dir, "exported_model", "deployment_model.pkl")
                    joblib.dump(deployment_bundle, deployment_path)
                    
                    st.success("✅ Model exported for deployment!")
                    st.info(f"Deployment model saved to: {deployment_path}")
                    
                else:
                    st.error("❌ Best model not found. Please train a model first.")
                    
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
        
        # Advanced configuration
        st.subheader("🔧 Advanced Configuration")
        
        with st.expander("Audio Settings"):
            st.write("**Default Recording Duration**")
            st.slider("Seconds", min_value=1, max_value=10, value=3)
            
            st.write("**Sample Rate**")
            st.slider("Hz", min_value=8000, max_value=48000, value=16000)
        
        with st.expander("Model Settings"):
            st.write("**PCA Components**")
            st.slider("Components", min_value=20, max_value=100, value=50)
            
            st.write("**Validation Set Size**")
            st.slider("Ratio", min_value=0.1, max_value=0.4, value=0.2)
        
        # System maintenance
        st.subheader("🧹 System Maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Temp Files"):
                try:
                    temp_files = os.listdir(self.temp_dir)
                    for file in temp_files:
                        if file.endswith(('.wav', '.mp3', '.flac')):
                            os.remove(os.path.join(self.temp_dir, file))
                    st.success("Temp files cleared")
                except Exception as e:
                    st.error(f"Error clearing temp files: {str(e)}")
        
        with col2:
            if st.button("🔄 Reload System"):
                try:
                    self.load_model()
                    st.success("System reloaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reloading system: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        if not MODULES_AVAILABLE:
            st.error("⚠️ Required modules not available. Please check your installation.")
            return
        
        # Sidebar navigation
        st.sidebar.title("🎤 Speaker Recognition")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["🏠 Home", "🔐 Login", "👤 Enrollment", "📊 Model Info", "📊 Dataset Management", "🎯 Model Training", "⚙️ System Config"],
            help="Select a page from the dropdown"
        )
        
        # Track page navigation
        page_key = page.split(" ")[-1].lower() if " " in page else page.lower()
        if 'current_page' not in st.session_state or st.session_state.get('current_page') != page_key:
            st.session_state['current_page'] = page_key
        
        # Audio availability warning
        if not AUDIO_AVAILABLE:
            st.sidebar.error("⚠️ Audio recording not available")
            st.sidebar.info("Install: pip install sounddevice soundfile")
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        
        model_status = "✅ Ready" if st.session_state.get('model_loaded', False) else "❌ No Model"
        audio_status = "✅ Available" if AUDIO_AVAILABLE else "❌ Not Available"
        
        st.sidebar.write(f"**Model:** {model_status}")
        st.sidebar.write(f"**Audio:** {audio_status}")
        
        speakers = st.session_state.get('speakers', [])
        st.sidebar.write(f"**Speakers:** {len(speakers)}")
        
        # Main content
        if page == "🏠 Home":
            self.home_page()
        elif page == "🔐 Login":
            self.login_page()
        elif page == "👤 Enrollment":
            self.enrollment_page()
        elif page == "📊 Model Info":
            self.model_info_page()
        elif page == "📊 Dataset Management":
            self.dataset_management_page()
        elif page == "🎯 Model Training":
            self.model_training_page()
        elif page == "⚙️ System Config":
            self.system_config_page()
    
    def home_page(self):
        """Home page with instructions"""
        st.title("🎤 Speaker Recognition System")
        st.markdown("### Welcome to the Complete Voice Authentication Platform")
        
        # Hero section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            This web application provides a complete speaker recognition system with:
            
            **🔐 Voice Authentication**
            - Secure login using voice biometrics
            - Real-time speaker identification
            - Confidence-based authentication
            
            **👤 Speaker Management**
            - Easy speaker enrollment
            - Voice sample recording
            - Speaker profile management
            
            **📊 Dataset Tools**
            - LibriSpeech & Common Voice integration
            - Automated sample data creation
            - Dataset statistics and management
            
            **🎯 Machine Learning**
            - Multiple model training (KNN, SVM, Random Forest, CNN)
            - Hyperparameter optimization
            - Real-time training progress
            
            **⚙️ System Configuration**
            - Model deployment tools
            - System diagnostics
            - Performance monitoring
            """)
        
        with col2:
            # System status card
            st.markdown("#### 📈 System Status")
            
            # Model status
            if st.session_state.get('model_loaded', False):
                st.success("✅ Model Ready")
                speakers = st.session_state.get('speakers', [])
                st.info(f"📊 {len(speakers)} Speakers")
                
                # Show latest model info
                model_info = st.session_state.get('model_info')
                if model_info:
                    st.metric("Accuracy", f"{model_info.get('accuracy', 0):.3f}")
                    st.metric("Model", model_info.get('model_type', 'Unknown').upper())
            else:
                st.warning("⚠️ No Model")
                st.info("Train a model first")
            
            # Audio status
            if AUDIO_AVAILABLE:
                st.success("🎤 Audio Ready")
            else:
                st.error("❌ Audio Issues")
        
        st.markdown("---")
        
        # Quick start guide
        st.subheader("🚀 Quick Start Guide")
        
        # Check current state and provide next steps
        model_loaded = st.session_state.get('model_loaded', False)
        speakers = st.session_state.get('speakers', [])
        
        if model_loaded and len(speakers) >= 2:
            # System is ready
            st.success("🎉 **System Ready!** You can now:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔐 Try Voice Login", use_container_width=True):
                    st.session_state['current_page'] = 'login'
                    st.rerun()
            
            with col2:
                if st.button("👤 Add More Speakers", use_container_width=True):
                    st.session_state['current_page'] = 'enrollment'
                    st.rerun()
            
            with col3:
                if st.button("📊 View Model Info", use_container_width=True):
                    st.session_state['current_page'] = 'info'
                    st.rerun()
        
        elif len(speakers) >= 2:
            # Has speakers but no model
            st.info("💡 **Next Step**: Train your model with the existing speakers")
            
            if st.button("🎯 Start Model Training", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'training'
                st.rerun()
        
        else:
            # No speakers or insufficient speakers
            st.info("🎯 **Get Started**: Set up your dataset and train your first model")
            
            # Setup steps
            steps = [
                ("1️⃣", "Create Dataset", "Use LibriSpeech or add your own speakers", "📊 Dataset Management"),
                ("2️⃣", "Train Model", "Train machine learning models on your data", "🎯 Model Training"),
                ("3️⃣", "Voice Authentication", "Try voice login with trained speakers", "🔐 Login"),
                ("4️⃣", "Add More Speakers", "Enroll additional users", "👤 Enrollment")
            ]
            
            for emoji, title, desc, page in steps:
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 2])
                    
                    with col1:
                        st.markdown(f"### {emoji}")
                    
                    with col2:
                        st.markdown(f"**{title}**")
                        st.write(desc)
                    
                    with col3:
                        if st.button(page, key=f"nav_{title}"):
                            page_key = page.split(" ")[-1].lower()
                            st.session_state['current_page'] = page_key
                            st.rerun()
        
        st.markdown("---")
        
        # Feature highlights
        st.subheader("✨ Key Features")
        
        features = [
            ("🔊 Advanced Audio Processing", "Voice Activity Detection, MFCC features, spectral analysis"),
            ("🤖 Multiple ML Models", "KNN, SVM, Random Forest, and CNN for best performance"),
            ("📈 Real-time Training", "Live progress tracking and model comparison"),
            ("🔧 Easy Configuration", "Web-based setup with auto-detection of datasets"),
            ("📊 Comprehensive Analytics", "Model performance metrics and confusion matrices"),
            ("🚀 Production Ready", "Model export and deployment tools")
        ]
        
        cols = st.columns(2)
        
        for i, (title, desc) in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.write(desc)
        
        # Footer info
        st.markdown("---")
        st.markdown("**💡 Tip**: All functionality is available through this web interface - no command line needed!")
        
        # Current system info
        if model_loaded:
            speakers_list = st.session_state.get('speakers', [])
            if speakers_list:
                st.info(f"**Enrolled Speakers**: {', '.join(speakers_list[:5])}{'...' if len(speakers_list) > 5 else ''}")
        else:
            st.warning("⚠️ No trained model found. Please create a dataset and train a model first.")

    def _show_dataset_adequacy_warning(self, speakers, total_files):
        """Show warnings about dataset size and adequacy for training"""
        num_speakers = len(speakers)
        
        # Calculate minimum files per speaker
        files_per_speaker = {}
        for speaker in speakers:
            speaker_dir = os.path.join(self.data_dir, speaker)
            try:
                files = os.listdir(speaker_dir)
                audio_files = [f for f in files 
                              if f.lower().endswith(('.mp3', '.wav', '.flac'))]
                files_per_speaker[speaker] = len(audio_files)
            except:
                files_per_speaker[speaker] = 0
        
        min_files_per_speaker = min(files_per_speaker.values()) if files_per_speaker else 0
        avg_files_per_speaker = total_files / num_speakers if num_speakers > 0 else 0
        
        # Determine dataset adequacy level
        if num_speakers == 0:
            st.info("📂 No speakers found in the dataset. Add speakers to get started.")
            return
        elif num_speakers == 1:
            st.error("🚫 **Dataset Too Small for Training**")
            st.error("   - Need at least **2 speakers** for speaker recognition")
            st.error("   - Current: 1 speaker")
            st.info("💡 **Solution**: Add more speakers using the dataset creation tools below")
            return
        elif num_speakers < 3:
            st.warning("⚠️ **Limited Dataset Size**")
            st.warning(f"   - Only {num_speakers} speakers (minimum for basic training)")
            st.warning("   - Recommend at least **3-5 speakers** for better accuracy")
        elif num_speakers < 5:
            st.info("ℹ️ **Acceptable Dataset Size**")
            st.info(f"   - {num_speakers} speakers (good for basic training)")
            st.info("   - Consider adding more speakers for production use")
        else:
            st.success("✅ **Good Dataset Size**")
            st.success(f"   - {num_speakers} speakers (excellent for training)")
        
        # Check files per speaker
        if min_files_per_speaker < 3:
            st.error("🚫 **Insufficient Audio Samples**")
            st.error(f"   - Minimum files per speaker: {min_files_per_speaker}")
            st.error("   - Need at least **3 files per speaker** for basic training")
            st.error("   - Recommend **5-10 files per speaker** for good accuracy")
            
            # Show which speakers need more files
            low_file_speakers = [speaker for speaker, count in files_per_speaker.items() if count < 3]
            if low_file_speakers:
                st.error(f"   - Speakers needing more files: {', '.join(low_file_speakers)}")
                
        elif min_files_per_speaker < 5:
            st.warning("⚠️ **Limited Audio Samples**")
            st.warning(f"   - Average files per speaker: {avg_files_per_speaker:.1f}")
            st.warning("   - Recommend **5-10 files per speaker** for better accuracy")
            
            # Show speakers with low file counts
            low_file_speakers = [f"{speaker} ({count})" for speaker, count in files_per_speaker.items() if count < 5]
            if low_file_speakers:
                st.warning(f"   - Speakers with few files: {', '.join(low_file_speakers)}")
                
        elif avg_files_per_speaker < 8:
            st.info("ℹ️ **Adequate Audio Samples**")
            st.info(f"   - Average files per speaker: {avg_files_per_speaker:.1f}")
            st.info("   - Consider adding more samples for production use")
        else:
            st.success("✅ **Excellent Audio Sample Count**")
            st.success(f"   - Average files per speaker: {avg_files_per_speaker:.1f}")
        
        # Overall assessment and recommendations
        st.markdown("---")
        
        # Critical issues
        if num_speakers < 2 or min_files_per_speaker < 3:
            st.error("🚫 **CRITICAL: Dataset Cannot Be Used for Training**")
            st.error("**Requirements:**")
            st.error("- Minimum 2 speakers")
            st.error("- Minimum 3 files per speaker")
            st.error("- Audio files in .mp3, .wav, or .flac format")
            
        # Warnings for small datasets
        elif num_speakers < 3 or avg_files_per_speaker < 5:
            st.warning("⚠️ **WARNING: Small Dataset May Cause Poor Performance**")
            st.warning("**Recommendations for Better Results:**")
            st.warning("- Add more speakers (target: 5-10 speakers)")
            st.warning("- Add more audio files per speaker (target: 5-10 files)")
            st.warning("- Ensure audio quality is good (clear speech, minimal background noise)")
            st.warning("- Each audio file should be 3-10 seconds long")
            
            # Show expected accuracy
            if num_speakers == 2 and avg_files_per_speaker < 5:
                st.warning("📊 **Expected Accuracy: 60-80%** (may not be reliable)")
            elif num_speakers < 4 and avg_files_per_speaker < 8:
                st.warning("📊 **Expected Accuracy: 70-85%** (acceptable for testing)")
            else:
                st.info("📊 **Expected Accuracy: 80-90%** (good for most applications)")
                
        # Good dataset
        else:
            st.success("✅ **EXCELLENT: Dataset Ready for High-Quality Training**")
            st.success("📊 **Expected Accuracy: 90-95%** (production-ready)")
            st.success("**Your dataset meets all recommendations for reliable speaker recognition!**")
    
def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state['model_loaded'] = False
    if 'speakers' not in st.session_state:
        st.session_state['speakers'] = []
    if 'model_info' not in st.session_state:
        st.session_state['model_info'] = None
    
    # Create and run the app
    app = StreamlitSpeakerApp()
    app.run()

if __name__ == "__main__":
    main()
