"""
Model Utilities for Speaker Recognition
Save and load trained models with preprocessing components
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model_bundle(self, model: Any, scaler: Any, label_names: List[str], 
                         model_name: str, metrics: Dict[str, float], 
                         bundle_name: str = "speaker_recognition_model") -> str:
        """
        Save complete model bundle with all components
        
        Args:
            model: Trained model
            scaler: Feature scaler
            label_names: List of speaker names
            model_name: Type of model (knn, svm, etc.)
            metrics: Performance metrics
            bundle_name: Name for the model bundle
            
        Returns:
            Path to saved bundle directory
        """
        bundle_dir = os.path.join(self.models_dir, bundle_name)
        os.makedirs(bundle_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(bundle_dir, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(bundle_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Save label names
        labels_path = os.path.join(bundle_dir, 'label_names.pkl')
        joblib.dump(label_names, labels_path)
        
        # Save metadata
        metadata = {
            'model_type': model_name,
            'accuracy': metrics.get('accuracy', 0.0),
            'f1_score': metrics.get('f1_score', 0.0),
            'speakers': label_names,
            'num_speakers': len(label_names),
            'feature_dim': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'unknown'
        }
        
        metadata_path = os.path.join(bundle_dir, 'metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Model bundle saved to: {bundle_dir}")
        return bundle_dir
    
    def load_model_bundle(self, bundle_name: str = "speaker_recognition_model") -> Dict[str, Any]:
        """
        Load complete model bundle
        
        Args:
            bundle_name: Name of the model bundle
            
        Returns:
            Dictionary containing model components
        """
        bundle_dir = os.path.join(self.models_dir, bundle_name)
        
        if not os.path.exists(bundle_dir):
            raise FileNotFoundError(f"Model bundle not found: {bundle_dir}")
        
        # Load components
        model_path = os.path.join(bundle_dir, 'model.pkl')
        scaler_path = os.path.join(bundle_dir, 'scaler.pkl')
        labels_path = os.path.join(bundle_dir, 'label_names.pkl')
        metadata_path = os.path.join(bundle_dir, 'metadata.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_names = joblib.load(labels_path)
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
        
        return {
            'model': model,
            'scaler': scaler,
            'label_names': label_names,
            'metadata': metadata
        }
    
    def save_best_model(self, trainer, results: Dict[str, Dict[str, float]]) -> str:
        """
        Save the best performing model from training results
        
        Args:
            trainer: SpeakerRecognitionTrainer instance
            results: Training results dictionary
            
        Returns:
            Path to saved model bundle
        """
        # Find best model
        best_accuracy = 0
        best_model_name = None
        
        for model_name, result in results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError("No valid model found in results")
        
        best_model = trainer.models[best_model_name]
        
        # Save the best model bundle
        bundle_path = self.save_model_bundle(
            model=best_model,
            scaler=trainer.scaler,
            label_names=trainer.label_names,
            model_name=best_model_name,
            metrics=results[best_model_name],
            bundle_name="best_model"
        )
        
        # Also save individual components for backward compatibility
        self.save_individual_components(best_model, trainer, results[best_model_name])
        
        return bundle_path
    
    def save_individual_components(self, model: Any, trainer, metrics: Dict[str, float]) -> None:
        """
        Save individual model components (for backward compatibility)
        
        Args:
            model: Trained model
            trainer: SpeakerRecognitionTrainer instance
            metrics: Performance metrics
        """
        # Save best model
        model_path = os.path.join(self.models_dir, 'best_model.pkl')
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        joblib.dump(trainer.scaler, scaler_path)
        
        # Save label names
        labels_path = os.path.join(self.models_dir, 'label_names.pkl')
        joblib.dump(trainer.label_names, labels_path)
        
        # Save metadata
        metadata = {
            'model_type': metrics.get('model_name', 'unknown'),  # Include model_type for consistency
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'speakers': trainer.label_names,
            'num_speakers': len(trainer.label_names)
        }
        
        metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
    
    def list_available_models(self) -> List[str]:
        """
        List all available model bundles
        
        Returns:
            List of model bundle names
        """
        if not os.path.exists(self.models_dir):
            return []
        
        bundles = []
        for item in os.listdir(self.models_dir):
            bundle_path = os.path.join(self.models_dir, item)
            if os.path.isdir(bundle_path):
                # Check if it's a valid model bundle
                required_files = ['model.pkl', 'scaler.pkl', 'label_names.pkl']
                if all(os.path.exists(os.path.join(bundle_path, f)) for f in required_files):
                    bundles.append(item)
        
        return bundles
    
    def get_model_info(self, bundle_name: str = "best_model") -> Optional[Dict[str, Any]]:
        """
        Get information about a model bundle or individual components
        
        Args:
            bundle_name: Name of the model bundle
            
        Returns:
            Model information dictionary
        """
        try:
            # First try to load as a bundle
            try:
                bundle_dir = os.path.join(self.models_dir, bundle_name)
                if os.path.exists(bundle_dir) and os.path.isdir(bundle_dir):
                    bundle = self.load_model_bundle(bundle_name)
                    metadata = bundle.get('metadata', {})
                    
                    info = {
                        'model_type': metadata.get('model_type', 'unknown'),
                        'accuracy': metadata.get('accuracy', 'unknown'),
                        'f1_score': metadata.get('f1_score', 'unknown'),
                        'speakers': metadata.get('speakers', []),
                        'num_speakers': metadata.get('num_speakers', 0),
                        'feature_dim': metadata.get('feature_dim', 'unknown')
                    }
                    
                    return info
            except Exception as e:
                print(f"Bundle load failed, trying individual components: {str(e)}")
                
            # If bundle loading fails, try to load individual components
            metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
            labels_path = os.path.join(self.models_dir, 'label_names.pkl')
            
            if os.path.exists(metadata_path) and os.path.exists(labels_path):
                metadata = joblib.load(metadata_path)
                label_names = joblib.load(labels_path)
                
                # Get the model_type field
                # Try to determine model_type from the actual model
                model_path = os.path.join(self.models_dir, 'best_model.pkl')
                model_type = 'unknown'
                
                try:
                    if os.path.exists(model_path):
                        model = joblib.load(model_path)
                        if hasattr(model, '__class__'):
                            if 'KNeighbors' in model.__class__.__name__:
                                model_type = 'knn'
                            elif 'SVC' in model.__class__.__name__:
                                model_type = 'svm'
                            elif 'RandomForest' in model.__class__.__name__:
                                model_type = 'random_forest'
                            elif 'Model' in model.__class__.__name__ or hasattr(model, 'predict'):
                                model_type = 'cnn'
                except:
                    pass  # If we can't determine from the model, use metadata
                    
                info = {
                    'model_type': metadata.get('model_type', model_type),
                    'accuracy': metadata.get('accuracy', 0.0),
                    'f1_score': metadata.get('f1_score', 0.0),
                    'speakers': label_names,
                    'num_speakers': len(label_names),
                    'feature_dim': metadata.get('num_features', 'unknown')
                }
                
                return info
                
            else:
                print("No model metadata or labels found")
                return None
            
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return None
    
    def export_model_for_deployment(self, bundle_name: str = "best_model", 
                                  export_path: str = "exported_model") -> str:
        """
        Export model in a format suitable for deployment
        
        Args:
            bundle_name: Name of the model bundle to export
            export_path: Path for exported model
            
        Returns:
            Path to exported model
        """
        export_dir = os.path.join(self.models_dir, export_path)
        os.makedirs(export_dir, exist_ok=True)
        
        # First try to load as bundle
        try:
            bundle_dir = os.path.join(self.models_dir, bundle_name)
            if os.path.exists(bundle_dir) and os.path.isdir(bundle_dir):
                bundle = self.load_model_bundle(bundle_name)
            else:
                raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
        except Exception as e:
            print(f"Bundle load failed, trying individual components: {e}")
            # Try to load individual components instead
            model_path = os.path.join(self.models_dir, 'best_model.pkl')
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            labels_path = os.path.join(self.models_dir, 'label_names.pkl')
            metadata_path = os.path.join(self.models_dir, 'model_metadata.pkl')
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_names = joblib.load(labels_path)
            metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
            
            bundle = {
                'model': model,
                'scaler': scaler,
                'label_names': label_names,
                'metadata': metadata
            }
        
        # Save in deployment format
        deployment_model = {
            'model': bundle['model'],
            'scaler': bundle['scaler'],
            'label_names': bundle['label_names'],
            'metadata': bundle['metadata']
        }
        
        deployment_path = os.path.join(export_dir, 'deployment_model.pkl')
        joblib.dump(deployment_model, deployment_path)
        
        # Create deployment info file
        info_file = os.path.join(export_dir, 'deployment_info.txt')
        with open(info_file, 'w') as f:
            f.write("Speaker Recognition Model - Deployment Package\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Type: {bundle['metadata'].get('model_type', 'unknown')}\n")
            f.write(f"Accuracy: {bundle['metadata'].get('accuracy', 'unknown')}\n")
            f.write(f"F1-Score: {bundle['metadata'].get('f1_score', 'unknown')}\n")
            f.write(f"Number of Speakers: {len(bundle['label_names'])}\n")
            f.write(f"Speakers: {', '.join(bundle['label_names'])}\n")
            f.write(f"\nFiles:\n")
            f.write(f"- deployment_model.pkl: Complete model package\n")
            f.write(f"- deployment_info.txt: This information file\n")
        
        print(f"Model exported for deployment to: {export_dir}")
        return export_dir
    
    def load_deployment_model(self, deployment_path: str) -> Dict[str, Any]:
        """
        Load a deployment model
        
        Args:
            deployment_path: Path to deployment model file
            
        Returns:
            Loaded model components
        """
        if os.path.isdir(deployment_path):
            model_file = os.path.join(deployment_path, 'deployment_model.pkl')
        else:
            model_file = deployment_path
            
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Deployment model not found: {model_file}")
        
        return joblib.load(model_file)
    
    def validate_model_compatibility(self, model_bundle: Dict[str, Any], 
                                   feature_dim: int) -> bool:
        """
        Validate if a model is compatible with given feature dimensions
        
        Args:
            model_bundle: Loaded model bundle
            feature_dim: Expected feature dimension
            
        Returns:
            True if compatible
        """
        scaler = model_bundle['scaler']
        
        if hasattr(scaler, 'n_features_in_'):
            return scaler.n_features_in_ == feature_dim
        
        # Fallback: try to check model input shape
        model = model_bundle['model']
        if hasattr(model, 'n_features_'):
            return model.n_features_ == feature_dim
        
        print("Warning: Could not verify model compatibility")
        return True  # Assume compatible if we can't check
    
    def cleanup_old_models(self, keep_latest: int = 3) -> None:
        """
        Clean up old model files, keeping only the latest versions
        
        Args:
            keep_latest: Number of latest models to keep
        """
        bundles = self.list_available_models()
        
        if len(bundles) <= keep_latest:
            return
        
        # Sort by modification time
        bundle_times = []
        for bundle in bundles:
            bundle_path = os.path.join(self.models_dir, bundle)
            mtime = os.path.getmtime(bundle_path)
            bundle_times.append((bundle, mtime))
        
        bundle_times.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old bundles
        for bundle, _ in bundle_times[keep_latest:]:
            if bundle not in ['best_model', 'deployment_model']:  # Protect important models
                bundle_path = os.path.join(self.models_dir, bundle)
                import shutil
                shutil.rmtree(bundle_path)
                print(f"Removed old model: {bundle}")
    
    def print_model_debug_info(self):
        """
        Print detailed debug information about the model files and metadata
        Useful for troubleshooting model info display issues
        """
        print("\n=== MODEL DEBUG INFORMATION ===")
        
        # Check if models directory exists
        if not os.path.exists(self.models_dir):
            print(f"Models directory not found: {self.models_dir}")
            return
        
        print(f"Models directory: {self.models_dir}")
        
        # List all files in the directory
        print("\nFiles in directory:")
        all_files = os.listdir(self.models_dir)
        for file in all_files:
            file_path = os.path.join(self.models_dir, file)
            if os.path.isdir(file_path):
                print(f"  📁 {file}/")
            else:
                print(f"  📄 {file}")
        
        # Check for individual model components
        model_file = os.path.join(self.models_dir, 'best_model.pkl')
        scaler_file = os.path.join(self.models_dir, 'scaler.pkl')
        labels_file = os.path.join(self.models_dir, 'label_names.pkl')
        metadata_file = os.path.join(self.models_dir, 'model_metadata.pkl')
        
        print("\nIndividual component files:")
        print(f"  Model file exists: {os.path.exists(model_file)}")
        print(f"  Scaler file exists: {os.path.exists(scaler_file)}")
        print(f"  Labels file exists: {os.path.exists(labels_file)}")
        print(f"  Metadata file exists: {os.path.exists(metadata_file)}")
        
        # Check for model bundle
        bundle_dir = os.path.join(self.models_dir, 'best_model')
        bundle_model = os.path.join(bundle_dir, 'model.pkl')
        bundle_scaler = os.path.join(bundle_dir, 'scaler.pkl')
        bundle_labels = os.path.join(bundle_dir, 'label_names.pkl')
        bundle_metadata = os.path.join(bundle_dir, 'metadata.pkl')
        
        print("\nBundle component files:")
        print(f"  Bundle directory exists: {os.path.exists(bundle_dir)}")
        if os.path.exists(bundle_dir):
            print(f"  Bundle model file exists: {os.path.exists(bundle_model)}")
            print(f"  Bundle scaler file exists: {os.path.exists(bundle_scaler)}")
            print(f"  Bundle labels file exists: {os.path.exists(bundle_labels)}")
            print(f"  Bundle metadata file exists: {os.path.exists(bundle_metadata)}")
        
        # Try to load and print metadata
        print("\nMetadata content:")
        try:
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                print("  Individual metadata:")
                for key, value in metadata.items():
                    if key == 'speakers':
                        print(f"    {key}: {len(value)} speakers")
                    else:
                        print(f"    {key}: {value}")
        except Exception as e:
            print(f"  Error loading individual metadata: {e}")
        
        try:
            if os.path.exists(bundle_metadata):
                metadata = joblib.load(bundle_metadata)
                print("  Bundle metadata:")
                for key, value in metadata.items():
                    if key == 'speakers':
                        print(f"    {key}: {len(value)} speakers")
                    else:
                        print(f"    {key}: {value}")
        except Exception as e:
            print(f"  Error loading bundle metadata: {e}")
        
        # Try to load and print label names
        print("\nLabel names:")
        try:
            if os.path.exists(labels_file):
                labels = joblib.load(labels_file)
                print(f"  Individual labels: {len(labels)} speakers")
                if len(labels) > 0:
                    print(f"  First few speakers: {labels[:5]}")
        except Exception as e:
            print(f"  Error loading individual labels: {e}")
        
        print("\n=== END DEBUG INFO ===\n")

def main():
    """
    Test model utilities
    """
    # Initialize model manager
    from config import Config
    models_dir = Config.MODELS_DIR
    manager = ModelManager(models_dir)
    
    import sys
    
    # Check for debug flag
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        print("Running in debug mode")
        manager.print_model_debug_info()
        return
    
    # List available models
    models = manager.list_available_models()
    print(f"Available models: {models}")
    
    # Get model info if models exist
    if models:
        for model in models:
            info = manager.get_model_info(model)
            if info:
                print(f"\nModel: {model}")
                print(f"Type: {info['model_type']}")
                print(f"Accuracy: {info['accuracy']}")
                print(f"Speakers: {info['speakers']}")
                print(f"Number of speakers: {info['num_speakers']}")
    else:
        print("No trained models found. Train a model first using model_training.py")
    
    print("\nTo see detailed model information, run with --debug flag:")
    print("python model_utils.py --debug")

if __name__ == "__main__":
    main()
