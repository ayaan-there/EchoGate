"""
Model Training Module for Speaker Recognition
Trains and evaluates KNN, SVM, and CNN models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Tuple, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# TensorFlow for CNN (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. CNN model will be skipped.")

class SpeakerRecognitionTrainer:
    def __init__(self):
        """
        Initialize the trainer with models and preprocessing tools
        """
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None  # Store PCA object for dimensionality reduction
        self.is_fitted = False
        self.label_names = []
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, label_names: List[str], 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            X: Feature matrix
            y: Labels
            label_names: List of speaker names
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Store label names
        self.label_names = label_names
        
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # For very small datasets, use stratified split with adjusted test size
        if n_samples < 10:
            # Use 1 sample per class for testing if possible
            test_size_adjusted = max(1/n_samples * n_classes, 0.1)  # At least 10% or 1 per class
            print(f"⚠️ Small dataset detected ({n_samples} samples). Using test_size={test_size_adjusted:.2f}")
        else:
            test_size_adjusted = test_size
        
        try:
            # Use stratified split to ensure each class is represented in both train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_adjusted, random_state=random_state, 
                stratify=y
            )
        except ValueError as e:
            # If stratified split fails, try without stratification
            print(f"⚠️ Stratified split failed: {e}")
            print("Using random split instead...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_adjusted, random_state=random_state
            )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_knn(self, X_train: np.ndarray, y_train: np.ndarray, 
                 optimize: bool = True) -> KNeighborsClassifier:
        """
        Train K-Nearest Neighbors classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Trained KNN model
        """
        print("Training KNN model...")
        
        if optimize and len(X_train) > 50:
            # Enhanced hyperparameter optimization
            param_grid = {
                'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
                'weights': ['distance'],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            }
            
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_knn = grid_search.best_estimator_
            print(f"Best KNN parameters: {grid_search.best_params_}")
        else:
            # Default parameters for small datasets - adjusted for robustness
            best_knn = KNeighborsClassifier(
                n_neighbors=min(3, len(np.unique(y_train))),
                weights='distance',
                metric='cosine'
            )
            best_knn.fit(X_train, y_train)
        
        self.models['knn'] = best_knn
        return best_knn
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, 
                 optimize: bool = True) -> SVC:
        """
        Train Support Vector Machine classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Trained SVM model
        """
        print("Training SVM model...")
        
        if optimize and len(X_train) > 50:
            # Hyperparameter optimization - expanded grid for better search
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            svm = SVC(probability=True, random_state=42, class_weight='balanced')
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_svm = grid_search.best_estimator_
            print(f"Best SVM parameters: {grid_search.best_params_}")
        else:
            # Default parameters - modified for better generalization
            best_svm = SVC(probability=True, random_state=42, class_weight='balanced', 
                         C=1, gamma='scale', kernel='rbf')
            best_svm.fit(X_train, y_train)
        
        self.models['svm'] = best_svm
        return best_svm
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained Random Forest model
        """
        print("Training Random Forest model...")
        
        # More robust parameters for Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_features='sqrt',
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        self.models['random_forest'] = rf
        return rf
    
    def train_cnn(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray) -> Optional[Any]:
        """
        Train 1D CNN for speaker recognition
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Trained CNN model or None if TensorFlow not available
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available. Skipping CNN training.")
            return None
            
        print("Training CNN model...")
        
        # Prepare data for CNN
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        # Reshape for 1D CNN (add time dimension)
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, n_classes)
        y_test_cat = keras.utils.to_categorical(y_test, n_classes)
        
        # Build improved CNN model
        model = keras.Sequential([
            # Input layer
            layers.Conv1D(64, 5, activation='relu', input_shape=(n_features, 1), padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            # Middle layers with residual connections
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Learning rate scheduler for better convergence
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model with more epochs and callbacks
        history = model.fit(
            X_train_cnn, y_train_cat,
            epochs=100,  # Increased from 50
            batch_size=32,
            validation_data=(X_test_cnn, y_test_cat),
            callbacks=[lr_scheduler, early_stopping],
            verbose=0
        )
        
        self.models['cnn'] = model
        return model
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        results = {}
        
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            print(f"\nEvaluating {model_name.upper()} model...")
            
            if model_name == 'cnn' and TF_AVAILABLE:
                # CNN evaluation
                X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                y_pred_prob = model.predict(X_test_cnn, verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                # Sklearn models
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_names))
        
        self.is_fitted = True
        return results
    
    def plot_confusion_matrices(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Plot confusion matrices for all models
        
        Args:
            results: Evaluation results from evaluate_models
        """
        valid_results = {k: v for k, v in results.items() if 'confusion_matrix' in v}
        n_models = len(valid_results)
        
        if n_models == 0:
            return
            
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(valid_results.items()):
            cm = result['confusion_matrix']
            # Ensure cm is a proper 2D array before plotting
            if isinstance(cm, np.ndarray) and cm.ndim == 2:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.label_names, 
                           yticklabels=self.label_names,
                           ax=axes[idx])
                axes[idx].set_title(f'{model_name.upper()} Confusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            else:
                print(f"Warning: Invalid confusion matrix for {model_name}")
                axes[idx].text(0.5, 0.5, f'Invalid\nConfusion Matrix\nfor {model_name}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, results: Dict[str, Dict[str, float]]) -> Tuple[str, Any]:
        """
        Get the best performing model based on accuracy
        
        Args:
            results: Evaluation results
            
        Returns:
            Tuple of (model_name, model_object)
        """
        best_accuracy = 0
        best_model_name = None
        
        for model_name, result in results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model_name = model_name
        
        if best_model_name is None:
            # Return a default model if no best model found
            if self.models:
                first_model_name = list(self.models.keys())[0]
                return first_model_name, self.models[first_model_name]
            else:
                return "unknown", None
        
        return best_model_name, self.models[best_model_name]
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, label_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Train all models and return evaluation results
        
        Args:
            X: Feature matrix
            y: Labels
            label_names: List of speaker names
            
        Returns:
            Dictionary with evaluation results
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, label_names)
        
        # Train models
        self.train_knn(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        
        # Train CNN if TensorFlow is available
        if TF_AVAILABLE:
            self.train_cnn(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # Plot results
        self.plot_confusion_matrices(results)
        
        return results

    def save_best_model(self, results: Dict[str, Dict[str, float]], X: np.ndarray, label_names: List[str], pca=None) -> None:
        """
        Save the best performing model to the models directory
        
        Args:
            results: Evaluation results
            X: Feature matrix (for feature count)
            label_names: List of speaker names
            pca: PCA object used for dimensionality reduction (if any)
        """
        # Store PCA object
        self.pca = pca
        
        # Get best model
        best_model_name, best_model = self.get_best_model(results)
        
        if best_model is None:
            print("No model to save.")
            return
        
        # Create models directory if it doesn't exist
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the best model
        best_model_path = os.path.join(models_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        
        # Save scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save PCA if available
        if pca is not None:
            pca_path = os.path.join(models_dir, "pca.pkl")
            joblib.dump(pca, pca_path)
        
        # Save label names
        label_names_path = os.path.join(models_dir, "label_names.pkl")
        joblib.dump(label_names, label_names_path)
        
        # Create metadata
        metadata = {
            'model_type': best_model_name,
            'accuracy': results[best_model_name]['accuracy'],
            'f1_score': results[best_model_name]['f1_score'],
            'speakers': label_names,
            'num_features': X.shape[1] if len(X.shape) > 1 else len(X[0]),
            'num_speakers': len(label_names),
            'has_pca': pca is not None,
            'pca_components': pca.n_components_ if pca is not None else None
        }
        
        # Save metadata
        metadata_path = os.path.join(models_dir, "model_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        # Save to best_model directory as well
        best_model_dir = os.path.join(models_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        joblib.dump(best_model, os.path.join(best_model_dir, "model.pkl"))
        joblib.dump(self.scaler, os.path.join(best_model_dir, "scaler.pkl"))
        joblib.dump(label_names, os.path.join(best_model_dir, "label_names.pkl"))
        joblib.dump(metadata, os.path.join(best_model_dir, "metadata.pkl"))
        
        # Save PCA to best_model directory if available
        if pca is not None:
            joblib.dump(pca, os.path.join(best_model_dir, "pca.pkl"))
        
        print(f"\n✅ Best model saved: {best_model_name}")
        print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
        print(f"   Speakers: {len(label_names)}")
        print(f"   Features: {X.shape[1] if len(X.shape) > 1 else len(X[0])}")
        if pca is not None:
            print(f"   PCA Components: {pca.n_components_}")
        print(f"   Saved to: {models_dir}/")
    
def main():
    """
    Main function - kept for compatibility
    Use the Streamlit app for training: streamlit run streamlit_app.py
    """
    print("Speaker Recognition Model Training")
    print("=" * 40)
    print("For training models, please use the Streamlit web interface:")
    print("streamlit run streamlit_app.py")
    print()
    print("The web interface provides:")
    print("- Model training with real-time progress")
    print("- Dataset management") 
    print("- Model evaluation and comparison")
    print("- Deployment management")

if __name__ == "__main__":
    main()
