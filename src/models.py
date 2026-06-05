# src/models.py
"""Model definitions for toxicity classification"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

class ToxicityClassifier:
    """WHO Hazard Class predictor for agricultural chemicals"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.is_trained = False
        
    def _get_model(self):
        """Initialize the chosen model"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the toxicity classifier"""
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_type.upper()} CLASSIFIER")
        print(f"{'='*60}")
        
        self.feature_cols = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {len(X_train)} chemicals")
        print(f"Test set: {len(X_test)} chemicals")
        print(f"Classes: {self.label_encoder.classes_.tolist()}")
        
        # Train model
        self.model = self._get_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Test Accuracy: {accuracy:.3f}")
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
        print(f"\n📊 5-Fold CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        self.is_trained = True
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n⭐ Top 5 Features:")
            for i, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': importance_df if hasattr(self.model, 'feature_importances_') else None
        }
    
    def predict(self, chemical_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict WHO class for new chemicals"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Ensure features match training
        chemical_features = chemical_features[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(chemical_features)
        
        # Predict
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        return y_pred, probabilities
    
    def predict_single(self, features: pd.Series) -> Dict[str, Any]:
        """Predict for a single chemical"""
        features_df = pd.DataFrame([features])[self.feature_cols]
        pred, probs = self.predict(features_df)
        
        result = {
            'predicted_class': pred[0],
            'confidence': max(probs[0]) if probs is not None else None,
            'all_probabilities': dict(zip(self.label_encoder.classes_, probs[0])) if probs is not None else None
        }
        
        return result
    
    def save(self, path: str = "models/saved"):
        """Save model and artifacts"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, path / 'model.pkl')
        joblib.dump(self.scaler, path / 'scaler.pkl')
        joblib.dump(self.label_encoder, path / 'label_encoder.pkl')
        joblib.dump(self.feature_cols, path / 'feature_columns.pkl')
        
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str = "models/saved"):
        """Load saved model"""
        path = Path(path)
        
        self.model = joblib.load(path / 'model.pkl')
        self.scaler = joblib.load(path / 'scaler.pkl')
        self.label_encoder = joblib.load(path / 'label_encoder.pkl')
        self.feature_cols = joblib.load(path / 'feature_columns.pkl')
        self.is_trained = True
        
        print(f"✅ Model loaded from {path}")
        return self