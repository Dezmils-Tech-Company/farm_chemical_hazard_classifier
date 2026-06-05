"""
src/model_training.py
Step 2: Train the WHO toxicity classifier
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class ToxicityModelTrainer:
    """Train and save the toxicity classification model"""
    
    def __init__(self, data_path="data/processed/master_dataset.csv"):
        self.project_root = Path.cwd()
        self.data_path = self.project_root / data_path
        self.models_path = self.project_root / "models"
        self.models_path.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = None
        
    def load_data(self):
        """Load preprocessed master dataset"""
        print("📂 Loading preprocessed data...")
        df = pd.read_csv(self.data_path)
        print(f"   ✅ Loaded {len(df)} chemicals")
        return df
    
    def prepare_features(self, df):
        """Select features and target"""
        
        # Define feature columns
        feature_cols = [
            'molecularweight', 'xlogp', 'tpsa', 'complexity',
            'hbonddonorcount', 'hbondacceptorcount', 'ld50_rat_oral_mgkg',
            'total_hazard_flags', 'is_carcinogen', 'is_neurotoxin',
            'is_endocrine_disruptor', 'is_reproductive_toxin',
            'name_length', 'has_number', 'has_hyphen'
        ]
        
        # Keep only columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].fillna(0)
        y = df['who_class']
        
        # Filter to known classes only
        valid_classes = ['U', 'III', 'II', 'Ib', 'Ia']
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]
        
        self.feature_cols = feature_cols
        
        print(f"\n📊 Dataset ready:")
        print(f"   Chemicals: {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Classes: {sorted(y.unique())}")
        
        return X, y
    
    def train(self, X, y, model_type='random_forest'):
        """Train the model"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {len(X_train)} chemicals")
        print(f"Test set: {len(X_test)} chemicals")
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42, 
                class_weight='balanced'
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                random_state=42
            )
        
        # Train
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
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n⭐ Top 8 Features:")
            importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in importance.head(8).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return accuracy
    
    def save(self):
        """Save model and artifacts"""
        joblib.dump(self.model, self.models_path / 'toxicity_model.pkl')
        joblib.dump(self.scaler, self.models_path / 'scaler.pkl')
        joblib.dump(self.label_encoder, self.models_path / 'label_encoder.pkl')
        joblib.dump(self.feature_cols, self.models_path / 'feature_columns.pkl')
        
        print(f"\n💾 Model saved to {self.models_path}")
    
    def run(self):
        """Execute full training pipeline"""
        df = self.load_data()
        X, y = self.prepare_features(df)
        self.train(X, y)
        self.save()
        return self.model


if __name__ == "__main__":
    trainer = ToxicityModelTrainer()
    trainer.run()
    print("\n✅ Model training complete!")