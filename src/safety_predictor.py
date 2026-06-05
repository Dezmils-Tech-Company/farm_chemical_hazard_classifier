"""
src/safety_predictor.py
Step 3: Predict chemical safety using trained model
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class SafetyPredictor:
    """Predict WHO hazard class for any chemical"""
    
    def __init__(self, model_path="models"):
        self.project_root = Path.cwd()
        self.model_path = self.project_root / model_path
        
        # Load model and artifacts
        print("📦 Loading Model 2 (Toxicity Classifier)...")
        self.model = joblib.load(self.model_path / 'toxicity_model.pkl')
        self.scaler = joblib.load(self.model_path / 'scaler.pkl')
        self.label_encoder = joblib.load(self.model_path / 'label_encoder.pkl')
        self.feature_cols = joblib.load(self.model_path / 'feature_columns.pkl')
        
        # Load chemical database
        self.db_path = self.project_root / 'data/processed/master_dataset.csv'
        if self.db_path.exists():
            self.database = pd.read_csv(self.db_path)
        else:
            self.database = None
            
        print("   ✅ Model loaded successfully")
    
    def _get_safety_info(self, who_class):
        """Convert WHO class to farmer-friendly information"""
        
        safety_info = {
            'U': {
                'level': 'SAFE',
                'message': '✅ SAFE - Can be used with standard precautions',
                'ppe': 'Gloves, wash hands after use',
                'action': 'APPROVED',
                'color': 'green',
                'reentry_hours': 4
            },
            'III': {
                'level': 'CAUTION',
                'message': '⚠️ CAUTION - Low toxicity, use with care',
                'ppe': 'Long sleeves, gloves, wash after use',
                'action': 'CAUTION',
                'color': 'yellow',
                'reentry_hours': 12
            },
            'II': {
                'level': 'WARNING',
                'message': '⚠️ WARNING - Moderately hazardous, requires protective equipment',
                'ppe': 'Coveralls, chemical gloves, N95 mask, boots',
                'action': 'WARNING',
                'color': 'orange',
                'reentry_hours': 24
            },
            'Ib': {
                'level': 'BLOCKED',
                'message': '🔴 BLOCKED - Highly hazardous, DO NOT USE',
                'ppe': 'Full protective gear required',
                'action': 'BLOCKED',
                'color': 'red',
                'reentry_hours': 48
            },
            'Ia': {
                'level': 'BLOCKED',
                'message': '🔴 BLOCKED - Extremely hazardous, DO NOT USE',
                'ppe': 'Full chemical suit + respirator',
                'action': 'BLOCKED',
                'color': 'red',
                'reentry_hours': 72
            }
        }
        
        return safety_info.get(who_class, {
            'level': 'UNKNOWN',
            'message': '⚠️ Unknown hazard level - consult official sources',
            'ppe': 'Consult label',
            'action': 'UNKNOWN',
            'color': 'gray',
            'reentry_hours': 24
        })
    
    def predict(self, chemical_name):
        """Predict safety for a single chemical"""
        
        if self.database is None:
            return {'error': True, 'message': 'Database not loaded'}
        
        # Find chemical in database
        chem_row = self.database[self.database['chemical_name'].str.lower() == chemical_name.lower()]
        
        if len(chem_row) == 0 and 'name_clean' in self.database.columns:
            chem_row = self.database[self.database['name_clean'] == chemical_name.lower()]
        
        if len(chem_row) == 0:
            return {
                'error': True,
                'chemical': chemical_name,
                'message': f"Chemical '{chemical_name}' not found in database"
            }
        
        # Prepare features
        features = chem_row[self.feature_cols].fillna(0)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        pred_encoded = self.model.predict(features_scaled)[0]
        pred_class = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
        
        # Get safety info
        safety = self._get_safety_info(pred_class)
        
        # Get additional properties
        ld50 = None
        if 'ld50_rat_oral_mgkg' in chem_row.columns:
            ld50_val = chem_row['ld50_rat_oral_mgkg'].iloc[0]
            ld50 = float(ld50_val) if pd.notna(ld50_val) else None
        
        return {
            'error': False,
            'chemical': chemical_name,
            'who_class': pred_class,
            'confidence': confidence,
            'safety_level': safety['level'],
            'farmer_message': safety['message'],
            'ppe_required': safety['ppe'],
            'action': safety['action'],
            'color': safety['color'],
            'reentry_hours': safety['reentry_hours'],
            'ld50_oral_mgkg': ld50
        }
    
    def predict_batch(self, chemical_list):
        """Predict for multiple chemicals"""
        results = []
        for chem in chemical_list:
            results.append(self.predict(chem))
        return pd.DataFrame(results)
    
    def compare(self, chemical1, chemical2):
        """Compare two chemicals"""
        pred1 = self.predict(chemical1)
        pred2 = self.predict(chemical2)
        
        print(f"\n{'='*50}")
        print(f"SAFETY COMPARISON")
        print(f"{'='*50}")
        print(f"\n{chemical1.upper()}:")
        print(f"   Class: {pred1.get('who_class', 'N/A')}")
        print(f"   Safety: {pred1.get('safety_level', 'N/A')}")
        print(f"   Action: {pred1.get('action', 'N/A')}")
        
        print(f"\n{chemical2.upper()}:")
        print(f"   Class: {pred2.get('who_class', 'N/A')}")
        print(f"   Safety: {pred2.get('safety_level', 'N/A')}")
        print(f"   Action: {pred2.get('action', 'N/A')}")
        
        # Recommend safer option
        rank = {'U': 0, 'III': 1, 'II': 2, 'Ib': 3, 'Ia': 4}
        rank1 = rank.get(pred1.get('who_class'), 5)
        rank2 = rank.get(pred2.get('who_class'), 5)
        
        if rank1 < rank2:
            print(f"\n✅ Recommendation: {chemical1} is SAFER than {chemical2}")
        elif rank2 < rank1:
            print(f"\n✅ Recommendation: {chemical2} is SAFER than {chemical1}")
        else:
            print(f"\n⚠️ Both have similar hazard levels")
        
        return pred1, pred2


if __name__ == "__main__":
    predictor = SafetyPredictor()
    
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    test_chemicals = ['glyphosate', 'chlorpyrifos', 'malathion', 'carbofuran']
    
    for chem in test_chemicals:
        result = predictor.predict(chem)
        if not result['error']:
            print(f"\n🔬 {chem.upper()}")
            print(f"   WHO Class: {result['who_class']} ({result['confidence']:.0f}% confidence)")
            print(f"   {result['farmer_message']}")
            print(f"   PPE: {result['ppe_required']}")