# src/predict.py
"""Inference pipeline for chemical safety prediction"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import sys

# Add src to path if running directly
sys.path.append(str(Path(__file__).parent))

from data_utils import DataLoader
from models import ToxicityClassifier

class SafetyPredictor:
    """Complete safety prediction pipeline for farmers"""
    
    def __init__(self, model_path: str = "models/saved"):
        self.project_root = Path.cwd()
        if self.project_root.name == "src":
            self.project_root = self.project_root.parent
            
        self.model_path = self.project_root / model_path
        self.loader = DataLoader()
        self.model = ToxicityClassifier()
        
        # Load model if it exists
        if self.model_path.exists():
            self.model.load(str(self.model_path))
        else:
            print("⚠️ Model not found. Train first using notebooks/04_baseline_models.ipynb")
    
    def predict_chemical(self, chemical_name: str) -> Dict:
        """
        Predict safety for a single chemical
        
        Args:
            chemical_name: e.g., "glyphosate", "chlorpyrifos"
            
        Returns:
            Dictionary with safety information
        """
        # Load master dataset to get features
        master = pd.read_csv(self.project_root / 'data' / 'processed' / 'master_dataset.csv')
        
        # Find chemical
        chem_row = master[master['chemical_name_clean'] == chemical_name.lower()]
        
        if len(chem_row) == 0:
            return {
                'error': True,
                'chemical': chemical_name,
                'message': f"Chemical '{chemical_name}' not found in database"
            }
        
        # Get features
        _, _, feature_cols = self.loader.get_features_and_target(master)
        features = chem_row[feature_cols].fillna(0)
        
        # Predict
        prediction = self.model.predict_single(features.iloc[0])
        
        # Add safety information
        safety_info = self._get_safety_info(prediction['predicted_class'])
        
        # Add chemical properties
        result = {
            'error': False,
            'chemical': chemical_name,
            'predicted_who_class': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'safety_level': safety_info['level'],
            'farmer_message': safety_info['message'],
            'ppe_required': safety_info['ppe'],
            'reentry_hours': safety_info['reentry_hours'],
            'is_blocked': safety_info['is_blocked'],
            'ld50_oral': float(chem_row['ld50_rat_oral_mgkg'].iloc[0]) if pd.notna(chem_row['ld50_rat_oral_mgkg'].iloc[0]) else None,
            'has_carcinogen_flag': bool(chem_row['is_carcinogen'].iloc[0]) if 'is_carcinogen' in chem_row else False,
            'has_neurotoxin_flag': bool(chem_row['is_neurotoxin'].iloc[0]) if 'is_neurotoxin' in chem_row else False,
        }
        
        return result
    
    def _get_safety_info(self, who_class: str) -> Dict:
        """Get farmer-friendly safety information"""
        
        safety_info = {
            'Ia': {
                'level': 'EXTREMELY HAZARDOUS',
                'message': '🔴 DO NOT USE - This chemical is deadly even in small amounts',
                'ppe': 'Full chemical suit + respirator (not recommended for farm use)',
                'reentry_hours': 72,
                'is_blocked': True
            },
            'Ib': {
                'level': 'HIGHLY HAZARDOUS',
                'message': '🔴 DO NOT USE - This chemical can cause death',
                'ppe': 'Chemical suit + respirator (not recommended for farm use)',
                'reentry_hours': 48,
                'is_blocked': True
            },
            'II': {
                'level': 'MODERATELY HAZARDOUS',
                'message': '⚠️ WARNING - Use with extreme caution. Wear protective equipment.',
                'ppe': 'Coveralls, chemical-resistant gloves, N95 mask, boots',
                'reentry_hours': 24,
                'is_blocked': False
            },
            'III': {
                'level': 'SLIGHTLY HAZARDOUS',
                'message': '⚠️ CAUTION - Generally safe but follow label instructions',
                'ppe': 'Long sleeves, gloves, wash after use',
                'reentry_hours': 12,
                'is_blocked': False
            },
            'U': {
                'level': 'UNLIKELY HAZARDOUS',
                'message': '✅ SAFE - Approved for use with standard precautions',
                'ppe': 'Standard hygiene, wash hands after use',
                'reentry_hours': 4,
                'is_blocked': False
            }
        }
        
        return safety_info.get(who_class, {
            'level': 'UNKNOWN',
            'message': '⚠️ Hazard level unknown - consult official sources',
            'ppe': 'Follow label instructions',
            'reentry_hours': 24,
            'is_blocked': False
        })
    
    def compare_chemicals(self, chemical1: str, chemical2: str) -> pd.DataFrame:
        """Compare two chemicals side by side"""
        pred1 = self.predict_chemical(chemical1)
        pred2 = self.predict_chemical(chemical2)
        
        comparison = pd.DataFrame({
            'Property': ['WHO Class', 'Safety Level', 'Confidence', 'PPE Required', 'Re-entry Hours', 'Blocked'],
            chemical1: [
                pred1.get('predicted_who_class', 'N/A'),
                pred1.get('safety_level', 'N/A'),
                f"{pred1.get('confidence', 0)*100:.1f}%" if pred1.get('confidence') else 'N/A',
                pred1.get('ppe_required', 'N/A'),
                pred1.get('reentry_hours', 'N/A'),
                '✅ Yes' if pred1.get('is_blocked') else '❌ No'
            ],
            chemical2: [
                pred2.get('predicted_who_class', 'N/A'),
                pred2.get('safety_level', 'N/A'),
                f"{pred2.get('confidence', 0)*100:.1f}%" if pred2.get('confidence') else 'N/A',
                pred2.get('ppe_required', 'N/A'),
                pred2.get('reentry_hours', 'N/A'),
                '✅ Yes' if pred2.get('is_blocked') else '❌ No'
            ]
        })
        
        return comparison
    
    def batch_predict(self, chemical_list: List[str]) -> pd.DataFrame:
        """Predict for multiple chemicals"""
        results = []
        for chem in chemical_list:
            results.append(self.predict_chemical(chem))
        
        df = pd.DataFrame(results)
        if 'error' in df.columns:
            df = df[~df['error']]
        
        return df[['chemical', 'predicted_who_class', 'safety_level', 'confidence', 'is_blocked']]


# CLI interface
if __name__ == "__main__":
    print("="*60)
    print("CHEMICAL SAFETY PREDICTOR")
    print("="*60)
    
    predictor = SafetyPredictor()
    
    # Test predictions
    test_chemicals = ['glyphosate', 'chlorpyrifos', 'mancozeb', 'carbofuran']
    
    for chem in test_chemicals:
        result = predictor.predict_chemical(chem)
        if not result['error']:
            print(f"\n🧪 {chem.upper()}")
            print(f"   WHO Class: {result['predicted_who_class']} ({result['confidence']:.1%} confidence)")
            print(f"   {result['farmer_message']}")
            print(f"   PPE: {result['ppe_required']}")