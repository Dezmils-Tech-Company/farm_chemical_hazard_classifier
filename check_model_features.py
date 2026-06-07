# diagnose_prediction.py
"""Diagnose why prediction is failing"""

import pandas as pd
import joblib
from pathlib import Path

print("="*60)
print("DIAGNOSING PREDICTION ISSUE")
print("="*60)

# Load model features
feature_cols = joblib.load('models/feature_columns.pkl')
print(f"\nModel expects {len(feature_cols)} features:")
print(feature_cols[:10])

# Load database
db = pd.read_csv('data/processed/master_preprocessed.csv')
print(f"\nDatabase has {len(db.columns)} columns")
print(f"Database columns: {db.columns.tolist()[:20]}...")

# Check if all features are in database
missing_features = [col for col in feature_cols if col not in db.columns]
if missing_features:
    print(f"\n❌ Missing features in database: {missing_features}")
else:
    print(f"\n✅ All features found in database!")

# Test a specific chemical
test_chem = 'glyphosate'
chem_row = db[db['chemical_name_clean'] == test_chem]
print(f"\nChemical '{test_chem}' found: {len(chem_row) > 0}")

if len(chem_row) > 0:
    print(f"\nFeatures for {test_chem}:")
    for col in feature_cols[:10]:
        if col in chem_row.columns:
            val = chem_row[col].iloc[0]
            print(f"   {col}: {val}")
        else:
            print(f"   {col}: NOT FOUND!")