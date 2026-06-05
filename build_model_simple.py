# build_model_simple.py
# Save this in: C:\Users\Dezmil\Downloads\agri_chem_safety_thesis_dataset\agri_chem_safety\
# (the same folder as your 'data' folder)

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("SIMPLE DATA PREPROCESSING & MODEL BUILDING")
print("="*70)

# Set the correct path - FIXED
data_dir = Path.cwd() / 'data' / 'raw'
print(f"\nLooking for data in: {data_dir}")

if not data_dir.exists():
    print(f"ERROR: Data directory not found!")
    print(f"Please make sure you're running this from: {Path.cwd()}")
    exit()

# Load data
print("\n📂 Loading files...")
who_df = pd.read_csv(data_dir / 'who_hazard_classifications.csv')
tox_df = pd.read_csv(data_dir / 'toxicity_flags.csv')
pubchem_df = pd.read_csv(data_dir / 'pubchem_properties.csv')

print(f"   WHO: {len(who_df)} chemicals")
print(f"   Toxicity: {len(tox_df)} chemicals")
print(f"   PubChem: {len(pubchem_df)} chemicals")

# Clean chemical names for merging
def clean_name(name):
    return str(name).lower().strip().replace(' ', '_')

who_df['name_clean'] = who_df['chemical_name'].apply(clean_name)
tox_df['name_clean'] = tox_df['chemical_name'].apply(clean_name)
pubchem_df['name_clean'] = pubchem_df['query_name'].apply(clean_name)

# Merge datasets
print("\n🔗 Merging datasets...")
master = who_df.merge(tox_df[['name_clean', 'is_carcinogen', 'is_neurotoxin', 
                               'is_endocrine_disruptor', 'is_reproductive_toxin', 
                               'is_pbt', 'total_hazard_flags']], 
                      on='name_clean', how='left')

master = master.merge(pubchem_df[['name_clean', 'molecularweight', 'xlogp', 'tpsa', 
                                   'complexity', 'hbonddonorcount', 'hbondacceptorcount']], 
                      on='name_clean', how='left')

print(f"   Master dataset: {len(master)} rows")

# Fill missing values
print("\n🔄 Handling missing values...")
master['is_carcinogen'] = master['is_carcinogen'].fillna(0)
master['is_neurotoxin'] = master['is_neurotoxin'].fillna(0)
master['is_endocrine_disruptor'] = master['is_endocrine_disruptor'].fillna(0)
master['is_reproductive_toxin'] = master['is_reproductive_toxin'].fillna(0)
master['total_hazard_flags'] = master['total_hazard_flags'].fillna(0)

# Fill numeric properties with median
numeric_cols = ['molecularweight', 'xlogp', 'tpsa', 'complexity', 
                'hbonddonorcount', 'hbondacceptorcount', 'ld50_rat_oral_mgkg']
for col in numeric_cols:
    if col in master.columns:
        master[col] = master[col].fillna(master[col].median())

# Create features
print("\n🔧 Creating features...")
master['name_length'] = master['name_clean'].str.len()
master['has_number'] = master['name_clean'].str.contains(r'\d').astype(int)
master['has_hyphen'] = master['name_clean'].str.contains('-').astype(int)

# Prepare for modeling
feature_cols = ['molecularweight', 'xlogp', 'tpsa', 'complexity',
                'hbonddonorcount', 'hbondacceptorcount', 'ld50_rat_oral_mgkg',
                'total_hazard_flags', 'is_carcinogen', 'is_neurotoxin',
                'is_endocrine_disruptor', 'is_reproductive_toxin',
                'name_length', 'has_number', 'has_hyphen']

# Keep only columns that exist
feature_cols = [col for col in feature_cols if col in master.columns]

X = master[feature_cols].fillna(0)
y = master['who_class']

# Remove unknowns
valid_mask = y.isin(['U', 'III', 'II', 'Ib', 'Ia'])
X = X[valid_mask]
y = y[valid_mask]

print(f"\n📊 Final dataset: {len(X)} chemicals")
print(f"   Features: {len(feature_cols)}")
print(f"   Target distribution:")
print(y.value_counts())

# Train model
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Test Accuracy: {accuracy:.3f}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance
print(f"\n⭐ Most Important Features:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in importance.head(8).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Save model
import joblib
Path('models').mkdir(exist_ok=True)
joblib.dump(model, 'models/toxicity_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print(f"\n💾 Model saved to models/")

# Test predictions
print("\n" + "="*70)
print("TESTING PREDICTIONS")
print("="*70)

test_chemicals = ['glyphosate', 'chlorpyrifos', 'malathion', 'carbofuran']
for chem in test_chemicals:
    chem_data = master[master['chemical_name'].str.lower() == chem]
    if len(chem_data) > 0:
        features = chem_data[feature_cols].fillna(0)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        pred_class = le.inverse_transform([pred])[0]
        actual = chem_data['who_class'].values[0]
        
        # Get probability
        proba = model.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
        
        # Safety message
        safety_msg = {
            'U': '✅ SAFE - Can be used',
            'III': '⚠️ CAUTION - Use with care',
            'II': '⚠️ WARNING - Requires protective equipment',
            'Ib': '🔴 BLOCKED - Highly hazardous',
            'Ia': '🔴 BLOCKED - Extremely hazardous'
        }
        
        print(f"\n🔬 {chem.upper()}")
        print(f"   Predicted: {pred_class} (confidence: {confidence:.1f}%)")
        print(f"   Actual: {actual}")
        print(f"   Farmer message: {safety_msg.get(pred_class, 'Unknown')}")
        print(f"   {'✅ CORRECT' if pred_class == actual else '❌ INCORRECT'}")

print("\n" + "="*70)
print("✅ MODEL 2 COMPLETE!")
print("="*70)