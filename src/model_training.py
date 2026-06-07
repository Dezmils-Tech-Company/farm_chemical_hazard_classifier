# retrain_full_model_fixed.py
"""Retrain model WITHOUT target leakage - CORRECT VERSION"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RETRAINING MODEL WITHOUT TARGET LEAKAGE")
print("="*70)

# Find the master_preprocessed.csv file
possible_paths = [
    Path('data/processed/master_preprocessed.csv'),
    Path('master_preprocessed.csv'),
]

df = None
for path in possible_paths:
    if path.exists():
        print(f"✅ Found file at: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print("❌ Could not find master_preprocessed.csv!")
    exit()

print(f"\n📊 Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# IMPORTANT: Remove target leakage columns
exclude_cols = [
    'chemical_name_clean', 'who_class_valid', 'who_class', 'name_clean', 
    'Unnamed: 0', 'chemical_name', 'who_description', 'xlogp_source',
    'who_class_numeric'  # <-- THIS IS LEAKAGE! Remove it
]

# Get only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"\n🔧 Using {len(feature_cols)} numeric features (no target leakage):")
print(f"   Features: {feature_cols[:15]}...")

# Prepare features and target
X = df[feature_cols].copy()
y = df['who_class_valid'].copy()

print(f"\n📊 Target distribution:")
print(y.value_counts())

# Handle missing values
print(f"\n🔄 Handling missing values...")
if X.isnull().sum().sum() > 0:
    print(f"   Missing values found: {X.isnull().sum().sum()}")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print(f"   Missing values after imputation: {X_imputed.isnull().sum().sum()}")
else:
    print(f"   No missing values found")
    X_imputed = X
    imputer = None

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Target encoding:")
for i, cls in enumerate(le.classes_):
    print(f"   {cls} → {i}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📊 Train set: {len(X_train)} chemicals")
print(f"   Test set: {len(X_test)} chemicals")

# Train model
print(f"\n🤖 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Test Accuracy: {accuracy:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
print(f"📊 CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Classification report
print(f"\n📈 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance
print(f"\n⭐ TOP 15 MOST IMPORTANT FEATURES:")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(15).iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"   {row['feature']:30s}: {row['importance']:.4f} {bar}")

# Test specific chemicals
print(f"\n🧪 TESTING SPECIFIC CHEMICALS:")
test_chemicals = ['glyphosate', 'chlorpyrifos', 'malathion', 'carbofuran', 'atrazine', 'mancozeb']

for chem in test_chemicals:
    chem_row = df[df['chemical_name_clean'].str.lower() == chem.lower()]
    
    if len(chem_row) > 0:
        features = chem_row[feature_cols].copy()
        
        if imputer is not None:
            features_imputed = pd.DataFrame(
                imputer.transform(features),
                columns=features.columns
            )
        else:
            features_imputed = features.fillna(0)
            
        features_scaled = scaler.transform(features_imputed)
        pred_encoded = model.predict(features_scaled)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        actual = chem_row['who_class_valid'].iloc[0]
        
        proba = model.predict_proba(features_scaled)[0]
        confidence = max(proba) * 100
        
        status = "✅" if pred_class == actual else "⚠️"
        print(f"   {status} {chem}: Predicted {pred_class} (Actual {actual}) - {confidence:.0f}%")

# Save all artifacts
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / 'toxicity_model.pkl')
joblib.dump(scaler, models_dir / 'scaler.pkl')
joblib.dump(le, models_dir / 'label_encoder.pkl')
joblib.dump(feature_cols, models_dir / 'feature_columns.pkl')
if imputer is not None:
    joblib.dump(imputer, models_dir / 'imputer.pkl')
else:
    # Create dummy imputer
    dummy_imputer = SimpleImputer(strategy='median')
    dummy_imputer.fit(X)
    joblib.dump(dummy_imputer, models_dir / 'imputer.pkl')

print(f"\n💾 Model and artifacts saved to {models_dir}")
print(f"\n✅ Model retrained without target leakage!")

# Save feature summary
feature_summary = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
feature_summary.to_csv('models/feature_summary.csv', index=False)
print(f"   Feature summary saved to models/feature_summary.csv")