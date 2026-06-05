# 01_preprocess_data.py
# Run this from the scripts folder

import pandas as pd
import numpy as np
from pathlib import Path
import os

print("="*70)
print("DATA PREPROCESSING PIPELINE")
print("="*70)

# Get the project root directory (parent of scripts folder)
current_dir = Path.cwd()
print(f"\nCurrent directory: {current_dir}")

# If we're in scripts folder, go up one level to project root
if current_dir.name == 'scripts':
    project_root = current_dir.parent
    print(f"Detected scripts folder → Project root: {project_root}")
else:
    project_root = current_dir
    print(f"Project root: {project_root}")

# Define data paths
data_dir = project_root / 'data' / 'raw'
processed_dir = project_root / 'data' / 'processed'

# Check if paths exist
print(f"\nChecking data path: {data_dir}")
if not data_dir.exists():
    print(f"❌ Data directory not found!")
    print(f"   Looking for: {data_dir}")
    print(f"\n   Please make sure your folder structure is:")
    print(f"   {project_root}/")
    print(f"   ├── data/")
    print(f"   │   ├── raw/")
    print(f"   │   │   ├── who_hazard_classifications.csv")
    print(f"   │   │   ├── toxicity_flags.csv")
    print(f"   │   │   ├── pubchem_properties.csv")
    print(f"   │   │   └── comptox_identifiers.csv")
    print(f"   └── scripts/")
    print(f"       └── 01_preprocess_data.py (this file)")
    exit()

# List files to confirm
print(f"\n📂 Files found in {data_dir}:")
for file in data_dir.glob('*.csv'):
    print(f"   ✅ {file.name}")

# Load all raw data files
print("\n📂 Loading files...")
who_df = pd.read_csv(data_dir / 'who_hazard_classifications.csv')
tox_df = pd.read_csv(data_dir / 'toxicity_flags.csv')
pubchem_df = pd.read_csv(data_dir / 'pubchem_properties.csv')
comptox_df = pd.read_csv(data_dir / 'comptox_identifiers.csv')

print(f"   WHO: {len(who_df)} rows")
print(f"   Toxicity: {len(tox_df)} rows")
print(f"   PubChem: {len(pubchem_df)} rows")
print(f"   CompTox: {len(comptox_df)} rows")

# ============================================
# 1. CLEAN WHO DATA
# ============================================
print("\n" + "="*50)
print("1. CLEANING WHO DATA")
print("="*50)

# Standardize chemical names
who_df['chemical_name_clean'] = who_df['chemical_name'].str.lower().str.strip()
who_df['chemical_name_clean'] = who_df['chemical_name_clean'].str.replace(r'[^\w\s-]', '', regex=True)

# Ensure WHO class is standardized
who_df['who_class'] = who_df['who_class'].astype(str).str.upper()

# Valid WHO classes
valid_classes = ['IA', 'IB', 'II', 'III', 'U', 'O']
who_df['who_class_valid'] = who_df['who_class'].apply(
    lambda x: x if x in valid_classes else 'UNKNOWN'
)

# Handle LD50 (oral toxicity)
who_df['ld50_rat_oral_mgkg'] = pd.to_numeric(who_df['ld50_rat_oral_mgkg'], errors='coerce')

print(f"   Cleaned {len(who_df)} chemicals")
print(f"   WHO classes: {sorted(who_df['who_class_valid'].unique())}")
print(f"   LD50 range: {who_df['ld50_rat_oral_mgkg'].min():.1f} - {who_df['ld50_rat_oral_mgkg'].max():.1f} mg/kg")

# ============================================
# 2. CLEAN TOXICITY FLAGS
# ============================================
print("\n" + "="*50)
print("2. CLEANING TOXICITY FLAGS")
print("="*50)

# Standardize chemical names
tox_df['chemical_name_clean'] = tox_df['chemical_name'].str.lower().str.strip()

# Ensure flags are binary (0/1)
flag_columns = ['is_carcinogen', 'is_endocrine_disruptor', 'is_neurotoxin', 
                'is_reproductive_toxin', 'is_pbt']

for col in flag_columns:
    if col in tox_df.columns:
        tox_df[col] = pd.to_numeric(tox_df[col], errors='coerce').fillna(0).astype(int)

# Recalculate total hazard flags
if all(col in tox_df.columns for col in flag_columns):
    tox_df['total_hazard_flags'] = tox_df[flag_columns].sum(axis=1)

print(f"   Cleaned {len(tox_df)} chemicals")
print(f"   Chemicals with flags: {(tox_df['total_hazard_flags'] > 0).sum()}")
print(f"   Average flags per chemical: {tox_df['total_hazard_flags'].mean():.2f}")

# ============================================
# 3. CLEAN PUBCHEM PROPERTIES
# ============================================
print("\n" + "="*50)
print("3. CLEANING PUBCHEM PROPERTIES")
print("="*50)

# Rename query_name to chemical_name
pubchem_df = pubchem_df.rename(columns={'query_name': 'chemical_name'})
pubchem_df['chemical_name_clean'] = pubchem_df['chemical_name'].str.lower().str.strip()

# Convert numeric columns
numeric_pubchem = ['cid', 'molecularweight', 'xlogp', 'tpsa', 'complexity', 
                   'hbonddonorcount', 'hbondacceptorcount', 'rotatablebondcount', 
                   'heavyatomcount', 'charge']

for col in numeric_pubchem:
    if col in pubchem_df.columns:
        pubchem_df[col] = pd.to_numeric(pubchem_df[col], errors='coerce')

# Handle special values
pubchem_df['xlogp'] = pubchem_df['xlogp'].replace([-999, -999.0], np.nan)

print(f"   Cleaned {len(pubchem_df)} chemicals")
print(f"   Properties available: {len(numeric_pubchem)} numeric columns")

# ============================================
# 4. MERGE ALL DATASETS
# ============================================
print("\n" + "="*50)
print("4. MERGING DATASETS")
print("="*50)

# Start with WHO data as base
master = who_df.copy()
print(f"   Base: {len(master)} chemicals")

# Merge toxicity flags
master = master.merge(
    tox_df[['chemical_name_clean', 'is_carcinogen', 'is_endocrine_disruptor', 
            'is_neurotoxin', 'is_reproductive_toxin', 'is_pbt', 'total_hazard_flags']],
    on='chemical_name_clean',
    how='left'
)
print(f"   After toxicity merge: {len(master)} chemicals")

# Merge PubChem properties
master = master.merge(
    pubchem_df[['chemical_name_clean', 'cid', 'molecularweight', 'xlogp', 'tpsa', 
                'complexity', 'hbonddonorcount', 'hbondacceptorcount', 
                'rotatablebondcount', 'heavyatomcount', 'charge']],
    on='chemical_name_clean',
    how='left'
)
print(f"   After PubChem merge: {len(master)} chemicals")

# ============================================
# 5. FEATURE ENGINEERING
# ============================================
print("\n" + "="*50)
print("5. FEATURE ENGINEERING")
print("="*50)

# Create chemical name features
master['name_length'] = master['chemical_name_clean'].str.len()
master['word_count'] = master['chemical_name_clean'].str.split().str.len()
master['has_number'] = master['chemical_name_clean'].str.contains(r'\d').astype(int)
master['has_hyphen'] = master['chemical_name_clean'].str.contains('-').astype(int)
master['has_parenthesis'] = master['chemical_name_clean'].str.contains('[()]').astype(int)
master['special_chars'] = master['chemical_name_clean'].str.count(r'[^a-z0-9]')

# Create log-transformed features for skewed data
master['log_molecularweight'] = np.log1p(master['molecularweight'])
master['log_ld50'] = np.log1p(master['ld50_rat_oral_mgkg'])

# Create interaction features
master['toxicity_weighted'] = master['total_hazard_flags'] * master['log_ld50'].fillna(0)

print(f"   Added 10 engineered features")
print(f"   Total features: {len(master.columns)}")

# ============================================
# 6. HANDLE MISSING VALUES
# ============================================
print("\n" + "="*50)
print("6. HANDLING MISSING VALUES")
print("="*50)

# Check missing values before imputation
missing_before = master.isnull().sum()
missing_cols = missing_before[missing_before > 0]
if len(missing_cols) > 0:
    print(f"\n   Missing values before imputation:")
    for col, count in missing_cols.items():
        print(f"      {col}: {count} ({count/len(master)*100:.1f}%)")

# Apply imputation
for col in master.columns:
    if master[col].dtype in ['float64', 'int64'] and master[col].isnull().any():
        if col in ['is_carcinogen', 'is_endocrine_disruptor', 'is_neurotoxin', 
                   'is_reproductive_toxin', 'is_pbt', 'total_hazard_flags']:
            master[col] = master[col].fillna(0)
            print(f"   {col}: filled with 0")
        else:
            median_val = master[col].median()
            master[col] = master[col].fillna(median_val)
            print(f"   {col}: filled with median ({median_val:.2f})")

print(f"\n   Missing values after imputation: {master.isnull().sum().sum()}")

# ============================================
# 7. REMOVE DUPLICATES
# ============================================
print("\n" + "="*50)
print("7. REMOVING DUPLICATES")
print("="*50)

before = len(master)
master = master.drop_duplicates(subset=['chemical_name_clean'], keep='first')
after = len(master)
print(f"   Removed {before - after} duplicate chemicals")
print(f"   Final unique chemicals: {after}")

# ============================================
# 8. SAVE PREPROCESSED DATA
# ============================================
print("\n" + "="*50)
print("8. SAVING PREPROCESSED DATA")
print("="*50)

# Create processed directory
processed_dir.mkdir(parents=True, exist_ok=True)

# Save full preprocessed dataset
master.to_csv(processed_dir / 'master_preprocessed.csv', index=False)
print(f"   ✅ Saved: {processed_dir / 'master_preprocessed.csv'}")

# Save a clean version with only features for modeling
feature_columns = [
    'chemical_name_clean', 'who_class_valid',
    'molecularweight', 'xlogp', 'tpsa', 'complexity',
    'hbonddonorcount', 'hbondacceptorcount', 'rotatablebondcount', 'heavyatomcount',
    'ld50_rat_oral_mgkg', 'log_ld50', 'log_molecularweight',
    'is_carcinogen', 'is_neurotoxin', 'is_endocrine_disruptor', 
    'is_reproductive_toxin', 'total_hazard_flags',
    'name_length', 'word_count', 'has_number', 'has_hyphen', 'special_chars'
]

# Only keep columns that exist
feature_columns = [col for col in feature_columns if col in master.columns]

master_features = master[feature_columns].copy()
master_features.to_csv(processed_dir / 'master_features.csv', index=False)
print(f"   ✅ Saved: {processed_dir / 'master_features.csv'} (for modeling)")

# ============================================
# 9. DATA SUMMARY REPORT
# ============================================
print("\n" + "="*50)
print("9. DATA SUMMARY REPORT")
print("="*50)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   Total unique chemicals: {len(master)}")
print(f"   Total features: {len(master.columns)}")
print(f"   Features for modeling: {len(feature_columns)}")

print(f"\n📊 TARGET VARIABLE (WHO Class):")
class_dist = master['who_class_valid'].value_counts()
for cls, count in class_dist.items():
    pct = count/len(master)*100
    bar = '█' * int(pct/2)
    print(f"   {cls}: {count:3d} ({pct:5.1f}%) {bar}")

print(f"\n📊 TOXICITY FLAGS (present):")
for flag in ['is_carcinogen', 'is_neurotoxin', 'is_endocrine_disruptor', 'is_reproductive_toxin']:
    if flag in master.columns:
        count = master[flag].sum()
        pct = count/len(master)*100
        print(f"   {flag}: {count:3d} chemicals ({pct:.1f}%)")

print(f"\n📊 PHYSICOCHEMICAL PROPERTIES (mean ± std):")
for prop in ['molecularweight', 'xlogp', 'tpsa', 'ld50_rat_oral_mgkg']:
    if prop in master.columns:
        mean_val = master[prop].mean()
        std_val = master[prop].std()
        print(f"   {prop}: {mean_val:.1f} ± {std_val:.1f}")

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE!")
print("="*70)

# Quick validation
print("\n🔍 VALIDATION CHECK:")
print(f"   No missing values: {master.isnull().sum().sum() == 0}")
print(f"   Valid WHO classes: {master['who_class_valid'].nunique()}")
print(f"   Ready for modeling: ✅")

# Show sample
print(f"\n📋 Sample of preprocessed data:")
sample_cols = ['chemical_name_clean', 'who_class_valid', 'molecularweight', 'xlogp', 'total_hazard_flags']
sample_cols = [c for c in sample_cols if c in master.columns]
print(master[sample_cols].head(10).to_string())