# create_master_data.py
"""Run this first to create the master dataset"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

print("="*60)
print("CREATING MASTER DATASET")
print("="*60)

# Set paths
data_dir = Path('data/raw')
processed_dir = Path('data/processed')
processed_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n📂 Loading raw data...")
who_df = pd.read_csv(data_dir / 'who_hazard_classifications.csv')
tox_df = pd.read_csv(data_dir / 'toxicity_flags.csv')
pubchem_df = pd.read_csv(data_dir / 'pubchem_properties.csv')

print(f"   WHO: {len(who_df)} chemicals")
print(f"   Toxicity: {len(tox_df)} chemicals")
print(f"   PubChem: {len(pubchem_df)} chemicals")

# Clean function
def clean_name(name):
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(' ', '_')

# Clean data
who_df['name_clean'] = who_df['chemical_name'].apply(clean_name)
tox_df['name_clean'] = tox_df['chemical_name'].apply(clean_name)
pubchem_df['name_clean'] = pubchem_df['query_name'].apply(clean_name)

# Standardize WHO classes
who_df['who_class'] = who_df['who_class'].astype(str).str.upper()
class_map = {'IA': 'Ia', 'IB': 'Ib', 'II': 'II', 'III': 'III', 'U': 'U'}
who_df['who_class'] = who_df['who_class'].map(class_map).fillna('Unknown')

# Merge datasets
print("\n🔗 Merging datasets...")
master = who_df.merge(tox_df[['name_clean', 'is_carcinogen', 'is_neurotoxin', 
                               'is_endocrine_disruptor', 'is_reproductive_toxin', 
                               'is_pbt', 'total_hazard_flags']], 
                      on='name_clean', how='left')

master = master.merge(pubchem_df[['name_clean', 'molecularweight', 'xlogp', 'tpsa', 
                                   'complexity', 'hbonddonorcount', 'hbondacceptorcount']], 
                      on='name_clean', how='left')

# Fill missing values
print("\n🔄 Handling missing values...")
flag_cols = ['is_carcinogen', 'is_neurotoxin', 'is_endocrine_disruptor', 
             'is_reproductive_toxin', 'is_pbt', 'total_hazard_flags']
for col in flag_cols:
    if col in master.columns:
        master[col] = master[col].fillna(0)

# Fill numeric with median
numeric_cols = ['molecularweight', 'xlogp', 'tpsa', 'complexity', 
                'hbonddonorcount', 'hbondacceptorcount', 'ld50_rat_oral_mgkg']
for col in numeric_cols:
    if col in master.columns:
        master[col] = master[col].fillna(master[col].median())

# Create features
master['name_length'] = master['name_clean'].str.len()
master['has_number'] = master['name_clean'].str.contains(r'\d').astype(int)
master['has_hyphen'] = master['name_clean'].str.contains('-').astype(int)

# Remove duplicates
master = master.drop_duplicates(subset=['name_clean'], keep='first')

# Save
master.to_csv(processed_dir / 'master_dataset.csv', index=False)
print(f"\n💾 Saved to: {processed_dir / 'master_dataset.csv'}")
print(f"   Shape: {master.shape}")
print(f"   Columns: {len(master.columns)}")

print("\n✅ Master dataset created successfully!")