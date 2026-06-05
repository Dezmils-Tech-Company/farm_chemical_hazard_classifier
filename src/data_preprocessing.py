"""
src/data_preprocessing.py
Step 1: Load and clean all raw data files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os

class DataPreprocessor:
    """Handles all data loading and cleaning"""
    
    def __init__(self, data_path=None):
        # Find project root (where 'data' folder is)
        current_file = Path(__file__).resolve()
        
        # Start from current file's location
        if current_file.parent.name == 'src':
            # We're in src folder, project root is parent
            self.project_root = current_file.parent.parent
        else:
            self.project_root = current_file.parent
        
        # If data_path is provided, use it; otherwise use project_root/data
        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = self.project_root / "data" / "raw"
        
        self.processed_path = self.project_root / "data" / "processed"
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📍 Project root: {self.project_root}")
        print(f"📍 Data path: {self.data_path}")
        
        # Check if data exists
        if not self.data_path.exists():
            print(f"❌ ERROR: Data folder not found at {self.data_path}")
            print(f"   Please make sure your data is in: {self.project_root}/data/raw/")
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
    def clean_chemical_name(self, name):
        """Standardize chemical names for matching"""
        if pd.isna(name):
            return ""
        name = str(name).lower().strip()
        name = re.sub(r'[^\w\s-]', '', name)
        return name
    
    def load_all_data(self):
        """Load all CSV files"""
        print("📂 Loading raw data...")
        
        who_file = self.data_path / 'who_hazard_classifications.csv'
        tox_file = self.data_path / 'toxicity_flags.csv'
        pubchem_file = self.data_path / 'pubchem_properties.csv'
        
        # Check if files exist
        for file in [who_file, tox_file, pubchem_file]:
            if not file.exists():
                print(f"❌ File not found: {file}")
                raise FileNotFoundError(f"Missing file: {file}")
        
        who_df = pd.read_csv(who_file)
        tox_df = pd.read_csv(tox_file)
        pubchem_df = pd.read_csv(pubchem_file)
        
        print(f"   ✅ WHO: {len(who_df)} chemicals")
        print(f"   ✅ Toxicity: {len(tox_df)} chemicals")
        print(f"   ✅ PubChem: {len(pubchem_df)} chemicals")
        
        return who_df, tox_df, pubchem_df
    
    def clean_who_data(self, df):
        """Clean WHO hazard classifications"""
        df = df.copy()
        df['name_clean'] = df['chemical_name'].apply(self.clean_chemical_name)
        df['who_class'] = df['who_class'].astype(str).str.upper()
        
        # Map to standard classes
        class_map = {'IA': 'Ia', 'IB': 'Ib', 'II': 'II', 'III': 'III', 'U': 'U'}
        df['who_class'] = df['who_class'].map(class_map).fillna('Unknown')
        
        # Convert LD50 to numeric
        df['ld50_rat_oral_mgkg'] = pd.to_numeric(df['ld50_rat_oral_mgkg'], errors='coerce')
        
        return df
    
    def clean_toxicity_data(self, df):
        """Clean toxicity flags"""
        df = df.copy()
        df['name_clean'] = df['chemical_name'].apply(self.clean_chemical_name)
        
        flag_cols = ['is_carcinogen', 'is_endocrine_disruptor', 'is_neurotoxin', 
                    'is_reproductive_toxin', 'is_pbt']
        
        for col in flag_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        if all(col in df.columns for col in flag_cols):
            df['total_hazard_flags'] = df[flag_cols].sum(axis=1)
        
        return df
    
    def clean_pubchem_data(self, df):
        """Clean PubChem properties"""
        df = df.copy()
        df = df.rename(columns={'query_name': 'chemical_name'})
        df['name_clean'] = df['chemical_name'].apply(self.clean_chemical_name)
        
        # Convert numeric columns
        numeric_cols = ['cid', 'molecularweight', 'xlogp', 'tpsa', 'complexity',
                       'hbonddonorcount', 'hbondacceptorcount', 'rotatablebondcount',
                       'heavyatomcount', 'charge']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'xlogp' in df.columns:
            df['xlogp'] = df['xlogp'].replace([-999, -999.0], np.nan)
        
        return df
    
    def merge_datasets(self, who_df, tox_df, pubchem_df):
        """Merge all datasets into master"""
        print("\n🔗 Merging datasets...")
        
        # Start with WHO
        master = who_df.copy()
        
        # Merge toxicity flags
        tox_cols = ['name_clean', 'is_carcinogen', 'is_neurotoxin', 
                   'is_endocrine_disruptor', 'is_reproductive_toxin', 
                   'is_pbt', 'total_hazard_flags']
        master = master.merge(tox_df[tox_cols], on='name_clean', how='left')
        
        # Merge PubChem properties
        pubchem_cols = ['name_clean', 'molecularweight', 'xlogp', 'tpsa', 'complexity',
                       'hbonddonorcount', 'hbondacceptorcount', 'rotatablebondcount',
                       'heavyatomcount']
        master = master.merge(pubchem_df[pubchem_cols], on='name_clean', how='left')
        
        # Remove duplicates
        master = master.drop_duplicates(subset=['name_clean'], keep='first')
        
        print(f"   ✅ Master dataset: {len(master)} chemicals, {len(master.columns)} columns")
        return master
    
    def handle_missing_values(self, df):
        """Fill missing values"""
        df = df.copy()
        
        # Fill toxicity flags with 0 (not present)
        flag_cols = ['is_carcinogen', 'is_neurotoxin', 'is_endocrine_disruptor',
                    'is_reproductive_toxin', 'is_pbt', 'total_hazard_flags']
        
        for col in flag_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_features(self, df):
        """Create additional features from chemical names"""
        df = df.copy()
        df['name_length'] = df['name_clean'].str.len()
        df['has_number'] = df['name_clean'].str.contains(r'\d').astype(int)
        df['has_hyphen'] = df['name_clean'].str.contains('-').astype(int)
        return df
    
    def run_pipeline(self):
        """Execute full preprocessing pipeline"""
        print("="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load
        who_df, tox_df, pubchem_df = self.load_all_data()
        
        # Clean
        who_df = self.clean_who_data(who_df)
        tox_df = self.clean_toxicity_data(tox_df)
        pubchem_df = self.clean_pubchem_data(pubchem_df)
        
        # Merge
        master = self.merge_datasets(who_df, tox_df, pubchem_df)
        
        # Handle missing values
        master = self.handle_missing_values(master)
        
        # Create features
        master = self.create_features(master)
        
        # Save
        master.to_csv(self.processed_path / 'master_dataset.csv', index=False)
        print(f"\n💾 Saved to: {self.processed_path / 'master_dataset.csv'}")
        
        return master


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.run_pipeline()
    print(f"\n✅ Final shape: {df.shape}")