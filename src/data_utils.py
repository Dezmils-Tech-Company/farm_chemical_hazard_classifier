# src/data_utils.py (FIXED VERSION - replace your entire file)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import re
import os

class DataLoader:
    """Load and preprocess all data sources for the thesis"""
    
    def __init__(self, data_dir: str = "data"):
        # Find project root regardless of where we're running from
        current_file = Path(__file__).resolve()
        
        # If we're in src directory, project root is parent
        if current_file.parent.name == 'src':
            self.project_root = current_file.parent.parent
        else:
            self.project_root = current_file.parent
        
        # Also check if we're in scripts folder
        if self.project_root.name == 'scripts':
            self.project_root = self.project_root.parent
        
        self.data_dir = self.project_root / data_dir
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        print(f"📍 Project root: {self.project_root}")
        print(f"📍 Raw data path: {self.raw_dir}")
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if raw data exists
        if not self.raw_dir.exists():
            print(f"⚠️ Warning: Raw data directory not found at {self.raw_dir}")
            print(f"   Looking for data in alternative locations...")
            
            # Try alternative locations
            alternatives = [
                Path.cwd() / 'data' / 'raw',
                self.project_root / 'agri_chem_safety' / 'data' / 'raw',
                Path.cwd().parent / 'data' / 'raw',
            ]
            
            for alt in alternatives:
                if alt.exists():
                    self.raw_dir = alt
                    print(f"   ✅ Found data at: {self.raw_dir}")
                    break
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files"""
        print("\nLoading raw data files...")
        
        # Check if files exist
        who_file = self.raw_dir / 'who_hazard_classifications.csv'
        tox_file = self.raw_dir / 'toxicity_flags.csv'
        pubchem_file = self.raw_dir / 'pubchem_properties.csv'
        comptox_file = self.raw_dir / 'comptox_identifiers.csv'
        
        if not who_file.exists():
            raise FileNotFoundError(f"Cannot find {who_file}")
        if not tox_file.exists():
            raise FileNotFoundError(f"Cannot find {tox_file}")
        if not pubchem_file.exists():
            raise FileNotFoundError(f"Cannot find {pubchem_file}")
        
        data = {
            'who': pd.read_csv(who_file),
            'toxicity': pd.read_csv(tox_file),
            'pubchem': pd.read_csv(pubchem_file),
            'comptox': pd.read_csv(comptox_file) if comptox_file.exists() else pd.DataFrame()
        }
        
        print(f"  ✅ WHO: {len(data['who'])} chemicals")
        print(f"  ✅ Toxicity flags: {len(data['toxicity'])} chemicals")
        print(f"  ✅ PubChem properties: {len(data['pubchem'])} chemicals")
        if len(data['comptox']) > 0:
            print(f"  ✅ CompTox: {len(data['comptox'])} chemicals")
        
        return data
    
    def clean_chemical_name(self, name: str) -> str:
        """Standardize chemical names for matching"""
        if pd.isna(name):
            return ""
        name = str(name).lower().strip()
        name = re.sub(r'[^\w\s-]', '', name)
        return name
    
    def preprocess_who_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean WHO hazard classification data"""
        df = df.copy()
        df['chemical_name_clean'] = df['chemical_name'].apply(self.clean_chemical_name)
        df['who_class'] = df['who_class'].astype(str).str.upper()
        
        # Map to standard classes
        class_map = {'IA': 'Ia', 'IB': 'Ib', 'II': 'II', 'III': 'III', 'U': 'U'}
        df['who_class'] = df['who_class'].map(class_map).fillna('Unknown')
        
        # Convert LD50 to numeric
        df['ld50_rat_oral_mgkg'] = pd.to_numeric(df['ld50_rat_oral_mgkg'], errors='coerce')
        
        print(f"   WHO classes found: {df['who_class'].unique()}")
        return df
    
    def preprocess_toxicity_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean toxicity flags data"""
        df = df.copy()
        df['chemical_name_clean'] = df['chemical_name'].apply(self.clean_chemical_name)
        
        flag_cols = ['is_carcinogen', 'is_endocrine_disruptor', 'is_neurotoxin', 
                    'is_reproductive_toxin', 'is_pbt']
        
        for col in flag_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        if all(col in df.columns for col in flag_cols):
            df['total_hazard_flags'] = df[flag_cols].sum(axis=1)
        
        return df
    
    def preprocess_pubchem(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean PubChem properties data"""
        df = df.copy()
        df = df.rename(columns={'query_name': 'chemical_name'})
        df['chemical_name_clean'] = df['chemical_name'].apply(self.clean_chemical_name)
        
        # Convert numeric columns
        numeric_cols = ['cid', 'molecularweight', 'xlogp', 'tpsa', 'complexity',
                       'hbonddonorcount', 'hbondacceptorcount', 'rotatablebondcount',
                       'heavyatomcount', 'charge']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fix xlogp special values
        if 'xlogp' in df.columns:
            df['xlogp'] = df['xlogp'].replace([-999, -999.0], np.nan)
        
        return df
    
    def create_master_dataset(self, save: bool = True) -> pd.DataFrame:
        """Merge all data sources into master dataset"""
        print("\n" + "="*60)
        print("CREATING MASTER DATASET")
        print("="*60)
        
        # Load and preprocess each source
        raw_data = self.load_raw_data()
        
        who_df = self.preprocess_who_data(raw_data['who'])
        tox_df = self.preprocess_toxicity_flags(raw_data['toxicity'])
        pubchem_df = self.preprocess_pubchem(raw_data['pubchem'])
        
        # Start with WHO data
        master = who_df.copy()
        print(f"\n📊 Base: {len(master)} chemicals")
        
        # Merge toxicity flags
        tox_cols = ['chemical_name_clean', 'is_carcinogen', 'is_endocrine_disruptor',
                   'is_neurotoxin', 'is_reproductive_toxin', 'is_pbt', 'total_hazard_flags']
        master = master.merge(tox_df[tox_cols], on='chemical_name_clean', how='left')
        print(f"   After toxicity flags: {len(master)} chemicals")
        
        # Merge PubChem properties
        pubchem_cols = ['chemical_name_clean', 'cid', 'molecularweight', 'xlogp', 'tpsa',
                       'complexity', 'hbonddonorcount', 'hbondacceptorcount',
                       'rotatablebondcount', 'heavyatomcount', 'charge']
        master = master.merge(pubchem_df[pubchem_cols], on='chemical_name_clean', how='left')
        print(f"   After PubChem: {len(master)} chemicals")
        
        # Remove duplicates
        master = master.drop_duplicates(subset=['chemical_name_clean'], keep='first')
        print(f"   After dedup: {len(master)} unique chemicals")
        
        # Handle missing values
        master = self.handle_missing_values(master)
        
        # Save if requested
        if save:
            master.to_csv(self.processed_dir / 'master_dataset.csv', index=False)
            print(f"\n💾 Saved to: {self.processed_dir / 'master_dataset.csv'}")
        
        return master
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Fill toxicity flags with 0 (meaning not present)
        flag_cols = ['is_carcinogen', 'is_endocrine_disruptor', 'is_neurotoxin',
                    'is_reproductive_toxin', 'is_pbt', 'total_hazard_flags']
        
        for col in flag_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill numeric properties with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def get_features_and_target(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Extract features and target for modeling"""
        if df is None:
            df = pd.read_csv(self.processed_dir / 'master_dataset.csv')
        
        # Define feature columns
        feature_cols = [
            'molecularweight', 'xlogp', 'tpsa', 'complexity',
            'hbonddonorcount', 'hbondacceptorcount', 'rotatablebondcount', 'heavyatomcount',
            'ld50_rat_oral_mgkg', 'total_hazard_flags',
            'is_carcinogen', 'is_neurotoxin', 'is_endocrine_disruptor', 'is_reproductive_toxin'
        ]
        
        # Add engineered features
        df['name_length'] = df['chemical_name_clean'].str.len()
        df['has_number'] = df['chemical_name_clean'].str.contains(r'\d').astype(int)
        df['has_hyphen'] = df['chemical_name_clean'].str.contains('-').astype(int)
        
        feature_cols.extend(['name_length', 'has_number', 'has_hyphen'])
        
        # Only keep columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].fillna(0)
        y = df['who_class']
        
        # Filter to known classes only
        valid_classes = ['Ia', 'Ib', 'II', 'III', 'U']
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]
        
        print(f"\n📊 Dataset ready for modeling:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target classes: {sorted(y.unique())}")
        print(f"   Total chemicals: {len(X)}")
        
        return X, y, feature_cols


# Quick test if run directly
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.create_master_dataset(save=True)
    print("\n✅ Data preprocessing complete!")
    print(f"   Final shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()[:10]}...")