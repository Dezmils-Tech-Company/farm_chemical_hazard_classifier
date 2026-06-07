# api/endpoints/chemicals.py
"""Chemicals listing endpoints"""

from fastapi import APIRouter, Query
from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

router = APIRouter()

# Global database cache
_database = None

def get_database():
    global _database
    if _database is None:
        db_path = project_root / 'data/processed/master_preprocessed.csv'
        if db_path.exists():
            _database = pd.read_csv(db_path)
    return _database

@router.get("/chemicals")
async def list_chemicals(
    limit: int = Query(50, description="Maximum number of chemicals to return", ge=1, le=200),
    search: str = Query(None, description="Search by chemical name")
):
    """List available chemicals in the database"""
    
    try:
        db = get_database()
        
        if db is None:
            return {
                "total": 0,
                "chemicals": [],
                "error": "Database not loaded"
            }
        
        # Get chemicals as list of strings
        chemicals = db['chemical_name'].tolist()
        
        if search:
            chemicals = [c for c in chemicals if search.lower() in c.lower()]
        
        # Ensure we return strings
        chemical_list = [str(c) for c in chemicals[:limit]]
        
        return {
            "total": len(chemicals),
            "chemicals": chemical_list
        }
        
    except Exception as e:
        print(f"Error in list_chemicals: {e}")
        return {
            "total": 0,
            "chemicals": [],
            "error": str(e)
        }