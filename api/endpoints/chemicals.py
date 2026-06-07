"""Endpoint for listing available chemicals"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import pandas as pd

router = APIRouter()

# Load your chemical database
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # Adjust this path to your actual data file
    df = pd.read_csv(project_root / "data" / "processed" / "master_dataset.csv")
    CHEMICALS_DB = df.to_dict('records')
except Exception as e:
    CHEMICALS_DB = []
    print(f"Warning: Could not load chemicals database: {e}")


@router.get("/chemicals")
async def list_chemicals(skip: int = 0, limit: int = 100):
    """List all available chemicals"""
    chemicals = CHEMICALS_DB[skip:skip + limit]
    return {
        "total": len(CHEMICALS_DB),
        "limit": limit,
        "skip": skip,
        "chemicals": chemicals
    }


@router.get("/chemicals/{chemical_name}")
async def get_chemical(chemical_name: str):
    """Get specific chemical details"""
    for chem in CHEMICALS_DB:
        if chem.get('name', '').lower() == chemical_name.lower():
            return chem
    raise HTTPException(status_code=404, detail=f"Chemical '{chemical_name}' not found")