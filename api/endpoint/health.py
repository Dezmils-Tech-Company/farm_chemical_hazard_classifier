# api/endpoints/health.py
"""Health check endpoint"""

import sys
from pathlib import Path
from fastapi import APIRouter
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API and model are ready"""
    
    # Check if model exists
    model_path = project_root / "models" / "toxicity_model.pkl"
    model_loaded = model_path.exists()
    
    # Check database
    db_path = project_root / "data" / "processed" / "master_dataset.csv"
    if db_path.exists():
        df = pd.read_csv(db_path)
        db_size = len(df)
    else:
        db_size = 0
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        database_size=db_size,
        version="1.0.0"
    )