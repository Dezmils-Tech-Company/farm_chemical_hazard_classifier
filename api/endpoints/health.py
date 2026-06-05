# api/endpoints/health.py
"""Health check endpoint"""

from fastapi import APIRouter
from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API, model, and database are ready."""
    model_path = project_root / "models" / "toxicity_model.pkl"
    db_path    = project_root / "data" / "processed" / "master_dataset.csv"

    model_loaded = model_path.exists()
    db_size      = len(pd.read_csv(db_path)) if db_path.exists() else 0

    return HealthResponse(
        status       = "healthy" if model_loaded else "degraded",
        model_loaded = model_loaded,
        database_size= db_size,
        version      = "1.0.0",
    )