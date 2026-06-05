# api/endpoints/predict.py
"""Prediction endpoints"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
from typing import List

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import (
    ChemicalPredictRequest, ChemicalPredictResponse,
    BatchPredictRequest
)
from src.safety_predictor import SafetyPredictor

router = APIRouter()

# Initialize predictor (singleton)
_predictor = None

def get_predictor():
    """Lazy load the predictor"""
    global _predictor
    if _predictor is None:
        try:
            _predictor = SafetyPredictor()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return _predictor

@router.post("/predict", response_model=ChemicalPredictResponse)
async def predict_chemical(request: ChemicalPredictRequest):
    """
    Predict safety for a single chemical
    
    Example:
    ```json
    {
        "chemical_name": "glyphosate"
    }
    ```
    """
    predictor = get_predictor()
    result = predictor.predict_chemical(request.chemical_name)
    
    if result.get('error'):
        raise HTTPException(status_code=404, detail=result['message'])
    
    return ChemicalPredictResponse(
        chemical=result['chemical'],
        hazard_class=result['hazard_class'],
        confidence=result['confidence']
    )