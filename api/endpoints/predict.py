"""Prediction endpoints"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.models import ChemicalPredictRequest, ChemicalPredictResponse, BatchPredictRequest

router = APIRouter()

# Global predictor
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            # Import here to avoid circular imports
            from src.safety_predictor import SafetyPredictor
            _predictor = SafetyPredictor()
            print("✅ Predictor loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load predictor: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return _predictor


@router.post("/predict", response_model=ChemicalPredictResponse)
async def predict_chemical(request: ChemicalPredictRequest):
    """
    Predict WHO hazard class for a single chemical.

    Example:
    ```json
    { "chemical_name": "glyphosate" }
    """
    predictor = get_predictor()
    result = predictor.predict(request.chemical_name)
    
    if result.get("error"):
        return ChemicalPredictResponse(
            success=False,
            chemical=request.chemical_name,
            error_message=result.get("message", "Unknown error")
        )
    
    return ChemicalPredictResponse(
        success=True,
        chemical=result["chemical"],
        who_class=result["who_class"],
        confidence=result["confidence"],
        safety_level=result["safety_level"],
        farmer_message=result["farmer_message"],
        ppe_required=result["ppe_required"],
        action=result["action"],
        reentry_hours=result["reentry_hours"],
        ld50_oral_mgkg=result.get("ld50_oral_mgkg"),
    )


@router.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """
    Predict WHO hazard class for multiple chemicals at once.

    Example:
    ```json
    { "chemicals": ["glyphosate", "chlorpyrifos", "mancozeb"] }
    """
    predictor = get_predictor()
    results = []
    
    for chemical in request.chemicals:
        result = predictor.predict(chemical)
        results.append({
            "chemical": chemical,
            "success": not result.get("error", False),
            "who_class": result.get("who_class"),
            "safety_level": result.get("safety_level"),
            "action": result.get("action"),
            "confidence": result.get("confidence"),
            "error_message": result.get("message") if result.get("error") else None,
        })
    
    return {
        "total": len(results),
        "results": results
    }

@router.get("/chemicals")
async def list_chemicals(limit: int = 50, search: str = None):
    """List available chemicals in the database"""
    
    try:
        from src.safety_predictor import SafetyPredictor
        predictor = SafetyPredictor()
        
        if predictor.database is None:
            return {
                "total": 0,
                "chemicals": [],
                "error": "Database not loaded"
            }
        
        # Get chemicals as list of strings (not dicts)
        chemicals = predictor.database['chemical_name'].tolist()
        
        if search:
            chemicals = [c for c in chemicals if search.lower() in c.lower()]
        
        # Ensure we return strings, not dicts
        chemical_list = [str(c) for c in chemicals[:limit]]
        
        return {
            "total": len(chemicals),
            "chemicals": chemical_list
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            "total": 0,
            "chemicals": [],
            "error": str(e)
        }