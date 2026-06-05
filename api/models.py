# api/models.py
"""Pydantic models for API request/response"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class WHOClass(str, Enum):
    """WHO Hazard Classes"""
    IA = "Ia"
    IB = "Ib"
    II = "II"
    III = "III"
    U = "U"

class SafetyLevel(str, Enum):
    """Safety levels for farmers"""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"

# Request Models
class ChemicalPredictRequest(BaseModel):
    """Request for single chemical prediction"""
    chemical_name: str = Field(..., description="Name of the chemical/pesticide", example="glyphosate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chemical_name": "chlorpyrifos"
            }
        }

class BatchPredictRequest(BaseModel):
    """Request for batch chemical predictions"""
    chemicals: List[str] = Field(..., description="List of chemical names", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chemicals": ["glyphosate", "chlorpyrifos", "mancozeb"]
            }
        }

class DiseaseRequest(BaseModel):
    """Request for disease treatment recommendation"""
    disease_name: str = Field(..., description="Name of the crop disease", example="maize leaf blight")
    confidence: Optional[float] = Field(0.85, description="Model 1 confidence score", ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "disease_name": "tomato late blight",
                "confidence": 0.92
            }
        }

# Response Models
class SafetyInfo(BaseModel):
    """Safety information for a chemical"""
    level: str
    message: str
    ppe: str
    action: str
    color: str
    reentry_hours: int

class ChemicalPredictResponse(BaseModel):
    """Response for single chemical prediction"""
    success: bool
    chemical: str
    who_class: Optional[str] = None
    confidence: Optional[float] = None
    safety_level: Optional[str] = None
    farmer_message: Optional[str] = None
    ppe_required: Optional[str] = None
    action: Optional[str] = None
    reentry_hours: Optional[int] = None
    ld50_oral_mgkg: Optional[float] = None
    error_message: Optional[str] = None

class TreatmentRecommendation(BaseModel):
    """Treatment recommendation for a disease"""
    chemical: str
    who_class: str
    safety_level: str
    action: str
    ppe: str
    reentry_hours: int
    confidence: float

class DiseaseResponse(BaseModel):
    """Response for disease treatment recommendation"""
    success: bool
    disease: str
    confidence: float
    safe_recommendations: List[TreatmentRecommendation]
    caution_recommendations: List[TreatmentRecommendation]
    blocked_recommendations: List[TreatmentRecommendation]
    organic_alternative: str
    total_chemicals_checked: int
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    database_size: int
    version: str