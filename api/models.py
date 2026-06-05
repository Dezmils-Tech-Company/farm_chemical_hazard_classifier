# api/models.py
"""Pydantic models for API request/response"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class WHOClass(str, Enum):
    IA = "Ia"
    IB = "Ib"
    II = "II"
    III = "III"
    U = "U"


class SafetyLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


# ── Request Models ──────────────────────────────────────────

class ChemicalPredictRequest(BaseModel):
    chemical_name: str = Field(..., description="Name of the chemical/pesticide")

    model_config = {
        "json_schema_extra": {"example": {"chemical_name": "glyphosate"}}
    }


class BatchPredictRequest(BaseModel):
    chemicals: List[str] = Field(..., description="List of chemical names", min_length=1, max_length=100)

    model_config = {
        "json_schema_extra": {"example": {"chemicals": ["glyphosate", "chlorpyrifos", "mancozeb"]}}
    }


class DiseaseRequest(BaseModel):
    disease_name: str = Field(..., description="Name of the crop disease")
    confidence: Optional[float] = Field(0.85, description="Model confidence score", ge=0, le=1)

    model_config = {
        "json_schema_extra": {"example": {"disease_name": "tomato late blight", "confidence": 0.92}}
    }


# ── Response Models ─────────────────────────────────────────

class ChemicalPredictResponse(BaseModel):
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
    chemical: str
    who_class: str
    safety_level: str
    action: str
    ppe: str
    reentry_hours: int
    confidence: float


class DiseaseResponse(BaseModel):
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
    status: str
    model_loaded: bool
    database_size: int
    version: str

    model_config = {"protected_namespaces": ()}