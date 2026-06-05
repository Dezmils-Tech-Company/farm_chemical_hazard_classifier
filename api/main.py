# api/main.py
"""FastAPI application for Agricultural Chemical Safety AI"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.models import (
    ChemicalPredictRequest, ChemicalPredictResponse,
    BatchPredictRequest, DiseaseRequest, DiseaseResponse,
    HealthResponse, TreatmentRecommendation
)
from api.endpoints import predict, diseases, health

# Create FastAPI app
app = FastAPI(
    title="Agricultural Chemical Safety AI",
    description="""
    ## Thesis Project: AI for Pesticide Safety in Kenya
    
    This API provides toxicity predictions for agricultural chemicals and 
    recommends safe pesticides for crop diseases.
    
    ### Features:
    - **Chemical Safety Prediction**: Predict WHO hazard class for any pesticide
    - **Disease Treatment Recommendations**: Get safe chemical recommendations for crop diseases
    - **Batch Processing**: Predict multiple chemicals at once
    
    ### WHO Hazard Classes:
    - **Ia**: Extremely hazardous (BLOCKED)
    - **Ib**: Highly hazardous (BLOCKED)
    - **II**: Moderately hazardous (WARNING)
    - **III**: Slightly hazardous (CAUTION)
    - **U**: Unlikely hazardous (SAFE)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (for web app integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
app.include_router(diseases.router, prefix="/api", tags=["Diseases"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Agricultural Chemical Safety AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "batch_predict": "/api/predict/batch",
            "disease": "/api/disease/recommend",
            "diseases_list": "/api/diseases"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )