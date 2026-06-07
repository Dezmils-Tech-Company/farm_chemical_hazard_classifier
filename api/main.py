"""FastAPI application for Agricultural Chemical Safety AI"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import routers
from api.endpoints import health, predict, diseases

# Create FastAPI app
app = FastAPI(
    title="Safe_Shamba AI",
    description="""
    ## Agricultural Chemical Safety AI for Kenyan Farmers
    
    This API helps farmers identify safe pesticides for crop diseases.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "message": "Safe_Shamba AI: Agricultural Chemical Safety API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "predict": "POST /api/predict",
            "batch_predict": "POST /api/predict/batch",
            "disease_recommend": "POST /api/disease/recommend",
            "list_diseases": "GET /api/diseases",
            "list_chemicals": "GET /api/chemicals"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )