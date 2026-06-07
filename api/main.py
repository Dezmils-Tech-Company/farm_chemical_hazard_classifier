# api/main.py
"""FastAPI application for Agricultural Chemical Safety AI"""

import sys
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:
    raise ImportError(
        "FastAPI is required to run this application. Install it with `pip install fastapi`."
    ) from exc

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.endpoints import predict, diseases, health, chemicals

app = FastAPI(
    title="Agricultural Chemical Safety AI",
    description="""
Safe_Shamba AI for Pesticide Safety in Kenya
Provides toxicity predictions for agricultural chemicals and recommends
safe pesticides for crop diseases.

### WHO Hazard Classes:
- **Ia**: Extremely hazardous — BLOCKED
- **Ib**: Highly hazardous — BLOCKED
- **II**: Moderately hazardous — WARNING
- **III**: Slightly hazardous — CAUTION
- **U**: Unlikely hazardous — SAFE
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,    prefix="/api", tags=["Health"])
app.include_router(predict.router,   prefix="/api", tags=["Prediction"])
app.include_router(diseases.router,  prefix="/api", tags=["Diseases"])
app.include_router(chemicals.router, prefix="/api", tags=["Chemicals"])

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Safe_Shamba AI: Agricultural Chemical Safety API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health":          "/api/health",
            "predict":         "/api/predict",
            "batch_predict":   "/api/predict/batch",
            "disease":         "/api/disease/recommend",
            "diseases_list":   "/api/diseases",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)