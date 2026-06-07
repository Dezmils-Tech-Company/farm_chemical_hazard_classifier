# api/endpoints/diseases.py
"""Safe_Shamba Disease treatment chemical recommendation endpoints"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import sys
import pandas as pd
import joblib
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

router = APIRouter()

# ── Disease → Chemical mapping ──────────────────────────────

DISEASE_MAP = {
    # MAIZE DISEASES (most common in Kenya)
    'maize_leaf_blight': {
        'common_name': 'Maize Leaf Blight',
        'pathogen': 'Fungal (Exserohilum turcicum)',
        'chemicals': ['mancozeb', 'azoxystrobin', 'propiconazole'],
        'organic_alternative': 'Neem oil spray + remove infected leaves',
        'min_confidence': 0.70
    },
    'maize_rust': {
        'common_name': 'Maize Rust',
        'pathogen': 'Fungal (Puccinia sorghi)',
        'chemicals': ['tebuconazole', 'propiconazole', 'azoxystrobin'],
        'organic_alternative': 'Sulfur spray + compost tea',
        'min_confidence': 0.70
    },
    'maize_streak_virus': {
        'common_name': 'Maize Streak Virus',
        'pathogen': 'Viral (transmitted by leafhoppers)',
        'chemicals': ['imidacloprid', 'acetamiprid'],
        'organic_alternative': 'Remove infected plants + neem oil for vectors',
        'min_confidence': 0.80
    },
    'maize_smut': {
        'common_name': 'Maize Smut',
        'pathogen': 'Fungal (Ustilago maydis)',
        'chemicals': ['mancozeb', 'chlorothalonil'],
        'organic_alternative': 'Remove galls + crop rotation',
        'min_confidence': 0.75
    },
    
    # TOMATO DISEASES
    'tomato_late_blight': {
        'common_name': 'Tomato Late Blight',
        'pathogen': 'Fungal (Phytophthora infestans)',
        'chemicals': ['mancozeb', 'chlorothalonil', 'metalaxyl'],
        'organic_alternative': 'Copper spray + remove infected leaves',
        'min_confidence': 0.75
    },
    'tomato_early_blight': {
        'common_name': 'Tomato Early Blight',
        'pathogen': 'Fungal (Alternaria solani)',
        'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
        'organic_alternative': 'Neem oil + baking soda spray',
        'min_confidence': 0.70
    },
    'tomato_leaf_curl': {
        'common_name': 'Tomato Leaf Curl',
        'pathogen': 'Viral (whitefly transmitted)',
        'chemicals': ['imidacloprid', 'deltamethrin'],
        'organic_alternative': 'Yellow sticky traps + neem oil',
        'min_confidence': 0.80
    },
    'tomato_bacterial_wilt': {
        'common_name': 'Tomato Bacterial Wilt',
        'pathogen': 'Bacterial (Ralstonia solanacearum)',
        'chemicals': ['copper_hydroxide', 'streptomycin'],
        'organic_alternative': 'Crop rotation + solarization',
        'min_confidence': 0.75
    },
    'tomato_spotted_wilt': {
        'common_name': 'Tomato Spotted Wilt Virus',
        'pathogen': 'Viral (thrips transmitted)',
        'chemicals': ['spinosad', 'imidacloprid'],
        'organic_alternative': 'Remove infected plants + thrips control',
        'min_confidence': 0.80
    },
    
    # POTATO DISEASES
    'potato_late_blight': {
        'common_name': 'Potato Late Blight',
        'pathogen': 'Fungal (Phytophthora infestans)',
        'chemicals': ['mancozeb', 'metalaxyl', 'chlorothalonil'],
        'organic_alternative': 'Copper hydroxide + hilling soil',
        'min_confidence': 0.75
    },
    'potato_early_blight': {
        'common_name': 'Potato Early Blight',
        'pathogen': 'Fungal (Alternaria solani)',
        'chemicals': ['azoxystrobin', 'chlorothalonil', 'mancozeb'],
        'organic_alternative': 'Neem oil + crop rotation',
        'min_confidence': 0.70
    },
    'potato_tuber_moth': {
        'common_name': 'Potato Tuber Moth',
        'pathogen': 'Insect (Phthorimaea operculella)',
        'chemicals': ['spinosad', 'bacillus_thuringiensis'],
        'organic_alternative': 'Hilling soil + pheromone traps',
        'min_confidence': 0.75
    },
    
    # BEAN DISEASES
    'bean_rust': {
        'common_name': 'Bean Rust',
        'pathogen': 'Fungal (Uromyces appendiculatus)',
        'chemicals': ['tebuconazole', 'azoxystrobin', 'mancozeb'],
        'organic_alternative': 'Sulfur + compost tea',
        'min_confidence': 0.70
    },
    'bean_anthracnose': {
        'common_name': 'Bean Anthracnose',
        'pathogen': 'Fungal (Colletotrichum lindemuthianum)',
        'chemicals': ['chlorothalonil', 'mancozeb', 'azoxystrobin'],
        'organic_alternative': 'Copper spray + resistant varieties',
        'min_confidence': 0.75
    },
    'bean_angular_leaf_spot': {
        'common_name': 'Bean Angular Leaf Spot',
        'pathogen': 'Bacterial (Pseudomonas syringae)',
        'chemicals': ['copper_hydroxide'],
        'organic_alternative': 'Remove infected leaves + copper spray',
        'min_confidence': 0.70
    },
    'bean_root_rot': {
        'common_name': 'Bean Root Rot',
        'pathogen': 'Fungal (Fusarium spp.)',
        'chemicals': ['metalaxyl', 'azoxystrobin'],
        'organic_alternative': 'Crop rotation + resistant varieties',
        'min_confidence': 0.75
    },
    
    # CASSAVA DISEASES
    'cassava_mosaic': {
        'common_name': 'Cassava Mosaic',
        'pathogen': 'Viral (whitefly transmitted)',
        'chemicals': ['imidacloprid', 'acetamiprid'],
        'organic_alternative': 'Use certified virus-free cuttings',
        'min_confidence': 0.85
    },
    'cassava_brown_streak': {
        'common_name': 'Cassava Brown Streak',
        'pathogen': 'Viral',
        'chemicals': [],
        'organic_alternative': 'Rogue infected plants + resistant varieties',
        'min_confidence': 0.80
    },
    'cassava_green_mite': {
        'common_name': 'Cassava Green Mite',
        'pathogen': 'Mite (Mononychellus tanajoa)',
        'chemicals': ['abamectin', 'spiromesifen'],
        'organic_alternative': 'Neem oil + predatory mites',
        'min_confidence': 0.75
    },
    
    # VEGETABLE DISEASES
    'cabbage_black_rot': {
        'common_name': 'Cabbage Black Rot',
        'pathogen': 'Bacterial (Xanthomonas campestris)',
        'chemicals': ['copper_hydroxide'],
        'organic_alternative': 'Crop rotation + resistant varieties',
        'min_confidence': 0.75
    },
    'cabbage_diamondback_moth': {
        'common_name': 'Cabbage Diamondback Moth',
        'pathogen': 'Insect (Plutella xylostella)',
        'chemicals': ['spinosad', 'bacillus_thuringiensis'],
        'organic_alternative': 'Netting + pheromone traps',
        'min_confidence': 0.70
    },
    'onion_downy_mildew': {
        'common_name': 'Onion Downy Mildew',
        'pathogen': 'Fungal (Peronospora destructor)',
        'chemicals': ['mancozeb', 'metalaxyl', 'chlorothalonil'],
        'organic_alternative': 'Good air circulation + neem oil',
        'min_confidence': 0.70
    },
    'onion_thrips': {
        'common_name': 'Onion Thrips',
        'pathogen': 'Insect (Thrips tabaci)',
        'chemicals': ['spinosad', 'imidacloprid'],
        'organic_alternative': 'Blue sticky traps + neem oil',
        'min_confidence': 0.70
    },
    'pepper_anthracnose': {
        'common_name': 'Pepper Anthracnose',
        'pathogen': 'Fungal (Colletotrichum species)',
        'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
        'organic_alternative': 'Copper spray + remove infected fruits',
        'min_confidence': 0.75
    },
    'pepper_leaf_curl': {
        'common_name': 'Pepper Leaf Curl',
        'pathogen': 'Viral (whitefly transmitted)',
        'chemicals': ['imidacloprid', 'acetamiprid'],
        'organic_alternative': 'Yellow sticky traps + neem oil',
        'min_confidence': 0.80
    },
    
    # FRUIT DISEASES
    'citrus_greening': {
        'common_name': 'Citrus Greening',
        'pathogen': 'Bacterial (Candidatus Liberibacter)',
        'chemicals': ['imidacloprid', 'abamectin'],
        'organic_alternative': 'Remove infected trees + vector control',
        'min_confidence': 0.85
    },
    'mango_anthracnose': {
        'common_name': 'Mango Anthracnose',
        'pathogen': 'Fungal (Colletotrichum gloeosporioides)',
        'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
        'organic_alternative': 'Copper spray + remove infected fruits',
        'min_confidence': 0.75
    },
    'banana_fusarium_wilt': {
        'common_name': 'Banana Fusarium Wilt (Panama disease)',
        'pathogen': 'Fungal (Fusarium oxysporum)',
        'chemicals': [],
        'organic_alternative': 'Use resistant varieties + soil solarization',
        'min_confidence': 0.85
    }
}

PPE_MAP = {
    "U": "Gloves, wash hands thoroughly with soap or detergent after use",
    "III": "Long sleeves cover body well, gloves, wash thoroughly with soap or detergent after use",
    "II": "Coveralls, chemical gloves, N95 mask, boots",
    "Ib": "Full protective gear required",
    "Ia": "Full chemical suit + respirator",
}

REENTRY_MAP = {"U": 4, "III": 12, "II": 24, "Ib": 48, "Ia": 72}

SAFETY_MAP = {
    "U": {"level": "SAFE", "action": "APPROVED"},
    "III": {"level": "CAUTION", "action": "CAUTION"},
    "II": {"level": "WARNING", "action": "WARNING"},
    "Ib": {"level": "BLOCKED", "action": "BLOCKED"},
    "Ia": {"level": "BLOCKED", "action": "BLOCKED"},
}

# ── Cached model loader ─────────────────────────────────────

_model_artifacts = None

def get_model_artifacts():
    global _model_artifacts
    if _model_artifacts is None:
        model_path = project_root / "models"
        db_path = project_root / "data" / "processed" / "master_dataset.csv"
        
        _model_artifacts = {
            "model": joblib.load(model_path / "toxicity_model.pkl"),
            "scaler": joblib.load(model_path / "scaler.pkl"),
            "label_encoder": joblib.load(model_path / "label_encoder.pkl"),
            "feature_cols": joblib.load(model_path / "feature_columns.pkl"),
            "database": pd.read_csv(db_path),
        }
    return _model_artifacts

# api/endpoints/diseases.py - Fix the predict_chemical_safety function

# Replace the predict_chemical_safety function with:

def predict_chemical_safety(chemical_name: str):
    """Return safety dict for a chemical, or None if not found."""
    try:
        # Import here to avoid circular imports
        from src.safety_predictor import SafetyPredictor
        
        # Create a new predictor instance (or use cached)
        if not hasattr(predict_chemical_safety, 'predictor'):
            predict_chemical_safety.predictor = SafetyPredictor()
        
        result = predict_chemical_safety.predictor.predict(chemical_name)
        
        if result.get('error'):
            return None
        
        # Standardize class names (IB -> Ib)
        who_class = result['who_class']
        if who_class == 'IB':
            who_class = 'Ib'
        
        return {
            "chemical": chemical_name,
            "who_class": who_class,
            "safety_level": result['safety_level'],
            "action": result['action'],
            "confidence": result['confidence'],
            "ppe": result['ppe_required'],
            "reentry_hours": result['reentry_hours'],
        }
    except Exception as e:
        print(f"⚠️ Error predicting {chemical_name}: {e}")
        return None
# ── Endpoints ───────────────────────────────────────────────

@router.post("/disease/recommend")
async def recommend_treatments(request: dict):
    """
    Get safe chemical recommendations for a crop disease.

    Example request:
    ```json
    { 
        "disease_name": "maize leaf blight", 
        "confidence": 0.87 
    }
    """
    disease_name = request.get("disease_name", "").strip().lower()
    confidence = float(request.get("confidence", 0.85))

    if not disease_name:
        return {
            "success": False,
            "error_message": "disease_name is required"
        }

    # Convert spaces to underscores for lookup
    disease_key = disease_name.replace(' ', '_')

    if disease_key not in DISEASE_MAP:
        # Try original format
        if disease_name not in DISEASE_MAP:
            # Return available diseases
            available = list(DISEASE_MAP.keys())
            available_display = [d.replace('_', ' ') for d in available[:10]]
            return {
                "success": False,
                "disease": disease_name,
                "confidence": confidence,
                "safe_recommendations": [],
                "caution_recommendations": [],
                "blocked_recommendations": [],
                "organic_alternative": "Remove infected plants and practice crop rotation",
                "total_chemicals_checked": 0,
                "error_message": f"Disease '{disease_name}' not found. Available: {', '.join(available_display)}...",
            }
        disease_key = disease_name

    info = DISEASE_MAP[disease_key]
    safe_recs, caution_recs, blocked_recs = [], [], []

    for chem in info["chemicals"]:
        safety = predict_chemical_safety(chem)
        if not safety:
            continue

        rec = {
            "chemical": safety["chemical"],
            "who_class": safety["who_class"],
            "safety_level": safety["safety_level"],
            "action": safety["action"],
            "ppe": safety["ppe"],
            "reentry_hours": safety["reentry_hours"],
            "confidence": safety["confidence"],
        }

        if safety["action"] == "APPROVED":
            safe_recs.append(rec)
        elif safety["action"] in ("CAUTION", "WARNING"):
            caution_recs.append(rec)
        else:
            blocked_recs.append(rec)

    return {
        "success": True,
        "disease": info["common_name"],
        "disease_key": disease_key,
        "pathogen": info["pathogen"],
        "confidence": confidence,
        "safe_recommendations": safe_recs,
        "caution_recommendations": caution_recs,
        "blocked_recommendations": blocked_recs,
        "organic_alternative": info["organic_alternative"],
        "total_chemicals_checked": len(info["chemicals"]),
    }


@router.get("/diseases")
async def list_diseases(
    crop_type: Optional[str] = Query(None, description="Filter by crop type (maize, tomato, potato, bean, cassava, vegetable, fruit)"),
    search: Optional[str] = Query(None, description="Search diseases by name"),
    limit: int = Query(50, description="Maximum number of diseases to return", ge=1, le=200)
):
    """
    List all crop diseases available in the system.

    Optional filters:

    crop_type: Filter by specific crop

    search: Search diseases by name

    limit: Limit number of results
    """
    disease_list = []

    crop_filters = {
        "maize": ["maize_"],
        "tomato": ["tomato_"],
        "potato": ["potato_"],
        "bean": ["bean_"],
        "cassava": ["cassava_"],
        "vegetable": ["cabbage_", "onion_", "pepper_"],
        "fruit": ["citrus_", "mango_", "banana_"]
    }

    for key, info in DISEASE_MAP.items():
        # Apply crop filter
        if crop_type and crop_type in crop_filters:
            if not any(key.startswith(prefix) for prefix in crop_filters[crop_type]):
                continue

        # Apply search filter
        if search and search.lower() not in info["common_name"].lower() and search.lower() not in key:
            continue

        disease_list.append({
            "key": key,
            "name": info["common_name"],
            "pathogen": info["pathogen"],
            "chemical_count": len(info["chemicals"]),
            "has_organic": bool(info["organic_alternative"]),
            "min_confidence": info.get("min_confidence", 0.75)
        })

    # Limit results
    disease_list = disease_list[:limit]

    return {
        "success": True,
        "total": len(disease_list),
        "total_available": len(DISEASE_MAP),
        "filters_applied": {
            "crop_type": crop_type,
            "search": search,
            "limit": limit
        },
        "diseases": disease_list
    }

@router.get("/diseases/crops")
async def list_crop_types():
    """List all available crop types with disease counts."""
    crop_counts = {
        "maize": 0,
        "tomato": 0,
        "potato": 0,
        "bean": 0,
        "cassava": 0,
        "vegetable": 0,
        "fruit": 0
    }

    for key in DISEASE_MAP.keys():
        if key.startswith("maize_"):
            crop_counts["maize"] += 1
        elif key.startswith("tomato_"):
            crop_counts["tomato"] += 1
        elif key.startswith("potato_"):
            crop_counts["potato"] += 1
        elif key.startswith("bean_"):
            crop_counts["bean"] += 1
        elif key.startswith("cassava_"):
            crop_counts["cassava"] += 1
        elif any(key.startswith(prefix) for prefix in ["cabbage_", "onion_", "pepper_"]):
            crop_counts["vegetable"] += 1
        elif any(key.startswith(prefix) for prefix in ["citrus_", "mango_", "banana_"]):
            crop_counts["fruit"] += 1

    return {
        "success": True,
        "total_diseases": len(DISEASE_MAP),
        "crops": crop_counts
    }

@router.get("/diseases/{disease_key}")
async def get_disease_detail(disease_key: str):
    """Get detailed information about a specific disease."""

    if disease_key not in DISEASE_MAP:
        return {
            "success": False,
            "error_message": f"Disease '{disease_key}' not found"
        }

    info = DISEASE_MAP[disease_key]

    # Get safety predictions for each chemical
    chemical_safety = []
    for chem in info["chemicals"]:
        safety = predict_chemical_safety(chem)
        if safety:
            chemical_safety.append(safety)

    return {
        "success": True,
        "disease_key": disease_key,
        "common_name": info["common_name"],
        "pathogen": info["pathogen"],
        "chemicals": info["chemicals"],
        "chemical_safety": chemical_safety,
        "organic_alternative": info["organic_alternative"],
        "min_confidence": info.get("min_confidence", 0.75),
        "total_chemicals": len(info["chemicals"])
    }

@router.get("/diseases/{disease_key}/chemicals")
async def get_disease_chemicals(disease_key: str):
    """Get all chemicals mapped to a specific disease with safety info."""

    if disease_key not in DISEASE_MAP:
        return {
            "success": False,
            "error_message": f"Disease '{disease_key}' not found"
        }

    info = DISEASE_MAP[disease_key]

    chemicals_with_safety = []
    for chem in info["chemicals"]:
        safety = predict_chemical_safety(chem)
        if safety:
            chemicals_with_safety.append(safety)

    return {
        "success": True,
        "disease": info["common_name"],
        "disease_key": disease_key,
        "pathogen": info["pathogen"],
        "chemicals": info["chemicals"],
        "chemicals_with_safety": chemicals_with_safety,
        "organic_alternative": info["organic_alternative"],
        "total_chemicals": len(info["chemicals"]),
        "safe_count": len([c for c in chemicals_with_safety if c["action"] == "APPROVED"]),
        "warning_count": len([c for c in chemicals_with_safety if c["action"] in ("CAUTION", "WARNING")]),
        "blocked_count": len([c for c in chemicals_with_safety if c["action"] == "BLOCKED"])
    }

@router.get("/health/diseases")
async def health_check_diseases():
    """Health check for diseases endpoint."""
    return {
        "status": "healthy",
        "total_diseases": len(DISEASE_MAP),
        "model_loaded": _model_artifacts is not None,
        "database_loaded": _model_artifacts is not None and _model_artifacts.get("database") is not None
    }