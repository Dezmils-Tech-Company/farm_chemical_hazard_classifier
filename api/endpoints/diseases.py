# api/endpoints/diseases.py
"""Safe_Shamba Disease treatment chemical recommendation endpoints"""

from fastapi import APIRouter
from pathlib import Path
import sys
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

router = APIRouter()

# ── Disease → Chemical mapping ──────────────────────────────

DISEASE_MAP ={
            # MAIZE DISEASES (most common in Kenya)
            'maize_leaf_blight': {
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
            
            # VEGETABLE DISEASES
            'cabbage_black_rot': {
                'common_name': 'Cabbage Black Rot',
                'pathogen': 'Bacterial (Xanthomonas campestris)',
                'chemicals': ['copper_hydroxide'],
                'organic_alternative': 'Crop rotation + resistant varieties',
                'min_confidence': 0.75
            },
            'onion_downy_mildew': {
                'common_name': 'Onion Downy Mildew',
                'pathogen': 'Fungal (Peronospora destructor)',
                'chemicals': ['mancozeb', 'metalaxyl', 'chlorothalonil'],
                'organic_alternative': 'Good air circulation + neem oil',
                'min_confidence': 0.70
            },
            'pepper_anthracnose': {
                'common_name': 'Pepper Anthracnose',
                'pathogen': 'Fungal (Colletotrichum species)',
                'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
                'organic_alternative': 'Copper spray + remove infected fruits',
                'min_confidence': 0.75
            }
        }
    

PPE_MAP = {
    "U":  "Gloves, wash hands thoroughly with soap or detergent after use",
    "III":"Long sleeves cover body well-, gloves, wash thoroughly with soap or detergent after use",
    "II": "Coveralls, chemical gloves, N95 mask, boots",
    "Ib": "Full protective gear required",
    "Ia": "Full chemical suit + respirator",
}

REENTRY_MAP = {"U": 4, "III": 12, "II": 24, "Ib": 48, "Ia": 72}

SAFETY_MAP = {
    "U":  {"level": "SAFE",    "action": "APPROVED"},
    "III":{"level": "CAUTION", "action": "CAUTION"},
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
        db_path    = project_root / "data" / "processed" / "master_dataset.csv"
        _model_artifacts = {
            "model":         joblib.load(model_path / "toxicity_model.pkl"),
            "scaler":        joblib.load(model_path / "scaler.pkl"),
            "label_encoder": joblib.load(model_path / "label_encoder.pkl"),
            "feature_cols":  joblib.load(model_path / "feature_columns.pkl"),
            "database":      pd.read_csv(db_path),
        }
    return _model_artifacts


def predict_chemical_safety(chemical_name: str):
    """Return safety dict for a chemical, or None if not found."""
    try:
        art = get_model_artifacts()
        db  = art["database"]

        row = db[db["chemical_name"].str.lower() == chemical_name.lower()]
        if row.empty:
            return None

        features        = row[art["feature_cols"]].fillna(0)
        features_scaled = art["scaler"].transform(features)
        pred_encoded    = art["model"].predict(features_scaled)[0]
        pred_class      = art["label_encoder"].inverse_transform([pred_encoded])[0]
        confidence      = float(max(art["model"].predict_proba(features_scaled)[0]) * 100)

        safety = SAFETY_MAP.get(pred_class, {"level": "UNKNOWN", "action": "UNKNOWN"})

        return {
            "chemical":     chemical_name,
            "who_class":    pred_class,
            "safety_level": safety["level"],
            "action":       safety["action"],
            "confidence":   confidence,
            "ppe":          PPE_MAP.get(pred_class, "Consult label"),
            "reentry_hours":REENTRY_MAP.get(pred_class, 24),
        }
    except Exception as e:
        print(f"⚠️  Error predicting {chemical_name}: {e}")
        return None


# ── Endpoints ───────────────────────────────────────────────

@router.post("/disease/recommend")
async def recommend_treatments(request: dict):
    """
    Get safe chemical recommendations for a crop disease.

    Example:
    ```json
    { "disease_name": "maize leaf blight", "confidence": 0.87 }
    ```
    """
    disease_name = request.get("disease_name", "").strip().lower()
    confidence   = float(request.get("confidence", 0.85))

    if not disease_name:
        return {"success": False, "error_message": "disease_name is required"}

    if disease_name not in DISEASE_MAP:
        return {
            "success": False,
            "disease": disease_name,
            "confidence": confidence,
            "safe_recommendations": [],
            "caution_recommendations": [],
            "blocked_recommendations": [],
            "organic_alternative": "Remove infected plants and practice crop rotation",
            "total_chemicals_checked": 0,
            "error_message": f"Disease '{disease_name}' not found. Call GET /api/diseases for the full list.",
        }

    info     = DISEASE_MAP[disease_name]
    safe_recs, caution_recs, blocked_recs = [], [], []

    for chem in info["chemicals"]:
        safety = predict_chemical_safety(chem)
        if not safety:
            continue

        rec = {
            "chemical":     safety["chemical"],
            "who_class":    safety["who_class"],
            "safety_level": safety["safety_level"],
            "action":       safety["action"],
            "ppe":          safety["ppe"],
            "reentry_hours":safety["reentry_hours"],
            "confidence":   safety["confidence"],
        }

        if safety["action"] == "APPROVED":
            safe_recs.append(rec)
        elif safety["action"] in ("CAUTION", "WARNING"):
            caution_recs.append(rec)
        else:
            blocked_recs.append(rec)

    return {
        "success": True,
        "disease": disease_name,
        "confidence": confidence,
        "safe_recommendations":    safe_recs,
        "caution_recommendations": caution_recs,
        "blocked_recommendations": blocked_recs,
        "organic_alternative":     info["organic"],
        "total_chemicals_checked": len(info["chemicals"]),
    }


@router.get("/diseases")
async def list_diseases():
    """List all crop diseases available in the system."""
    return {
        "total": len(DISEASE_MAP),
        "diseases": [
            {
                "name":           k.title(),
                "key":            k,
                "chemical_count": len(v["chemicals"]),
                "has_organic":    bool(v["organic"]),
            }
            for k, v in DISEASE_MAP.items()
        ],
    }


@router.get("/disease/{disease_name}/chemicals")
async def get_disease_chemicals(disease_name: str):
    """Get all chemicals mapped to a specific disease."""
    key = disease_name.lower()

    if key not in DISEASE_MAP:
        return {"success": False, "error_message": f"Disease '{disease_name}' not found"}

    info = DISEASE_MAP[key]
    return {
        "success":            True,
        "disease":            disease_name,
        "Pathogen":           info["pathogen"],
        "chemicals":          info["chemicals"],
        "organic_alternative":info["organic_alternative"],
        "total_chemicals":    len(info["chemicals"]),
    }