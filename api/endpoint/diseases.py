
from fastapi import APIRouter
from pathlib import Path
import pandas as pd
import joblib

router = APIRouter()

# Global variables
_disease_map = None
_predictor = None

def get_disease_map():
    """Get disease-chemical mapping"""
    global _disease_map
    
    if _disease_map is None:
        _disease_map = {
            'maize leaf blight': {
                'chemicals': ['mancozeb', 'azoxystrobin', 'propiconazole'],
                'organic': 'Neem oil spray + remove infected leaves'
            },
            'maize rust': {
                'chemicals': ['tebuconazole', 'propiconazole', 'azoxystrobin'],
                'organic': 'Sulfur spray + compost tea'
            },
            'maize streak virus': {
                'chemicals': ['imidacloprid', 'acetamiprid'],
                'organic': 'Remove infected plants + neem oil'
            },
            'tomato late blight': {
                'chemicals': ['mancozeb', 'chlorothalonil', 'metalaxyl'],
                'organic': 'Copper spray + remove infected leaves'
            },
            'tomato early blight': {
                'chemicals': ['chlorothalonil', 'azoxystrobin', 'mancozeb'],
                'organic': 'Neem oil + baking soda spray'
            },
            'tomato leaf curl': {
                'chemicals': ['imidacloprid', 'deltamethrin'],
                'organic': 'Yellow sticky traps + neem oil'
            },
            'potato late blight': {
                'chemicals': ['mancozeb', 'metalaxyl', 'chlorothalonil'],
                'organic': 'Copper hydroxide + hilling soil'
            },
            'bean rust': {
                'chemicals': ['tebuconazole', 'azoxystrobin', 'mancozeb'],
                'organic': 'Sulfur + compost tea'
            },
            'bean anthracnose': {
                'chemicals': ['chlorothalonil', 'mancozeb', 'azoxystrobin'],
                'organic': 'Copper spray + resistant varieties'
            },
            'cassava mosaic': {
                'chemicals': ['imidacloprid', 'acetamiprid'],
                'organic': 'Use certified virus-free cuttings'
            }
        }
    
    return _disease_map

def get_safety_prediction(chemical_name):
    """Get safety prediction for a chemical"""
    try:
        # Get project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        # Load model
        model_path = project_root / "models"
        model = joblib.load(model_path / 'toxicity_model.pkl')
        scaler = joblib.load(model_path / 'scaler.pkl')
        label_encoder = joblib.load(model_path / 'label_encoder.pkl')
        feature_cols = joblib.load(model_path / 'feature_columns.pkl')
        
        # Load database
        db_path = project_root / "data" / "processed" / "master_dataset.csv"
        database = pd.read_csv(db_path)
        
        # Find chemical
        chem_row = database[database['chemical_name'].str.lower() == chemical_name.lower()]
        
        if len(chem_row) == 0:
            return None
        
        # Prepare features
        features = chem_row[feature_cols].fillna(0)
        features_scaled = scaler.transform(features)
        
        # Predict
        pred_encoded = model.predict(features_scaled)[0]
        pred_class = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get confidence
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(max(proba) * 100)
        
        # Safety mapping
        safety_map = {
            'U': {'level': 'SAFE', 'action': 'APPROVED'},
            'III': {'level': 'CAUTION', 'action': 'CAUTION'},
            'II': {'level': 'WARNING', 'action': 'WARNING'},
            'Ib': {'level': 'BLOCKED', 'action': 'BLOCKED'},
            'Ia': {'level': 'BLOCKED', 'action': 'BLOCKED'}
        }
        
        safety = safety_map.get(pred_class, {'level': 'UNKNOWN', 'action': 'UNKNOWN'})
        
        ppe_map = {
            'U': 'Gloves, wash hands after use',
            'III': 'Long sleeves, gloves, wash after use',
            'II': 'Coveralls, chemical gloves, N95 mask, boots',
            'Ib': 'Full protective gear required',
            'Ia': 'Full chemical suit + respirator'
        }
        
        return {
            'chemical': chemical_name,
            'who_class': pred_class,
            'safety_level': safety['level'],
            'action': safety['action'],
            'confidence': confidence,
            'ppe': ppe_map.get(pred_class, 'Consult label'),
            'reentry_hours': {'U': 4, 'III': 12, 'II': 24, 'Ib': 48, 'Ia': 72}.get(pred_class, 24)
        }
        
    except Exception as e:
        print(f"Error predicting {chemical_name}: {e}")
        return None

@router.post("/disease/recommend")
async def recommend_treatments(request: dict):
    """
    Get safe chemical recommendations for a crop disease
    
    Example:
    ```json
    {
        "disease_name": "maize leaf blight",
        "confidence": 0.87
    }
    ```
    """
    disease_name = request.get('disease_name', '').lower()
    confidence = request.get('confidence', 0.85)

    if not disease_name:
        return {
            "success": False,
            "error_message": "disease_name is required"
        }

    disease_map = get_disease_map()

    if disease_name not in disease_map:
        return {
            "success": False,
            "disease": disease_name,
            "confidence": confidence,
            "safe_recommendations": [],
            "caution_recommendations": [],
            "blocked_recommendations": [],
            "organic_alternative": "Remove infected plants and practice crop rotation",
            "total_chemicals_checked": 0,
            "error_message": f"Disease '{disease_name}' not found"
        }

    disease_info = disease_map[disease_name]
    chemicals = disease_info['chemicals']
    organic = disease_info['organic']

    safe_recs = []
    caution_recs = []
    blocked_recs = []

    for chem in chemicals:
        safety = get_safety_prediction(chem)

        if safety:
            rec = {
                "chemical": chem,
                "who_class": safety['who_class'],
                "safety_level": safety['safety_level'],
                "action": safety['action'],
                "ppe": safety['ppe'],
                "reentry_hours": safety['reentry_hours'],
                "confidence": safety['confidence']
            }

            if safety['action'] == 'APPROVED':
                safe_recs.append(rec)
            elif safety['action'] in ['CAUTION', 'WARNING']:
                caution_recs.append(rec)
            elif safety['action'] == 'BLOCKED':
                blocked_recs.append(rec)

    return {
        "success": True,
        "disease": disease_name,
        "confidence": confidence,
        "safe_recommendations": safe_recs,
        "caution_recommendations": caution_recs,
        "blocked_recommendations": blocked_recs,
        "organic_alternative": organic,
        "total_chemicals_checked": len(chemicals)
    }

@router.get("/diseases")
async def list_diseases():
    """Get all available crop diseases in the system"""
    disease_map = get_disease_map()

    disease_list = []
    for disease_key, info in disease_map.items():
        disease_list.append({
            "name": disease_key.title(),
            "key": disease_key,
            "chemical_count": len(info['chemicals']),
            "has_organic": bool(info['organic'])
        })

    return {
        "total": len(disease_list),
        "diseases": disease_list
    }

@router.get("/disease/{disease_name}/chemicals")
async def get_disease_chemicals(disease_name: str):
    """Get all chemicals used for a specific disease"""
    disease_map = get_disease_map()
    disease_key = disease_name.lower()

    if disease_key not in disease_map:
        return {
            "success": False,
            "error_message": f"Disease '{disease_name}' not found"
        }

    disease_info = disease_map[disease_key]

    return {
        "success": True,
        "disease": disease_name,
        "chemicals": disease_info['chemicals'],
        "organic_alternative": disease_info['organic'],
        "total_chemicals": len(disease_info['chemicals'])
    }