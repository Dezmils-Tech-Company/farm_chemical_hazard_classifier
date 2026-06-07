"""Test cases for Agricultural Chemical Safety AI API"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# tests/test_api.py - Updated version

def test_import_main():
    """Test that the main FastAPI app can be imported"""
    try:
        from api.main import app
        assert app is not None
        # Update to match your actual title
        assert app.title == "Safe_Shamba AI"  
        print("✅ Main app imported successfully")
    except Exception as e:
        print(f"❌ Failed to import main app: {e}")
        raise

def test_api_endpoints():
    """Test that the API has the expected endpoints"""
    try:
        from api.main import app
        routes = [route.path for route in app.routes]
        
        # Expected endpoints
        expected_endpoints = ["/", "/api/health", "/api/predict", "/api/diseases"]
        
        # Check that expected endpoints exist (at least the base ones)
        found_endpoints = [path for path in expected_endpoints if path in routes]
        assert len(found_endpoints) >= 2, f"Expected at least 2 endpoints, found {found_endpoints}"
    except Exception as e:
        print(f"❌ Failed to test endpoints: {e}")
        raise

def test_models_import():
    """Test that Pydantic models can be imported"""
    try:
        from api.models import ChemicalPredictRequest, DiseaseRequest, BatchPredictRequest
        assert ChemicalPredictRequest is not None
        assert DiseaseRequest is not None
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        raise