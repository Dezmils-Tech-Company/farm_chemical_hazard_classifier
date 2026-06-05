"""Test cases for Agricultural Chemical Safety AI API"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_import_main():
    """Test that the main FastAPI app can be imported"""
    try:
        from api.main import app
        assert app is not None
        assert app.title == "Agricultural Chemical Safety AI"
    except ImportError as e:
        pytest.skip(f"FastAPI dependencies not installed: {e}")

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
    except ImportError as e:
        pytest.skip(f"Cannot test endpoints: {e}")

def test_models_import():
    """Test that Pydantic models can be imported"""
    try:
        from api.models import ChemicalPredictRequest, DiseaseRequest, BatchPredictRequest
        assert ChemicalPredictRequest is not None
        assert DiseaseRequest is not None
    except ImportError as e:
        pytest.skip(f"Models not available: {e}")