"""Integration tests against running API server"""

import requests
import time

BASE_URL = "http://localhost:8000"

def test_server_health():
    """Test server is running"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200

def test_predict_endpoint_integration():
    """Test prediction with real HTTP requests"""
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json={"chemical_name": "glyphosate"}
    )
    assert response.status_code in [200, 404]

def test_batch_prediction():
    """Test batch prediction endpoint"""
    response = requests.post(
        f"{BASE_URL}/api/predict/batch",
        json={"chemicals": ["glyphosate", "atrazine", "chlorpyrifos"]}
    )
    assert response.status_code in [200, 404]