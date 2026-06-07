```markdown

# 🌾 Safe_Shamba AI

### *AI-Powered Pesticide Safety System for Kenyan Farmers*

```
Overview
--------
Safe_Shamba AI  assists farmers by predicting hazard classifications for agricultural chemicals and delivering concise, actionable safety guidance. The system is implemented with a Random Forest classifier and served via a FastAPI-based REST API.

Highlights
----------
- Chemicals in database: 84
- Crop diseases covered: 28
- Test accuracy: 100% (on provided test set)
- 5-fold CV: 95% ± 8%
- Typical API response time: ~2.4s

Goals and Problem Statement
---------------------------
Many smallholder farmers lack easy access to pesticide safety information, face language and literacy barriers, and are exposed to hazardous chemicals. Safe_Shamba AI provides:

- Instant hazard predictions
- Farmer-friendly safety messages (English/Swahili)
- Recommended precautions (PPE, re-entry intervals)
- Organic alternatives where available

System Architecture (high level)
-------------------------------
Components:

- Mobile/web client
- FastAPI REST service
- Random Forest toxicity classifier
- Safety-message and recommendation engine

Key Features
------------
- Predict WHO hazard class (e.g., U, III, II, Ib, Ia)
- Confidence scores and LD50 values
- Batch prediction (up to 50 chemicals)
- Farmer-friendly outputs and bilingual support

Installation
------------
1. Clone the repository
2. Create a Python virtual environment
3. Install dependencies: pip install -r requirements.txt

Usage
-----
- Start the API server (uvicorn) and call endpoints documented in the API section.
- Example calls and payloads are provided in the repository's docs and tests.

Testing and Validation
----------------------
- Unit and integration tests are included. Run via pytest.
- Model evaluation results and cross-validation details are in the Thesis Results section.

Project Structure
-----------------
- app/: API and application code
- models/: trained model artifacts
- data/: datasets and preprocessing scripts
- docs/: design notes, API docs and thesis materials
- tests/: automated tests

License
-------
See the LICENSE file in the repository.

Contact
-------
Project lead: contact details in the repo metadata.

For full documentation and deployment instructions, refer to the docs/ directory and the Thesis Results section of this repository.
- Simple PPE recommendations
- Re-entry interval in hours
- Color-coded responses (green, yellow, orange, red)

---
```
```
### Classification Report

| WHO Class | Precision | Recall | F1-Score | Support |
|-----------|----------|--------|----------|---------|
| U (SAFE) | 1.00 | 1.00 | 1.00 | 4 |
| III (CAUTION) | 1.00 | 1.00 | 1.00 | 2 |
| II (WARNING) | 1.00 | 1.00 | 1.00 | 2 |
| Ib (BLOCKED) | 1.00 | 1.00 | 1.00 | 2 |
| **Average** | **1.00** | **1.00** | **1.00** | **10** |

```
```
### Test Results (7 Chemicals)

| Chemical | Predicted | Actual | Confidence | Status |
|----------|-----------|--------|------------|--------|
| Glyphosate | U (SAFE) | U | 100% | ✅ |
| Chlorpyrifos | II (WARNING) | II | 100% | ✅ |
| Malathion | III (CAUTION) | III | 100% | ✅ |
| Carbofuran | Ib (BLOCKED) | Ib | 90% | ✅ |
| Mancozeb | U (SAFE) | U | 100% | ✅ |
| Atrazine | III (CAUTION) | III | 85% | ✅ |
| Cypermethrin | II (WARNING) | II | 100% | ✅ |
```
---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/safe-shamba-ai.git
cd safe-shamba-ai

# Install dependencies
pip install -r requirements.txt

# Run the API
python api/main.py

# Open in browser
http://localhost:8000/docs
```

### Docker Setup

```bash
# Build the image
docker build -t safe-shamba-api .

# Run the container
docker run -p 8000:8000 safe-shamba-api

# Or use docker-compose
docker-compose up -d
```

### Cloud Deployment (Render.com)

1. Push code to GitHub
2. Create `render.yaml` in root
3. Connect repository to Render
4. Deploy automatically

---

## 📡 API Usage

### Base URL
```
http://localhost:8000/api
```

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| POST | `/predict` | Predict chemical safety |
| POST | `/predict/batch` | Batch prediction |
| POST | `/disease/recommend` | Get disease treatments |
| GET | `/diseases` | List all diseases |
| GET | `/diseases/crops` | List by crop type |
| GET | `/diseases/{key}` | Disease details |
| GET | `/diseases/{key}/chemicals` | Chemicals for disease |
| GET | `/chemicals` | List all chemicals |

### Request/Response Examples

#### 1. Predict Chemical Safety

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"chemical_name": "glyphosate"}'
```

**Response:**
```json
{
  "success": true,
  "chemical": "glyphosate",
  "who_class": "U",
  "confidence": 100.0,
  "safety_level": "SAFE",
  "farmer_message": "✅ SAFE - Can be used with standard precautions",
  "ppe_required": "Gloves, wash hands after use",
  "action": "APPROVED",
  "reentry_hours": 4,
  "ld50_oral_mgkg": 5600.0
}
```

#### 2. Disease Treatment Recommendation

**Request:**
```bash
curl -X POST http://localhost:8000/api/disease/recommend \
  -H "Content-Type: application/json" \
  -d '{"disease_name": "maize leaf blight", "confidence": 0.87}'
```

**Response:**
```json
{
  "success": true,
  "disease": "Maize Leaf Blight",
  "confidence": 0.87,
  "safe_recommendations": [
    {
      "chemical": "mancozeb",
      "who_class": "U",
      "safety_level": "SAFE",
      "action": "APPROVED",
      "ppe": "Gloves, wash hands after use",
      "reentry_hours": 4,
      "confidence": 100.0
    }
  ],
  "organic_alternative": "Neem oil spray + remove infected leaves"
}
```

#### 3. Batch Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"chemicals": ["glyphosate", "chlorpyrifos", "mancozeb"]}'
```

**Response:**
```json
{
  "total": 3,
  "results": [
    {"chemical": "glyphosate", "who_class": "U", "action": "APPROVED"},
    {"chemical": "chlorpyrifos", "who_class": "II", "action": "WARNING"},
    {"chemical": "mancozeb", "who_class": "U", "action": "APPROVED"}
  ]
}
```

#### 4. List All Diseases

```bash
curl http://localhost:8000/api/diseases
```

#### 5. List All Chemicals

```bash
curl http://localhost:8000/api/chemicals?limit=20
```

### Interactive Documentation

Open your browser to: **http://localhost:8000/docs**

---

## 🧪 Testing

### Run Full Test Suite

```bash
python test_api_endpoints.py
```

### Quick Test

```bash
python test_api_endpoints.py --quick
```

### Expected Output

```
======================================================================
SAFE_SHAMBA AI - API TEST SUITE
======================================================================
✅ Health Check
✅ Root Endpoint
✅ Single Prediction (7/7 correct)
✅ Batch Prediction
✅ Disease Recommendation
✅ List Diseases
✅ Crop Types
✅ Disease Detail
✅ Disease Chemicals
✅ List Chemicals
✅ Invalid Chemical Handling
✅ Invalid Disease Handling

RESULTS: 12/13 tests passed
🎉 API is working perfectly!
```

---

## 🐳 Deployment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| API_TITLE | "Safe_Shamba AI" | API display name |
| API_VERSION | "1.0.0" | Version number |
| API_HOST | "0.0.0.0" | Host address |
| API_PORT | 8000 | Port number |
| MODEL_PATH | "models" | Model directory |
| LOG_LEVEL | "info" | Logging level |



## 📁 Project Structure

```
safe-shamba-ai/
│
├── api/                          # FastAPI application
│   ├── endpoints/                # API endpoints
│   │   ├── health.py            # Health check
│   │   ├── predict.py           # Prediction endpoints
│   │   ├── diseases.py          # Disease endpoints
│   │   └── chemicals.py         # Chemicals endpoints
│   ├── main.py                  # FastAPI entry point
│   └── models.py                # Pydantic models
│
├── src/                          # Core ML code
│   ├── safety_predictor.py      # Model inference
│   ├── data_preprocessing.py    # Data cleaning
│   └── model_training.py        # Model training
│
├── data/                         # Datasets
│   ├── raw/                     # Original CSV files
│   └── processed/               # Cleaned master dataset
│
├── models/                       # Trained models
│   ├── toxicity_model.pkl       # Random Forest model
│   ├── scaler.pkl              # StandardScaler
│   ├── label_encoder.pkl       # Label encoder
│   └── feature_columns.pkl     # Feature list
│
├── reports/                      # Thesis outputs
│   └── figures/                 # Performance graphs
│
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container config
├── docker-compose.yml           # Multi-container setup
├── test_api_endpoints.py        # Test suite
└── README.md                    # This file
```

---

## 🙏 Acknowledgments

- WHO for hazard classification data
- PubChem for chemical properties
- Kenya PCPB for regulatory information

---

<div align="center">
  
**Made with ❤️ for Kenyan Farmers**

*"Safeguarding harvests, protecting lives"*

**Version 1.0.0 | © 2024 Safe_Shamba AI**

</div>
