"""
Heart Disease Predictor — FastAPI Backend
Run: uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import predict_single, FEATURE_INFO, FEATURES, load_data, eda_summary, train_models

app = FastAPI(title="Heart Disease Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve frontend
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
    model: str = "logistic_regression"


@app.get("/")
def root():
    return {"status": "Heart Disease Predictor API running", "version": "1.0"}


@app.get("/api/features")
def get_features():
    """Return feature metadata for building the form."""
    return FEATURE_INFO


@app.get("/api/metrics")
def get_metrics():
    """Return model evaluation metrics."""
    try:
        with open("data/metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run model.py first to train models")


@app.get("/api/eda")
def get_eda():
    """Return EDA summary statistics."""
    try:
        with open("data/eda_summary.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run model.py first")


@app.post("/api/predict")
def predict(patient: PatientData):
    """Predict heart disease risk for a patient."""
    try:
        data = patient.dict()
        model_name = data.pop("model")
        result = predict_single(data, model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample")
def get_sample():
    """Return a sample patient for demo purposes."""
    return {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
        "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
        "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
    }


from fastapi.responses import HTMLResponse
import os

@app.get("/app", response_class=HTMLResponse)
def serve_ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "..", "frontend", "cardiosense-ui.html")
    print(f"Looking for file at: {html_path}")  # ← add this line
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()