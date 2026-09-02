"""FastAPI service for loan default prediction.

Run from the repo root:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.loan_default.predict import get_pipeline

from .schemas import ApplicantFeatures, ModelMetadata, PredictionResponse

METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "metrics.json"

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts the probability that a borrower defaults on a loan.",
    version="1.0.0",
)


def _risk_band(probability: float) -> str:
    if probability < 0.2:
        return "low"
    if probability < 0.5:
        return "medium"
    return "high"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/metadata", response_model=ModelMetadata)
def model_metadata():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503, detail="Model metrics not found — train the model first."
        )
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return ModelMetadata(
        model_type=metrics["selected_model"],
        test_metrics=metrics["test_metrics"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures):
    try:
        pipeline = get_pipeline()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, detail="Model not found — train the model first."
        )

    row = pd.DataFrame([applicant.model_dump()])
    probability = float(pipeline.predict_proba(row)[0, 1])
    return PredictionResponse(
        default_probability=round(probability, 4),
        predicted_status="default" if probability >= 0.5 else "no_default",
        risk_band=_risk_band(probability),
    )
