"""Load the persisted pipeline and score new applicants.

Used by both the FastAPI service (api/main.py) and the batch scoring
script (scripts/score_loan_requests.py) so there is one source of truth
for inference.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_default_probability(applicant: dict) -> float:
    row = pd.DataFrame([{col: applicant.get(col) for col in FEATURE_COLUMNS}])
    pipeline = get_pipeline()
    return float(pipeline.predict_proba(row)[0, 1])


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    pipeline = get_pipeline()
    X = df[FEATURE_COLUMNS]
    proba = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["default_probability"] = proba
    out["predicted_status"] = (proba >= 0.5).astype(int)
    return out
