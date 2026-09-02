"""Pydantic request/response models for the loan default API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(None, ge=0)
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: Optional[float] = Field(None, ge=0)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 29,
                "person_income": 82450,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 11,
                "loan_intent": "MEDICAL",
                "loan_grade": "E",
                "loan_amnt": 7500,
                "loan_int_rate": 14.5,
                "loan_percent_income": 0.08,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 9,
            }
        }
    }


class PredictionResponse(BaseModel):
    default_probability: float
    predicted_status: Literal["default", "no_default"]
    risk_band: Literal["low", "medium", "high"]


class ModelMetadata(BaseModel):
    model_type: str
    test_metrics: dict
