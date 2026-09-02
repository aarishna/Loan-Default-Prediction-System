"""Loading and cleaning for the credit-risk dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "loan_status"

# person_age has entries up to 144 in the raw data (data-entry errors), and
# person_emp_length has a known bad outlier far beyond a plausible career
# length. These are dropped as impossible rather than imputed. Real missing
# values (NaN) are left for the pipeline's imputers to handle.
MAX_PLAUSIBLE_AGE = 100
MAX_PLAUSIBLE_EMP_LENGTH = 60


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["person_age"] <= MAX_PLAUSIBLE_AGE]
    if "person_emp_length" in df.columns:
        df = df[
            df["person_emp_length"].isna()
            | (df["person_emp_length"] <= MAX_PLAUSIBLE_EMP_LENGTH)
        ]
    return df.reset_index(drop=True)


def load_train() -> pd.DataFrame:
    return clean(load_csv(DATA_DIR / "credit_risk_train.csv"))


def load_test() -> pd.DataFrame:
    return clean(load_csv(DATA_DIR / "credit_risk_test.csv"))


def load_loan_requests() -> pd.DataFrame:
    return load_csv(DATA_DIR / "loan_requests.csv")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return X, y
