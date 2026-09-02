"""Train, tune, and compare candidate models; persist the best pipeline + metrics.

Run from the repo root:  python -m src.loan_default.train
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import load_test, load_train, split_features_target
from .evaluate import (
    compute_metrics,
    save_curve_plots,
    save_metrics,
    save_shap_summary,
)
from .features import build_preprocessor

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
RANDOM_STATE = 42


def _model_candidates(y_train) -> dict:
    """Three deliberately different model families:
    - LogisticRegression: interpretable linear baseline.
    - RandomForest: bagged trees, robust to outliers/non-linearity.
    - XGBoost: boosted trees, usually the strongest tabular performer.
    Each gets class-imbalance handling appropriate to its API."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos

    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"model__C": uniform(0.01, 10)},
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(3, 20),
                "model__min_samples_leaf": randint(1, 10),
            },
        ),
        "xgboost": (
            XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": randint(2, 10),
                "model__learning_rate": uniform(0.01, 0.3),
                "model__subsample": uniform(0.6, 0.4),
            },
        ),
    }


def train_and_select(X_train, y_train, n_iter: int = 20, cv_folds: int = 5):
    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_name, best_search, best_score = None, None, -np.inf

    for name, (estimator, param_dist) in _model_candidates(y_train).items():
        pipeline = Pipeline(
            steps=[("preprocess", preprocessor), ("model", estimator)]
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        results[name] = {
            "cv_roc_auc": search.best_score_,
            "best_params": search.best_params_,
        }
        print(
            f"[{name}] cv ROC-AUC: {search.best_score_:.4f}  "
            f"params: {search.best_params_}"
        )

        if search.best_score_ > best_score:
            best_name, best_search, best_score = name, search, search.best_score_

    return best_name, best_search.best_estimator_, results


def main():
    print("Loading data...")
    train_df = load_train()
    test_df = load_test()
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print(
        "Training & tuning candidate models "
        "(Logistic Regression, Random Forest, XGBoost)..."
    )
    best_name, best_pipeline, cv_results = train_and_select(X_train, y_train)
    print(f"\nSelected model: {best_name}")

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    print(
        f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}  "
        f"PR-AUC: {test_metrics['pr_auc']:.4f}  "
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )

    print("Saving plots (ROC, PR, confusion matrix, SHAP summary)...")
    save_curve_plots(y_test, y_proba, y_pred)
    shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    save_shap_summary(best_pipeline, shap_sample)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "model.joblib")

    save_metrics(
        {
            "selected_model": best_name,
            "cv_results": cv_results,
            "test_metrics": test_metrics,
        },
        MODELS_DIR / "metrics.json",
    )
    print(f"\nSaved pipeline to {MODELS_DIR / 'model.joblib'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
