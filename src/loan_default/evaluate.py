"""Evaluation utilities: metrics, curve plots, and SHAP explainability."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never blocks on plt.show()
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_curve_plots(y_true, y_proba, y_pred) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve — Loan Default Prediction")
    plt.savefig(FIGURES_DIR / "roc_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve — Loan Default Prediction")
    plt.savefig(FIGURES_DIR / "pr_curve.png", bbox_inches="tight", dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No Default", "Default"]
    )
    plt.title("Confusion Matrix (test set)")
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_shap_summary(pipeline, X_sample) -> None:
    """SHAP summary plot for the fitted model inside `pipeline`. Prefers the
    tree explainer (fast, exact for tree models); falls back to a
    model-agnostic explainer for anything else (e.g. logistic regression)."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        background = shap.sample(X_transformed, min(100, X_transformed.shape[0]))
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(X_transformed)[..., 1].values

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names, show=False
    )
    plt.savefig(FIGURES_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
