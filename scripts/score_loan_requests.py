"""Batch-score the sample applicants in data/loan_requests.csv.

Replaces the original interactive CLI carousel: runs the trained pipeline
over every applicant and prints a ranked results table.

Run from the repo root:  python -m scripts.score_loan_requests
"""
from __future__ import annotations

from src.loan_default.data import load_loan_requests
from src.loan_default.predict import predict_batch


def main():
    applicants = load_loan_requests()
    scored = predict_batch(applicants).sort_values(
        "default_probability", ascending=False
    )

    print(f"{'Borrower':<22}{'Prob. Default':>15}{'Recommendation':>18}")
    print("-" * 55)
    for _, row in scored.iterrows():
        recommendation = "REJECT" if row["predicted_status"] == 1 else "ACCEPT"
        print(
            f"{row['borrower']:<22}{row['default_probability']:>14.1%}"
            f"{recommendation:>18}"
        )


if __name__ == "__main__":
    main()
