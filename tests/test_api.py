from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_probability():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["predicted_status"] in {"default", "no_default"}
    assert body["risk_band"] in {"low", "medium", "high"}


def test_model_metadata():
    response = client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "model_type" in body
    assert "test_metrics" in body
