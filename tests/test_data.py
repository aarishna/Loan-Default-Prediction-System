import pandas as pd

from src.loan_default.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean,
    load_train,
    split_features_target,
)


def test_clean_drops_impossible_ages():
    df = pd.DataFrame(
        {
            "person_age": [25, 144, 40],
            "person_emp_length": [2, 5, 3],
            "loan_status": [0, 1, 0],
        }
    )
    cleaned = clean(df)
    assert cleaned["person_age"].max() <= 100
    assert len(cleaned) == 2


def test_load_train_has_expected_columns():
    df = load_train()
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns
    assert len(df) > 0


def test_split_features_target():
    df = load_train()
    X, y = split_features_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}
