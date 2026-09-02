from src.loan_default.data import load_train, split_features_target
from src.loan_default.features import build_preprocessor


def test_preprocessor_transforms_without_error():
    df = load_train().head(200)
    X, _ = split_features_target(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(X.columns)  # one-hot expands categoricals
