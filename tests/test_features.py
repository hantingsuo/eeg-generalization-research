import numpy as np
from pcma.data.features import extract_rich_features, N_FEATURES


def test_rich_features_shape_and_finite():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 30, 2500))   # 8 trials, 30 ch, 10s @250Hz
    F = extract_rich_features(X, sfreq=250.0)
    assert F.shape == (8, N_FEATURES)
    assert np.isfinite(F).all()


def test_rich_features_deterministic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 30, 2500))
    a = extract_rich_features(X); b = extract_rich_features(X)
    assert np.allclose(a, b)
