import inspect
import numpy as np
from pcma.model.rich import fit_predict_rich, per_subject_zscore

def _make(rng, n_per, d=20, sep=1.2):
    F, y, sid = [], [], []
    for s in range(3):
        for lab in (0, 1):
            base = rng.normal(0, 1, (n_per, d)) + (sep if lab else 0.0) + 0.4 * s
            F.append(base); y += [lab] * n_per; sid += [s] * n_per
    return np.vstack(F), np.array(y), np.array(sid)

def test_per_subject_zscore_is_unit():
    rng = np.random.default_rng(0)
    F = rng.normal(5, 3, (30, 4)); subject = np.array([0]*15 + [1]*15)
    Z = per_subject_zscore(F, subject)
    for s in (0, 1):
        assert np.allclose(Z[subject == s].mean(0), 0, atol=1e-6)
        assert np.allclose(Z[subject == s].std(0), 1, atol=1e-6)

def test_fit_predict_rich_beats_chance_and_valid():
    rng = np.random.default_rng(1)
    Ftr, ytr, str_ = _make(rng, 40); Fte, yte, ste = _make(rng, 40)
    pred = fit_predict_rich(Ftr, ytr, str_, Fte, ste)
    assert pred.shape == yte.shape and set(np.unique(pred)) <= {0, 1}
    assert (pred == yte).mean() > 0.7

def test_fit_predict_rich_no_target_label():
    params = list(inspect.signature(fit_predict_rich).parameters)
    assert "yte" not in params and "y_target" not in params

def test_fit_predict_rich_can_return_proba():
    rng = np.random.default_rng(3)
    Ftr, ytr, str_ = _make(rng, 30); Fte, yte, ste = _make(rng, 20)
    pred, proba, classes = fit_predict_rich(Ftr, ytr, str_, Fte, ste, return_proba=True)
    assert pred.shape == yte.shape
    assert proba.shape == (len(yte), 2)
    assert set(classes.tolist()) == {0, 1}
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_ensemble_beats_chance_and_valid():
    from pcma.model.rich import fit_predict_rich_ensemble
    rng = np.random.default_rng(2)
    Ftr, ytr, str_ = _make(rng, 40); Fte, yte, ste = _make(rng, 40)
    pred = fit_predict_rich_ensemble(Ftr, ytr, str_, Fte, ste, k=15)
    assert pred.shape == yte.shape and set(np.unique(pred)) <= {0, 1}
    assert (pred == yte).mean() > 0.7

def test_ensemble_no_target_label():
    import inspect
    from pcma.model.rich import fit_predict_rich_ensemble
    params = list(inspect.signature(fit_predict_rich_ensemble).parameters)
    assert "yte" not in params and "y_target" not in params
