import inspect

import numpy as np
import pytest

from pcma.model.ot_da import barycentric_target_to_source, fit_predict_ot_da_proba


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    Xs, y, subject = [], [], []
    for s in range(3):
        for lab in (0, 1):
            block = rng.normal(0, 0.2, size=(8, 6)) + lab * 1.2 + s * 0.1
            Xs.append(block)
            y.extend([lab] * len(block))
            subject.extend([f"S{s}"] * len(block))
    Xt = np.vstack([rng.normal(0, 0.2, size=(8, 6)), rng.normal(1.2, 0.2, size=(8, 6))]) + 0.2
    return np.vstack(Xs), np.array(y), np.array(subject), Xt, np.array(["T"] * len(Xt))


def test_ot_signature_has_no_target_labels():
    params = list(inspect.signature(fit_predict_ot_da_proba).parameters)
    assert "yte" not in params and "y_target" not in params


def test_barycentric_target_to_source_shape():
    pytest.importorskip("ot")
    Xs, y, subject, Xt, ste = _toy()
    mapped = barycentric_target_to_source(Xs, Xt, reg=0.1, cost_dim=3)
    assert mapped.shape == Xt.shape
    assert np.isfinite(mapped).all()


def test_fit_predict_ot_da_proba_shape():
    pytest.importorskip("ot")
    Xs, y, subject, Xt, ste = _toy(1)
    pred, proba, classes = fit_predict_ot_da_proba(Xs, y, subject, Xt, ste, reg=0.1, cost_dim=3)
    assert pred.shape == (len(Xt),)
    assert proba.shape == (len(Xt), 2)
    assert set(classes.tolist()) == {0, 1}
