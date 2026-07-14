import inspect

import numpy as np

from pcma.model.clinical_prior import (
    clinical_marker_from_scaled,
    clinical_prior_weights,
    fit_predict_clinical_prior_proba,
    frontal_alpha_asymmetry_indices,
    posterior_alpha_indices,
)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    X, y, subject = [], [], []
    for s in range(4):
        for lab in (0, 1):
            block = rng.normal(size=(10, 570)) * 0.2
            block[:, 510 + 2] += lab * 1.0
            block[:, 22 * 5 + 2] += lab * 0.5
            X.append(block)
            y.extend([lab] * len(block))
            subject.extend([f"S{s}"] * len(block))
    Xt = rng.normal(size=(12, 570)) * 0.2
    return np.vstack(X), np.array(y), np.array(subject), Xt, np.array(["T"] * len(Xt))


def test_clinical_indices_are_in_rich_feature_range():
    assert frontal_alpha_asymmetry_indices().min() >= 510
    assert frontal_alpha_asymmetry_indices().max() < 570
    assert posterior_alpha_indices().min() >= 0
    assert posterior_alpha_indices().max() < 300
    w = clinical_prior_weights(570)
    assert w[frontal_alpha_asymmetry_indices()].mean() > 1.0
    assert w[posterior_alpha_indices()].mean() > 1.0


def test_clinical_prior_signature_has_no_target_labels():
    params = list(inspect.signature(fit_predict_clinical_prior_proba).parameters)
    assert "yte" not in params and "y_target" not in params


def test_clinical_marker_and_predict_proba():
    X, y, subject, Xt, ste = _toy()
    marker = clinical_marker_from_scaled(X)
    assert marker.shape == (len(X),)
    pred, proba, classes = fit_predict_clinical_prior_proba(X, y, subject, Xt, ste)
    assert pred.shape == (len(Xt),)
    assert proba.shape == (len(Xt), 2)
    assert set(classes.tolist()) == {0, 1}
    assert np.allclose(proba.sum(axis=1), 1.0)
