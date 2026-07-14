import importlib.util

import numpy as np

from pcma.model.seed_svm import (
    fit_predict_seed_hgb,
    fit_predict_seed_lightgbm,
    fit_predict_seed_logreg,
    fit_predict_seed_linear_svm,
    fit_predict_seed_svm,
    loso_splits,
)


def _toy_seed_data():
    rng = np.random.default_rng(7)
    X, y, subject = [], [], []
    for s in range(4):
        for lab in (0, 1, 2):
            block = rng.normal(0, 0.2, size=(8, 12))
            block[:, lab * 4 : (lab + 1) * 4] += 2.0
            block += s * 0.05
            X.append(block)
            y.extend([lab] * len(block))
            subject.extend([str(s)] * len(block))
    return np.vstack(X), np.array(y), np.array(subject)


def test_loso_splits_leave_one_subject_out():
    _, _, subject = _toy_seed_data()
    folds = list(loso_splits(subject))
    assert len(folds) == 4
    for tr, te in folds:
        assert set(subject[tr]).isdisjoint(set(subject[te]))
        assert len(np.unique(subject[te])) == 1


def test_fit_predict_seed_svm_multiclass_signal():
    X, y, subject = _toy_seed_data()
    train = subject != "0"
    test = ~train
    pred = fit_predict_seed_svm(X[train], y[train], subject[train], X[test], subject[test], C=1.0)
    assert pred.shape == y[test].shape
    assert (pred == y[test]).mean() > 0.9


def test_fit_predict_seed_hgb_multiclass_signal():
    X, y, subject = _toy_seed_data()
    train = subject != "0"
    test = ~train
    pred = fit_predict_seed_hgb(X[train], y[train], subject[train], X[test], subject[test])
    assert pred.shape == y[test].shape
    assert (pred == y[test]).mean() > 0.8


def test_fit_predict_seed_linear_svm_multiclass_signal():
    X, y, subject = _toy_seed_data()
    train = subject != "0"
    test = ~train
    pred = fit_predict_seed_linear_svm(X[train], y[train], subject[train], X[test], subject[test], C=1.0)
    assert pred.shape == y[test].shape
    assert (pred == y[test]).mean() > 0.9


def test_fit_predict_seed_logreg_multiclass_signal():
    X, y, subject = _toy_seed_data()
    train = subject != "0"
    test = ~train
    pred = fit_predict_seed_logreg(X[train], y[train], subject[train], X[test], subject[test], C=1.0)
    assert pred.shape == y[test].shape
    assert (pred == y[test]).mean() > 0.9


def test_fit_predict_seed_lightgbm_multiclass_signal():
    if importlib.util.find_spec("lightgbm") is None:
        return
    X, y, subject = _toy_seed_data()
    train = subject != "0"
    test = ~train
    pred = fit_predict_seed_lightgbm(
        X[train],
        y[train],
        subject[train],
        X[test],
        subject[test],
        n_estimators=20,
    )
    assert pred.shape == y[test].shape
    assert (pred == y[test]).mean() > 0.8
