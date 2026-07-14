# pcma/model/seed_svm.py
"""SEED-family cross-subject classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from pcma.model.rich import per_subject_zscore


def fit_predict_seed_svm(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    C: float = 10.0,
    gamma="scale",
):
    """Per-subject z-score -> train-only scaling -> RBF-SVM."""
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    clf = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")
    clf.fit(scaler.transform(Ftr), ytr)
    return clf.predict(scaler.transform(Fte))


def fit_predict_seed_hgb(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    max_iter: int = 200,
    learning_rate: float = 0.05,
):
    """Per-subject z-score -> train-only scaling -> compact boosting baseline."""
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=0.01,
        random_state=0,
    )
    clf.fit(scaler.transform(Ftr), ytr)
    return clf.predict(scaler.transform(Fte))


def fit_predict_seed_linear_svm(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    C: float = 1.0,
):
    """Linear SVM baseline for large window-level SEED features."""
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    clf = LinearSVC(C=C, class_weight="balanced", dual="auto", random_state=0, max_iter=5000)
    clf.fit(scaler.transform(Ftr), ytr)
    return clf.predict(scaler.transform(Fte))


def fit_predict_seed_logreg(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    C: float = 1.0,
):
    """Per-subject z-score -> train-only scaling -> balanced multinomial logistic regression."""
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=0)
    clf.fit(scaler.transform(Ftr), ytr)
    return clf.predict(scaler.transform(Fte))


def fit_predict_seed_lightgbm(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
):
    """Optional LightGBM baseline; kept isolated so imports fail only when used."""
    import lightgbm as lgb

    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=0,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(scaler.transform(Ftr), ytr)
    return clf.predict(scaler.transform(Fte))


def loso_splits(subject):
    subject = np.asarray(subject)
    idx = np.arange(len(subject))
    for s in np.unique(subject):
        test = idx[subject == s]
        train = idx[subject != s]
        yield train, test
