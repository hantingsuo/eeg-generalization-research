"""Clinical-prior classifier for depression-positive EEG transfer."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.data.features import LR_PAIRS
from pcma.model.novelty_eval import align_proba, source_subject_validation_mask
from pcma.model.rich import per_subject_zscore


FRONTAL = set(range(0, 12))
POSTERIOR = set(range(22, 30))
ALPHA = 2


def frontal_alpha_asymmetry_indices() -> np.ndarray:
    return np.asarray(
        [510 + pair_i * 5 + ALPHA for pair_i, (left, right) in enumerate(LR_PAIRS) if left in FRONTAL and right in FRONTAL],
        dtype=int,
    )


def posterior_alpha_indices() -> np.ndarray:
    idx = []
    for ch in POSTERIOR:
        idx.append(ch * 5 + ALPHA)
        idx.append(150 + ch * 5 + ALPHA)
    return np.asarray(idx, dtype=int)


def clinical_prior_weights(n_features: int, faa_weight: float = 2.5, posterior_alpha_weight: float = 1.5) -> np.ndarray:
    w = np.ones(n_features, dtype=float)
    faa = frontal_alpha_asymmetry_indices()
    post = posterior_alpha_indices()
    w[faa[faa < n_features]] *= faa_weight
    w[post[post < n_features]] *= posterior_alpha_weight
    return w


def clinical_marker_from_scaled(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    parts = []
    faa = frontal_alpha_asymmetry_indices()
    faa = faa[faa < X.shape[1]]
    if len(faa):
        parts.append(X[:, faa].mean(axis=1))
    post = posterior_alpha_indices()
    post = post[post < X.shape[1]]
    if len(post):
        parts.append(0.5 * X[:, post].mean(axis=1))
    if not parts:
        return np.zeros(X.shape[0], dtype=float)
    marker = np.sum(parts, axis=0)
    return (marker - marker.mean()) / (marker.std() + 1e-8)


def _prepare(Ftr, subject_tr, Fte, subject_te):
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    Xs = scaler.transform(Ftr)
    Xt = scaler.transform(Fte)
    weights = clinical_prior_weights(Xs.shape[1])
    return Xs, Xs * weights, Xt, Xt * weights


def _adjust_binary_proba(proba, marker, beta: float, sign: float):
    p = align_proba(proba, np.asarray([0, 1]), target_classes=(0, 1))
    logit = np.log((p[:, 1] + 1e-8) / (p[:, 0] + 1e-8))
    p1 = 1.0 / (1.0 + np.exp(-(logit + sign * beta * marker)))
    return np.column_stack([1.0 - p1, p1])


def _choose_beta(Xw, Xraw, y, subject, beta_grid, sign_grid):
    train_mask, val_mask = source_subject_validation_mask(subject, val_fraction=0.25, seed=3)
    if val_mask.sum() == 0 or len(np.unique(y[val_mask])) < 2:
        return 0.0, 1.0
    clf = SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, random_state=0).fit(Xw[train_mask], y[train_mask])
    base = align_proba(clf.predict_proba(Xw[val_mask]), clf.classes_, target_classes=(0, 1))
    marker = clinical_marker_from_scaled(Xraw[val_mask])
    best = (-np.inf, 0.0, 1.0)
    for beta in beta_grid:
        for sign in sign_grid:
            adj = _adjust_binary_proba(base, marker, beta, sign)
            pred = adj.argmax(axis=1)
            score = balanced_accuracy_score(y[val_mask], pred)
            if score > best[0] + 1e-12:
                best = (float(score), float(beta), float(sign))
    return best[1], best[2]


def fit_predict_clinical_prior_proba(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    C: float = 1.0,
    beta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    sign_grid=(-1.0, 1.0),
):
    """Clinical-prior weighted SVM; target labels are never accepted."""
    Xraw_s, Xs, Xraw_t, Xt = _prepare(Ftr, subject_tr, Fte, subject_te)
    ytr = np.asarray(ytr)
    classes = np.unique(ytr)
    if set(classes.tolist()) != {0, 1}:
        clf = SVC(C=C, gamma="scale", kernel="rbf", probability=True, random_state=0).fit(Xs, ytr)
        proba = clf.predict_proba(Xt)
        return clf.predict(Xt), proba, clf.classes_
    beta, sign = _choose_beta(Xs, Xraw_s, ytr, subject_tr, beta_grid, sign_grid)
    clf = SVC(C=C, gamma="scale", kernel="rbf", probability=True, random_state=0).fit(Xs, ytr)
    base = align_proba(clf.predict_proba(Xt), clf.classes_, target_classes=(0, 1))
    marker_t = clinical_marker_from_scaled(Xraw_t)
    proba = _adjust_binary_proba(base, marker_t, beta, sign)
    pred = np.asarray([0, 1])[proba.argmax(axis=1)]
    return pred, proba, np.asarray([0, 1])
