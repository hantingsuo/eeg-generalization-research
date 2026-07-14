# pcma/model/seed_da.py
"""Domain-adaptation helpers for SEED-family vector features."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.model.pipelines import coral_align
from pcma.model.rich import per_subject_zscore


def _zscore_scale(Ftr, subject_tr, Fte, subject_te):
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    return scaler.transform(Ftr), scaler.transform(Fte)


def fit_predict_seed_coral_svm(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    C: float = 1.0,
    gamma="scale",
    shrink: float = 0.1,
):
    """Per-subject z-score -> train scaling -> CORAL target-to-source -> RBF-SVM."""
    Xtr, Xte = _zscore_scale(Ftr, subject_tr, Fte, subject_te)
    Xte = coral_align(Xtr, Xte, shrink=shrink)
    clf = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def fit_predict_seed_svm_proba(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    C: float = 1.0,
    gamma="scale",
):
    """Baseline RBF-SVM with probabilities for window/trial aggregation experiments."""
    Xtr, Xte = _zscore_scale(Ftr, subject_tr, Fte, subject_te)
    clf = SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced", probability=True, random_state=0)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte), clf.predict_proba(Xte), clf.classes_


def trial_group_keys(subject, session, trial) -> np.ndarray:
    subject = np.asarray(subject).astype(str)
    session = np.asarray(session).astype(int)
    trial = np.asarray(trial).astype(int)
    return np.asarray([f"{s}_s{se}_t{tr}" for s, se, tr in zip(subject, session, trial)])


def select_evenly_per_group(groups, max_per_group: int) -> np.ndarray:
    """Deterministically select up to max_per_group rows from each group."""
    groups = np.asarray(groups)
    selected = []
    for group in dict.fromkeys(groups.tolist()):
        idx = np.where(groups == group)[0]
        if len(idx) <= max_per_group:
            selected.extend(idx.tolist())
        else:
            pos = np.linspace(0, len(idx) - 1, max_per_group).round().astype(int)
            selected.extend(idx[pos].tolist())
    return np.asarray(sorted(selected), dtype=int)


def aggregate_scores_by_group(scores, classes, groups, y_true):
    """Average per-window scores within each trial group."""
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes)
    groups = np.asarray(groups)
    y_true = np.asarray(y_true)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    out_y, out_pred, out_group = [], [], []
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        labels = np.unique(y_true[mask])
        if len(labels) != 1:
            raise ValueError(f"{group}: y_true is not constant")
        mean_score = scores[mask].mean(axis=0)
        out_y.append(labels[0])
        out_pred.append(classes[int(np.argmax(mean_score))])
        out_group.append(group)
    return np.asarray(out_y), np.asarray(out_pred), np.asarray(out_group)
