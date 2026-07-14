# pcma/model/seed_spd.py
"""Fast Euclidean-alignment SPD classifiers for SEED-family covariance features."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _sym_invsqrt(M, eps=1e-8):
    w, V = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, eps, None)
    return (V * (w ** -0.5)) @ V.T


def _sym_log(M, eps=1e-8):
    w, V = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, eps, None)
    return (V * np.log(w)) @ V.T


def euclidean_align_bands(C, subject):
    """Whiten each subject's per-band covariances by that subject's arithmetic mean."""
    C = np.asarray(C, dtype=float)
    subject = np.asarray(subject)
    out = np.empty_like(C)
    for s in np.unique(subject):
        mask = subject == s
        for b in range(C.shape[1]):
            M = C[mask, b].mean(axis=0)
            W = _sym_invsqrt(M)
            out[mask, b] = W @ C[mask, b] @ W
    return out


def logeuclid_band_features(C):
    """Vectorize per-band matrix logarithms using upper triangles."""
    C = np.asarray(C, dtype=float)
    iu = np.triu_indices(C.shape[-1])
    feats = []
    for sample in C:
        blocks = [_sym_log(sample[b])[iu] for b in range(C.shape[1])]
        feats.append(np.concatenate(blocks))
    return np.asarray(feats, dtype=np.float32)


def fit_predict_logsvm_bands(Ctr, ytr, Cte, C: float = 1.0):
    Xtr = logeuclid_band_features(Ctr)
    Xte = logeuclid_band_features(Cte)
    scaler = StandardScaler().fit(Xtr)
    clf = LinearSVC(C=C, class_weight="balanced", dual="auto", random_state=0, max_iter=10000)
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict(scaler.transform(Xte))


def fit_predict_ea_logsvm_bands(Ctr, ytr, subject_tr, Cte, subject_te, C: float = 1.0):
    Ctr = euclidean_align_bands(Ctr, subject_tr)
    Cte = euclidean_align_bands(Cte, subject_te)
    return fit_predict_logsvm_bands(Ctr, ytr, Cte, C=C)
