"""Optimal-transport domain adaptation for vector EEG features."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.model.novelty_eval import align_proba
from pcma.model.rich import per_subject_zscore


def _prepare(Ftr, subject_tr, Fte, subject_te):
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    return scaler.transform(Ftr), scaler.transform(Fte)


def _transport_plan(Xt_cost, Xs_cost, y_source=None, pseudo_target=None, alpha: float = 0.0, reg: float = 0.05):
    try:
        import ot
    except ImportError as exc:
        raise ImportError("POT is required for OT-DA. Install with: pip install pot") from exc

    M = ot.dist(Xt_cost, Xs_cost, metric="sqeuclidean")
    scale = np.median(M[M > 0]) if np.any(M > 0) else 1.0
    M = M / (scale + 1e-12)
    if y_source is not None and pseudo_target is not None and alpha > 0:
        label_cost = (np.asarray(pseudo_target)[:, None] != np.asarray(y_source)[None, :]).astype(float)
        M = M + alpha * label_cost
    a = np.full(Xt_cost.shape[0], 1.0 / Xt_cost.shape[0])
    b = np.full(Xs_cost.shape[0], 1.0 / Xs_cost.shape[0])
    return ot.sinkhorn(a, b, M, reg=reg, numItermax=300, stopThr=1e-6, warn=False)


def barycentric_target_to_source(Xs, Xt, y_source=None, pseudo_target=None, alpha: float = 0.0, reg: float = 0.05, cost_dim: int = 50):
    """Map target features into the source support by Sinkhorn barycentric projection."""
    Xs = np.asarray(Xs, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    if cost_dim and min(Xs.shape[1], len(Xs) + len(Xt) - 1) > cost_dim:
        pca = PCA(n_components=cost_dim, random_state=0).fit(np.vstack([Xs, Xt]))
        Xs_cost = pca.transform(Xs)
        Xt_cost = pca.transform(Xt)
    else:
        Xs_cost, Xt_cost = Xs, Xt
    gamma = _transport_plan(Xt_cost, Xs_cost, y_source, pseudo_target, alpha=alpha, reg=reg)
    row_sum = gamma.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return (gamma @ Xs) / row_sum


def fit_predict_ot_da_proba(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    variant: str = "sinkhorn",
    C: float = 1.0,
    reg: float = 0.05,
    alpha: float = 0.5,
    cost_dim: int = 50,
):
    """Source SVM on labeled source; classify target after unsupervised OT mapping."""
    Xs, Xt = _prepare(Ftr, subject_tr, Fte, subject_te)
    ytr = np.asarray(ytr)
    clf0 = SVC(C=C, gamma="scale", kernel="rbf", probability=True, random_state=0).fit(Xs, ytr)
    pseudo = None
    label_alpha = 0.0
    if variant == "classaware":
        pseudo = clf0.predict(Xt)
        label_alpha = alpha
    elif variant != "sinkhorn":
        raise ValueError(f"unknown OT-DA variant: {variant}")
    Xt_map = barycentric_target_to_source(
        Xs,
        Xt,
        y_source=ytr if pseudo is not None else None,
        pseudo_target=pseudo,
        alpha=label_alpha,
        reg=reg,
        cost_dim=cost_dim,
    )
    clf = SVC(C=C, gamma="scale", kernel="rbf", probability=True, random_state=0).fit(Xs, ytr)
    proba = align_proba(clf.predict_proba(Xt_map), clf.classes_, target_classes=np.unique(ytr))
    classes = np.unique(ytr)
    pred = classes[proba.argmax(axis=1)]
    return pred, proba, classes
