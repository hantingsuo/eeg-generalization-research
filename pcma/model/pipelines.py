# pcma/model/pipelines.py
import numpy as np
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pcma.manifold.recenter import recenter_per_subject
from pyriemann.utils.mean import mean_riemann

def _classify(Ctr, ytr, Cte):
    ts = TangentSpace(metric="riemann").fit(Ctr)
    clf = LinearDiscriminantAnalysis().fit(ts.transform(Ctr), ytr)
    return clf.predict(ts.transform(Cte))

def fit_predict_noadapt(Ctr, ytr, Cte):
    """No adaptation: tangent space fit on source, applied to target."""
    return _classify(Ctr, ytr, Cte)

def fit_predict_ra(Ctr, ytr, subject_tr, Cte, subject_te):
    """RA: recenter each subject (train and test) unsupervised, then classify."""
    Ctr_r = recenter_per_subject(Ctr, subject_tr)
    Cte_r = recenter_per_subject(Cte, subject_te)
    return _classify(Ctr_r, ytr, Cte_r)

def band_tangent_features(Ctr, Cte, subject_tr=None, subject_te=None,
                          ref_means_per_band=None, recenter=False):
    """Per-band tangent features, concatenated -> (Xtr, Xte). If recenter=True,
    apply per-subject Riemannian recentering per band first (RA)."""
    nb = Ctr.shape[1]
    if recenter:
        def _rm(b):
            return None if ref_means_per_band is None else ref_means_per_band[b]
        Ctr = np.stack([recenter_per_subject(Ctr[:, b], subject_tr, ref_means=_rm(b)) for b in range(nb)], axis=1)
        Cte = np.stack([recenter_per_subject(Cte[:, b], subject_te, ref_means=_rm(b)) for b in range(nb)], axis=1)
    ftr, fte = [], []
    for b in range(nb):
        ts = TangentSpace(metric="riemann").fit(Ctr[:, b])
        ftr.append(ts.transform(Ctr[:, b]))
        fte.append(ts.transform(Cte[:, b]))
    return np.concatenate(ftr, axis=1), np.concatenate(fte, axis=1)

def _classify_bands(Ctr, ytr, Cte):
    """Ctr/Cte: (n_trials, n_bands, ch, ch). Per-band tangent, concat, shrinkage LDA."""
    Xtr, Xte = band_tangent_features(Ctr, Cte, recenter=False)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    return clf.predict(Xte)

def fit_predict_noadapt_bands(Ctr, ytr, Cte):
    """Per-band, no adaptation."""
    return _classify_bands(Ctr, ytr, Cte)

def subject_band_means(C, subject):
    """Per-band, per-subject Riemannian means. C: (n_trials, n_bands, ch, ch).
    Returns a list of length n_bands; each element is a dict {subject: mean_matrix}.
    A subject's mean is over ALL its trials -- valid to reuse across subject-level
    folds because each subject lives entirely in train OR test."""
    nb = C.shape[1]
    out = []
    for b in range(nb):
        means = {s: mean_riemann(C[subject == s, b]) for s in np.unique(subject)}
        out.append(means)
    return out

def fit_predict_ra_bands(Ctr, ytr, subject_tr, Cte, subject_te, ref_means_per_band=None):
    """Per-band RA: recenter each band per-subject (unsupervised), then classify.
    If ref_means_per_band is given (list of {subject: mean} per band, e.g. from
    subject_band_means over the full dataset), reuse those means instead of
    recomputing per fold -- identical result, much faster."""
    nb = Ctr.shape[1]
    def _rm(b):
        return None if ref_means_per_band is None else ref_means_per_band[b]
    Ctr_r = np.stack([recenter_per_subject(Ctr[:, b], subject_tr, ref_means=_rm(b)) for b in range(nb)], axis=1)
    Cte_r = np.stack([recenter_per_subject(Cte[:, b], subject_te, ref_means=_rm(b)) for b in range(nb)], axis=1)
    return _classify_bands(Ctr_r, ytr, Cte_r)

def global_mean_align(Xs, Xt):
    """Global (class-agnostic) first-moment alignment ablation baseline."""
    Xt = np.asarray(Xt, float)
    return Xt - Xt.mean(axis=0) + np.asarray(Xs, float).mean(axis=0)

def conditional_mean_align(Xs, ys, Xt, yt_pseudo, conf=None, conf_thresh=0.0, min_count=1):
    """Class-conditional first-moment alignment. Each target class mean is estimated
    ONLY from confident target samples (conf >= conf_thresh); a class with fewer than
    min_count confident samples is left unaligned. All target samples of an aligned
    class are shifted by (mu_source,c - mu_target_confident,c). Uses source true
    labels + target pseudo-labels/confidence only -- never target true labels."""
    Xt = np.asarray(Xt, float)
    Xt_a = Xt.copy()
    conf = np.ones(len(Xt)) if conf is None else np.asarray(conf, float)
    yt_pseudo = np.asarray(yt_pseudo)
    for c in np.unique(ys):
        ms = Xs[ys == c].mean(axis=0)
        cls = yt_pseudo == c
        confident = cls & (conf >= conf_thresh)
        if confident.sum() >= min_count:
            mt = Xt[confident].mean(axis=0)
            Xt_a[cls] = Xt[cls] - mt + ms
    return Xt_a

def fit_predict_pcma_bands(Ctr, ytr, subject_tr, Cte, subject_te,
                           ref_means_per_band=None, mode="iterative",
                           n_iter=3, conf_thresh=0.6, min_count=10):
    """PCMA: per-band RA recenter -> concat tangent -> source shrinkage-LDA ->
    alignment on the UNLABELED target. mode in {global, conditional, iterative}.
    'conditional' = one confidence-gated class-conditional pass; 'iterative' = up to
    n_iter passes, stopping on pseudo-label stability. Target labels never used."""
    Xtr, Xte = band_tangent_features(Ctr, Cte, subject_tr, subject_te,
                                     ref_means_per_band, recenter=True)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    if mode == "global":
        return clf.predict(global_mean_align(Xtr, Xte))
    yt = clf.predict(Xte)
    conf = clf.predict_proba(Xte).max(axis=1)
    iters = 1 if mode == "conditional" else int(n_iter)
    for _ in range(iters):
        Xte_a = conditional_mean_align(Xtr, ytr, Xte, yt, conf, conf_thresh, min_count)
        yt_new = clf.predict(Xte_a)
        conf = clf.predict_proba(Xte_a).max(axis=1)
        if np.array_equal(yt_new, yt):
            break
        yt = yt_new
    return yt

def _sym_pow(M, p, shrink):
    """Symmetric matrix power of a shrinkage-regularized covariance (PSD)."""
    d = M.shape[0]
    M = (1.0 - shrink) * M + shrink * (np.trace(M) / d) * np.eye(d)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** p)) @ V.T

def coral_align(Xs, Xt, shrink=0.1):
    """CORAL: align target second-order stats to source (unsupervised, no labels).
    Whiten target covariance then recolor to source covariance."""
    Xs = np.asarray(Xs, float); Xt = np.asarray(Xt, float)
    ms = Xs.mean(0); mt = Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False); Ct = np.cov(Xt, rowvar=False)
    Wt = _sym_pow(np.atleast_2d(Ct), -0.5, shrink)   # whiten target
    Rs = _sym_pow(np.atleast_2d(Cs),  0.5, shrink)   # recolor to source
    return (Xt - mt) @ Wt @ Rs + ms

def band_tangent_blocks(Ctr, Cte, subject_tr=None, subject_te=None,
                        ref_means_per_band=None, recenter=False):
    """Like band_tangent_features but returns per-band lists (before concat):
    (list_of_Xtr_per_band, list_of_Xte_per_band)."""
    nb = Ctr.shape[1]
    if recenter:
        def _rm(b):
            return None if ref_means_per_band is None else ref_means_per_band[b]
        Ctr = np.stack([recenter_per_subject(Ctr[:, b], subject_tr, ref_means=_rm(b)) for b in range(nb)], axis=1)
        Cte = np.stack([recenter_per_subject(Cte[:, b], subject_te, ref_means=_rm(b)) for b in range(nb)], axis=1)
    ftr, fte = [], []
    for b in range(nb):
        ts = TangentSpace(metric="riemann").fit(Ctr[:, b])
        ftr.append(ts.transform(Ctr[:, b]))
        fte.append(ts.transform(Cte[:, b]))
    return ftr, fte

def fit_predict_coral_bands(Ctr, ytr, subject_tr, Cte, subject_te,
                            ref_means_per_band=None, shrink=0.1):
    """Per-band RA recenter -> per-band tangent -> per-band CORAL (target->source,
    unsupervised) -> concat -> source shrinkage-LDA. No target labels used."""
    ftr, fte = band_tangent_blocks(Ctr, Cte, subject_tr, subject_te, ref_means_per_band, recenter=True)
    fte_c = [coral_align(ftr[b], fte[b], shrink) for b in range(len(ftr))]
    Xtr = np.concatenate(ftr, axis=1)
    Xte = np.concatenate(fte_c, axis=1)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    return clf.predict(Xte)
