# pcma/model/rich.py
"""Strong classifier on rich features: per-subject z-score (unsupervised) ->
global standardize -> RBF-SVM. Cross-subject / cross-population ready."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def per_subject_zscore(F, subject):
    """Standardize each subject's feature rows by that subject's own mean/std.
    Unsupervised (uses only subject ids), so it is leakage-free for held-out subjects."""
    F = np.asarray(F, float)
    out = np.empty_like(F)
    subject = np.asarray(subject)
    for s in np.unique(subject):
        m = subject == s
        mu = F[m].mean(axis=0)
        sd = F[m].std(axis=0) + 1e-8
        out[m] = (F[m] - mu) / sd
    return out

def fit_predict_rich(Ftr, ytr, subject_tr, Fte, subject_te, C=1.0, gamma="scale", return_proba=False):
    """Per-subject z-score -> global StandardScaler (fit on train) -> RBF-SVM.
    Target labels never used."""
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    sc = StandardScaler().fit(Ftr)
    clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=return_proba, random_state=0).fit(sc.transform(Ftr), ytr)
    Xte = sc.transform(Fte)
    pred = clf.predict(Xte)
    if return_proba:
        return pred, clf.predict_proba(Xte), clf.classes_
    return pred

def fit_predict_rich_ensemble(Ftr, ytr, subject_tr, Fte, subject_te, k=200, C=1.0, gamma="scale"):
    """Per-subject z-score -> global StandardScaler -> SelectKBest(f_classif, k)
    -> soft-vote of RBF-SVM and HistGradientBoosting. Target labels never used."""
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    sc = StandardScaler().fit(Ftr)
    Xtr = sc.transform(Ftr); Xte = sc.transform(Fte)
    kk = min(k, Xtr.shape[1])
    sel = SelectKBest(f_classif, k=kk).fit(Xtr, ytr)
    Xtr = sel.transform(Xtr); Xte = sel.transform(Xte)
    svm = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=0).fit(Xtr, ytr)
    hgb = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    proba = 0.5 * svm.predict_proba(Xte) + 0.5 * hgb.predict_proba(Xte)
    return svm.classes_[proba.argmax(axis=1)]
