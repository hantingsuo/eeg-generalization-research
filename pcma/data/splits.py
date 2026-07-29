# pcma/data/splits.py
import numpy as np
from sklearn.model_selection import GroupKFold


def p1_hc_to_dep(subject, population):
    tr = np.where(population == "HC")[0]
    te = np.where(population == "DEP")[0]
    return tr, te


def p2_dep_to_hc(subject, population):
    tr = np.where(population == "DEP")[0]
    te = np.where(population == "HC")[0]
    return tr, te


def p3_mixed_cv(subject, n_splits=5):
    """Same-population-agnostic cross-subject CV; groups = subject (no leakage)."""
    idx = np.arange(len(subject))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(idx, groups=subject):
        yield tr, te


def p4_leave_one_dep(subject, population):
    """One fold per DEP subject: test = that DEP subject, train = everyone else."""
    dep_subjects = np.unique(subject[population == "DEP"])
    idx = np.arange(len(subject))
    for s in dep_subjects:
        te = idx[subject == s]
        tr = idx[subject != s]
        yield tr, te
