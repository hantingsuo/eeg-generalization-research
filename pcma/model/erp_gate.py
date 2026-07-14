"""Gated LOSO decoding utilities for MODMA ERP features."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

from pcma.model.rich import per_subject_zscore


def bootstrap_ci(values, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        stats[i] = values[idx].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(stats.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def loso_erp_decode(F, y, subject, C=1.0):
    """Leave-one-subject-out ERP cue decoding with unsupervised subject z-score."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(y).astype(str)
    subject = np.asarray(subject).astype(str)
    subjects = np.asarray(list(dict.fromkeys(subject.tolist())))
    X = per_subject_zscore(F, subject)
    all_true: list[str] = []
    all_pred: list[str] = []
    rows = []
    labels_global = sorted(np.unique(y).tolist())
    for sid in subjects:
        train = subject != sid
        test = subject == sid
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=0)
        clf.fit(scaler.transform(X[train]), y[train])
        pred = clf.predict(scaler.transform(X[test]))
        labels, counts = np.unique(y[test], return_counts=True)
        majority = float(np.max(counts) / np.sum(counts))
        acc = float(accuracy_score(y[test], pred))
        bacc = float(recall_score(y[test], pred, labels=labels_global, average="macro", zero_division=0))
        rows.append(
            {
                "subject": str(sid),
                "n_trials": int(test.sum()),
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "majority_baseline": majority,
                "accuracy_minus_majority": acc - majority,
            }
        )
        all_true.extend(y[test].tolist())
        all_pred.extend(pred.tolist())

    accs = np.asarray([r["accuracy"] for r in rows], dtype=float)
    baccs = np.asarray([r["balanced_accuracy"] for r in rows], dtype=float)
    baselines = np.asarray([r["majority_baseline"] for r in rows], dtype=float)
    diffs = accs - baselines
    try:
        w = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        wilcoxon_p = float(w.pvalue)
        wilcoxon_stat = float(w.statistic)
    except ValueError:
        wilcoxon_p = 1.0
        wilcoxon_stat = 0.0
    return {
        "subjects": rows,
        "classes": labels_global,
        "n_subjects": int(len(subjects)),
        "n_trials": int(len(y)),
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "mean_balanced_accuracy": float(baccs.mean()),
        "std_balanced_accuracy": float(baccs.std(ddof=1)) if len(baccs) > 1 else 0.0,
        "mean_majority_baseline": float(baselines.mean()),
        "mean_accuracy_minus_majority": float(diffs.mean()),
        "accuracy_bootstrap_ci": bootstrap_ci(accs, seed=0),
        "accuracy_minus_majority_bootstrap_ci": bootstrap_ci(diffs, seed=1),
        "wilcoxon_accuracy_gt_majority": {"statistic": wilcoxon_stat, "p_value": wilcoxon_p},
        "confusion_matrix": confusion_matrix(all_true, all_pred, labels=labels_global).astype(int).tolist(),
    }


def gate1_passed(result, alpha=0.05):
    """Fixed GATE1: subject-level accuracy must beat majority baseline."""
    diff_ci = result["accuracy_minus_majority_bootstrap_ci"]
    return bool(
        result["mean_accuracy_minus_majority"] > 0.0
        and diff_ci["ci_low"] > 0.0
        and result["wilcoxon_accuracy_gt_majority"]["p_value"] < alpha
    )
