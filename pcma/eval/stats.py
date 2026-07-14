# pcma/eval/stats.py
import numpy as np
from scipy.stats import wilcoxon

def paired_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.all(a == b):
        return 0.0, 1.0
    stat, p = wilcoxon(a, b)
    return float(stat), float(p)

def holm_correction(pvalues):
    """Holm-Bonferroni. Returns adjusted p-values in the ORIGINAL order."""
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()

def bootstrap_paired_accuracy(y_true, pred_a, pred_b, n_boot=2000, seed=0):
    """Paired bootstrap of acc(b)-acc(a) on the same test set. Returns mean diff,
    95% CI, and fraction of resamples where b beats a."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = (np.asarray(pred_a) == y); b = (np.asarray(pred_b) == y)
    n = len(y); diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "frac_b_gt_a": float((diffs > 0).mean())}


def bootstrap_two_sample_mean_difference(a, b, n_boot=10000, seed=0):
    """Bootstrap mean(a)-mean(b) when rows are independent analysis units.

    This helper is intended for subject-level summaries.  It deliberately does
    not accept segment-level predictions because repeated observations from one
    subject must first be reduced to a single subject-level estimand.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(b) == 0:
        raise ValueError("a and b must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("a and b must contain only finite values")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        diffs[i] = aa.mean() - bb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    observed = float(a.mean() - b.mean())
    return {
        "mean_diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def permutation_two_sample_mean_difference(a, b, n_perm=10000, seed=0, alternative="two-sided"):
    """Permutation test for mean(a)-mean(b) on independent analysis units."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(b) == 0:
        raise ValueError("a and b must be non-empty one-dimensional arrays")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[: len(a)].mean() - perm[len(a) :].mean())
        if alternative == "two-sided":
            extreme += abs(diff) >= abs(observed)
        elif alternative == "greater":
            extreme += diff >= observed
        else:
            extreme += diff <= observed
    return {
        "mean_diff": observed,
        "p_value": float((extreme + 1) / (n_perm + 1)),
        "alternative": alternative,
        "n_perm": int(n_perm),
    }
