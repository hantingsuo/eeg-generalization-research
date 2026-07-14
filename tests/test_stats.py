# tests/test_stats.py
import numpy as np
from pcma.eval.stats import paired_wilcoxon, holm_correction

def test_wilcoxon_detects_difference():
    a = np.array([0.70, 0.72, 0.68, 0.71, 0.69])
    b = np.array([0.60, 0.62, 0.59, 0.61, 0.58])   # a consistently higher
    stat, p = paired_wilcoxon(a, b)
    assert p < 0.1

def test_holm_orders_and_bounds():
    ps = [0.001, 0.04, 0.03]
    adj = holm_correction(ps)
    assert len(adj) == 3
    assert all(0.0 <= x <= 1.0 for x in adj)
    assert adj[0] <= adj[1]  # smallest raw p gets smallest adjusted-ish ordering preserved

def test_bootstrap_paired_accuracy_detects_improvement():
    import numpy as np
    from pcma.eval.stats import bootstrap_paired_accuracy
    y = np.array([0, 1] * 50)
    a = y.copy(); a[:20] = 1 - a[:20]           # method A: 20 wrong
    b = y.copy(); b[:5] = 1 - b[:5]             # method B: 5 wrong (better)
    r = bootstrap_paired_accuracy(y, a, b, n_boot=500)
    assert r["mean_diff"] > 0 and r["frac_b_gt_a"] > 0.9
    assert r["ci_low"] <= r["mean_diff"] <= r["ci_high"]

def test_paired_wilcoxon_all_equal_returns_p1():
    import numpy as np
    from pcma.eval.stats import paired_wilcoxon
    a = [0.6, 0.7, 0.65]; b = list(a)   # identical -> all-zero differences
    stat, p = paired_wilcoxon(a, b)
    assert p == 1.0
