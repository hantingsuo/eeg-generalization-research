"""Common-time-range grid for the within- and cross-session trajectories.

The rotation (300 trajectories) and cross-session (276 trajectories) studies store,
in every trace.npz, the window accuracy of each of the 80 epochs on the validation,
A and B pools. That is all the selection rule needs, so the common grid of
jne_common_range.py can be applied to them without refitting.

Step 1 rebuilds the frozen-grid own/cross scores from those per-epoch accuracies and
checks them against the values stored in every result.json; the run stops if any
differs. Step 2 applies the common grid (epochs 16..80, K = 5 -> 65) with the same
selection rule (highest score, earliest epoch on ties) and the same participant
averaging and bootstrap as jne_retention_curve.py.

Post hoc sensitivity analysis on trajectories whose results were already known.

Usage: python experiments/jne_common_range_sessions.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    "same_session": ROOT / "results/revision_2026-09-10/selection_rotation_v2/cells",
    "cross_session": ROOT / "results/revision_2026-09-11/seed_family_extension_v4/cells",
}
OUT = ROOT / "reports/jne_revision_2026-09/common_time_range_sessions.json"

FROZEN = {k: [80 // k * i for i in range(1, k + 1)] for k in (5, 10, 20, 40, 80)}
COMMON = {k: list(range(16, 81, step)) for k, step in ((5, 16), (9, 8), (17, 4), (33, 2), (65, 1))}
LO, HI = 5, 65
POOL = {"validation": 0, "A": 1, "B": 2}
BOOT_SEED = 20260916
N_BOOT = 10000


def pick(scores, epochs):
    """Highest score among the candidate epochs, earliest on ties (epochs are 1-based)."""
    best = max(scores[e - 1] for e in epochs)
    return next(e for e in epochs if scores[e - 1] == best)


def own_cross(window, epochs):
    """Select on A, score A and B; select on B, score B and A; average the two roles."""
    a, b = window[:, POOL["A"]], window[:, POOL["B"]]
    ea, eb = pick(a, epochs), pick(b, epochs)
    return (a[ea - 1] + b[eb - 1]) / 2, (b[ea - 1] + a[eb - 1]) / 2


def boot(num, den, rng):
    """Participant bootstrap of the aggregate ratio and of the mean retained gain."""
    ratios, means = [], []
    n = len(num)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        means.append(num[i].mean())
        d = den[i].mean()
        if abs(d) > 1e-9:
            ratios.append(num[i].mean() / d)
    return (np.percentile(ratios, [2.5, 97.5]).tolist(),
            np.percentile(means, [2.5, 97.5]).tolist())


def main():
    groups = defaultdict(lambda: defaultdict(list))
    checks = {}
    for study, root in SETS.items():
        cells = sorted(root.glob("*/result.json"))
        worst = 0.0
        for p in cells:
            r = json.loads(p.read_text())
            with np.load(p.parent / "trace.npz") as z:
                w = z["window"]
            for k, ep in FROZEN.items():
                o, c = own_cross(w, ep)
                ref = r["results"][str(k)]["window"]
                worst = max(worst, abs(o - ref["own_selected"]), abs(c - ref["cross_selected"]))
            key = f"{study}/{r.get('dataset', 'seed')}/{r['model']}"
            groups[key][r["subject"]].append({k: own_cross(w, ep) for k, ep in COMMON.items()})
        if worst != 0.0:
            raise SystemExit(f"{study}: frozen-grid reconstruction differs by {worst}")
        checks[study] = {"cells": len(cells), "frozen_grid_max_abs_diff": worst}

    rng = np.random.default_rng(BOOT_SEED)
    out = {
        "provenance": {
            "script": "experiments/jne_common_range_sessions.py",
            "reconstruction_check": checks,
            "grid": {str(k): v for k, v in COMMON.items()},
            "comparison": f"K = {LO} -> {HI}",
            "bootstrap_seed": BOOT_SEED,
            "n_bootstrap": N_BOOT,
            "boundary": ("Post hoc sensitivity analysis of trajectories whose results were "
                         "already known; no model is refitted."),
        },
        "groups": {},
    }
    for key in sorted(groups):
        subj = groups[key]
        ids = sorted(subj)

        def arr(k, j):
            return np.array([np.mean([rec[k][j] for rec in subj[s]]) for s in ids])

        dS = arr(HI, 0) - arr(LO, 0)
        dC = arr(HI, 1) - arr(LO, 1)
        ratio_ci, ret_ci = boot(dC, dS, rng)
        out["groups"][key] = {
            "n_participants": len(ids),
            "apparent_gain_pp": float(dS.mean() * 100),
            "retained_gain_pp": float(dC.mean() * 100),
            "retained_gain_ci_pp": [x * 100 for x in ret_ci],
            "retention_ratio": float(dC.mean() / dS.mean()),
            "retention_ratio_ci": ratio_ci,
        }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key, g in out["groups"].items():
        print(f"{key:30s} app {g['apparent_gain_pp']:5.2f}  ret {g['retained_gain_pp']:+5.2f} "
              f"({g['retained_gain_ci_pp'][0]:+.2f} to {g['retained_gain_ci_pp'][1]:+.2f})  "
              f"ratio {g['retention_ratio']:+.2f} ({g['retention_ratio_ci'][0]:+.2f} to "
              f"{g['retention_ratio_ci'][1]:+.2f})")
    print("written", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
