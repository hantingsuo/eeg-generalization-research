"""Analysis of the FACED cells, following the frozen FACED plan.

Standard condition (all 28 clips seen by source participants):
  C_r = one clip per class from calibration.standard.permutations[c][r], r = 0..2
  B   = the target's other 19 clips; U selects on all validation clips.
Stimulus contrast on B* (one held-out clip per class):
  seen   = standard cells, unseen = holdout cells
  C_r = calibration.holdout.permutations[c][r], r = 0..1 (never B*)
  B   = B*; U selects on validation clips excluding B* in both conditions.

Usage: python experiments/jne_faced_analysis.py [--allow-partial]   (FACED_SMOKE=1 for the synthetic layout)
"""
from pathlib import Path
import sys, json, argparse
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import jne_e1_faced as faced  # noqa: E402
from experiments.jne_e1_analysis import (candidates, pick, acc_curve, balanced_curve, trial_curve,  # noqa: E402
                                         summarise, boot_mean_ci, signflip_p, holm)

REPORT = ROOT / "reports/jne_revision_2026-09"


def records(trace, val_clips, cal_sets, b_clips, budgets):
    """cal_sets: list over rotations of clip-id sets; b_clips: set or None (= complement of C)."""
    vu, vc, vn = trace["validation_units"], trace["validation_correct"], trace["validation_count"]
    vsel = np.isin(vu[:, 1], list(val_clips))
    val_acc = acc_curve(vc, vn, vsel)
    tu, tc, tn = trace["target_units"], trace["target_correct"], trace["target_count"]
    tl, tlab = trace["target_trial_logits"], trace["target_label"]
    epochs = tc.shape[0]
    out = []
    for p in np.unique(tu[:, 0]):
        own = tu[:, 0] == p
        for r, cset in enumerate(cal_sets):
            is_c = own & np.isin(tu[:, 1], list(cset))
            is_b = own & (np.isin(tu[:, 1], list(b_clips)) if b_clips is not None else ~is_c)
            assert not (is_c & is_b).any() and is_c.sum() == len(cset)
            accB, accC = acc_curve(tc, tn, is_b), acc_curve(tc, tn, is_c)
            balB, triB = balanced_curve(tc, tn, tlab, is_b), trial_curve(tl, tlab, is_b)
            rec = {"participant": int(p), "k": 1, "rotation": r, "K": {}}
            for K in budgets:
                cand = candidates(K, epochs)
                hV, hC, hB = pick(val_acc, cand), pick(accC, cand), pick(accB, cand)
                rec["K"][K] = {"U": accB[hV], "A": accB[hC], "O": accB[hB], "F": accB[-1],
                               "U_bal": balB[hV], "A_bal": balB[hC], "F_bal": balB[-1],
                               "U_trial": triB[hV], "A_trial": triB[hC], "F_trial": triB[-1],
                               "apparent_U": val_acc[hV], "apparent_A": accC[hC],
                               "epoch_V": hV + 1, "epoch_C": hC + 1}
            out.append(rec)
    return out


def load_cells(cond):
    rows = {}
    for rp in sorted((faced.OUT / "cells").glob(f"{cond}_*/result.json")):
        meta = json.loads(rp.read_text())
        with np.load(rp.parent / "trace.npz") as z:
            rows[(meta["model"], meta["seed"])] = {k: z[k] for k in z.files}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(faced.PLAN.read_text(encoding="utf-8"))
    budgets = cfg["budgets"]
    expected = len(cfg["conditions"]) * len(cfg["models"]) * len(cfg["optimization_seeds"])
    n_done = len(list((faced.OUT / "cells").glob("*/result.json")))
    partial = n_done != expected
    if partial and not a.allow_partial:
        raise SystemExit(f"only {n_done}/{expected} cells complete")
    classes = range(len(cfg["classes"]))
    all_clips = set(range(1, faced.N_CLIPS + 1))
    bstar = {int(v) for v in cfg["stimulus_holdout"]["clip_per_class"].values()}
    ps = cfg["calibration"]["standard"]["permutations"]
    ph = cfg["calibration"]["holdout"]["permutations"]
    cal_std = [{ps[str(c)][r] for c in classes} for r in range(cfg["calibration"]["standard"]["rotations"])]
    cal_ho = [{ph[str(c)][r] for c in classes} for r in range(cfg["calibration"]["holdout"]["rotations"])]
    scfg = {"budgets": budgets, "optimization_seeds": cfg["optimization_seeds"]}

    std_cells, ho_cells = load_cells("standard"), load_cells("holdout")
    sets = {"standard": [], "seen": [], "unseen": []}
    for (m, s), tr in std_cells.items():
        for r in records(tr, all_clips, cal_std, None, budgets):
            sets["standard"].append({**r, "model": m, "seed": s})
        for r in records(tr, all_clips - bstar, cal_ho, bstar, budgets):
            sets["seen"].append({**r, "model": m, "seed": s})
    for (m, s), tr in ho_cells.items():
        for r in records(tr, all_clips - bstar, cal_ho, bstar, budgets):
            sets["unseen"].append({**r, "model": m, "seed": s})

    summary = {"status": "PARTIAL - do not report" if partial else "complete", "cells": n_done,
               "plan": faced.PLAN.relative_to(ROOT).as_posix(), "results": {}}
    rng = np.random.default_rng(20260917)
    lo, hi = min(b for b in budgets if b > 1), max(budgets)
    for m in cfg["models"]:
        res = {}
        for name, rows in sets.items():
            rr = [r for r in rows if r["model"] == m]
            if rr:
                res[name] = summarise(scfg, rr, m, 1)
        if all(sets[x] for x in ("seen", "unseen")) and any(r["model"] == m for r in sets["unseen"]):
            def pv(name, K, f):
                by = defaultdict(list)
                for r in sets[name]:
                    if r["model"] == m:
                        by[r["participant"]].append(r["K"][K][f])
                return np.array([np.mean(by[p]) for p in sorted(by)])
            contr = {
                "(A_hi-A_lo)_unseen-(A_hi-A_lo)_seen": (pv("unseen", hi, "A") - pv("unseen", lo, "A"))
                                                       - (pv("seen", hi, "A") - pv("seen", lo, "A")),
                "(U_hi-U_lo)_unseen-(U_hi-U_lo)_seen": (pv("unseen", hi, "U") - pv("unseen", lo, "U"))
                                                       - (pv("seen", hi, "U") - pv("seen", lo, "U")),
            }
            pvals = [signflip_p(v) for v in contr.values()]
            res["stimulus_contrast"] = {
                n: {"mean_pp": float(v.mean() * 100), "ci_pp": [x * 100 for x in boot_mean_ci(v, rng)],
                    "signflip_p": p, "holm_p": h}
                for (n, v), p, h in zip(contr.items(), pvals, holm(pvals))}
            res["stimulus_levels_pp"] = {
                f"{pol}{K}_unseen-seen": float((pv("unseen", K, pol) - pv("seen", K, pol)).mean() * 100)
                for pol in ("U", "A", "F") for K in (hi,)}
        summary["results"][m] = res

    name = f"{faced.TAG}_summary{'_partial' if partial else ''}.json"
    (faced.OUT / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if not partial and not faced.SMOKE:
        (REPORT / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("written", faced.OUT / name)


if __name__ == "__main__":
    main()
