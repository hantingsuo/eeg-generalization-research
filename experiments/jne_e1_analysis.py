"""Analysis of the E1 SEED cross-subject anchor, exactly as frozen in the plan.

Selection always uses window accuracy on the frozen candidate grid (earliest
maximum). Participants are the inference unit: seeds and calibration rotations are
averaged within participant first.

  U_K  zero target labels: select on pooled validation participants, score on B
  A_K  calibration labels: select on the target's C trials, score on B
  O_K  diagnostic: select on B itself (not a deployable policy)
  F    fixed epoch 80, scored on B

Usage: python experiments/jne_e1_analysis.py [--plan P --out DIR --tag NAME] [--allow-partial]
The defaults analyse E1 (DGCNN/MLP); E2 (EEGNet) reuses the same frozen rules.
--allow-partial exists only to exercise the code on the pilot cell; its output is
marked partial and must not be reported.
"""
from pathlib import Path
import sys, json, argparse
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.jne_retention_curve import boot_ratio, boot_slope  # noqa: E402

PLAN = ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json"
OUT = ROOT / "results/jne_revision_2026-09/e1_seed_cross_subject"
REPORT = ROOT / "reports/jne_revision_2026-09"
BOOT_SEED = 20260917
N_BOOT = 10000


GRID = "nested"           # frozen plan grid; set to "common" for the common-time-range check
COMMON_FIRST_EPOCH = 16   # every budget starts here, so only the candidate count changes
COMMON_STEP = {5: 16, 9: 8, 17: 4, 33: 2, 65: 1}
COMMON_BUDGETS = [1, 5, 9, 17, 33, 65]


def candidates(k, epochs=80):
    """Candidate epochs (0-based) for budget k.

    "nested" is the frozen grid: epochs/k, 2*epochs/k, ..., epochs. It is nested,
    but the earliest candidate moves with k, so a wider budget also reaches
    earlier epochs. "common" keeps the same first and last epoch for every budget
    (COMMON_FIRST_EPOCH .. epochs) and only changes how densely that range is
    sampled; the sets are still nested. The battle plan of 2026-09-16 asked for
    the second as a check on the first.
    """
    if GRID == "common":
        if k == 1:
            return np.array([epochs - 1])
        c = np.arange(COMMON_FIRST_EPOCH - 1, epochs, COMMON_STEP[k])
        assert len(c) == k, (k, len(c))
        return c
    c = np.arange(epochs // k - 1, epochs, epochs // k)
    assert len(c) == k
    return c


def pick(curve, cands):
    return int(cands[int(np.argmax(curve[cands]))])  # argmax = earliest maximum


def acc_curve(correct, count, sel):
    return correct[:, sel].sum(1) / count[sel].sum()


def balanced_curve(correct, count, label, sel):
    per = []
    for c in np.unique(label[sel]):
        s = sel & (label == c)
        per.append(correct[:, s].sum(1) / count[s].sum())
    return np.mean(per, axis=0)


def trial_curve(logits, label, sel):
    return (logits[:, sel].argmax(2) == label[sel]).mean(1)


def cell_records(cfg, trace, budgets):
    """One record per (target participant, k, rotation) with every policy at every K."""
    perms = {int(c): v for c, v in cfg["calibration"]["class_trial_permutations"].items()}
    vc, vn = trace["validation_correct"], trace["validation_count"]
    val_acc = vc.sum(1) / vn.sum()
    tc, tn = trace["target_correct"], trace["target_count"]
    tl, tlab, units = trace["target_trial_logits"], trace["target_label"], trace["target_units"]
    last = tc.shape[0] - 1
    out = []
    for p in np.unique(units[:, 0]):
        own = units[:, 0] == p
        for k in (1, 2):
            for r in range(cfg["calibration"]["rotations"]):
                cal_trials = {perms[c][(r + j) % 5] for c in perms for j in range(k)}
                is_c = own & np.isin(units[:, 1], list(cal_trials))
                is_b = own & ~is_c
                assert is_c.sum() == 3 * k and is_b.sum() == 15 - 3 * k
                accB = acc_curve(tc, tn, is_b)
                accC = acc_curve(tc, tn, is_c)
                balB = balanced_curve(tc, tn, tlab, is_b)
                triB = trial_curve(tl, tlab, is_b)
                rec = {"participant": int(p), "k": k, "rotation": r, "K": {}}
                for K in budgets:
                    cand = candidates(K)
                    hV, hC, hB = pick(val_acc, cand), pick(accC, cand), pick(accB, cand)
                    rec["K"][K] = {
                        "U": accB[hV], "A": accB[hC], "O": accB[hB], "F": accB[last],
                        "U_bal": balB[hV], "A_bal": balB[hC], "F_bal": balB[last],
                        "U_trial": triB[hV], "A_trial": triB[hC], "F_trial": triB[last],
                        "apparent_U": val_acc[hV], "apparent_A": accC[hC],
                        "epoch_V": hV + 1, "epoch_C": hC + 1,
                    }
                out.append(rec)
    return out


def boot_mean_ci(v, rng):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    m = v[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def signflip_p(v, n_random=100000, seed=20260917):
    """Two-sided sign-flip permutation p for a participant-level mean.

    Exact enumeration up to 20 participants; beyond that, n_random random flips
    (including the observed sign pattern, so p is never zero).
    """
    v = np.asarray(v, float)
    n = len(v)
    if n > 20:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_random, n))
        stats = np.abs((signs * v).mean(1))
        return float(((stats >= abs(v.mean()) - 1e-12).sum() + 1) / (n_random + 1))
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    stats = np.abs((signs * v).mean(1))
    return float((stats >= abs(v.mean()) - 1e-12).mean())


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def summarise(cfg, rows, model, k, lo=5, hi=80):
    budgets = cfg["budgets"]
    by_p = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["k"] == k:
            by_p[r["participant"]].append(r)
    parts = sorted(by_p)
    fields = list(rows[0]["K"][budgets[0]].keys())

    def pv(K, field):
        return np.array([np.mean([r["K"][K][field] for r in by_p[p]]) for p in parts])

    rng = np.random.default_rng(BOOT_SEED)
    levels = {K: {f: {"mean": float(pv(K, f).mean()), "ci": boot_mean_ci(pv(K, f), rng)} for f in fields}
              for K in budgets}
    contrasts = {
        f"U{hi}-U{lo}": pv(hi, "U") - pv(lo, "U"),
        f"A{hi}-A{lo}": pv(hi, "A") - pv(lo, "A"),
        f"A{hi}-U{hi}": pv(hi, "A") - pv(hi, "U"),
    }
    ps = [signflip_p(v) for v in contrasts.values()]
    primary = {}
    for (name, v), p, pa in zip(contrasts.items(), ps, holm(ps)):
        primary[name] = {"mean_pp": float(v.mean() * 100), "ci_pp": [x * 100 for x in boot_mean_ci(v, rng)],
                         "signflip_p": p, "holm_p": pa, "participant_pp": (v * 100).tolist()}
    diagnostic = {
        f"O{hi}-U{hi}": pv(hi, "O") - pv(hi, "U"),
        f"O{hi}-O{lo}": pv(hi, "O") - pv(lo, "O"),
        f"U{hi}-F": pv(hi, "U") - pv(hi, "F"),
        f"A{hi}-F": pv(hi, "A") - pv(hi, "F"),
        f"U{hi}-U{lo}_balanced": pv(hi, "U_bal") - pv(lo, "U_bal"),
        f"A{hi}-A{lo}_balanced": pv(hi, "A_bal") - pv(lo, "A_bal"),
        f"U{hi}-U{lo}_trial": pv(hi, "U_trial") - pv(lo, "U_trial"),
        f"A{hi}-A{lo}_trial": pv(hi, "A_trial") - pv(lo, "A_trial"),
    }
    diagnostic = {n: {"mean_pp": float(v.mean() * 100), "ci_pp": [x * 100 for x in boot_mean_ci(v, rng)]}
                  for n, v in diagnostic.items()}
    retention = {}
    for policy in ("U", "A"):
        retention[policy] = {}
        for K in budgets:
            if K <= lo:
                continue
            dS = pv(K, f"apparent_{policy}") - pv(lo, f"apparent_{policy}")
            dC = pv(K, policy) - pv(lo, policy)
            mS = float(dS.mean())
            vx = dS.var()
            retention[policy][K] = {
                "apparent_pp": mS * 100, "retained_pp": float(dC.mean() * 100),
                "ratio": float(dC.mean() / mS) if abs(mS) > 1e-9 else None,
                "ratio_ci": list(boot_ratio(dC, dS, rng)) if abs(mS) > 1e-9 else [None, None],
                "slope": float(np.cov(dS, dC, bias=True)[0, 1] / vx) if vx > 1e-12 else None,
                "slope_ci": list(boot_slope(dS, dC, rng)) if vx > 1e-12 else [None, None],
            }
    seeds = {}
    for s in cfg["optimization_seeds"]:
        sub = defaultdict(list)
        for r in rows:
            if r["model"] == model and r["k"] == k and r["seed"] == s:
                sub[r["participant"]].append(r)
        u = np.mean([np.mean([x["K"][hi]["U"] - x["K"][lo]["U"] for x in sub[p]]) for p in sub])
        a = np.mean([np.mean([x["K"][hi]["A"] - x["K"][lo]["A"] for x in sub[p]]) for p in sub])
        f = np.mean([np.mean([x["K"][hi]["F"] for x in sub[p]]) for p in sub])
        seeds[s] = {f"U{hi}-U{lo}_pp": float(u * 100), f"A{hi}-A{lo}_pp": float(a * 100),
                    "F_acc": float(f)}
    return {"n_participants": len(parts), "levels": levels, "primary": primary,
            "diagnostic": diagnostic, "retention": retention, "seeds": seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="e1")
    a = ap.parse_args()
    plan, out = a.plan, a.out
    cfg = json.loads(plan.read_text(encoding="utf-8"))
    expected = len(cfg["folds"]) * len(cfg["models"]) * len(cfg["optimization_seeds"])
    results = sorted((out / "cells").glob("*/result.json"))
    partial = len(results) != expected
    if partial and not a.allow_partial:
        raise SystemExit(f"only {len(results)}/{expected} cells complete")
    rows = []
    for rp in results:
        meta = json.loads(rp.read_text())
        with np.load(rp.parent / "trace.npz") as z:
            trace = {k: z[k] for k in z.files}
        for rec in cell_records(cfg, trace, cfg["budgets"]):
            rec.update(model=meta["model"], seed=meta["seed"], fold=meta["fold"])
            rows.append(rec)
    summary = {"status": "PARTIAL - do not report" if partial else "complete",
               "cells": len(results), "plan": plan.resolve().relative_to(ROOT).as_posix(), "results": {}}
    for model in cfg["models"]:
        if not any(r["model"] == model for r in rows):
            continue
        summary["results"][model] = {f"k{k}": summarise(cfg, rows, model, k) for k in (1, 2)}
    name = f"{a.tag}_summary_partial.json" if partial else f"{a.tag}_summary.json"
    (out / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if partial:
        print("partial summary written for code checking only:", out / name)
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("written", REPORT / name)


if __name__ == "__main__":
    main()
