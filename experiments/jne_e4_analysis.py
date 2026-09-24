"""Analysis of E4 (matched same- vs cross-session scoring), as frozen in the plan.

For one trajectory and one audit session q, with pools A and B of the rotation:
  S_K = (a_A(h_A) + a_B(h_B)) / 2      own-pool selected score
  C_K = (a_B(h_A) + a_A(h_B)) / 2      exchanged-pool score
  V_K = (a_A(h_V) + a_B(h_V)) / 2      validation-selected
These are the rotation_v2 definitions. The same relation uses q = s; the cross
relation averages the two sessions q != s within the cell.

Also checks that the seed-2024 MLP same-relation values reproduce rotation_v2.

Usage: python experiments/jne_e4_analysis.py [--allow-partial]
"""
from pathlib import Path
import sys, json, argparse
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.jne_e1_analysis import candidates, pick, boot_mean_ci, signflip_p, holm, N_BOOT  # noqa: E402
from experiments.jne_retention_curve import boot_ratio, boot_slope  # noqa: E402

PLAN = ROOT / "plans/2026-09-17-jne-e4-matched-session.json"
OUT = ROOT / "results/jne_revision_2026-09/e4_matched_session"
V2 = ROOT / "results/revision_2026-09-10/selection_rotation_v2/cells"
REPORT = ROOT / "reports/jne_revision_2026-09"
BOOT_SEED = 20260917


def pool_curve(correct, count, units, session, trials):
    sel = (units[:, 0] == session) & np.isin(units[:, 1], trials)
    assert sel.sum() == len(trials)
    return correct[:, sel].sum(1) / count[sel].sum()


def quantities(aA, aB, aV, budgets):
    out = {}
    last = len(aA) - 1
    for K in budgets:
        c = candidates(K, len(aA))
        hA, hB, hV = pick(aA, c), pick(aB, c), pick(aV, c)
        out[K] = {"S": (aA[hA] + aB[hB]) / 2, "C": (aB[hA] + aA[hB]) / 2,
                  "V": (aA[hV] + aB[hV]) / 2, "F": (aA[last] + aB[last]) / 2}
    return out


def cell_values(cfg, meta, trace):
    rot = cfg["rotations"][meta["rotation"]]
    s = meta["train_session"]
    vc, vn = trace["validation_correct"], trace["validation_count"]
    aV = vc.sum(1) / vn.sum()
    ac, an, au = trace["audit_correct"], trace["audit_count"], trace["audit_units"]
    per_q = {}
    for q in cfg["audit_sessions"]:
        aA = pool_curve(ac, an, au, q, rot["A"])
        aB = pool_curve(ac, an, au, q, rot["B"])
        per_q[q] = quantities(aA, aB, aV, cfg["budgets"])
    same = per_q[s]
    others = [per_q[q] for q in cfg["audit_sessions"] if q != s]
    cross = {K: {m: float(np.mean([o[K][m] for o in others])) for m in ("S", "C", "V", "F")}
             for K in cfg["budgets"]}
    return {"same": same, "cross": cross}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    expected = 5 * len(cfg["models"]) * len(cfg["training_sessions"]) * 15 * len(cfg["optimization_seeds"])
    paths = sorted((OUT / "cells").glob("*/result.json"))
    partial = len(paths) != expected
    if partial and not a.allow_partial:
        raise SystemExit(f"only {len(paths)}/{expected} cells complete")
    rows = []
    for rp in paths:
        meta = json.loads(rp.read_text())
        with np.load(rp.parent / "trace.npz") as z:
            trace = {k: z[k] for k in z.files}
        rows.append({**meta, "values": cell_values(cfg, meta, trace)})

    # Reproduction check against rotation_v2 (seed 2024 MLP, same relation).
    repro = {"cells": 0, "max_abs_diff": 0.0, "mismatches": []}
    std_effect_rows = []
    for r in rows:
        if r["seed"] != 2024:
            continue
        v2p = V2 / f"r{r['rotation']}_{r['model']}_s{r['train_session']}_p{r['subject']:02d}" / "result.json"
        if not v2p.exists():
            continue
        v2 = json.loads(v2p.read_text())["results"]
        if r["model"] == "mlp":
            repro["cells"] += 1
            for K in cfg["budgets"]:
                for m, key in (("S", "own_selected"), ("C", "cross_selected"), ("V", "validation"), ("F", "fixed")):
                    d = abs(r["values"]["same"][K][m] - v2[str(K)]["window"][key])
                    repro["max_abs_diff"] = max(repro["max_abs_diff"], d)
                    if d > 1e-9 and len(repro["mismatches"]) < 20:
                        repro["mismatches"].append([r["cell"], K, m, d])
        else:
            std_effect_rows.append((r, v2))

    summary = {"status": "PARTIAL - do not report" if partial else "complete", "cells": len(paths),
               "reproduction_check_mlp_seed2024_vs_rotation_v2": repro, "results": {}}
    rng = np.random.default_rng(BOOT_SEED)
    for model in cfg["models"]:
        by_p = defaultdict(list)
        for r in rows:
            if r["model"] == model:
                by_p[r["subject"]].append(r)
        if not by_p:
            continue
        parts = sorted(by_p)

        def pv(rel, K, m):
            return np.array([np.mean([x["values"][rel][K][m] for x in by_p[p]]) for p in parts])

        res = {"n_participants": len(parts), "relations": {}}
        for rel in ("same", "cross"):
            item = {"levels": {K: {m: float(pv(rel, K, m).mean()) for m in ("S", "C", "V", "F")}
                               for K in cfg["budgets"]}}
            for name, m in (("S80-S5", "S"), ("C80-C5", "C"), ("V80-V5", "V")):
                v = pv(rel, 80, m) - pv(rel, 5, m)
                item[name] = {"mean_pp": float(v.mean() * 100), "ci_pp": [x * 100 for x in boot_mean_ci(v, rng)]}
            item["retention"] = {}
            for K in cfg["budgets"]:
                if K <= 5:
                    continue
                dS = pv(rel, K, "S") - pv(rel, 5, "S")
                dC = pv(rel, K, "C") - pv(rel, 5, "C")
                vx = dS.var()
                item["retention"][K] = {
                    "apparent_pp": float(dS.mean() * 100), "retained_pp": float(dC.mean() * 100),
                    "ratio": float(dC.mean() / dS.mean()) if abs(dS.mean()) > 1e-9 else None,
                    "ratio_ci": list(boot_ratio(dC, dS, rng)),
                    "slope": float(np.cov(dS, dC, bias=True)[0, 1] / vx) if vx > 1e-12 else None,
                    "slope_ci": list(boot_slope(dS, dC, rng)) if vx > 1e-12 else [None, None],
                }
            res["relations"][rel] = item
        prim = {
            "dC_cross_minus_same": (pv("cross", 80, "C") - pv("cross", 5, "C")) - (pv("same", 80, "C") - pv("same", 5, "C")),
            "dS_cross_minus_same": (pv("cross", 80, "S") - pv("cross", 5, "S")) - (pv("same", 80, "S") - pv("same", 5, "S")),
        }
        ps = [signflip_p(v) for v in prim.values()]
        res["primary"] = {n: {"mean_pp": float(v.mean() * 100), "ci_pp": [x * 100 for x in boot_mean_ci(v, rng)],
                              "signflip_p": p, "holm_p": h, "participant_pp": (v * 100).tolist()}
                          for (n, v), p, h in zip(prim.items(), ps, holm(ps))}
        res["level_gap_cross_minus_same_F"] = float((pv("cross", 80, "F") - pv("same", 80, "F")).mean() * 100)
        res["seeds"] = {}
        for sd in cfg["optimization_seeds"]:
            sub = [x for x in rows if x["model"] == model and x["seed"] == sd]
            if sub:
                res["seeds"][sd] = {rel: float(np.mean([x["values"][rel][80]["C"] - x["values"][rel][5]["C"] for x in sub]) * 100)
                                    for rel in ("same", "cross")}
        summary["results"][model] = res

    if std_effect_rows:
        by_p = defaultdict(list)
        for r, v2 in std_effect_rows:
            e4d = r["values"]["same"][80]["C"] - r["values"]["same"][5]["C"]
            v2d = v2["80"]["window"]["cross_selected"] - v2["5"]["window"]["cross_selected"]
            e4s = r["values"]["same"][80]["S"] - r["values"]["same"][5]["S"]
            v2s = v2["80"]["window"]["own_selected"] - v2["5"]["window"]["own_selected"]
            by_p[r["subject"]].append((e4d - v2d, e4s - v2s))
        dc = np.array([np.mean([t[0] for t in by_p[p]]) for p in sorted(by_p)])
        ds = np.array([np.mean([t[1] for t in by_p[p]]) for p in sorted(by_p)])
        summary["dgcnn_standardisation_effect_seed2024_same_session"] = {
            "n_participants": len(dc),
            "dC_standardised_minus_raw_pp": {"mean": float(dc.mean() * 100), "ci": [x * 100 for x in boot_mean_ci(dc, rng)]},
            "dS_standardised_minus_raw_pp": {"mean": float(ds.mean() * 100), "ci": [x * 100 for x in boot_mean_ci(ds, rng)]},
        }

    name = "e4_summary_partial.json" if partial else "e4_summary.json"
    (OUT / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if not partial:
        (REPORT / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("written", name, "| reproduction:", repro["cells"], "cells, max |diff|", repro["max_abs_diff"])


if __name__ == "__main__":
    main()
