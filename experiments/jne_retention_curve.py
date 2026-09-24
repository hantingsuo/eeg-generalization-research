"""Retention of checkpoint-search gains: how much of an apparent improvement survives.

Pure re-analysis of the 576 frozen trajectories (300 same-session rotation_v2 +
276 cross-session extension_v4). No model is refitted; every number is read from
the per-cell result.json contrasts that were written when those runs completed.

The quantity of interest is not whether selection inflates a score -- that sign is
algebraic. It is the *retention* of an apparent gain:

    dS_K = S_K - S_5   apparent gain from widening the candidate budget 5 -> K,
                       measured on the pool that did the selecting
    dC_K = C_K - C_5   the same widening, measured on the exchanged pool

    rho_K = mean(dC_K) / mean(dS_K)          aggregate retention ratio
    beta  = OLS slope of dC_K on dS_K        per-participant retention slope

beta is the deliverable: for every one point of apparent gain a reader sees
reported, beta points are what showed up on trials that did not do the selecting.
Ratios are formed from aggregated numerator/denominator, never per participant --
per-participant dS_K sits near zero often enough that individual ratios diverge.

Participants are the inference unit. Rotations and sessions are dependent repeats
and are averaged within participant before anything else happens.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAME = ROOT / "results/revision_2026-09-10/selection_rotation_v2/cells"
CROSS = ROOT / "results/revision_2026-09-11/seed_family_extension_v4/cells"
OUT = ROOT / "results/jne_revision_2026-09"

BUDGETS = [1, 5, 10, 20, 40, 80]
BASE = 5  # reference budget; gains are measured relative to this
METRIC = "window"
BOOT_SEED = 20260916
N_BOOT = 10000


def load_cells():
    """Read every frozen cell into flat records keyed by its participant group."""
    rows = []
    for p in sorted(SAME.glob("*/result.json")):
        r = json.loads(p.read_text())
        rows.append(
            dict(
                study="same_session",
                dataset="seed",
                model=r["model"],
                # sessions are separate recordings of the same person: the paper
                # treats participant as the unit, so session folds in as a repeat
                subject=r["subject"],
                repeat=(r["rotation"], r["session"]),
                results=r["results"],
            )
        )
    for p in sorted(CROSS.glob("*/result.json")):
        r = json.loads(p.read_text())
        rows.append(
            dict(
                study="cross_session",
                dataset=r["dataset"],
                model=r["model"],
                subject=r["subject"],
                repeat=(r["rotation"],),
                results=r["results"],
            )
        )
    return rows


def per_participant(rows, study, dataset, model):
    """Average dependent repeats within participant, then return the paired table.

    Returns {budget: {'own': array_over_participants, 'cross': ..., ...}}.
    """
    sel = [
        r
        for r in rows
        if r["study"] == study and r["dataset"] == dataset and r["model"] == model
    ]
    if not sel:
        return None, 0
    bucket = defaultdict(list)
    for r in sel:
        bucket[r["subject"]].append(r)
    subjects = sorted(bucket)
    table = {}
    for k in BUDGETS:
        cols = {}
        for policy in ("own_selected", "cross_selected", "validation", "fixed"):
            cols[policy] = np.array(
                [
                    np.mean([c["results"][str(k)][METRIC][policy] for c in bucket[s]])
                    for s in subjects
                ]
            )
        table[k] = cols
    return table, len(subjects)


def boot_ratio(num, den, rng):
    """Bootstrap the aggregate ratio by resampling participants, not observations."""
    n = len(num)
    out = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d = den[idx].mean()
        if abs(d) < 1e-9:
            continue
        out.append(num[idx].mean() / d)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_slope(x, y, rng):
    """Bootstrap the OLS slope of y on x over participants."""
    n = len(x)
    out = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        vx = xi.var()
        if vx < 1e-12:
            continue
        out.append(np.cov(xi, yi, bias=True)[0, 1] / vx)
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def analyse(table, n_subj):
    rng = np.random.default_rng(BOOT_SEED)
    base_own = table[BASE]["own_selected"]
    base_cross = table[BASE]["cross_selected"]
    curve = {}
    for k in BUDGETS:
        if k <= BASE:
            continue
        dS = table[k]["own_selected"] - base_own
        dC = table[k]["cross_selected"] - base_cross
        mS, mC = float(dS.mean()), float(dC.mean())
        rho = mC / mS if abs(mS) > 1e-9 else None
        rlo, rhi = boot_ratio(dC, dS, rng) if rho is not None else (None, None)
        vx = dS.var()
        slope = float(np.cov(dS, dC, bias=True)[0, 1] / vx) if vx > 1e-12 else None
        slo, shi = boot_slope(dS, dC, rng) if slope is not None else (None, None)
        r = (
            float(np.corrcoef(dS, dC)[0, 1])
            if dS.var() > 1e-12 and dC.var() > 1e-12
            else None
        )
        curve[k] = dict(
            n_participants=n_subj,
            apparent_gain_pp=mS * 100,
            retained_gain_pp=mC * 100,
            retention_ratio=rho,
            retention_ratio_ci=[rlo, rhi],
            retention_slope=slope,
            retention_slope_ci=[slo, shi],
            correlation=r,
            participant_dS_pp=(dS * 100).tolist(),
            participant_dC_pp=(dC * 100).tolist(),
        )
    return curve


def main():
    rows = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "provenance": {
            "same_session_cells": len(list(SAME.glob("*/result.json"))),
            "cross_session_cells": len(list(CROSS.glob("*/result.json"))),
            "metric": METRIC,
            "base_budget": BASE,
            "bootstrap_seed": BOOT_SEED,
            "n_bootstrap": N_BOOT,
            "refitting": "none; read from frozen per-cell result.json contrasts",
            "boundary": (
                "Post hoc re-analysis of trajectories whose results were already "
                "known. Both pools supply labels for their own selection, so this "
                "measures retention across trials of the same participant, not "
                "zero-label cross-subject generalization."
            ),
        },
        "groups": {},
    }
    groups = [
        ("same_session", "seed", "dgcnn"),
        ("same_session", "seed", "mlp"),
        ("cross_session", "seed", "dgcnn"),
        ("cross_session", "seed", "mlp"),
        ("cross_session", "seediv", "dgcnn"),
        ("cross_session", "seediv", "mlp"),
        ("cross_session", "seedv", "dgcnn"),
        ("cross_session", "seedv", "mlp"),
    ]
    for study, dataset, model in groups:
        table, n = per_participant(rows, study, dataset, model)
        if table is None:
            continue
        key = f"{study}/{dataset}/{model}"
        report["groups"][key] = analyse(table, n)

    (OUT / "retention_curve.json").write_text(json.dumps(report, indent=1))

    lines = [
        "# 检查点搜索收益的保留率 (retention)",
        "",
        f"来源:{report['provenance']['same_session_cells']} 条同会话 + "
        f"{report['provenance']['cross_session_cells']} 条跨会话冻结轨迹,纯重分析,未重训。",
        f"基准预算 K={BASE};participant bootstrap {N_BOOT} 次,seed {BOOT_SEED}。",
        "",
        "`apparent` = 选优池上看到的提升 S_K-S_5;`retained` = 同一次扩预算在交换池上的 C_K-C_5。",
        "`ratio` = 两者总体均值之比;`slope` = 跨参与者 dC 对 dS 的 OLS 斜率(每 1 点表观提升留下几点)。",
        "",
        "| 组 | n | K | apparent (pp) | retained (pp) | ratio [95% CI] | slope [95% CI] | r |",
        "|---|--:|--:|--:|--:|---|---|--:|",
    ]
    for key, curve in report["groups"].items():
        for k, v in curve.items():
            def fmt(x, nd=2):
                return "n/a" if x is None else f"{x:.{nd}f}"

            def fmtci(ci):
                if ci[0] is None:
                    return "n/a"
                return f"{ci[0]:.2f} 到 {ci[1]:.2f}"

            lines.append(
                f"| {key} | {v['n_participants']} | {k} | "
                f"{v['apparent_gain_pp']:.2f} | {v['retained_gain_pp']:.2f} | "
                f"{fmt(v['retention_ratio'])} [{fmtci(v['retention_ratio_ci'])}] | "
                f"{fmt(v['retention_slope'])} [{fmtci(v['retention_slope_ci'])}] | "
                f"{fmt(v['correlation'])} |"
            )
    lines += [
        "",
        "## 边界",
        "",
        report["provenance"]["boundary"],
    ]
    (OUT / "retention_curve.md").write_text("\n".join(lines), encoding="utf-8")
    # Windows consoles default to GBK here; never let a console codec break the run
    print("\n".join(lines).encode("utf-8", "replace").decode("utf-8", "replace"),
          flush=True)


if __name__ == "__main__":
    main()
