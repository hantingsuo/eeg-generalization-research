"""Common-time-range check on the candidate grid (battle plan 2026-09-16, E3).

The frozen grid is nested but its earliest candidate moves with the budget: K = 5
looks at epochs 16, 32, 48, 64, 80 while K = 80 looks at every epoch from 1. Part
of the apparent gain from widening the budget therefore comes from reaching
earlier epochs, not from having more candidates. The plan asked for a grid with a
common first and last epoch as the check on that.

This script re-derives every frozen conclusion on

    common grid:  epochs 16 .. 80, sampled with step 16, 8, 4, 2, 1
                  -> budgets 5, 9, 17, 33, 65 (plus K = 1 = epoch 80 only)

from the same stored per-epoch records. Nothing is refitted, no frozen output is
touched, and the headline comparison becomes K = 5 -> 65 with both ends fixed.

Usage: python experiments/jne_common_range.py
"""
from pathlib import Path
import sys, json
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import jne_e1_analysis as A  # noqa: E402
from experiments import jne_e4_analysis as E4  # noqa: E402
from experiments.jne_faced_analysis import records as faced_records  # noqa: E402

REPORT = ROOT / "reports/jne_revision_2026-09"
RES = ROOT / "results/jne_revision_2026-09"
LO, HI = 5, 65
NESTED_LO, NESTED_HI = 5, 80


def load_traces(cells_dir, pattern="*"):
    out = []
    for rp in sorted(Path(cells_dir).glob(f"{pattern}/result.json")):
        meta = json.loads(rp.read_text())
        with np.load(rp.parent / "trace.npz") as z:
            out.append((meta, {k: z[k] for k in z.files}))
    return out


def seed_like(plan_path, cells_dir, models):
    """E1 and E2 share the same frozen record builder."""
    cfg = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    scfg = {"budgets": A.COMMON_BUDGETS, "optimization_seeds": cfg["optimization_seeds"]}
    rows = []
    for meta, trace in load_traces(cells_dir):
        for rec in A.cell_records(cfg, trace, A.COMMON_BUDGETS):
            rec.update(model=meta["model"], seed=meta["seed"], fold=meta["fold"])
            rows.append(rec)
    return {m: A.summarise(scfg, rows, m, 1, lo=LO, hi=HI) for m in models
            if any(r["model"] == m for r in rows)}


def faced(plan_path, cells_dir):
    cfg = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    scfg = {"budgets": A.COMMON_BUDGETS, "optimization_seeds": cfg["optimization_seeds"]}
    classes = range(len(cfg["classes"]))
    all_clips = set(range(1, 29))
    ps = cfg["calibration"]["standard"]["permutations"]
    cal_std = [{ps[str(c)][r] for c in classes} for r in range(cfg["calibration"]["standard"]["rotations"])]
    rows = []
    for meta, trace in load_traces(cells_dir, "standard_*"):
        for rec in faced_records(trace, all_clips, cal_std, None, A.COMMON_BUDGETS):
            rec.update(model=meta["model"], seed=meta["seed"])
            rows.append(rec)
    return {m: A.summarise(scfg, rows, m, 1, lo=LO, hi=HI) for m in cfg["models"]
            if any(r["model"] == m for r in rows)}


def matched_session(plan_path, cells_dir):
    """E4: same-session versus other-session retention on the common grid."""
    cfg = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    cfg_common = dict(cfg, budgets=A.COMMON_BUDGETS)
    by_model = defaultdict(lambda: defaultdict(list))
    for meta, trace in load_traces(cells_dir):
        by_model[meta["model"]][meta["subject"]].append(E4.cell_values(cfg_common, meta, trace))
    rng = np.random.default_rng(A.BOOT_SEED)
    out = {}
    for model, by_p in by_model.items():
        parts = sorted(by_p)

        def pv(rel, K, m):
            return np.array([np.mean([v[rel][K][m] for v in by_p[p]]) for p in parts])

        res = {"n_participants": len(parts), "relations": {}}
        for rel in ("same", "cross"):
            dS, dC = pv(rel, HI, "S") - pv(rel, LO, "S"), pv(rel, HI, "C") - pv(rel, LO, "C")
            mS = float(dS.mean())
            res["relations"][rel] = {
                "apparent_pp": mS * 100, "retained_pp": float(dC.mean() * 100),
                "retained_ci_pp": [x * 100 for x in A.boot_mean_ci(dC, rng)],
                "ratio": float(dC.mean() / mS) if abs(mS) > 1e-9 else None,
            }
        d = ((pv("cross", HI, "C") - pv("cross", LO, "C"))
             - (pv("same", HI, "C") - pv("same", LO, "C")))
        res["cross_minus_same_retained"] = {
            "mean_pp": float(d.mean() * 100), "ci_pp": [x * 100 for x in A.boot_mean_ci(d, rng)],
            "signflip_p": A.signflip_p(d)}
        out[model] = res
    return out


def main():
    A.GRID = "common"
    assert list(A.candidates(5)) == [15, 31, 47, 63, 79]
    assert list(A.candidates(65))[0] == 15 and list(A.candidates(65))[-1] == 79
    for k in (9, 17, 33):
        assert set(A.candidates(5)) <= set(A.candidates(k)) <= set(A.candidates(65))

    out = {
        "status": "sensitivity analysis specified in the battle plan of 2026-09-16 "
                  "(committed before every frozen plan) and run after the primary results",
        "grid": {"first_epoch": A.COMMON_FIRST_EPOCH, "last_epoch": 80,
                 "budgets": A.COMMON_BUDGETS, "steps": A.COMMON_STEP,
                 "headline": f"K = {LO} -> {HI}"},
        "frozen_grid_headline": f"K = {NESTED_LO} -> {NESTED_HI}",
        "results": {},
    }
    out["results"]["e1_seed"] = seed_like(
        ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json",
        RES / "e1_seed_cross_subject/cells", ["dgcnn", "mlp"])
    out["results"]["e2_seed_eegnet"] = seed_like(
        ROOT / "plans/2026-09-17-jne-e2-eegnet-seed.json",
        RES / "e2_eegnet_seed/cells", ["eegnet"])
    out["results"]["faced_v1"] = faced(ROOT / "plans/2026-09-18-jne-e1-faced.json",
                                       RES / "e1_faced/cells")
    out["results"]["faced_v2"] = faced(ROOT / "plans/2026-09-20-jne-e1-faced-v2.json",
                                       RES / "e1_faced_v2/cells")
    out["results"]["e4_matched_session"] = matched_session(
        ROOT / "plans/2026-09-17-jne-e4-matched-session.json", RES / "e4_matched_session/cells")

    p = REPORT / "common_time_range.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", p)

    frozen = {k: json.loads((REPORT / f"{k}.json").read_text())["results"]
              for k in ("e1_summary", "e2_summary", "faced_summary", "faced_v2_summary")}
    src = {"e1_seed": ("e1_summary", "k1"), "e2_seed_eegnet": ("e2_summary", "k1"),
           "faced_v1": ("faced_summary", "standard"), "faced_v2": ("faced_v2_summary", "standard")}
    def fmt(prim, retn):
        r = "   na" if retn["ratio"] is None else f"{retn['ratio']:5.2f}"
        return (f"{prim['mean_pp']:+6.2f} [{prim['ci_pp'][0]:+6.2f},{prim['ci_pp'][1]:+6.2f}] "
                f"app {retn['apparent_pp']:+5.2f} r{r}")

    print(f"\n{'experiment':<16}{'model':<8}{'pol':<4}{'frozen  K 5->80':<44}{'common  K 5->65 (epochs 16..80)'}")
    for exp, (key, sub) in src.items():
        for m, res in out["results"][exp].items():
            f = frozen[key][m][sub]
            for pol in ("U", "A"):
                print(f"{exp:<16}{m:<8}{pol:<4}"
                      f"{fmt(f['primary'][f'{pol}80-{pol}5'], f['retention'][pol]['80']):<44}"
                      f"{fmt(res['primary'][f'{pol}{HI}-{pol}{LO}'], res['retention'][pol][HI])}")
    print("\nE4 (matched session), common grid:")
    for m, r in out["results"]["e4_matched_session"].items():
        for rel in ("same", "cross"):
            v = r["relations"][rel]
            print(f"  {m:<7}{rel:<6} apparent {v['apparent_pp']:+6.2f} retained {v['retained_pp']:+6.2f} "
                  f"[{v['retained_ci_pp'][0]:+6.2f},{v['retained_ci_pp'][1]:+6.2f}] "
                  f"ratio {'na' if v['ratio'] is None else round(v['ratio'], 2)}")
        d = r["cross_minus_same_retained"]
        print(f"  {m:<7}cross-same retained {d['mean_pp']:+6.2f} "
              f"[{d['ci_pp'][0]:+6.2f},{d['ci_pp'][1]:+6.2f}] p={d['signflip_p']:.4f}")


if __name__ == "__main__":
    main()
