"""Post hoc: does retention track how much data drives the checkpoint choice?

EXPLORATORY. Designed after the FACED results were known, to test one reading of
why SEED and FACED differ: the retained part of a checkpoint-search gain may
depend on how informative the selection data are about the target, not on the
generalisation label attached to them. Nothing is refitted: every number is
re-derived from the per-epoch, per-clip records of the frozen runs.

Two axes, both scored on a fixed evaluation set per participant:
  zero-label (U):  select on a random subset of n validation participants
                   (20 draws per n, seed 20260920), score on all target clips
  calibration (A): select on k labelled clips per class of the target
                   (k = 1, 2 from the frozen permutations), score on the clips
                   that are never used as calibration, so the scored set is the
                   same for both k

Reported: apparent gain, retained gain and retention ratio for K = 5 -> 80.
--faced-version selects the FACED run: v1 (training-set normalisation) or v2
(per-participant normalisation). SEED is the same run in both cases.
"""
from pathlib import Path
import sys, json, argparse
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.jne_e1_analysis import candidates, pick, acc_curve, boot_mean_ci  # noqa: E402

REPORT = ROOT / "reports/jne_revision_2026-09"
RES = ROOT / "results/jne_revision_2026-09"
DRAWS, SEED = 20, 20260920
LO, HI = 5, 80


def load(cells_dir, pattern="*"):
    out = []
    for rp in sorted(Path(cells_dir).glob(f"{pattern}/result.json")):
        meta = json.loads(rp.read_text())
        with np.load(rp.parent / "trace.npz") as z:
            out.append((meta, {k: z[k] for k in z.files}))
    return out


def ratio_row(dS, dC):
    mS = float(np.mean(dS))
    rng = np.random.default_rng(SEED)
    return {"apparent_pp": mS * 100, "retained_pp": float(np.mean(dC)) * 100,
            "retained_ci_pp": [x * 100 for x in boot_mean_ci(np.asarray(dC), rng)],
            "ratio": float(np.mean(dC) / mS) if abs(mS) > 1e-9 else None}


def val_size_curve(cells, sizes):
    """Retention of the zero-label policy as the validation pool grows."""
    rng = np.random.default_rng(SEED)
    per_size = {n: {"dS": defaultdict(list), "dC": defaultdict(list)} for n in sizes}
    for meta, tr in cells:
        vu, vc, vn = tr["validation_units"], tr["validation_correct"], tr["validation_count"]
        tu, tc, tn = tr["target_units"], tr["target_correct"], tr["target_count"]
        val_ids = np.unique(vu[:, 0])
        for n in sizes:
            for _ in range(DRAWS):
                subset = rng.choice(val_ids, size=n, replace=False)
                vsel = np.isin(vu[:, 0], subset)
                va = acc_curve(vc, vn, vsel)
                h = {K: pick(va, candidates(K, tc.shape[0])) for K in (LO, HI)}
                for p in np.unique(tu[:, 0]):
                    ta = acc_curve(tc, tn, tu[:, 0] == p)
                    per_size[n]["dS"][int(p)].append(va[h[HI]] - va[h[LO]])
                    per_size[n]["dC"][int(p)].append(ta[h[HI]] - ta[h[LO]])
    out = {}
    for n in sizes:
        dS = [np.mean(v) for v in per_size[n]["dS"].values()]
        dC = [np.mean(per_size[n]["dC"][p]) for p in per_size[n]["dS"]]
        out[n] = {**ratio_row(dS, dC), "n_participants": len(dC)}
    return out


def calib_size_curve(cells, perms, ks):
    """Retention of the calibration policy as the calibration set grows.

    The scored set excludes every clip that any k could use, so it is identical
    across k (unlike the main analysis, where B is the complement of C).
    """
    reserved = {c: perms[c][: max(ks)] for c in perms}
    out = {}
    for k in ks:
        dS, dC = defaultdict(list), defaultdict(list)
        for meta, tr in cells:
            tu, tc, tn = tr["target_units"], tr["target_correct"], tr["target_count"]
            cal = {perms[c][i] for c in perms for i in range(k)}
            score = set(np.unique(tu[:, 1]).tolist()) - {t for v in reserved.values() for t in v}
            for p in np.unique(tu[:, 0]):
                own = tu[:, 0] == p
                ca = acc_curve(tc, tn, own & np.isin(tu[:, 1], list(cal)))
                ba = acc_curve(tc, tn, own & np.isin(tu[:, 1], list(score)))
                h = {K: pick(ca, candidates(K, tc.shape[0])) for K in (LO, HI)}
                dS[int(p)].append(ca[h[HI]] - ca[h[LO]])
                dC[int(p)].append(ba[h[HI]] - ba[h[LO]])
        out[k] = {**ratio_row([np.mean(v) for v in dS.values()], [np.mean(v) for v in dC.values()]),
                  "n_participants": len(dC), "n_calibration_clips": k * len(perms)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dgcnn")
    ap.add_argument("--faced-version", default="v1", choices=["v1", "v2"],
                    help="v1 = training-set normalisation, v2 = per-participant normalisation")
    a = ap.parse_args()
    faced_plan_path = ROOT / ("plans/2026-09-20-jne-e1-faced-v2.json" if a.faced_version == "v2"
                              else "plans/2026-09-18-jne-e1-faced.json")
    faced_cells = RES / ("e1_faced_v2/cells" if a.faced_version == "v2" else "e1_faced/cells")
    faced_plan = json.loads(faced_plan_path.read_text(encoding="utf-8"))
    seed_plan = json.loads((ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json").read_text(encoding="utf-8"))
    faced = [c for c in load(faced_cells, f"standard_{a.model}_*")]
    seed = [c for c in load(RES / "e1_seed_cross_subject/cells", f"*_{a.model}_*")]
    summary = {
        "status": "EXPLORATORY - designed after the FACED results were known; no refitting",
        "model": a.model, "faced_version": a.faced_version,
        "faced_plan": faced_plan_path.name, "draws_per_size": DRAWS, "seed": SEED, "budgets": [LO, HI],
        "faced": {"cells": len(faced),
                  "validation_pool": val_size_curve(faced, [1, 3, 6, 12, 23]),
                  "calibration": calib_size_curve(faced, {int(c): v for c, v in
                                                          faced_plan["calibration"]["standard"]["permutations"].items()},
                                                  [1, 2])},
        "seed": {"cells": len(seed),
                 "validation_pool": val_size_curve(seed, [1, 2, 3]),
                 "calibration": calib_size_curve(seed, {int(c): v for c, v in
                                                        seed_plan["calibration"]["class_trial_permutations"].items()},
                                                 [1, 2])},
    }
    suffix = "" if a.faced_version == "v1" else f"_{a.faced_version}"
    p = REPORT / f"selection_informativeness_{a.model}{suffix}.json"
    p.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("written", p)
    for ds in ("faced", "seed"):
        print(f"== {ds} ({summary[ds]['cells']} cells)")
        for n, v in summary[ds]["validation_pool"].items():
            print(f"   val n={n:>2}: apparent {v['apparent_pp']:+.2f} retained {v['retained_pp']:+.2f} "
                  f"[{v['retained_ci_pp'][0]:+.2f},{v['retained_ci_pp'][1]:+.2f}] ratio "
                  f"{'na' if v['ratio'] is None else round(v['ratio'], 2)}")
        for k, v in summary[ds]["calibration"].items():
            print(f"   cal k={k} ({v['n_calibration_clips']} clips): apparent {v['apparent_pp']:+.2f} "
                  f"retained {v['retained_pp']:+.2f} [{v['retained_ci_pp'][0]:+.2f},{v['retained_ci_pp'][1]:+.2f}] "
                  f"ratio {'na' if v['ratio'] is None else round(v['ratio'], 2)}")


if __name__ == "__main__":
    main()
