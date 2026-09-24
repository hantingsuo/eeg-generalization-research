"""Unit-level simulation behind the apparent/retained accounting (battle plan E6).

The identity is arithmetic, not a mechanism. For a selection pool P, a scored pool
Q that took no part in the selection, and the checkpoint h_P(K) chosen on P from K
candidates, write the pool discrepancy of a checkpoint as

    b(h) = a_P(h) - a_Q(h).

Then, by adding and subtracting a_P,

    retained(K)  =  a_Q(h_P(K)) - a_Q(h_P(5))
                 =  [a_P(h_P(K)) - a_P(h_P(5))] - [b(h_P(K)) - b(h_P(5))]
                 =  apparent(K) - [b(h_P(K)) - b(h_P(5))],

so the gap between an apparent and a retained gain is exactly how much more the
chosen checkpoint favours P at the larger budget, and the retention ratio is
1 - [b(h_P(K)) - b(h_P(5))] / apparent(K). This says nothing about why; it only
fixes what the two numbers mean relative to each other.

The simulation below asks what the two numbers do in four regimes that the
experiments suggest, using the same candidate grid and tie rule as the analyses:

  A  no participant-specific optimum, independent checkpoint noise
  B  no participant-specific optimum, correlated checkpoint noise
  C  a participant-specific optimum exists; the selection pool sees it with
     varying amounts of noise
  D  the selection noise is fixed and the optimum itself varies in height

Nothing here is evidence about EEG; it is a check that the measured pattern is
what this account predicts. It makes two checkable statements: a large apparent
gain with nothing retained needs no real optimum at all (A, B), and reproducing
the retention measured on SEED with the target's own calibration clips needs a
participant-specific optimum of about 20 accuracy points (D).

Usage: python experiments/jne_selection_simulation.py
"""
from pathlib import Path
import sys, json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import jne_e1_analysis as A  # noqa: E402

REPORT = ROOT / "reports/jne_revision_2026-09"
EPOCHS, N_PART, N_REP = 80, 15, 4000
LO, HI = 5, 80
SEED = 20260921


def ar1(rng, shape, phi, sd):
    """Noise correlated across checkpoints: adjacent saves resemble each other."""
    n = shape[-1]
    e = rng.normal(0.0, 1.0, shape)
    out = np.empty(shape)
    out[..., 0] = e[..., 0]
    for h in range(1, n):
        out[..., h] = phi * out[..., h - 1] + np.sqrt(1 - phi ** 2) * e[..., h]
    return out * sd


def run(rng, true_curve, sd_p, sd_q, phi):
    """Return (apparent, retained) in percentage points, averaged over participants."""
    shape = (N_REP, N_PART, EPOCHS)
    noise_p = ar1(rng, shape, phi, sd_p) if phi else rng.normal(0, sd_p, shape)
    noise_q = ar1(rng, shape, phi, sd_q) if phi else rng.normal(0, sd_q, shape)
    aP, aQ = true_curve + noise_p, true_curve + noise_q
    cand = {K: A.candidates(K, EPOCHS) for K in (LO, HI)}
    app, ret = [], []
    for K in (LO, HI):
        c = cand[K]
        h = c[np.argmax(aP[..., c], axis=-1)]                  # earliest maximum, as in the analyses
        app.append(np.take_along_axis(aP, h[..., None], -1)[..., 0])
        ret.append(np.take_along_axis(aQ, h[..., None], -1)[..., 0])
    d_app = (app[1] - app[0]).mean(1)     # participant mean, per replicate
    d_ret = (ret[1] - ret[0]).mean(1)
    return d_app * 100, d_ret * 100


def summarise(d_app, d_ret):
    return {"apparent_pp": float(d_app.mean()), "retained_pp": float(d_ret.mean()),
            "retained_pct2.5": float(np.percentile(d_ret, 2.5)),
            "retained_pct97.5": float(np.percentile(d_ret, 97.5)),
            "ratio": float(d_ret.mean() / d_app.mean())}


def main():
    rng = np.random.default_rng(SEED)
    h = np.arange(EPOCHS)
    flat = np.full((1, 1, EPOCHS), 0.50)
    out = {"epochs": EPOCHS, "participants": N_PART, "replicates": N_REP,
           "budgets": [LO, HI], "grid": A.GRID, "seed": SEED,
           "identity": "retained(K) = apparent(K) - [b(h_K) - b(h_5)],  b(h) = a_P(h) - a_Q(h)",
           "note": "illustration of the accounting, not EEG evidence", "regimes": {}}

    out["regimes"]["A_flat_independent"] = summarise(
        *run(rng, flat, 0.05, 0.02, phi=0.0))
    out["regimes"]["B_flat_correlated"] = summarise(
        *run(rng, flat, 0.05, 0.02, phi=0.9))

    # C:每名参与者有自己的最优轮次
    peak = rng.integers(20, EPOCHS, size=(1, N_PART, 1))
    bump = 0.50 + 0.06 * np.exp(-((h[None, None, :] - peak) ** 2) / (2 * 8.0 ** 2))
    for sd_p in (0.08, 0.05, 0.03, 0.015, 0.005):
        out["regimes"][f"C_optimum_selection_noise_{sd_p}"] = summarise(
            *run(rng, bump, sd_p, 0.02, phi=0.9))

    # D:固定选择噪声,改变真实最优的高度
    for amp in (0.03, 0.06, 0.12, 0.20):
        curve = 0.50 + amp * np.exp(-((h[None, None, :] - peak) ** 2) / (2 * 8.0 ** 2))
        out["regimes"][f"D_optimum_height_{amp}"] = summarise(
            *run(rng, curve, 0.05, 0.02, phi=0.9))

    p = REPORT / "selection_simulation.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", p)
    for name, v in out["regimes"].items():
        print(f"  {name:<34} apparent {v['apparent_pp']:+6.2f}  retained {v['retained_pp']:+6.2f} "
              f"[{v['retained_pct2.5']:+6.2f},{v['retained_pct97.5']:+6.2f}]  ratio {v['ratio']:5.2f}")


if __name__ == "__main__":
    main()
