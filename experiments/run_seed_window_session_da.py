# experiments/run_seed_window_session_da.py
"""SEED-only session-wise window training with trial aggregation."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import SeedData, load_seed_family
from run_seed_window_trial_da import _evaluate


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _subset(data: SeedData, mask) -> SeedData:
    mask = np.asarray(mask, dtype=bool)
    return SeedData(
        X=data.X[mask],
        y=data.y[mask],
        subject=data.subject[mask],
        session=data.session[mask],
        trial=data.trial[mask],
        dataset=data.dataset,
    )


def main():
    t0 = time.time()
    data = load_seed_family("SEED", unit="window")
    configs = [
        ("rbf8", 8, False),
        ("coral_rbf8", 8, True),
        ("rbf16", 16, False),
        ("coral_rbf16", 16, True),
    ]
    results = {}
    print(f"[t={time.time()-t0:.0f}s] loaded SEED window X={data.X.shape}", flush=True)
    for config_name, max_windows, use_coral in configs:
        by_session = {}
        for session in sorted(np.unique(data.session).tolist()):
            ds = _subset(data, data.session == session)
            res = _evaluate(ds, max_windows=max_windows, coral=use_coral)
            by_session[str(session)] = res
            print(
                f"[t={time.time()-t0:.0f}s] {config_name:12s} session={session} "
                f"trial={res['trial']['accuracy']:.4f} window={res['window']['accuracy']:.4f}",
                flush=True,
            )
        results[config_name] = {
            "by_session": by_session,
            "session_mean_trial_acc": float(np.mean([r["trial"]["accuracy"] for r in by_session.values()])),
            "session_std_trial_acc": float(np.std([r["trial"]["accuracy"] for r in by_session.values()], ddof=1)),
        }
        print(
            f"[t={time.time()-t0:.0f}s] {config_name:12s} "
            f"session_mean_trial={results[config_name]['session_mean_trial_acc']:.4f}",
            flush=True,
        )

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_window_session_da.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Config | Session mean trial acc | Session std | S1 | S2 | S3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config_name, res in results.items():
        by = res["by_session"]
        lines.append(
            f"| {config_name} | {res['session_mean_trial_acc']:.3f} | "
            f"{res['session_std_trial_acc']:.3f} | "
            f"{by['1']['trial']['accuracy']:.3f} | {by['2']['trial']['accuracy']:.3f} | {by['3']['trial']['accuracy']:.3f} |"
        )
    with open(os.path.join(OUT, "seed_window_session_da.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
