# experiments/run_seed_session_loso.py
"""Session-wise trial-level LOSO for SEED-family datasets."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import SeedData, load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.seed_svm import fit_predict_seed_svm, loso_splits


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


def _evaluate(data: SeedData, C: float):
    pred = np.empty_like(data.y)
    fold_acc = []
    for tr, te in loso_splits(data.subject):
        p = fit_predict_seed_svm(data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], C=C)
        pred[te] = p
        fold_acc.append(float((p == data.y[te]).mean()))
    summary = summarize(data.y, pred, data.subject)
    summary["loso_mean"] = float(np.mean(fold_acc))
    summary["loso_std"] = float(np.std(fold_acc, ddof=1))
    return summary


def main():
    t0 = time.time()
    configs = [
        ("mean_svm_c1", "mean", 1.0),
        ("meanstd_svm_c1", "mean_std", 1.0),
        ("meanstd_svm_c10", "mean_std", 10.0),
    ]
    datasets = ["SEED", "SEED-IV", "SEED-V"]
    results = {}
    for dataset_name in datasets:
        results[dataset_name] = {}
        cache = {}
        for config_name, aggregate, C in configs:
            if aggregate not in cache:
                cache[aggregate] = load_seed_family(dataset_name, aggregate=aggregate)
                d = cache[aggregate]
                print(f"[t={time.time()-t0:.0f}s] loaded {dataset_name} {aggregate} X={d.X.shape}", flush=True)
            data = cache[aggregate]
            by_session = {}
            for session in sorted(np.unique(data.session).tolist()):
                ds = _subset(data, data.session == session)
                by_session[str(session)] = _evaluate(ds, C=C)
            pooled_mean = float(np.mean([v["accuracy"] for v in by_session.values()]))
            pooled_std = float(np.std([v["accuracy"] for v in by_session.values()], ddof=1))
            results[dataset_name][config_name] = {
                "by_session": by_session,
                "session_mean_accuracy": pooled_mean,
                "session_std_accuracy": pooled_std,
            }
            print(
                f"[t={time.time()-t0:.0f}s] {dataset_name:7s} {config_name:16s} "
                f"session_mean={pooled_mean:.4f}+/-{pooled_std:.4f} "
                + " ".join(f"s{s}={by_session[str(s)]['accuracy']:.4f}" for s in sorted(np.unique(data.session).tolist())),
                flush=True,
            )

    os.makedirs(OUT, exist_ok=True)
    json_path = os.path.join(OUT, "seed_session_loso.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Dataset | Config | Session mean +/- std | Session 1 | Session 2 | Session 3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset_name, by_config in results.items():
        for config_name, res in by_config.items():
            by = res["by_session"]
            lines.append(
                f"| {dataset_name} | {config_name} | "
                f"{res['session_mean_accuracy']:.3f} +/- {res['session_std_accuracy']:.3f} | "
                f"{by['1']['accuracy']:.3f} | {by['2']['accuracy']:.3f} | {by['3']['accuracy']:.3f} |"
            )
    with open(os.path.join(OUT, "seed_session_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[t={time.time()-t0:.0f}s] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
