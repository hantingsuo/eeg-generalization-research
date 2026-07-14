# experiments/run_seed_coral_loso.py
"""Track A: DE-feature CORAL LOSO for SEED / SEED-IV / SEED-V."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.seed_da import fit_predict_seed_coral_svm
from pcma.model.seed_svm import fit_predict_seed_svm, loso_splits


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _evaluate(data, predictor, **params):
    pred = np.empty_like(data.y)
    fold_acc = []
    for tr, te in loso_splits(data.subject):
        p = predictor(data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], **params)
        pred[te] = p
        fold_acc.append(float((p == data.y[te]).mean()))
    summary = summarize(data.y, pred, data.subject)
    summary["loso_mean"] = float(np.mean(fold_acc))
    summary["loso_std"] = float(np.std(fold_acc, ddof=1))
    return summary


def main():
    t0 = time.time()
    configs = [
        ("mean_svm_c1", "mean", fit_predict_seed_svm, {"C": 1.0}),
        ("mean_coral_c1", "mean", fit_predict_seed_coral_svm, {"C": 1.0, "shrink": 0.1}),
        ("meanstd_svm_c10", "mean_std", fit_predict_seed_svm, {"C": 10.0}),
        ("meanstd_coral_c10", "mean_std", fit_predict_seed_coral_svm, {"C": 10.0, "shrink": 0.1}),
    ]
    results = {}
    for dataset_name in ("SEED", "SEED-IV", "SEED-V"):
        results[dataset_name] = {}
        cache = {}
        for config_name, aggregate, predictor, params in configs:
            if aggregate not in cache:
                cache[aggregate] = load_seed_family(dataset_name, aggregate=aggregate)
                d = cache[aggregate]
                print(f"[t={time.time()-t0:.0f}s] loaded {dataset_name} {aggregate} X={d.X.shape}", flush=True)
            d = cache[aggregate]
            res = _evaluate(d, predictor, **params)
            results[dataset_name][config_name] = res
            print(
                f"[t={time.time()-t0:.0f}s] {dataset_name:7s} {config_name:18s} "
                f"acc={res['accuracy']:.4f} loso={res['loso_mean']:.4f}+/-{res['loso_std']:.4f}",
                flush=True,
            )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_coral_loso.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Dataset | Config | Accuracy | LOSO mean +/- std |",
        "|---|---|---:|---:|",
    ]
    for dataset_name, by_config in results.items():
        for config_name, res in by_config.items():
            lines.append(f"| {dataset_name} | {config_name} | {res['accuracy']:.3f} | {res['loso_mean']:.3f} +/- {res['loso_std']:.3f} |")
    with open(os.path.join(OUT, "seed_coral_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
