# experiments/run_seed_loso.py
"""Track A: SEED / SEED-IV / SEED-V LOSO baselines on DE features."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.seed_svm import fit_predict_seed_hgb, fit_predict_seed_svm, loso_splits


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _evaluate(data, model: str, **params):
    pred = np.empty_like(data.y)
    fold_acc = []
    fold_subject = []
    for tr, te in loso_splits(data.subject):
        if model == "svm":
            p = fit_predict_seed_svm(data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], **params)
        elif model == "hgb":
            p = fit_predict_seed_hgb(data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], **params)
        else:
            raise ValueError(model)
        pred[te] = p
        fold_acc.append(float((p == data.y[te]).mean()))
        fold_subject.append(str(np.unique(data.subject[te])[0]))
    summary = summarize(data.y, pred, data.subject)
    summary["loso_mean"] = float(np.mean(fold_acc))
    summary["loso_std"] = float(np.std(fold_acc, ddof=1))
    summary["fold_accuracy"] = {s: a for s, a in zip(fold_subject, fold_acc)}
    return summary


def main():
    t0 = time.time()
    configs = [
        ("mean_svm_c1", "mean", "svm", {"C": 1.0}),
        ("mean_svm_c10", "mean", "svm", {"C": 10.0}),
        ("meanstd_svm_c10", "mean_std", "svm", {"C": 10.0}),
        ("mean_hgb", "mean", "hgb", {}),
    ]
    datasets = ["SEED", "SEED-IV", "SEED-V"]
    results = {}
    for dataset_name in datasets:
        results[dataset_name] = {}
        cache = {}
        for config_name, aggregate, model, params in configs:
            if aggregate not in cache:
                cache[aggregate] = load_seed_family(dataset_name, aggregate=aggregate)
                d = cache[aggregate]
                print(
                    f"[t={time.time()-t0:.0f}s] loaded {dataset_name} aggregate={aggregate} "
                    f"X={d.X.shape} subjects={d.n_subjects} classes={sorted(np.unique(d.y).tolist())}",
                    flush=True,
                )
            d = cache[aggregate]
            summary = _evaluate(d, model=model, **params)
            results[dataset_name][config_name] = summary
            print(
                f"[t={time.time()-t0:.0f}s] {dataset_name:7s} {config_name:15s} "
                f"acc={summary['accuracy']:.4f} loso={summary['loso_mean']:.4f}+/-{summary['loso_std']:.4f} "
                f"bal={summary['balanced_accuracy']:.4f}",
                flush=True,
            )
    os.makedirs(OUT, exist_ok=True)
    json_path = os.path.join(OUT, "seed_loso.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [
        "| Dataset | Config | Accuracy | LOSO mean +/- std | Balanced acc |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_name, by_config in results.items():
        for config_name, summary in by_config.items():
            lines.append(
                f"| {dataset_name} | {config_name} | {summary['accuracy']:.3f} | "
                f"{summary['loso_mean']:.3f} +/- {summary['loso_std']:.3f} | "
                f"{summary['balanced_accuracy']:.3f} |"
            )
    with open(os.path.join(OUT, "seed_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[t={time.time()-t0:.0f}s] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
