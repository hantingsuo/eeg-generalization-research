# experiments/run_seed_window_loso.py
"""Window-level SEED-family LOSO baselines.

This keeps subject-level holdout but evaluates the 1s/4s DE windows directly,
which is closer to the common sample-level SEED reporting protocol than the
trial-mean experiment in run_seed_loso.py.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
import time

import numpy as np

from pcma.data.seed import SeedData, load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.seed_svm import (
    fit_predict_seed_lightgbm,
    fit_predict_seed_linear_svm,
    loso_splits,
)


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _subset(data: SeedData, mask) -> SeedData:
    mask = np.asarray(mask, dtype=bool)
    return replace(
        data,
        X=data.X[mask],
        y=data.y[mask],
        subject=data.subject[mask],
        session=data.session[mask],
        trial=data.trial[mask],
    )


def _predict(model: str, data: SeedData, tr, te, params):
    if model == "linear_svm":
        return fit_predict_seed_linear_svm(
            data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], **params
        )
    if model == "lightgbm":
        return fit_predict_seed_lightgbm(
            data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], **params
        )
    raise ValueError(model)


def _evaluate(data: SeedData, model: str, **params):
    pred = np.empty_like(data.y)
    fold_acc = []
    fold_subject = []
    for tr, te in loso_splits(data.subject):
        p = _predict(model, data, tr, te, params)
        pred[te] = p
        fold_acc.append(float((p == data.y[te]).mean()))
        fold_subject.append(str(np.unique(data.subject[te])[0]))
    summary = summarize(data.y, pred, data.subject)
    summary["loso_mean"] = float(np.mean(fold_acc))
    summary["loso_std"] = float(np.std(fold_acc, ddof=1))
    summary["fold_accuracy"] = {s: a for s, a in zip(fold_subject, fold_acc)}
    return summary


def _evaluate_by_session(data: SeedData, model: str, params):
    out = {}
    for session in sorted(np.unique(data.session).tolist()):
        ds = _subset(data, data.session == session)
        out[str(session)] = _evaluate(ds, model, **params)
    out["mean_accuracy"] = float(np.mean([v["accuracy"] for k, v in out.items() if k != "mean_accuracy"]))
    return out


def main():
    t0 = time.time()
    configs = [
        ("window_linear_c1", "linear_svm", {"C": 1.0}),
    ]
    datasets = ["SEED", "SEED-IV", "SEED-V"]
    results = {}
    for dataset_name in datasets:
        data = load_seed_family(dataset_name, unit="window")
        print(
            f"[t={time.time()-t0:.0f}s] loaded {dataset_name} window X={data.X.shape} "
            f"subjects={data.n_subjects} classes={sorted(np.unique(data.y).tolist())}",
            flush=True,
        )
        results[dataset_name] = {}
        for config_name, model, params in configs:
            pooled = _evaluate(data, model=model, **params)
            by_session = _evaluate_by_session(data, model=model, params=params)
            results[dataset_name][config_name] = {"pooled": pooled, "by_session": by_session}
            print(
                f"[t={time.time()-t0:.0f}s] {dataset_name:7s} {config_name:16s} "
                f"pooled={pooled['accuracy']:.4f} loso={pooled['loso_mean']:.4f}+/-{pooled['loso_std']:.4f} "
                f"session_mean={by_session['mean_accuracy']:.4f}",
                flush=True,
            )
    os.makedirs(OUT, exist_ok=True)
    json_path = os.path.join(OUT, "seed_window_loso.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [
        "| Dataset | Config | Pooled window acc | Pooled LOSO mean +/- std | Session-wise mean acc |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_name, by_config in results.items():
        for config_name, res in by_config.items():
            pooled = res["pooled"]
            by_session = res["by_session"]
            lines.append(
                f"| {dataset_name} | {config_name} | {pooled['accuracy']:.3f} | "
                f"{pooled['loso_mean']:.3f} +/- {pooled['loso_std']:.3f} | "
                f"{by_session['mean_accuracy']:.3f} |"
            )
    with open(os.path.join(OUT, "seed_window_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[t={time.time()-t0:.0f}s] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
