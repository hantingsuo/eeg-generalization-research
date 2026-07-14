"""Track A: strong linear SEED-family LOSO baseline on mean+std DE features."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import SeedData, load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.seed_svm import fit_predict_seed_logreg, loso_splits


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


def _evaluate(data: SeedData, C: float) -> dict[str, object]:
    pred = np.empty_like(data.y)
    fold_acc = []
    for tr, te in loso_splits(data.subject):
        p = fit_predict_seed_logreg(data.X[tr], data.y[tr], data.subject[tr], data.X[te], data.subject[te], C=C)
        pred[te] = p
        fold_acc.append(float((p == data.y[te]).mean()))
    summary = summarize(data.y, pred, data.subject)
    summary["loso_mean"] = float(np.mean(fold_acc))
    summary["loso_std"] = float(np.std(fold_acc, ddof=1))
    return summary


def _evaluate_sessions(data: SeedData, C: float) -> dict[str, object]:
    by_session = {}
    for session in sorted(np.unique(data.session).tolist()):
        ds = _subset(data, data.session == session)
        by_session[str(session)] = _evaluate(ds, C=C)
    return {
        "by_session": by_session,
        "session_mean_accuracy": float(np.mean([v["accuracy"] for v in by_session.values()])),
        "session_std_accuracy": float(np.std([v["accuracy"] for v in by_session.values()], ddof=1)),
    }


def main() -> None:
    t0 = time.time()
    configs = [
        ("meanstd_logreg_c0p1", 0.1),
        ("meanstd_logreg_c1", 1.0),
        ("meanstd_logreg_c10", 10.0),
    ]
    results: dict[str, dict[str, object]] = {}
    for dataset_name in ("SEED", "SEED-IV", "SEED-V"):
        data = load_seed_family(dataset_name, aggregate="mean_std")
        print(f"[t={time.time()-t0:.0f}s] loaded {dataset_name} mean_std X={data.X.shape}", flush=True)
        results[dataset_name] = {}
        for config_name, C in configs:
            pooled = _evaluate(data, C=C)
            session = _evaluate_sessions(data, C=C)
            results[dataset_name][config_name] = {"pooled": pooled, "session": session}
            print(
                f"[t={time.time()-t0:.0f}s] {dataset_name:7s} {config_name:20s} "
                f"pooled={pooled['accuracy']:.4f} "
                f"session={session['session_mean_accuracy']:.4f}+/-{session['session_std_accuracy']:.4f}",
                flush=True,
            )

    os.makedirs(OUT, exist_ok=True)
    json_path = os.path.join(OUT, "seed_logreg_loso.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Dataset | Config | Pooled acc | Session mean +/- std | Session 1 | Session 2 | Session 3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, by_config in results.items():
        for config_name, res in by_config.items():
            session = res["session"]
            by = session["by_session"]
            lines.append(
                f"| {dataset_name} | {config_name} | {res['pooled']['accuracy']:.3f} | "
                f"{session['session_mean_accuracy']:.3f} +/- {session['session_std_accuracy']:.3f} | "
                f"{by['1']['accuracy']:.3f} | {by['2']['accuracy']:.3f} | {by['3']['accuracy']:.3f} |"
            )
    with open(os.path.join(OUT, "seed_logreg_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[t={time.time()-t0:.0f}s] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
