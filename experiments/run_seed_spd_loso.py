# experiments/run_seed_spd_loso.py
"""Track A: raw SEED band-covariance LOSO with no-adapt / RA / CORAL."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import SeedRawData, load_seed_preprocessed, raw_band_covariances
from pcma.eval.metrics import summarize
from pcma.model.pipelines import (
    fit_predict_coral_bands,
    fit_predict_noadapt_bands,
    fit_predict_ra_bands,
    subject_band_means,
)
from pcma.model.seed_svm import loso_splits


OUT = os.path.join(os.path.dirname(__file__), "..", "results")
METHODS = ("noadapt", "ra", "coral")


def _subset(raw: SeedRawData, mask) -> SeedRawData:
    mask = np.asarray(mask, dtype=bool)
    return SeedRawData(
        signals=[x for x, keep in zip(raw.signals, mask) if keep],
        y=raw.y[mask],
        subject=raw.subject[mask],
        session=raw.session[mask],
        trial=raw.trial[mask],
        dataset=raw.dataset,
        sfreq=raw.sfreq,
    )


def _predict(method, C, y, subject, tr, te, sbm):
    if method == "noadapt":
        return fit_predict_noadapt_bands(C[tr], y[tr], C[te])
    if method == "ra":
        return fit_predict_ra_bands(C[tr], y[tr], subject[tr], C[te], subject[te], ref_means_per_band=sbm)
    if method == "coral":
        return fit_predict_coral_bands(C[tr], y[tr], subject[tr], C[te], subject[te], ref_means_per_band=sbm)
    raise ValueError(method)


def _evaluate(C, y, subject):
    sbm = subject_band_means(C, subject)
    out = {}
    for method in METHODS:
        pred = np.empty_like(y)
        fold_acc = []
        for tr, te in loso_splits(subject):
            p = _predict(method, C, y, subject, tr, te, sbm)
            pred[te] = p
            fold_acc.append(float((p == y[te]).mean()))
        summary = summarize(y, pred, subject)
        summary["loso_mean"] = float(np.mean(fold_acc))
        summary["loso_std"] = float(np.std(fold_acc, ddof=1))
        out[method] = summary
    return out


def main():
    t0 = time.time()
    raw = load_seed_preprocessed()
    print(
        f"[t={time.time()-t0:.0f}s] loaded raw trials n={len(raw.signals)} "
        f"subjects={raw.n_subjects} sfreq={raw.sfreq}",
        flush=True,
    )
    C = raw_band_covariances(raw, shrink=0.02, decim=2, per_trial_zscore=False)
    print(f"[t={time.time()-t0:.0f}s] covariances {C.shape} computed", flush=True)

    results = {"pooled": _evaluate(C, raw.y, raw.subject), "by_session": {}}
    for method, res in results["pooled"].items():
        print(
            f"[t={time.time()-t0:.0f}s] pooled {method:7s} "
            f"acc={res['accuracy']:.4f} loso={res['loso_mean']:.4f}+/-{res['loso_std']:.4f}",
            flush=True,
        )

    for session in sorted(np.unique(raw.session).tolist()):
        m = raw.session == session
        sess_res = _evaluate(C[m], raw.y[m], raw.subject[m])
        results["by_session"][str(session)] = sess_res
        for method, res in sess_res.items():
            print(
                f"[t={time.time()-t0:.0f}s] session={session} {method:7s} "
                f"acc={res['accuracy']:.4f} loso={res['loso_mean']:.4f}+/-{res['loso_std']:.4f}",
                flush=True,
            )

    results["session_mean"] = {
        method: {
            "accuracy": float(np.mean([results["by_session"][str(s)][method]["accuracy"] for s in sorted(np.unique(raw.session))])),
            "std": float(np.std([results["by_session"][str(s)][method]["accuracy"] for s in sorted(np.unique(raw.session))], ddof=1)),
        }
        for method in METHODS
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_spd_loso.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Scope | Method | Accuracy | LOSO mean +/- std |",
        "|---|---|---:|---:|",
    ]
    for method in METHODS:
        res = results["pooled"][method]
        lines.append(f"| pooled | {method} | {res['accuracy']:.3f} | {res['loso_mean']:.3f} +/- {res['loso_std']:.3f} |")
    for session in sorted(np.unique(raw.session).tolist()):
        for method in METHODS:
            res = results["by_session"][str(session)][method]
            lines.append(f"| session {session} | {method} | {res['accuracy']:.3f} | {res['loso_mean']:.3f} +/- {res['loso_std']:.3f} |")
    for method in METHODS:
        res = results["session_mean"][method]
        lines.append(f"| session mean | {method} | {res['accuracy']:.3f} | {res['accuracy']:.3f} +/- {res['std']:.3f} |")
    with open(os.path.join(OUT, "seed_spd_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
