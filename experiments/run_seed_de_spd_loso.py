# experiments/run_seed_de_spd_loso.py
"""SEED DE-window covariance + Euclidean Alignment LOSO."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.seed import SeedData, load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.seed_da import trial_group_keys
from pcma.model.seed_spd import fit_predict_ea_logsvm_bands, fit_predict_logsvm_bands
from pcma.model.seed_svm import loso_splits


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def de_window_covariances(data: SeedData, shrink: float = 0.05):
    groups = trial_group_keys(data.subject, data.session, data.trial)
    seen = list(dict.fromkeys(groups.tolist()))
    C, y, subject, session, trial = [], [], [], [], []
    eye = np.eye(62)
    for group in seen:
        mask = groups == group
        x = data.X[mask].reshape(mask.sum(), 62, 5)
        labels = np.unique(data.y[mask])
        if len(labels) != 1:
            raise ValueError(f"{group}: nonconstant labels")
        bands = []
        for b in range(5):
            cov = np.cov(x[:, :, b], rowvar=False)
            cov = (cov + cov.T) / 2
            tr = float(np.trace(cov)) / cov.shape[0]
            cov = (1 - shrink) * cov + shrink * tr * eye + 1e-8 * eye
            bands.append(cov)
        C.append(np.stack(bands))
        y.append(labels[0])
        s, rest = group.split("_s", 1)
        se, trl = rest.split("_t", 1)
        subject.append(s)
        session.append(int(se))
        trial.append(int(trl))
    return np.asarray(C), np.asarray(y, dtype=int), np.asarray(subject), np.asarray(session), np.asarray(trial)


def _evaluate(C, y, subject, method):
    pred = np.empty_like(y)
    fold_acc = []
    for tr, te in loso_splits(subject):
        if method == "logsvm":
            p = fit_predict_logsvm_bands(C[tr], y[tr], C[te])
        elif method == "ea_logsvm":
            p = fit_predict_ea_logsvm_bands(C[tr], y[tr], subject[tr], C[te], subject[te])
        else:
            raise ValueError(method)
        pred[te] = p
        fold_acc.append(float((p == y[te]).mean()))
    res = summarize(y, pred, subject)
    res["loso_mean"] = float(np.mean(fold_acc))
    res["loso_std"] = float(np.std(fold_acc, ddof=1))
    return res


def main():
    t0 = time.time()
    win = load_seed_family("SEED", unit="window")
    print(f"[t={time.time()-t0:.0f}s] loaded SEED windows {win.X.shape}", flush=True)
    C, y, subject, session, trial = de_window_covariances(win)
    print(f"[t={time.time()-t0:.0f}s] DE-window covariances {C.shape}", flush=True)
    results = {"pooled": {}, "by_session": {}}
    for method in ("logsvm", "ea_logsvm"):
        results["pooled"][method] = _evaluate(C, y, subject, method)
        print(
            f"[t={time.time()-t0:.0f}s] pooled {method:9s} "
            f"acc={results['pooled'][method]['accuracy']:.4f}",
            flush=True,
        )
    for sess in sorted(np.unique(session).tolist()):
        m = session == sess
        results["by_session"][str(sess)] = {}
        for method in ("logsvm", "ea_logsvm"):
            results["by_session"][str(sess)][method] = _evaluate(C[m], y[m], subject[m], method)
            print(
                f"[t={time.time()-t0:.0f}s] session={sess} {method:9s} "
                f"acc={results['by_session'][str(sess)][method]['accuracy']:.4f}",
                flush=True,
            )
    results["session_mean"] = {
        method: float(np.mean([results["by_session"][str(s)][method]["accuracy"] for s in sorted(np.unique(session))]))
        for method in ("logsvm", "ea_logsvm")
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_de_spd_loso.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Scope | logSVM | EA+logSVM |",
        "|---|---:|---:|",
        f"| pooled | {results['pooled']['logsvm']['accuracy']:.3f} | {results['pooled']['ea_logsvm']['accuracy']:.3f} |",
    ]
    for sess in sorted(np.unique(session).tolist()):
        lines.append(
            f"| session {sess} | {results['by_session'][str(sess)]['logsvm']['accuracy']:.3f} | "
            f"{results['by_session'][str(sess)]['ea_logsvm']['accuracy']:.3f} |"
        )
    lines.append(f"| session mean | {results['session_mean']['logsvm']:.3f} | {results['session_mean']['ea_logsvm']:.3f} |")
    with open(os.path.join(OUT, "seed_de_spd_loso.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
