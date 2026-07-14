# experiments/run_seed_window_trial_da.py
"""Window-level DE training with trial-level aggregation for SEED-family LOSO."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.data.seed import load_seed_family
from pcma.eval.metrics import summarize
from pcma.model.pipelines import coral_align
from pcma.model.rich import per_subject_zscore
from pcma.model.seed_da import aggregate_scores_by_group, select_evenly_per_group, trial_group_keys
from pcma.model.seed_svm import loso_splits


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _fit_scores(Ftr, ytr, subject_tr, Fte, subject_te, train_groups, max_windows, coral=False):
    Xtr = per_subject_zscore(Ftr, subject_tr)
    Xte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    pick = select_evenly_per_group(train_groups, max_windows)
    Xtr_fit = Xtr[pick]
    ytr_fit = ytr[pick]
    if coral:
        Xte = coral_align(Xtr_fit, Xte, shrink=0.1)
    clf = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced", decision_function_shape="ovr")
    clf.fit(Xtr_fit, ytr_fit)
    scores = clf.decision_function(Xte)
    return scores, clf.classes_, len(pick)


def _evaluate(data, max_windows: int, coral: bool):
    pred_window = np.empty_like(data.y)
    trial_true, trial_pred, trial_subject = [], [], []
    sample_counts = []
    groups_all = trial_group_keys(data.subject, data.session, data.trial)
    for tr, te in loso_splits(data.subject):
        scores, classes, n_fit = _fit_scores(
            data.X[tr],
            data.y[tr],
            data.subject[tr],
            data.X[te],
            data.subject[te],
            groups_all[tr],
            max_windows=max_windows,
            coral=coral,
        )
        sample_counts.append(n_fit)
        if scores.ndim == 1:
            pred_window[te] = classes[(scores > 0).astype(int)]
        else:
            pred_window[te] = classes[np.argmax(scores, axis=1)]
        y_trial, p_trial, trial_groups = aggregate_scores_by_group(scores, classes, groups_all[te], data.y[te])
        trial_true.append(y_trial)
        trial_pred.append(p_trial)
        trial_subject.extend([g.split("_s", 1)[0] for g in trial_groups])
    trial_true = np.concatenate(trial_true)
    trial_pred = np.concatenate(trial_pred)
    trial_subject = np.asarray(trial_subject)
    return {
        "window": summarize(data.y, pred_window, data.subject),
        "trial": summarize(trial_true, trial_pred, trial_subject),
        "mean_train_windows": float(np.mean(sample_counts)),
    }


def main():
    t0 = time.time()
    configs = [
        ("rbf8", 8, False),
        ("coral_rbf8", 8, True),
        ("rbf16", 16, False),
        ("coral_rbf16", 16, True),
    ]
    results = {}
    for dataset_name in ("SEED", "SEED-IV", "SEED-V"):
        data = load_seed_family(dataset_name, unit="window")
        print(f"[t={time.time()-t0:.0f}s] loaded {dataset_name} window X={data.X.shape}", flush=True)
        results[dataset_name] = {}
        for config_name, max_windows, use_coral in configs:
            res = _evaluate(data, max_windows=max_windows, coral=use_coral)
            results[dataset_name][config_name] = res
            print(
                f"[t={time.time()-t0:.0f}s] {dataset_name:7s} {config_name:12s} "
                f"trial={res['trial']['accuracy']:.4f} window={res['window']['accuracy']:.4f} "
                f"fit_n~{res['mean_train_windows']:.0f}",
                flush=True,
            )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_window_trial_da.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Dataset | Config | Trial acc | Window acc | Mean train windows/fold |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_name, by_config in results.items():
        for config_name, res in by_config.items():
            lines.append(
                f"| {dataset_name} | {config_name} | {res['trial']['accuracy']:.3f} | "
                f"{res['window']['accuracy']:.3f} | {res['mean_train_windows']:.0f} |"
            )
    with open(os.path.join(OUT, "seed_window_trial_da.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
