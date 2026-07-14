# experiments/run_rich_video_aggregation.py
"""Track B: segment-level rich-SVM vs 50s video-level probability aggregation."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.competition import load_competition
from pcma.data.features import extract_rich_features
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.eval.metrics import summarize
from pcma.eval.stats import paired_wilcoxon
from pcma.model.rich import fit_predict_rich


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def aggregate_video_predictions(y_true, proba, classes, video_id, subject, population):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    classes = np.asarray(classes)
    video_id = np.asarray(video_id).astype(str)
    subject = np.asarray(subject)
    population = np.asarray(population)

    seen = list(dict.fromkeys(video_id.tolist()))
    yv, pv, sv, popv = [], [], [], []
    for vid in seen:
        m = video_id == vid
        if len(np.unique(y_true[m])) != 1:
            raise ValueError(f"{vid}: labels are not constant within video")
        if len(np.unique(subject[m])) != 1:
            raise ValueError(f"{vid}: subjects are not constant within video")
        yv.append(y_true[m][0])
        pv.append(classes[np.mean(proba[m], axis=0).argmax()])
        sv.append(subject[m][0])
        popv.append(population[m][0])
    return np.asarray(yv), np.asarray(pv), np.asarray(sv), np.asarray(popv)


def _fit_segment_and_video(F, data, tr, te):
    pred, proba, classes = fit_predict_rich(
        F[tr], data.y[tr], data.subject[tr], F[te], data.subject[te], return_proba=True
    )
    seg_summary = summarize(data.y[te], pred, data.subject[te], data.population[te])
    yv, pv, sv, popv = aggregate_video_predictions(
        data.y[te], proba, classes, data.video_id[te], data.subject[te], data.population[te]
    )
    vid_summary = summarize(yv, pv, sv, popv)
    return seg_summary, vid_summary


def main():
    t0 = time.time()
    data = load_competition()
    F = extract_rich_features(data.X, sfreq=250.0)
    print(f"[t={time.time()-t0:.0f}s] features {F.shape} loaded", flush=True)
    results = {}

    for name, split_fn in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = split_fn(data.subject, data.population)
        seg, vid = _fit_segment_and_video(F, data, tr, te)
        results[name] = {"segment": seg, "video": vid}
        print(
            f"[t={time.time()-t0:.0f}s] {name} segment={seg['accuracy']:.4f} "
            f"video={vid['accuracy']:.4f} video_n={len(np.unique(data.video_id[te]))}",
            flush=True,
        )

    seg_fold_acc, vid_fold_acc = [], []
    fold_rows = []
    for i, (tr, te) in enumerate(p3_mixed_cv(data.subject, n_splits=5), start=1):
        seg, vid = _fit_segment_and_video(F, data, tr, te)
        seg_fold_acc.append(seg["accuracy"])
        vid_fold_acc.append(vid["accuracy"])
        fold_rows.append({"fold": i, "segment": seg, "video": vid})
        print(
            f"[t={time.time()-t0:.0f}s] P3 fold {i}/5 segment={seg['accuracy']:.4f} "
            f"video={vid['accuracy']:.4f}",
            flush=True,
        )
    _, wp = paired_wilcoxon(vid_fold_acc, seg_fold_acc)
    results["P3_mixed_cv"] = {
        "segment_mean": float(np.mean(seg_fold_acc)),
        "segment_std": float(np.std(seg_fold_acc, ddof=1)),
        "video_mean": float(np.mean(vid_fold_acc)),
        "video_std": float(np.std(vid_fold_acc, ddof=1)),
        "wilcoxon_video_vs_segment_p": float(wp),
        "folds": fold_rows,
    }
    print(
        f"[t={time.time()-t0:.0f}s] P3 segment={np.mean(seg_fold_acc):.4f} "
        f"video={np.mean(vid_fold_acc):.4f} wilcoxon_p={wp:.4f}",
        flush=True,
    )

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rich_video_aggregation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    lines = [
        "| Protocol | Segment acc | Video acc |",
        "|---|---:|---:|",
        f"| P1 HC->DEP | {results['P1_HC2DEP']['segment']['accuracy']:.3f} | {results['P1_HC2DEP']['video']['accuracy']:.3f} |",
        f"| P2 DEP->HC | {results['P2_DEP2HC']['segment']['accuracy']:.3f} | {results['P2_DEP2HC']['video']['accuracy']:.3f} |",
        f"| P3 mixed mean | {results['P3_mixed_cv']['segment_mean']:.3f} | {results['P3_mixed_cv']['video_mean']:.3f} |",
        "",
        f"P3 paired Wilcoxon p(video vs segment) = {results['P3_mixed_cv']['wilcoxon_video_vs_segment_p']:.4f}",
    ]
    with open(os.path.join(OUT, "rich_video_aggregation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
