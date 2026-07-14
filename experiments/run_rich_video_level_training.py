"""Video-level rich-SVM training vs segment-trained video aggregation baseline."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.competition import aggregate_by_video, load_competition
from pcma.data.features import extract_rich_features
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.eval.metrics import summarize
from pcma.eval.stats import bootstrap_paired_accuracy, paired_wilcoxon
from pcma.model.novelty_eval import aggregate_proba_by_group, align_proba, keep_from_gate
from pcma.model.rich import fit_predict_rich


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _segment_baseline_video(F, data, tr, te):
    pred, proba, classes = fit_predict_rich(
        F[tr], data.y[tr], data.subject[tr], F[te], data.subject[te], return_proba=True
    )
    proba = align_proba(proba, classes, target_classes=(0, 1))
    return aggregate_proba_by_group(
        data.y[te],
        proba,
        np.asarray([0, 1]),
        data.video_id[te],
        data.subject[te],
        data.population[te],
    )


def _video_training(Fv, yv, subject_v, population_v, tr, te):
    pred, proba, classes = fit_predict_rich(
        Fv[tr], yv[tr], subject_v[tr], Fv[te], subject_v[te], return_proba=True
    )
    return {
        "y": yv[te],
        "pred": pred,
        "proba": align_proba(proba, classes, target_classes=(0, 1)),
        "classes": np.asarray([0, 1]),
        "subject": subject_v[te],
        "population": population_v[te],
    }


def _aligned_method_predictions(baseline_group, method_pred, method_video_id):
    method_by_vid = {str(vid): int(pred) for vid, pred in zip(method_video_id, method_pred)}
    return np.asarray([method_by_vid[str(vid)] for vid in baseline_group], dtype=int)


def _eval_fixed_split(F, data, Fv, yv, subject_v, population_v, video_id_v, tr_seg, te_seg, tr_vid, te_vid):
    baseline = _segment_baseline_video(F, data, tr_seg, te_seg)
    method = _video_training(Fv, yv, subject_v, population_v, tr_vid, te_vid)
    method_pred_aligned = _aligned_method_predictions(baseline["group"], method["pred"], video_id_v[te_vid])
    base_summary = summarize(baseline["y"], baseline["pred"], baseline["subject"], baseline["population"])
    method_summary = summarize(baseline["y"], method_pred_aligned, baseline["subject"], baseline["population"])
    return {
        "baseline": base_summary,
        "video_train": method_summary,
        "bootstrap_video_train_vs_baseline": bootstrap_paired_accuracy(
            baseline["y"], baseline["pred"], method_pred_aligned
        ),
        "n_videos": int(len(baseline["y"])),
    }


def main():
    t0 = time.time()
    data = load_competition()
    F = extract_rich_features(data.X, sfreq=250.0)
    Fv, yv, subject_v, population_v, video_id_v = aggregate_by_video(
        F, data.y, data.subject, data.population, data.video_id
    )
    print(
        f"[t={time.time()-t0:.0f}s] segment features {F.shape}; video features {Fv.shape}",
        flush=True,
    )

    results = {
        "method": "video_level_training",
        "baseline": "segment-trained rich-SVM with probability-mean video aggregation",
        "n_segments": int(len(data.y)),
        "n_videos": int(len(yv)),
        "protocols": {},
    }

    for name, split_fn in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr_seg, te_seg = split_fn(data.subject, data.population)
        tr_vid, te_vid = split_fn(subject_v, population_v)
        res = _eval_fixed_split(F, data, Fv, yv, subject_v, population_v, video_id_v, tr_seg, te_seg, tr_vid, te_vid)
        results["protocols"][name] = res
        print(
            f"[t={time.time()-t0:.0f}s] {name} baseline_video={res['baseline']['accuracy']:.4f} "
            f"video_train={res['video_train']['accuracy']:.4f} "
            f"CI[{res['bootstrap_video_train_vs_baseline']['ci_low']:+.4f},"
            f"{res['bootstrap_video_train_vs_baseline']['ci_high']:+.4f}]",
            flush=True,
        )

    fold_rows, base_acc, method_acc = [], [], []
    idx_seg = np.arange(len(data.y))
    idx_vid = np.arange(len(yv))
    for fold, (_, te_seg) in enumerate(p3_mixed_cv(data.subject, n_splits=5), start=1):
        test_subjects = np.unique(data.subject[te_seg])
        te_seg = idx_seg[np.isin(data.subject, test_subjects)]
        tr_seg = idx_seg[~np.isin(data.subject, test_subjects)]
        te_vid = idx_vid[np.isin(subject_v, test_subjects)]
        tr_vid = idx_vid[~np.isin(subject_v, test_subjects)]
        res = _eval_fixed_split(F, data, Fv, yv, subject_v, population_v, video_id_v, tr_seg, te_seg, tr_vid, te_vid)
        base_acc.append(res["baseline"]["accuracy"])
        method_acc.append(res["video_train"]["accuracy"])
        fold_rows.append({"fold": fold, "test_subjects": test_subjects.astype(str).tolist(), **res})
        print(
            f"[t={time.time()-t0:.0f}s] P3 fold {fold}/5 baseline_video={base_acc[-1]:.4f} "
            f"video_train={method_acc[-1]:.4f}",
            flush=True,
        )

    _, p3_p = paired_wilcoxon(method_acc, base_acc)
    results["protocols"]["P3_mixed_cv"] = {
        "folds": fold_rows,
        "baseline_video_mean": float(np.mean(base_acc)),
        "baseline_video_std": float(np.std(base_acc, ddof=1)),
        "method_video_mean": float(np.mean(method_acc)),
        "method_video_std": float(np.std(method_acc, ddof=1)),
        "wilcoxon_method_vs_baseline_p": float(p3_p),
    }
    retained, reason = keep_from_gate(
        results["protocols"]["P1_HC2DEP"]["bootstrap_video_train_vs_baseline"],
        results["protocols"]["P3_mixed_cv"],
    )
    results["retained"] = bool(retained)
    results["retention_reason"] = reason
    print(
        f"[t={time.time()-t0:.0f}s] P3 baseline_video={np.mean(base_acc):.4f} "
        f"video_train={np.mean(method_acc):.4f} p={p3_p:.4f}; {reason}",
        flush=True,
    )

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rich_video_level_training.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    p1 = results["protocols"]["P1_HC2DEP"]
    p2 = results["protocols"]["P2_DEP2HC"]
    p3 = results["protocols"]["P3_mixed_cv"]
    lines = [
        "# Rich Video-Level Training Check",
        "",
        f"- baseline: {results['baseline']}",
        f"- method: direct rich-SVM training on averaged 5-segment video features",
        f"- retained: {results['retained']} ({results['retention_reason']})",
        "",
        "| Protocol | Baseline video | Video-level training | Delta | Gate evidence |",
        "|---|---:|---:|---:|---|",
        (
            f"| P1 HC->DEP | {p1['baseline']['accuracy']:.3f} | {p1['video_train']['accuracy']:.3f} | "
            f"{p1['video_train']['accuracy'] - p1['baseline']['accuracy']:+.3f} | "
            f"bootstrap CI [{p1['bootstrap_video_train_vs_baseline']['ci_low']:+.3f}, "
            f"{p1['bootstrap_video_train_vs_baseline']['ci_high']:+.3f}] |"
        ),
        (
            f"| P2 DEP->HC | {p2['baseline']['accuracy']:.3f} | {p2['video_train']['accuracy']:.3f} | "
            f"{p2['video_train']['accuracy'] - p2['baseline']['accuracy']:+.3f} | "
            f"bootstrap CI [{p2['bootstrap_video_train_vs_baseline']['ci_low']:+.3f}, "
            f"{p2['bootstrap_video_train_vs_baseline']['ci_high']:+.3f}] |"
        ),
        (
            f"| P3 mixed mean | {p3['baseline_video_mean']:.3f} | {p3['method_video_mean']:.3f} | "
            f"{p3['method_video_mean'] - p3['baseline_video_mean']:+.3f} | "
            f"Wilcoxon p={p3['wilcoxon_method_vs_baseline_p']:.4f} |"
        ),
    ]
    with open(os.path.join(OUT, "rich_video_level_training.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
