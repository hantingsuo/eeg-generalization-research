"""SEED-vs-competition contrast for the same vector CORAL adaptation method."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.data.competition import load_competition
from pcma.data.features import extract_rich_features
from pcma.data.seed import load_seed_family
from pcma.data.splits import p1_hc_to_dep
from pcma.eval.metrics import summarize
from pcma.model.pipelines import coral_align
from pcma.model.rich import fit_predict_rich, per_subject_zscore
from pcma.model.seed_da import fit_predict_seed_coral_svm
from pcma.model.seed_svm import fit_predict_seed_svm, loso_splits


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _seed_loso(data, predictor, **params):
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


def _fit_predict_rich_coral(Ftr, ytr, subject_tr, Fte, subject_te, C=1.0, shrink=0.1):
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    Xtr = scaler.transform(Ftr)
    Xte = coral_align(Xtr, scaler.transform(Fte), shrink=shrink)
    clf = SVC(C=C, gamma="scale", kernel="rbf", probability=True, random_state=0).fit(Xtr, ytr)
    return clf.predict(Xte), clf.predict_proba(Xte), clf.classes_


def _aggregate_video(y_true, proba, classes, video_id, subject, population):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    classes = np.asarray(classes)
    video_id = np.asarray(video_id).astype(str)
    subject = np.asarray(subject)
    population = np.asarray(population)
    yv, pv, sv, popv = [], [], [], []
    for vid in dict.fromkeys(video_id.tolist()):
        mask = video_id == vid
        if len(np.unique(y_true[mask])) != 1:
            raise ValueError(f"{vid}: labels are not constant within video")
        if len(np.unique(subject[mask])) != 1:
            raise ValueError(f"{vid}: subjects are not constant within video")
        yv.append(y_true[mask][0])
        pv.append(classes[np.mean(proba[mask], axis=0).argmax()])
        sv.append(subject[mask][0])
        popv.append(population[mask][0])
    return np.asarray(yv), np.asarray(pv), np.asarray(sv), np.asarray(popv)


def _competition_p1(F, data, use_coral: bool):
    tr, te = p1_hc_to_dep(data.subject, data.population)
    if use_coral:
        pred, proba, classes = _fit_predict_rich_coral(F[tr], data.y[tr], data.subject[tr], F[te], data.subject[te])
    else:
        pred, proba, classes = fit_predict_rich(
            F[tr], data.y[tr], data.subject[tr], F[te], data.subject[te], return_proba=True
        )
    segment = summarize(data.y[te], pred, data.subject[te], data.population[te])
    yv, pv, sv, popv = _aggregate_video(data.y[te], proba, classes, data.video_id[te], data.subject[te], data.population[te])
    video = summarize(yv, pv, sv, popv)
    return {"segment": segment, "video": video}


def main() -> None:
    t0 = time.time()
    seed = load_seed_family("SEED", aggregate="mean_std")
    print(f"[t={time.time()-t0:.0f}s] loaded SEED mean_std X={seed.X.shape}", flush=True)
    seed_baseline = _seed_loso(seed, fit_predict_seed_svm, C=10.0)
    print(f"[t={time.time()-t0:.0f}s] SEED baseline acc={seed_baseline['accuracy']:.4f}", flush=True)
    seed_coral = _seed_loso(seed, fit_predict_seed_coral_svm, C=10.0, shrink=0.1)
    print(f"[t={time.time()-t0:.0f}s] SEED CORAL acc={seed_coral['accuracy']:.4f}", flush=True)

    comp = load_competition()
    F = extract_rich_features(comp.X, sfreq=250.0)
    print(f"[t={time.time()-t0:.0f}s] loaded competition rich features X={F.shape}", flush=True)
    comp_baseline = _competition_p1(F, comp, use_coral=False)
    print(
        f"[t={time.time()-t0:.0f}s] competition P1 baseline "
        f"segment={comp_baseline['segment']['accuracy']:.4f} video={comp_baseline['video']['accuracy']:.4f}",
        flush=True,
    )
    comp_coral = _competition_p1(F, comp, use_coral=True)
    print(
        f"[t={time.time()-t0:.0f}s] competition P1 CORAL "
        f"segment={comp_coral['segment']['accuracy']:.4f} video={comp_coral['video']['accuracy']:.4f}",
        flush=True,
    )

    results = {
        "method": "per-subject zscore + train scaling + vector CORAL(target->source) + RBF-SVM",
        "seed_meanstd_loso": {"baseline": seed_baseline, "coral": seed_coral},
        "competition_p1_hc_to_dep": {"baseline": comp_baseline, "coral": comp_coral},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "seed_competition_coral_contrast.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = [
        (
            "SEED",
            "LOSO pooled trial DE mean+std",
            seed_baseline["accuracy"],
            seed_coral["accuracy"],
        ),
        (
            "Competition P1",
            "HC->DEP segment rich features",
            comp_baseline["segment"]["accuracy"],
            comp_coral["segment"]["accuracy"],
        ),
        (
            "Competition P1",
            "HC->DEP video probability mean",
            comp_baseline["video"]["accuracy"],
            comp_coral["video"]["accuracy"],
        ),
    ]
    lines = [
        "| Domain | Protocol | Baseline | CORAL | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for domain, protocol, baseline, coral in rows:
        lines.append(f"| {domain} | {protocol} | {baseline:.3f} | {coral:.3f} | {coral - baseline:+.3f} |")
    seed_delta = seed_coral["accuracy"] - seed_baseline["accuracy"]
    comp_seg_delta = comp_coral["segment"]["accuracy"] - comp_baseline["segment"]["accuracy"]
    comp_vid_delta = comp_coral["video"]["accuracy"] - comp_baseline["video"]["accuracy"]
    lines += [
        "",
        (
            "Interpretation: fixed vector-CORAL deltas are "
            f"SEED {seed_delta:+.3f}, competition P1 segment {comp_seg_delta:+.3f}, "
            f"competition P1 video {comp_vid_delta:+.3f}."
        ),
    ]
    with open(os.path.join(OUT, "seed_competition_coral_contrast.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[t={time.time()-t0:.0f}s] wrote contrast results", flush=True)


if __name__ == "__main__":
    main()
