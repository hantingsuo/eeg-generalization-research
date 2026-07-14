"""Common runner utilities for G1/G2/G3 novelty-method evaluations."""
from __future__ import annotations

import json
import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np

from pcma.data.competition import load_competition
from pcma.data.features import extract_rich_features
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.eval.metrics import summarize
from pcma.eval.stats import bootstrap_paired_accuracy, paired_wilcoxon
from pcma.model.novelty_eval import aggregate_proba_by_group, align_proba, keep_from_gate, pred_from_proba
from pcma.model.rich import fit_predict_rich


OUT = Path(__file__).resolve().parents[1] / "results"


def load_features():
    t0 = time.time()
    data = load_competition()
    F = extract_rich_features(data.X, sfreq=250.0)
    data.X = np.empty((0,), dtype=np.float32)
    gc.collect()
    print(f"[t={time.time()-t0:.0f}s] loaded competition rich features {F.shape}", flush=True)
    return data, F, t0


def rich_proba(F, data, tr, te):
    pred, proba, classes = fit_predict_rich(
        F[tr], data.y[tr], data.subject[tr], F[te], data.subject[te], return_proba=True
    )
    proba = align_proba(proba, classes, target_classes=(0, 1))
    return pred, proba, np.asarray([0, 1])


def segment_video_eval(y, subject, population, video_id, baseline, method):
    b_pred, b_proba, b_classes = baseline
    m_pred, m_proba, m_classes = method
    b_proba = align_proba(b_proba, b_classes, target_classes=(0, 1))
    m_proba = align_proba(m_proba, m_classes, target_classes=(0, 1))
    b_pred = np.asarray(b_pred)
    m_pred = np.asarray(m_pred)
    b_video = aggregate_proba_by_group(y, b_proba, np.asarray([0, 1]), video_id, subject, population)
    m_video = aggregate_proba_by_group(y, m_proba, np.asarray([0, 1]), video_id, subject, population)
    return {
        "segment": {
            "baseline": summarize(y, b_pred, subject, population),
            "method": summarize(y, m_pred, subject, population),
            "bootstrap_method_vs_baseline": bootstrap_paired_accuracy(y, b_pred, m_pred),
        },
        "video": {
            "baseline": summarize(b_video["y"], b_video["pred"], b_video["subject"], b_video["population"]),
            "method": summarize(m_video["y"], m_video["pred"], m_video["subject"], m_video["population"]),
            "bootstrap_method_vs_baseline": bootstrap_paired_accuracy(b_video["y"], b_video["pred"], m_video["pred"]),
        },
    }


def evaluate_method(method_name: str, predictor, data, F, t0=None) -> dict:
    t0 = time.time() if t0 is None else t0
    result = {"method": method_name, "protocols": {}}
    for name, split_fn in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = split_fn(data.subject, data.population)
        baseline = rich_proba(F, data, tr, te)
        method = predictor(F, data, tr, te)
        res = segment_video_eval(data.y[te], data.subject[te], data.population[te], data.video_id[te], baseline, method)
        result["protocols"][name] = res
        print(
            f"[t={time.time()-t0:.0f}s] {method_name} {name} "
            f"seg {res['segment']['baseline']['accuracy']:.4f}->{res['segment']['method']['accuracy']:.4f} "
            f"vid {res['video']['baseline']['accuracy']:.4f}->{res['video']['method']['accuracy']:.4f} "
            f"vidCI[{res['video']['bootstrap_method_vs_baseline']['ci_low']:+.4f},"
            f"{res['video']['bootstrap_method_vs_baseline']['ci_high']:+.4f}]",
            flush=True,
        )

    fold_rows, b_seg, m_seg, b_vid, m_vid = [], [], [], [], []
    for fold, (tr, te) in enumerate(p3_mixed_cv(data.subject, n_splits=5), start=1):
        baseline = rich_proba(F, data, tr, te)
        method = predictor(F, data, tr, te)
        res = segment_video_eval(data.y[te], data.subject[te], data.population[te], data.video_id[te], baseline, method)
        bsa = res["segment"]["baseline"]["accuracy"]
        msa = res["segment"]["method"]["accuracy"]
        bva = res["video"]["baseline"]["accuracy"]
        mva = res["video"]["method"]["accuracy"]
        b_seg.append(bsa)
        m_seg.append(msa)
        b_vid.append(bva)
        m_vid.append(mva)
        fold_rows.append({"fold": fold, "segment": res["segment"], "video": res["video"]})
        print(
            f"[t={time.time()-t0:.0f}s] {method_name} P3 fold {fold}/5 "
            f"seg {bsa:.4f}->{msa:.4f} vid {bva:.4f}->{mva:.4f}",
            flush=True,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sample size too small.*")
        _, p_seg = paired_wilcoxon(m_seg, b_seg)
        _, p_vid = paired_wilcoxon(m_vid, b_vid)
    result["protocols"]["P3_mixed_cv"] = {
        "folds": fold_rows,
        "baseline_segment_mean": float(np.mean(b_seg)),
        "method_segment_mean": float(np.mean(m_seg)),
        "baseline_video_mean": float(np.mean(b_vid)),
        "method_video_mean": float(np.mean(m_vid)),
        "wilcoxon_method_vs_baseline_segment_p": float(p_seg),
        "wilcoxon_method_vs_baseline_p": float(p_vid),
    }
    p1_boot = result["protocols"]["P1_HC2DEP"]["video"]["bootstrap_method_vs_baseline"]
    retained, reason = keep_from_gate(p1_boot, result["protocols"]["P3_mixed_cv"])
    result["retained"] = bool(retained)
    result["retention_reason"] = reason
    print(
        f"[t={time.time()-t0:.0f}s] {method_name} P3 video "
        f"{np.mean(b_vid):.4f}->{np.mean(m_vid):.4f} p={p_vid:.4f}; {reason}",
        flush=True,
    )
    return result


def write_json_md(result: dict, stem: str):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    entries = result.get("variants", {result["method"]: result})
    lines = [
        f"# {stem}",
        "",
        "| Method | Retain? | P1 seg | P1 video | P2 seg | P2 video | P3 seg | P3 video | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, res in entries.items():
        p1 = res["protocols"]["P1_HC2DEP"]
        p2 = res["protocols"]["P2_DEP2HC"]
        p3 = res["protocols"]["P3_mixed_cv"]
        lines.append(
            f"| {name} | {res['retained']} | "
            f"{p1['segment']['method']['accuracy']:.3f} | {p1['video']['method']['accuracy']:.3f} | "
            f"{p2['segment']['method']['accuracy']:.3f} | {p2['video']['method']['accuracy']:.3f} | "
            f"{p3['method_segment_mean']:.3f} | {p3['method_video_mean']:.3f} | "
            f"{res['retention_reason']} |"
        )
    with open(OUT / f"{stem}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_result(stem: str):
    with open(OUT / f"{stem}.json", "r", encoding="utf-8") as f:
        return json.load(f)
