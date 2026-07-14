# experiments/run_rich_asymmetry_analysis.py
"""Track B: feature-subset analysis for cross-population asymmetry."""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np

from pcma.data.competition import load_competition
from pcma.data.features import LR_PAIRS, extract_rich_features
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.eval.metrics import summarize
from pcma.eval.stats import bootstrap_paired_accuracy, paired_wilcoxon
from pcma.model.rich import fit_predict_rich


OUT = os.path.join(os.path.dirname(__file__), "..", "results")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHANNEL_NAMES = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]
REGIONS = {
    "frontal": list(range(0, 12)),
    "central": list(range(12, 22)),
    "posterior": list(range(22, 30)),
}


def _unique_sorted(idx):
    return np.asarray(sorted(set(int(i) for i in idx)), dtype=int)


def family_indices(name: str):
    if name == "de":
        return np.arange(0, 150)
    if name == "rel_power":
        return np.arange(150, 300)
    if name == "ratios":
        return np.arange(300, 450)
    if name == "hjorth":
        return np.arange(450, 510)
    if name == "asymmetry":
        return np.arange(510, 570)
    raise ValueError(name)


def band_indices(band: int):
    idx = []
    for ch in range(30):
        idx.append(ch * 5 + band)          # DE
        idx.append(150 + ch * 5 + band)    # relative power
    for pair_i in range(len(LR_PAIRS)):
        idx.append(510 + pair_i * 5 + band)  # DE left-right asymmetry
    return _unique_sorted(idx)


def region_indices(region: str):
    channels = set(REGIONS[region])
    idx = []
    for ch in channels:
        idx.extend(range(ch * 5, ch * 5 + 5))              # DE
        idx.extend(range(150 + ch * 5, 150 + ch * 5 + 5))  # relative power
        idx.extend(range(300 + ch * 5, 300 + ch * 5 + 5))  # ratios
        idx.append(450 + ch)                               # Hjorth mobility
        idx.append(480 + ch)                               # Hjorth complexity
    for pair_i, (left, right) in enumerate(LR_PAIRS):
        if left in channels and right in channels:
            idx.extend(range(510 + pair_i * 5, 510 + pair_i * 5 + 5))
    return _unique_sorted(idx)


def frontal_alpha_asym_indices():
    alpha = 2
    frontal = set(REGIONS["frontal"])
    idx = [
        510 + pair_i * 5 + alpha
        for pair_i, (left, right) in enumerate(LR_PAIRS)
        if left in frontal and right in frontal
    ]
    return _unique_sorted(idx)


def build_subsets():
    subsets = {"full": np.arange(570)}
    for fam in ("de", "asymmetry", "hjorth"):
        subsets[f"family_{fam}"] = family_indices(fam)
    for i, name in enumerate(BAND_NAMES):
        subsets[f"band_{name}"] = band_indices(i)
    for name in REGIONS:
        subsets[f"region_{name}"] = region_indices(name)
    subsets["marker_frontal_alpha_asym"] = frontal_alpha_asym_indices()
    return subsets


def _predict(F, data, tr, te, cols):
    return fit_predict_rich(F[tr][:, cols], data.y[tr], data.subject[tr], F[te][:, cols], data.subject[te])


def _eval_p1_p2(F, data, cols):
    out = {}
    for name, split_fn in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = split_fn(data.subject, data.population)
        pred = _predict(F, data, tr, te, cols)
        out[name] = {
            "summary": summarize(data.y[te], pred, data.subject[te], data.population[te]),
            "y_true": data.y[te].astype(int).tolist(),
            "pred": pred.astype(int).tolist(),
        }
    return out


def _eval_p3(F, data, cols):
    acc = []
    for tr, te in p3_mixed_cv(data.subject, n_splits=5):
        pred = _predict(F, data, tr, te, cols)
        acc.append(float((pred == data.y[te]).mean()))
    return {"fold_accuracy": acc, "mean": float(np.mean(acc)), "std": float(np.std(acc, ddof=1))}


def main():
    t0 = time.time()
    data = load_competition()
    F = extract_rich_features(data.X, sfreq=250.0)
    subsets = build_subsets()
    print(f"[t={time.time()-t0:.0f}s] features {F.shape}; subsets={len(subsets)}", flush=True)

    raw = {}
    for name, cols in subsets.items():
        p12 = _eval_p1_p2(F, data, cols)
        p3 = _eval_p3(F, data, cols)
        raw[name] = {"n_features": int(len(cols)), **p12, "P3_mixed_cv": p3}
        print(
            f"[t={time.time()-t0:.0f}s] {name:28s} n={len(cols):3d} "
            f"P1={p12['P1_HC2DEP']['summary']['accuracy']:.4f} "
            f"P1_posrec={p12['P1_HC2DEP']['summary']['recall_per_class'].get(1, float('nan')):.4f} "
            f"P2={p12['P2_DEP2HC']['summary']['accuracy']:.4f} P3={p3['mean']:.4f}",
            flush=True,
        )

    full = raw["full"]
    results = {}
    for name, res in raw.items():
        compact = {
            "n_features": res["n_features"],
            "P1_HC2DEP": res["P1_HC2DEP"]["summary"],
            "P2_DEP2HC": res["P2_DEP2HC"]["summary"],
            "P3_mixed_cv": res["P3_mixed_cv"],
        }
        if name != "full":
            for protocol in ("P1_HC2DEP", "P2_DEP2HC"):
                compact[protocol]["bootstrap_full_vs_subset"] = bootstrap_paired_accuracy(
                    res[protocol]["y_true"],
                    res[protocol]["pred"],
                    full[protocol]["pred"],
                )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Sample size too small.*")
                _, wp = paired_wilcoxon(
                    full["P3_mixed_cv"]["fold_accuracy"],
                    res["P3_mixed_cv"]["fold_accuracy"],
                )
            compact["P3_mixed_cv"]["wilcoxon_full_vs_subset_p"] = float(wp)
        results[name] = compact

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rich_asymmetry_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    def row(name):
        res = results[name]
        p1 = res["P1_HC2DEP"]
        p2 = res["P2_DEP2HC"]
        p3 = res["P3_mixed_cv"]
        return (
            f"| {name} | {res['n_features']} | {p1['accuracy']:.3f} | "
            f"{p1['recall_per_class'].get(1, float('nan')):.3f} | "
            f"{p2['accuracy']:.3f} | {p2['recall_per_class'].get(1, float('nan')):.3f} | "
            f"{p3['mean']:.3f} |"
        )

    lines = [
        "# Rich Feature-Subset Asymmetry Analysis",
        "",
        f"Full P1 confusion [[TN, FP], [FN, TP]]: {results['full']['P1_HC2DEP']['confusion']}",
        "",
        "| Subset | n feat | P1 acc | P1 positive recall | P2 acc | P2 positive recall | P3 mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        ["full"]
        + [f"family_{x}" for x in ("de", "asymmetry", "hjorth")]
        + [f"band_{x}" for x in BAND_NAMES]
        + [f"region_{x}" for x in REGIONS]
        + ["marker_frontal_alpha_asym"]
    )
    lines.extend(row(name) for name in order)
    lines += [
        "",
        "Band P1 positive recall ranking:",
    ]
    band_rank = sorted(
        ((name, results[name]["P1_HC2DEP"]["recall_per_class"].get(1, 0.0)) for name in results if name.startswith("band_")),
        key=lambda x: x[1],
        reverse=True,
    )
    lines.extend(f"- {name}: {value:.3f}" for name, value in band_rank)
    with open(os.path.join(OUT, "rich_asymmetry_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
