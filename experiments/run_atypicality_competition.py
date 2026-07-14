"""Clinical-pivot sanity check: subject-level emotional EEG atypicality on competition data."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from pcma.data.competition import load_competition
from pcma.data.features import extract_rich_features
from pcma.model.atypicality import group_contrast, subject_atypicality


OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    t0 = time.time()
    data = load_competition()
    F = extract_rich_features(data.X, sfreq=250.0)
    data.X = np.empty((0,), dtype=np.float32)
    print(f"[t={time.time()-t0:.0f}s] competition features {F.shape}", flush=True)
    atyp = subject_atypicality(F, data.y, data.subject, data.population, healthy_label="HC", max_components=20)
    rows = atyp["subjects"]
    contrasts = {
        "atypicality": group_contrast(rows, score="atypicality", target_label="DEP", reference_label="HC", n_perm=5000, seed=0),
        "manifold_distance": group_contrast(rows, score="manifold_distance", target_label="DEP", reference_label="HC", n_perm=5000, seed=1),
        "classifier_error": group_contrast(rows, score="classifier_error", target_label="DEP", reference_label="HC", n_perm=5000, seed=2),
        "low_confidence": group_contrast(rows, score="low_confidence", target_label="DEP", reference_label="HC", n_perm=5000, seed=3),
    }
    result = {
        "purpose": "competition sanity check only; no severity labels available",
        "index_definition": "HC-referenced composite of manifold distance, healthy-trained classifier error, and low confidence",
        "atypicality": atyp,
        "contrasts": contrasts,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "atypicality_competition.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = [
        "# Competition Emotional-EEG Atypicality Sanity Check",
        "",
        "This is a binary HC/DEP sanity check only; it does not test severity.",
        "",
        "| Score | HC mean | DEP mean | DEP-HC diff | 95% bootstrap CI | Mann-Whitney p(greater) | Permutation p(greater) | Cliff delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for score, c in contrasts.items():
        b = c["bootstrap_mean_diff"]
        p = c["permutation_mean_diff"]
        lines.append(
            f"| {score} | {c['reference_mean']:.3f} | {c['target_mean']:.3f} | "
            f"{c['target_mean'] - c['reference_mean']:+.3f} | "
            f"[{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
            f"{c['mannwhitney_p_greater']:.4f} | {p['p_value']:.4f} | {c['cliffs_delta']:.3f} |"
        )
    lines += [
        "",
        "Subject-level rows are stored in `results/atypicality_competition.json`.",
    ]
    with open(os.path.join(OUT, "atypicality_competition.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    c = contrasts["atypicality"]
    print(
        f"[t={time.time()-t0:.0f}s] atypicality HC={c['reference_mean']:.4f} DEP={c['target_mean']:.4f} "
        f"diff={c['target_mean'] - c['reference_mean']:+.4f} perm_p={c['permutation_mean_diff']['p_value']:.4f}",
        flush=True,
    )
    print(f"[t={time.time()-t0:.0f}s] wrote results/atypicality_competition.json", flush=True)


if __name__ == "__main__":
    main()
