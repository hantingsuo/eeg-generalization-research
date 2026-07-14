"""Run MODMA atypicality-vs-clinical-scale analysis."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pcma.data.modma import (
    DEFAULT_MODMA_ROOT,
    ModmaFeatureConfig,
    load_modma_feature_cache,
    load_modma_raw_features,
    save_modma_feature_cache,
)
from pcma.model.atypicality import group_contrast, severity_correlations, subject_atypicality


OUT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CACHE = DEFAULT_MODMA_ROOT / "modma_features_verified.npz"
SCALES = ("PHQ-9", "GAD-7", "PSQI")


def _blocked_result(root: Path, cache: Path):
    return {
        "status": "blocked",
        "reason": "MODMA root not found",
        "root": str(root),
        "expected_cache": str(cache),
    }


def _scale_map(data, scale):
    if scale == "PHQ-9":
        return dict(data.phq9_by_subject)
    return {
        sid: scores[scale]
        for sid, scores in data.scales_by_subject.items()
        if scale in scores
    }


def _safe_corr(rows, severity, score, population_filter, seed):
    try:
        return severity_correlations(
            rows,
            severity,
            score=score,
            population_filter=population_filter,
            n_perm=5000,
            n_boot=2000,
            seed=seed,
        )
    except ValueError as exc:
        return {
            "score": score,
            "population_filter": population_filter,
            "status": "skipped",
            "reason": str(exc),
        }


def _run_correlations(rows, data):
    scores = ["atypicality"] + [c for c in ("manifold_distance", "classifier_error", "low_confidence") if c in rows[0]]
    out = {}
    seed = 0
    for scale in SCALES:
        severity = _scale_map(data, scale)
        out[scale] = {}
        for population_filter in (None, "MDD"):
            key = "all_subjects" if population_filter is None else "MDD_only"
            out[scale][key] = {}
            for score in scores:
                out[scale][key][score] = _safe_corr(rows, severity, score, population_filter, seed)
                seed += 1
    return out


def _run_group_contrasts(rows):
    contrasts = {}
    labels = sorted({r["population"] for r in rows})
    if "HC" not in labels or "MDD" not in labels:
        return contrasts
    for score in ["atypicality"] + [c for c in ("manifold_distance", "classifier_error", "low_confidence") if c in rows[0]]:
        contrasts[score] = group_contrast(rows, score=score, target_label="MDD", reference_label="HC", n_perm=5000, seed=10)
    return contrasts


def _build_or_load_cache(root: Path, cache: Path):
    rebuild = os.environ.get("MODMA_REBUILD_CACHE", "0") == "1"
    if cache.exists() and not rebuild:
        print(f"loading feature cache {cache}", flush=True)
        return load_modma_feature_cache(cache), False
    max_events = int(os.environ.get("MODMA_MAX_EVENTS_PER_LABEL", "30"))
    config = ModmaFeatureConfig(max_events_per_label=max_events)
    print(f"building MODMA feature cache from raw, max_events_per_label={max_events}", flush=True)

    def progress(msg):
        print(msg, flush=True)

    data = load_modma_raw_features(root, config=config, progress=progress)
    save_modma_feature_cache(cache, data)
    print(f"saved feature cache {cache} with F={data.F.shape}", flush=True)
    return data, True


def _write_markdown(result):
    if result["status"] != "complete":
        return f"# MODMA Severity Analysis\n\nBlocked: {result['reason']}.\n\nRoot: `{result['root']}`\n"
    lines = [
        "# MODMA Severity Analysis",
        "",
        f"- feature cache: `{result['feature_cache']}`",
        f"- built cache this run: {result['built_cache']}",
        f"- feature shape: `{tuple(result['feature_shape'])}`",
        f"- classifier for event difficulty: {result['atypicality']['classifier']}",
        "",
        "## PHQ-9 Primary Correlations",
        "",
        "| Cohort | Score | Spearman r | 95% CI | perm p | Pearson r | 95% CI | perm p | n |",
        "|---|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for cohort in ("MDD_only", "all_subjects"):
        for score, c in result["correlations"]["PHQ-9"][cohort].items():
            if c.get("status") == "skipped":
                lines.append(f"| {cohort} | {score} | skipped | {c['reason']} |  |  |  |  |  |")
                continue
            sp_ci = c["spearman_bootstrap_ci"]
            pr_ci = c["pearson_bootstrap_ci"]
            lines.append(
                f"| {cohort} | {score} | {c['spearman_r']:.3f} | "
                f"[{sp_ci['ci_low']:.3f}, {sp_ci['ci_high']:.3f}] | {c['spearman_permutation']['p_value']:.4f} | "
                f"{c['pearson_r']:.3f} | [{pr_ci['ci_low']:.3f}, {pr_ci['ci_high']:.3f}] | "
                f"{c['pearson_permutation']['p_value']:.4f} | {c['n']} |"
            )
    lines += [
        "",
        "## Secondary Scales",
        "",
        "| Scale | Cohort | Score | Spearman r | perm p | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scale in ("GAD-7", "PSQI"):
        for cohort in ("MDD_only", "all_subjects"):
            for score, c in result["correlations"][scale][cohort].items():
                if c.get("status") == "skipped":
                    lines.append(f"| {scale} | {cohort} | {score} | skipped |  |  |")
                else:
                    lines.append(
                        f"| {scale} | {cohort} | {score} | {c['spearman_r']:.3f} | "
                        f"{c['spearman_permutation']['p_value']:.4f} | {c['n']} |"
                    )
    if result["contrasts"]:
        lines += [
            "",
            "## MDD vs HC Atypicality",
            "",
            "| Score | HC mean | MDD mean | MDD-HC | perm p | Cliff delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for score, c in result["contrasts"].items():
            lines.append(
                f"| {score} | {c['reference_mean']:.3f} | {c['target_mean']:.3f} | "
                f"{c['target_mean'] - c['reference_mean']:+.3f} | {c['permutation_mean_diff']['p_value']:.4f} | "
                f"{c['cliffs_delta']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    root = Path(os.environ.get("MODMA_ROOT", DEFAULT_MODMA_ROOT))
    cache = Path(os.environ.get("MODMA_FEATURE_CACHE", DEFAULT_CACHE))
    OUT.mkdir(exist_ok=True)
    if not root.exists():
        result = _blocked_result(root, cache)
    else:
        data, built = _build_or_load_cache(root, cache)
        print(f"[t={time.time()-t0:.0f}s] feature table {data.F.shape}", flush=True)
        atyp = subject_atypicality(
            data.F,
            data.y_emotion,
            data.subject,
            data.population,
            healthy_label="HC",
            max_components=20,
            classifier="logreg",
        )
        rows = atyp["subjects"]
        correlations = _run_correlations(rows, data)
        contrasts = _run_group_contrasts(rows)
        result = {
            "status": "complete",
            "root": str(root),
            "feature_cache": str(cache),
            "built_cache": built,
            "feature_shape": list(data.F.shape),
            "feature_definition": {
                "events": ["fcue", "hcue", "scue"],
                "window_sec": [0.0, 1.0],
                "max_events_per_label_per_subject": int(os.environ.get("MODMA_MAX_EVENTS_PER_LABEL", "30")),
                "features": "per-EEG-channel five-band differential entropy and relative power",
            },
            "atypicality": atyp,
            "correlations": correlations,
            "contrasts": contrasts,
        }
        primary = correlations["PHQ-9"]["MDD_only"]["atypicality"]
        if primary.get("status") != "skipped":
            print(
                f"[t={time.time()-t0:.0f}s] PHQ-9 MDD-only atypicality spearman={primary['spearman_r']:.4f} "
                f"perm_p={primary['spearman_permutation']['p_value']:.4f} n={primary['n']}",
                flush=True,
            )

    with open(OUT / "modma_severity.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(OUT / "modma_severity.md", "w", encoding="utf-8") as f:
        f.write(_write_markdown(result))
    print(json.dumps({"status": result["status"], "result": str(OUT / "modma_severity.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
