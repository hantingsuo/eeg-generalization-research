"""MODMA ERP v2 gated analysis.

GATE1 must show cue-class decoding above subject-level majority baseline before
any PHQ-9 atypicality analysis is run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pcma.data.modma import DEFAULT_MODMA_ROOT, load_modma_feature_cache, save_modma_feature_cache
from pcma.data.modma_erp import ModmaErpConfig, load_modma_erp_features
from pcma.model.atypicality import group_contrast, severity_correlations, subject_atypicality
from pcma.model.erp_gate import gate1_passed, loso_erp_decode


OUT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CACHE = DEFAULT_MODMA_ROOT / "modma_erp_features_v2.npz"


def _build_or_load_cache(root: Path, cache: Path):
    rebuild = os.environ.get("MODMA_ERP_REBUILD_CACHE", "0") == "1"
    if cache.exists() and not rebuild:
        print(f"loading ERP cache {cache}", flush=True)
        return load_modma_feature_cache(cache), False
    config = ModmaErpConfig()
    print("building ERP v2 cache from raw", flush=True)

    def progress(msg):
        print(msg, flush=True)

    data = load_modma_erp_features(root, config=config, progress=progress)
    save_modma_feature_cache(cache, data)
    print(f"saved ERP cache {cache} with F={data.F.shape}", flush=True)
    return data, True


def _scale_map(data, scale):
    if scale == "PHQ-9":
        return dict(data.phq9_by_subject)
    return {
        sid: scores[scale]
        for sid, scores in data.scales_by_subject.items()
        if scale in scores
    }


def _run_gate2(data):
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
    correlations = {}
    seed = 0
    for scale in ("PHQ-9", "GAD-7", "PSQI"):
        correlations[scale] = {}
        sev = _scale_map(data, scale)
        for cohort, population_filter in (("MDD_only", "MDD"), ("all_subjects", None)):
            correlations[scale][cohort] = {}
            for score in ["atypicality"] + atyp["components"]:
                correlations[scale][cohort][score] = severity_correlations(
                    rows,
                    sev,
                    score=score,
                    population_filter=population_filter,
                    n_perm=5000,
                    n_boot=2000,
                    seed=seed,
                )
                seed += 1
    contrasts = {}
    labels = sorted({r["population"] for r in rows})
    if "HC" in labels and "MDD" in labels:
        for score in ["atypicality"] + atyp["components"]:
            contrasts[score] = group_contrast(rows, score=score, target_label="MDD", reference_label="HC", seed=seed)
            seed += 1
    return {"atypicality": atyp, "correlations": correlations, "contrasts": contrasts}


def _write_markdown(result):
    definition = result.get("feature_definition") or {}
    reject_summary = definition.get("reject_summary") or {}
    total_epochs = sum(int(v.get("total", 0)) for v in reject_summary.values())
    kept_epochs = sum(int(v.get("kept", 0)) for v in reject_summary.values())
    min_label_rows = []
    for sid, summary in reject_summary.items():
        kept_by_label = summary.get("kept_by_label", {})
        if kept_by_label:
            min_label_rows.append((min(int(v) for v in kept_by_label.values()), sid, kept_by_label))
    min_label_rows = sorted(min_label_rows)[:5]
    lines = [
        "# MODMA ERP v2 Gated Analysis",
        "",
        f"- status: {result['status']}",
        f"- feature cache: `{result['feature_cache']}`",
        f"- built cache this run: {result['built_cache']}",
        f"- feature shape: `{tuple(result['feature_shape'])}`",
        f"- cluster channels: `{definition.get('cluster_channels', [])}`",
        f"- ERP windows: `{definition.get('windows', [])}`",
        f"- artifact rejection kept: {kept_epochs}/{total_epochs}" if total_epochs else "- artifact rejection kept: unavailable",
        "",
        "## GATE1 Cue Decoding",
        "",
    ]
    g = result["gate1"]
    lines += [
        f"- classes: `{g['classes']}`",
        f"- n subjects: {g['n_subjects']}",
        f"- n trials after artifact rejection: {g['n_trials']}",
        f"- mean accuracy: {g['mean_accuracy']:.3f} +/- {g['std_accuracy']:.3f}",
        f"- mean balanced accuracy: {g['mean_balanced_accuracy']:.3f} +/- {g['std_balanced_accuracy']:.3f}",
        f"- mean majority baseline: {g['mean_majority_baseline']:.3f}",
        f"- mean accuracy - majority baseline: {g['mean_accuracy_minus_majority']:+.3f}",
        f"- bootstrap CI for improvement: [{g['accuracy_minus_majority_bootstrap_ci']['ci_low']:+.3f}, {g['accuracy_minus_majority_bootstrap_ci']['ci_high']:+.3f}]",
        f"- Wilcoxon p(greater): {g['wilcoxon_accuracy_gt_majority']['p_value']:.4f}",
        f"- GATE1 passed: {result['gate1_passed']}",
        "",
    ]
    if not result["gate1_passed"]:
        lines += [
            "GATE2 was not run because cue decoding did not pass the fixed gate.",
            "",
        ]
        if min_label_rows:
            lines += [
                "Lowest retained per-condition counts after artifact rejection:",
                "",
                "| Subject | minimum retained condition count | retained by condition |",
                "|---|---:|---|",
            ]
            for min_count, sid, kept_by_label in min_label_rows:
                lines.append(f"| {sid} | {min_count} | `{kept_by_label}` |")
            lines.append("")
        return "\n".join(lines)

    c = result["gate2"]["correlations"]["PHQ-9"]["MDD_only"]["atypicality"]
    ci = c["spearman_bootstrap_ci"]
    lines += [
        "## GATE2 PHQ-9",
        "",
        "| Cohort | Score | Spearman r | 95% CI | perm p | n |",
        "|---|---|---:|---|---:|---:|",
    ]
    for cohort in ("MDD_only", "all_subjects"):
        for score, corr in result["gate2"]["correlations"]["PHQ-9"][cohort].items():
            sci = corr["spearman_bootstrap_ci"]
            lines.append(
                f"| {cohort} | {score} | {corr['spearman_r']:.3f} | "
                f"[{sci['ci_low']:.3f}, {sci['ci_high']:.3f}] | {corr['spearman_permutation']['p_value']:.4f} | {corr['n']} |"
            )
    lines += [
        "",
        f"Primary MDD-only composite: Spearman r={c['spearman_r']:.3f}, "
        f"CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], p={c['spearman_permutation']['p_value']:.4f}.",
        "",
    ]
    return "\n".join(lines)


def main():
    t0 = time.time()
    root = Path(os.environ.get("MODMA_ROOT", DEFAULT_MODMA_ROOT))
    cache = Path(os.environ.get("MODMA_ERP_FEATURE_CACHE", DEFAULT_CACHE))
    OUT.mkdir(exist_ok=True)
    if not root.exists():
        result = {
            "status": "blocked",
            "reason": "MODMA root not found",
            "root": str(root),
            "feature_cache": str(cache),
            "built_cache": False,
            "feature_shape": [],
            "feature_definition": {},
            "gate1": {},
            "gate1_passed": False,
        }
    else:
        data, built = _build_or_load_cache(root, cache)
        print(f"[t={time.time()-t0:.0f}s] ERP feature table {data.F.shape}", flush=True)
        gate1 = loso_erp_decode(data.F, data.y_emotion, data.subject)
        passed = gate1_passed(gate1)
        print(
            f"[t={time.time()-t0:.0f}s] GATE1 acc={gate1['mean_accuracy']:.4f} "
            f"baseline={gate1['mean_majority_baseline']:.4f} "
            f"diff={gate1['mean_accuracy_minus_majority']:+.4f} "
            f"ci=[{gate1['accuracy_minus_majority_bootstrap_ci']['ci_low']:+.4f},"
            f"{gate1['accuracy_minus_majority_bootstrap_ci']['ci_high']:+.4f}] "
            f"wilcoxon_p={gate1['wilcoxon_accuracy_gt_majority']['p_value']:.4f} passed={passed}",
            flush=True,
        )
        result = {
            "status": "gate1_passed" if passed else "gate1_failed",
            "root": str(root),
            "feature_cache": str(cache),
            "built_cache": built,
            "feature_shape": list(data.F.shape),
            "feature_definition": data.config["erp_v2"] if data.config and "erp_v2" in data.config else {},
            "gate1": gate1,
            "gate1_passed": passed,
            "gate2": None,
        }
        if passed:
            print(f"[t={time.time()-t0:.0f}s] running GATE2 severity correlations", flush=True)
            result["gate2"] = _run_gate2(data)
            c = result["gate2"]["correlations"]["PHQ-9"]["MDD_only"]["atypicality"]
            print(
                f"[t={time.time()-t0:.0f}s] GATE2 PHQ9 MDD atypicality spearman={c['spearman_r']:.4f} "
                f"perm_p={c['spearman_permutation']['p_value']:.4f}",
                flush=True,
            )

    with open(OUT / "modma_erp_v2.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(OUT / "modma_erp_v2.md", "w", encoding="utf-8") as f:
        f.write(_write_markdown(result) + "\n")
    print(json.dumps({"status": result["status"], "result": str(OUT / "modma_erp_v2.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
