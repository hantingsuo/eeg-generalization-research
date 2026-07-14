"""Gate A: target-conditioned, matched-size HC/DEP transfer decomposition.

This is a confirmatory re-analysis of an already inspected dataset, not a
preregistration.  The frozen design is documented in
plans/2026-07-10-advisor-aligned-stage1-blueprint.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.data.competition import load_competition
from pcma.data.features import extract_rich_features
from pcma.eval.stats import (
    bootstrap_two_sample_mean_difference,
    permutation_two_sample_mean_difference,
)
from pcma.model.rich import per_subject_zscore


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "cross_population_gate_a.json"
OUT_MD = ROOT / "results" / "cross_population_gate_a.md"
POPULATIONS = ("HC", "DEP")
METRICS = ("segment", "video")


def balanced_target_schedule(subjects, target_n, repeats, rng):
    """Return target groups with near-equal subject exposure across repeats."""
    subjects = np.asarray(sorted(np.asarray(subjects).astype(str).tolist()))
    if len(subjects) % target_n != 0:
        raise ValueError("target_n must divide the number of subjects")
    groups = []
    while len(groups) < repeats:
        order = rng.permutation(subjects)
        groups.extend(order[i : i + target_n] for i in range(0, len(order), target_n))
    return [np.asarray(group).astype(str) for group in groups[:repeats]]


def extract_rich_features_batched(X, sfreq=250.0, batch_size=120):
    """Extract the unchanged rich feature definition without a >1 GiB temporary."""
    rows = []
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        rows.append(extract_rich_features(X[start:stop], sfreq=sfreq))
        print(f"feature batch {start}:{stop}/{len(X)}", flush=True)
    return np.vstack(rows)


def fit_predict_scores(Ftr, ytr, Fte):
    """Fixed train-only scaling + RBF-SVM; return labels and decision scores."""
    scaler = StandardScaler().fit(Ftr)
    model = SVC(C=1.0, gamma="scale", kernel="rbf", random_state=0)
    model.fit(scaler.transform(Ftr), ytr)
    Xte = scaler.transform(Fte)
    if model.classes_.tolist() != [0, 1]:
        raise ValueError(f"Expected binary classes [0, 1], got {model.classes_.tolist()}")
    return model.predict(Xte), model.decision_function(Xte)


def subject_metric_rows(y, pred, score, subject, video_id):
    """Compute one 10-s segment and one 50-s video accuracy per subject."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    subject = np.asarray(subject).astype(str)
    video_id = np.asarray(video_id).astype(str)
    rows = {}
    for sid in np.unique(subject):
        sm = subject == sid
        segment_acc = float(np.mean(pred[sm] == y[sm]))
        yv, pv = [], []
        for vid in dict.fromkeys(video_id[sm].tolist()):
            vm = video_id == vid
            labels = np.unique(y[vm])
            if len(labels) != 1:
                raise ValueError(f"{vid}: labels are not constant")
            yv.append(int(labels[0]))
            pv.append(int(np.mean(score[vm]) > 0.0))
        rows[sid] = {
            "segment": segment_acc,
            "video": float(np.mean(np.asarray(pv) == np.asarray(yv))),
        }
    return rows


def _empty_accumulator(subjects_by_population):
    return {
        target_pop: {
            source_pop: {
                metric: {str(s): [] for s in subjects_by_population[target_pop]}
                for metric in METRICS
            }
            for source_pop in POPULATIONS
        }
        for target_pop in POPULATIONS
    }


def evaluate_variant(F, data, n_source, target_n, repeats, seed, n_resamples, variant_name):
    rng = np.random.default_rng(seed)
    subjects_by_population = {
        pop: np.asarray(sorted(np.unique(data.subject[data.population == pop]).astype(str).tolist()))
        for pop in POPULATIONS
    }
    schedules = {
        pop: balanced_target_schedule(subjects_by_population[pop], target_n, repeats, rng)
        for pop in POPULATIONS
    }
    acc = _empty_accumulator(subjects_by_population)

    for repeat in range(repeats):
        target_subjects = {pop: schedules[pop][repeat] for pop in POPULATIONS}
        source_subjects = {}
        for pop in POPULATIONS:
            candidates = np.setdiff1d(subjects_by_population[pop], target_subjects[pop])
            if len(candidates) < n_source:
                raise ValueError(f"{pop}: only {len(candidates)} source candidates for n_source={n_source}")
            source_subjects[pop] = rng.choice(candidates, size=n_source, replace=False)

        all_targets = np.concatenate([target_subjects["HC"], target_subjects["DEP"]])
        te = np.isin(data.subject.astype(str), all_targets)
        for source_pop in POPULATIONS:
            tr = np.isin(data.subject.astype(str), source_subjects[source_pop])
            pred, score = fit_predict_scores(F[tr], data.y[tr], F[te])
            rows = subject_metric_rows(
                data.y[te], pred, score, data.subject[te], data.video_id[te]
            )
            for target_pop in POPULATIONS:
                for sid in target_subjects[target_pop]:
                    for metric in METRICS:
                        acc[target_pop][source_pop][metric][str(sid)].append(rows[str(sid)][metric])

        if (repeat + 1) % max(1, repeats // 8) == 0 or repeat + 1 == repeats:
            print(f"{variant_name}: repeat {repeat + 1}/{repeats}", flush=True)

    result = {
        "normalization": variant_name,
        "n_source_per_population": int(n_source),
        "n_target_per_population_per_repeat": int(target_n),
        "repeats": int(repeats),
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        per_subject = {pop: {} for pop in POPULATIONS}
        for target_pop in POPULATIONS:
            within_source = target_pop
            cross_source = "DEP" if target_pop == "HC" else "HC"
            for sid in subjects_by_population[target_pop]:
                sid = str(sid)
                within_values = np.asarray(acc[target_pop][within_source][metric][sid], dtype=float)
                cross_values = np.asarray(acc[target_pop][cross_source][metric][sid], dtype=float)
                if len(within_values) == 0:
                    continue
                if len(within_values) != len(cross_values):
                    raise RuntimeError(f"Missing or unpaired evaluations for {sid}")
                within = float(within_values.mean())
                cross = float(cross_values.mean())
                per_subject[target_pop][sid] = {
                    "n_evaluations": int(len(within_values)),
                    "within_accuracy": within,
                    "cross_accuracy": cross,
                    "transfer_penalty": within - cross,
                }

        hc_rows = list(per_subject["HC"].values())
        dep_rows = list(per_subject["DEP"].values())
        hc_penalty = np.asarray([row["transfer_penalty"] for row in hc_rows])
        dep_penalty = np.asarray([row["transfer_penalty"] for row in dep_rows])
        within_hc = np.asarray([row["within_accuracy"] for row in hc_rows])
        within_dep = np.asarray([row["within_accuracy"] for row in dep_rows])
        cross_hc = np.asarray([row["cross_accuracy"] for row in hc_rows])
        cross_dep = np.asarray([row["cross_accuracy"] for row in dep_rows])

        asymmetry = bootstrap_two_sample_mean_difference(
            dep_penalty, hc_penalty, n_boot=n_resamples, seed=seed + 101 + metric_index
        )
        asymmetry["definition"] = "mean_DEP_transfer_penalty - mean_HC_transfer_penalty"
        asymmetry["permutation_greater"] = permutation_two_sample_mean_difference(
            dep_penalty,
            hc_penalty,
            n_perm=n_resamples,
            seed=seed + 201 + metric_index,
            alternative="greater",
        )
        target_difficulty = bootstrap_two_sample_mean_difference(
            within_hc, within_dep, n_boot=n_resamples, seed=seed + 301 + metric_index
        )
        target_difficulty["definition"] = "mean_HC_within_accuracy - mean_DEP_within_accuracy"
        cross_target_gap = bootstrap_two_sample_mean_difference(
            cross_hc, cross_dep, n_boot=n_resamples, seed=seed + 401 + metric_index
        )
        cross_target_gap["definition"] = "mean_HC_cross_accuracy - mean_DEP_cross_accuracy"

        if asymmetry["mean_diff"] >= 0.03 and asymmetry["ci_low"] > 0:
            gate = "directional_transfer_supported"
        elif target_difficulty["ci_low"] > 0 and asymmetry["ci_low"] <= 0 <= asymmetry["ci_high"]:
            gate = "target_difficulty_not_directional_transfer"
        else:
            gate = "inconclusive"

        result["metrics"][metric] = {
            "per_subject": per_subject,
            "target_subject_coverage": {
                pop: int(len(per_subject[pop])) for pop in POPULATIONS
            },
            "population_summary": {
                "HC": {
                    "within_accuracy": float(within_hc.mean()),
                    "cross_accuracy": float(cross_hc.mean()),
                    "transfer_penalty": float(hc_penalty.mean()),
                },
                "DEP": {
                    "within_accuracy": float(within_dep.mean()),
                    "cross_accuracy": float(cross_dep.mean()),
                    "transfer_penalty": float(dep_penalty.mean()),
                },
            },
            "differential_transfer_penalty": asymmetry,
            "within_target_difficulty_gap": target_difficulty,
            "cross_target_accuracy_gap": cross_target_gap,
            "gate": gate,
        }
    return result


def overall_gate(variants):
    primary = variants["global_train_only"]["metrics"]["segment"]
    sensitivity = variants["subject_batch_transductive"]["metrics"]["segment"]
    if (
        primary["gate"] == "directional_transfer_supported"
        and sensitivity["differential_transfer_penalty"]["mean_diff"] > 0
    ):
        return "directional_transfer_supported"
    if (
        primary["gate"] == "target_difficulty_not_directional_transfer"
        and sensitivity["within_target_difficulty_gap"]["mean_diff"] > 0
    ):
        return "target_difficulty_not_directional_transfer"
    return "inconclusive_or_normalization_sensitive"


def render_markdown(result):
    lines = [
        "# Cross-Population Gate A",
        "",
        f"Overall gate: **{result['overall_gate']}**",
        "",
        "Primary endpoint: 10-s segment accuracy under global train-only normalization. ",
        "The subject-batch result is a declared transductive sensitivity analysis.",
        "",
        "| Normalization | Unit | HC within | HC cross | HC penalty | DEP within | DEP cross | DEP penalty | Differential penalty [95% CI] | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant_name, variant in result["variants"].items():
        for metric, res in variant["metrics"].items():
            hc = res["population_summary"]["HC"]
            dep = res["population_summary"]["DEP"]
            diff = res["differential_transfer_penalty"]
            lines.append(
                f"| {variant_name} | {metric} | {hc['within_accuracy']:.3f} | "
                f"{hc['cross_accuracy']:.3f} | {hc['transfer_penalty']:+.3f} | "
                f"{dep['within_accuracy']:.3f} | {dep['cross_accuracy']:.3f} | "
                f"{dep['transfer_penalty']:+.3f} | {diff['mean_diff']:+.3f} "
                f"[{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}] | {res['gate']} |"
            )
    lines.extend(
        [
            "",
            "Transfer penalty is within-population accuracy minus cross-population accuracy on the same target subjects.",
            "Differential penalty is DEP-target penalty minus HC-target penalty.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-source", type=int, default=15)
    parser.add_argument("--target-n", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--feature-batch-size", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    data = load_competition()
    F_raw = extract_rich_features_batched(
        data.X, sfreq=250.0, batch_size=args.feature_batch_size
    )
    F_subject = per_subject_zscore(F_raw, data.subject)
    print(f"features={F_raw.shape} elapsed={time.time() - t0:.1f}s", flush=True)

    variants = {
        "global_train_only": evaluate_variant(
            F_raw,
            data,
            args.n_source,
            args.target_n,
            args.repeats,
            args.seed,
            args.n_resamples,
            "global_train_only",
        ),
        "subject_batch_transductive": evaluate_variant(
            F_subject,
            data,
            args.n_source,
            args.target_n,
            args.repeats,
            args.seed,
            args.n_resamples,
            "subject_batch_transductive",
        ),
    }
    result = {
        "status": "complete",
        "analysis_scope": "post-hoc confirmatory re-analysis frozen before this runner was executed",
        "primary_endpoint": "global_train_only 10-s segment accuracy",
        "config": vars(args),
        "variants": variants,
        "overall_gate": overall_gate(variants),
        "elapsed_seconds": float(time.time() - t0),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_markdown(result), encoding="utf-8")
    print(f"overall_gate={result['overall_gate']}", flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"elapsed={result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
