"""Production-independent closure audit for the frozen CF-TRE G0-B matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


COMPONENTS = (
    "linear_de310",
    "linear_structural245",
    "linear_all555",
    "rbf_de310",
    "rbf_all555",
    "xgboost_all555",
    "lightgbm_all555",
)
MODELS = ("dgcnn", "gcbnet")
ANCHORS = {
    "dgcnn": {"seed": 0.8255, "seediv": 0.5239},
    "gcbnet": {"seed": 0.8056, "seediv": 0.5328},
}


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("invalid probability matrix")
    values = np.clip(values, 1e-6, 1.0)
    return values / values.sum(axis=1, keepdims=True)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probs = _normalize(probabilities)
    predictions = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
    }


def _trial_metrics(labels: np.ndarray, probabilities: np.ndarray, trials: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probs = _normalize(probabilities)
    groups = np.asarray(trials, dtype=np.int64)
    trial_y: list[int] = []
    trial_probabilities: list[np.ndarray] = []
    for trial in sorted(np.unique(groups).tolist()):
        mask = groups == trial
        values = np.unique(y[mask])
        if len(values) != 1:
            raise ValueError(f"trial {trial} contains multiple labels")
        trial_y.append(int(values[0]))
        trial_probabilities.append(probs[mask].mean(axis=0))
    return _metrics(np.asarray(trial_y), np.stack(trial_probabilities))


def _assert_metrics(actual: Mapping[str, Any], expected: Mapping[str, float], context: str) -> None:
    for key, expected_value in expected.items():
        actual_value = float(actual[key])
        if not np.isclose(actual_value, expected_value, rtol=0, atol=1e-12):
            raise ValueError(f"{context} {key} mismatch: {actual_value} != {expected_value}")


def subject_equal_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    by_subject: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(float(row[metric]))
    if set(by_subject) != set(range(1, 16)):
        raise ValueError(f"subject coverage mismatch: {sorted(by_subject)}")
    if any(len(values) != 3 for values in by_subject.values()):
        raise ValueError("each subject must contribute exactly three sessions")
    return float(np.mean([np.mean(by_subject[subject]) for subject in range(1, 16)]))


def _load_splits(split_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in ("seed", "seediv"):
        path = split_root / f"track_a_{dataset}_split_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "cf_tre_seed_family_v1" or payload.get("dataset") != dataset:
            raise ValueError(f"bad split authority {path}")
        units = {str(unit["unit_id"]): unit for unit in payload["units"]}
        if len(units) != 45:
            raise ValueError(f"{dataset} split has {len(units)} units")
        result[dataset] = units
    return result


def _common_json_audit(payload: Mapping[str, Any], dataset: str, session: int, subject: int) -> None:
    expected = {
        "protocol": "cf_tre_seed_family_v1",
        "stage": "G0-B",
        "dataset": dataset,
        "session": session,
        "subject": subject,
        "unit_id": f"{dataset}:s{session:02d}:sub{subject:02d}",
        "optimization_seed": 2024,
        "validation_only": False,
        "test_contacted": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{expected['unit_id']} {key} mismatch")


def audit_matrix(root: Path, split_root: Path) -> dict[str, Any]:
    splits = _load_splits(split_root)
    summaries: dict[str, dict[str, list[dict[str, Any]]]] = {
        "classical": {dataset: [] for dataset in ("seed", "seediv")},
        "dgcnn": {dataset: [] for dataset in ("seed", "seediv")},
        "gcbnet": {dataset: [] for dataset in ("seed", "seediv")},
    }
    artifact_counts = {"classical_json": 0, "classical_npz": 0, "deep_json": 0, "deep_npz": 0, "deep_pt": 0}

    for dataset in ("seed", "seediv"):
        expected_classes = 3 if dataset == "seed" else 4
        for session in range(1, 4):
            for subject in range(1, 16):
                unit_id = f"{dataset}:s{session:02d}:sub{subject:02d}"
                split = splits[dataset][unit_id]
                stem = f"s{session:02d}_sub{subject:02d}"

                classical_json = root / "classical" / dataset / f"{stem}.json"
                classical_npz = root / "classical" / dataset / f"{stem}.npz"
                if not classical_json.is_file() or not classical_npz.is_file():
                    raise FileNotFoundError(f"incomplete classical artifacts for {unit_id}")
                payload = json.loads(classical_json.read_text(encoding="utf-8"))
                _common_json_audit(payload, dataset, session, subject)
                if tuple(payload.get("components", {})) != COMPONENTS:
                    raise ValueError(f"{unit_id} component ordering/coverage mismatch")
                with np.load(classical_npz, allow_pickle=False) as predictions:
                    labels = predictions["test_labels"]
                    trials = predictions["test_trials"]
                    if set(np.unique(trials).tolist()) != set(split["test_trials"]):
                        raise ValueError(f"{unit_id} classical test trial mismatch")
                    if set(np.unique(labels).tolist()) != set(range(expected_classes)):
                        raise ValueError(f"{unit_id} classical test class mismatch")
                    for component in COMPONENTS:
                        probabilities = predictions[f"test_probabilities__{component}"]
                        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
                            raise ValueError(f"{unit_id} {component} probability simplex failure")
                        window = _metrics(labels, probabilities)
                        trial = _trial_metrics(labels, probabilities, trials)
                        _assert_metrics(payload["components"][component]["test_window"], window, f"{unit_id} {component} window")
                        _assert_metrics(payload["components"][component]["test_trial_mean_probability"], trial, f"{unit_id} {component} trial")
                        summaries["classical"][dataset].append(
                            {
                                "component": component,
                                "session": session,
                                "subject": subject,
                                **window,
                                "trial_accuracy": trial["accuracy"],
                            }
                        )
                artifact_counts["classical_json"] += 1
                artifact_counts["classical_npz"] += 1

                for model in MODELS:
                    deep_dir = root / "deep" / model / dataset
                    json_path = deep_dir / f"{stem}.json"
                    npz_path = deep_dir / f"{stem}.npz"
                    pt_path = deep_dir / f"{stem}.pt"
                    if not json_path.is_file() or not npz_path.is_file() or not pt_path.is_file():
                        raise FileNotFoundError(f"incomplete {model} artifacts for {unit_id}")
                    deep_payload = json.loads(json_path.read_text(encoding="utf-8"))
                    _common_json_audit(deep_payload, dataset, session, subject)
                    if deep_payload.get("model") != model or deep_payload.get("epochs") != 150:
                        raise ValueError(f"{unit_id} {model} model/epoch mismatch")
                    with np.load(npz_path, allow_pickle=False) as predictions:
                        labels = predictions["test_labels"]
                        trials = predictions["test_trials"]
                        probabilities = predictions["test_probabilities"]
                        if set(np.unique(trials).tolist()) != set(split["test_trials"]):
                            raise ValueError(f"{unit_id} {model} test trial mismatch")
                        if set(np.unique(labels).tolist()) != set(range(expected_classes)):
                            raise ValueError(f"{unit_id} {model} test class mismatch")
                        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
                            raise ValueError(f"{unit_id} {model} probability simplex failure")
                        window = _metrics(labels, probabilities)
                        trial = _trial_metrics(labels, probabilities, trials)
                        _assert_metrics(deep_payload["test"]["window"], window, f"{unit_id} {model} window")
                        _assert_metrics(deep_payload["test"]["trial_mean_probability"], trial, f"{unit_id} {model} trial")
                        summaries[model][dataset].append(
                            {
                                "session": session,
                                "subject": subject,
                                **window,
                                "trial_accuracy": trial["accuracy"],
                            }
                        )
                    artifact_counts["deep_json"] += 1
                    artifact_counts["deep_npz"] += 1
                    artifact_counts["deep_pt"] += 1

    aggregate: dict[str, Any] = {"classical": {}, "deep": {}}
    for dataset in ("seed", "seediv"):
        aggregate["classical"][dataset] = {}
        rows = summaries["classical"][dataset]
        for component in COMPONENTS:
            component_rows = [row for row in rows if row["component"] == component]
            aggregate["classical"][dataset][component] = {
                "subject_equal_window_accuracy": subject_equal_mean(component_rows, "accuracy"),
                "subject_equal_macro_f1": subject_equal_mean(component_rows, "macro_f1"),
                "subject_equal_balanced_accuracy": subject_equal_mean(component_rows, "balanced_accuracy"),
                "subject_equal_log_loss": subject_equal_mean(component_rows, "log_loss"),
                "subject_equal_trial_accuracy": subject_equal_mean(component_rows, "trial_accuracy"),
                "unit_count": len(component_rows),
            }
        aggregate["deep"][dataset] = {}
        for model in MODELS:
            model_rows = summaries[model][dataset]
            accuracy = subject_equal_mean(model_rows, "accuracy")
            aggregate["deep"][dataset][model] = {
                "subject_equal_window_accuracy": accuracy,
                "subject_equal_macro_f1": subject_equal_mean(model_rows, "macro_f1"),
                "subject_equal_balanced_accuracy": subject_equal_mean(model_rows, "balanced_accuracy"),
                "subject_equal_log_loss": subject_equal_mean(model_rows, "log_loss"),
                "subject_equal_trial_accuracy": subject_equal_mean(model_rows, "trial_accuracy"),
                "official_anchor": ANCHORS[model][dataset],
                "anchor_gap": accuracy - ANCHORS[model][dataset],
                "within_two_percentage_points": abs(accuracy - ANCHORS[model][dataset]) <= 0.02,
                "unit_count": len(model_rows),
            }

    dgcnn_anchor_gate = all(
        aggregate["deep"][dataset]["dgcnn"]["within_two_percentage_points"]
        for dataset in ("seed", "seediv")
    )
    proposed_outer_candidates = list((root / "cf_tre").glob("**/*")) if (root / "cf_tre").exists() else []
    status = "PASS" if dgcnn_anchor_gate and not proposed_outer_candidates else "PAUSE_ROOT_CAUSE"
    strongest = {
        dataset: max(
            aggregate["classical"][dataset].items(),
            key=lambda item: item[1]["subject_equal_window_accuracy"],
        )[0]
        for dataset in ("seed", "seediv")
    }
    return {
        "protocol": "cf_tre_seed_family_v1",
        "stage": "G0-B",
        "status": status,
        "audit": "production-independent metric and coverage recomputation",
        "artifact_counts": artifact_counts,
        "aggregate": aggregate,
        "strongest_classical_component": strongest,
        "gates": {
            "all_90_classical_units_complete": artifact_counts["classical_json"] == 90,
            "all_180_deep_units_complete": artifact_counts["deep_json"] == 180,
            "dgcnn_within_two_percentage_points_both_datasets": dgcnn_anchor_gate,
            "no_cf_tre_outer_artifact": not proposed_outer_candidates,
        },
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CF-TRE G0-B clean baseline matrix audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "No CF-TRE outer-test score is included in this report.",
        "",
        "## Deep references",
        "",
        "| Dataset | Model | Subject-equal window accuracy | Official anchor | Gap | Within +/-2 pp |",
        "|---|---|---:|---:|---:|---|",
    ]
    for dataset in ("seed", "seediv"):
        for model in MODELS:
            row = report["aggregate"]["deep"][dataset][model]
            lines.append(
                f"| {dataset.upper()} | {model.upper()} | {row['subject_equal_window_accuracy']:.4f} | "
                f"{row['official_anchor']:.4f} | {row['anchor_gap']:+.4f} | "
                f"{'yes' if row['within_two_percentage_points'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Classical components",
            "",
            "| Dataset | Component | Subject-equal window accuracy | Macro-F1 | Trial accuracy |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset in ("seed", "seediv"):
        for component in COMPONENTS:
            row = report["aggregate"]["classical"][dataset][component]
            lines.append(
                f"| {dataset.upper()} | {component} | {row['subject_equal_window_accuracy']:.4f} | "
                f"{row['subject_equal_macro_f1']:.4f} | {row['subject_equal_trial_accuracy']:.4f} |"
            )
    lines.extend(["", "## Gates", ""])
    for key, value in report["gates"].items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cf_tre/g0b/formal"))
    parser.add_argument("--split-root", type=Path, default=Path("results/cf_tre/g0a"))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g0b/g0b_matrix_audit.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/2026-07-14-cf-tre-g0b-validation.md"))
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite G0-B closure artifacts")
    report = audit_matrix(args.root, args.split_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
