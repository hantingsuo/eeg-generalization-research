"""Independent prediction-level closure audit for LibEER compatibility C1."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


PROTOCOL = "libeer_39dc27e_clean_table_compat_v1"
ANCHORS = {"seed": {"accuracy": 0.8255, "sd": 0.1561}, "seediv": {"accuracy": 0.5239, "sd": 0.2432}}
SETTINGS = {"seed": {"epochs": 80, "learning_rate": 0.0015}, "seediv": {"epochs": 150, "learning_rate": 0.001}}


def _metric(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or len(probs) != len(y) or not np.all(np.isfinite(probs)):
        raise ValueError("invalid predictions")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("probabilities do not sum to one")
    predicted = probs.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0))}


def _assert_close(actual: Mapping[str, Any], expected: Mapping[str, float], context: str) -> None:
    for key, value in expected.items():
        if not np.isclose(float(actual[key]), value, rtol=0, atol=1e-12):
            raise ValueError(f"{context} {key} mismatch")


def audit(root: Path, split_root: Path) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    counts = {"json": 0, "npz": 0, "pt": 0}
    for dataset in ("seed", "seediv"):
        split_payload = json.loads((split_root / f"libeer_clean_{dataset}_split_v1.json").read_text(encoding="utf-8"))
        split_by_id = {row["unit_id"]: row for row in split_payload["units"]}
        by_subject: dict[int, list[dict[str, float]]] = defaultdict(list)
        unit_rows: list[dict[str, float]] = []
        for session in range(1, 4):
            for subject in range(1, 16):
                unit_id = f"{dataset}:s{session:02d}:sub{subject:02d}"
                stem = f"s{session:02d}_sub{subject:02d}"
                directory = root / "dgcnn" / dataset
                json_path, npz_path, pt_path = directory / f"{stem}.json", directory / f"{stem}.npz", directory / f"{stem}.pt"
                if not all(path.is_file() for path in (json_path, npz_path, pt_path)):
                    raise FileNotFoundError(f"incomplete artifacts: {unit_id}")
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                expected_header = {"protocol": PROTOCOL, "stage": "C1", "dataset": dataset, "session": session, "subject": subject, "model": "dgcnn", "optimization_seed": 2024, "sampler_semantics": "upstream_global_rng", "test_contacted": True}
                for key, value in expected_header.items():
                    if payload.get(key) != value:
                        raise ValueError(f"{unit_id} {key} mismatch")
                for key, value in SETTINGS[dataset].items():
                    if payload.get(key) != value:
                        raise ValueError(f"{unit_id} {key} mismatch")
                with np.load(npz_path, allow_pickle=False) as arrays:
                    labels = arrays["test_labels"]
                    trials = arrays["test_trials"]
                    probabilities = arrays["test_probabilities"]
                    if set(np.unique(trials).tolist()) != set(split_by_id[unit_id]["test_trials"]):
                        raise ValueError(f"{unit_id} test trial mismatch")
                    metrics = _metric(labels, probabilities)
                _assert_close(payload["test"]["window"], metrics, unit_id)
                by_subject[subject].append(metrics)
                unit_rows.append(metrics)
                counts = {key: value + 1 for key, value in counts.items()}
        if set(by_subject) != set(range(1, 16)) or any(len(rows) != 3 for rows in by_subject.values()):
            raise ValueError(f"{dataset} subject/session coverage failure")
        subject_accuracy = [float(np.mean([row["accuracy"] for row in by_subject[subject]])) for subject in range(1, 16)]
        subject_f1 = [float(np.mean([row["macro_f1"] for row in by_subject[subject]])) for subject in range(1, 16)]
        unit_accuracy = [row["accuracy"] for row in unit_rows]
        mean_accuracy = float(np.mean(subject_accuracy))
        gap = mean_accuracy - ANCHORS[dataset]["accuracy"]
        aggregates[dataset] = {
            "subject_equal_window_accuracy": mean_accuracy,
            "unit_population_sd": float(np.std(unit_accuracy)),
            "subject_equal_macro_f1": float(np.mean(subject_f1)),
            "official_accuracy_anchor": ANCHORS[dataset]["accuracy"],
            "official_sd_anchor": ANCHORS[dataset]["sd"],
            "anchor_gap": gap,
            "within_two_percentage_points": abs(gap) <= 0.02,
            "subject_accuracy": subject_accuracy,
            "unit_accuracy": unit_accuracy,
        }
    gate = all(aggregates[dataset]["within_two_percentage_points"] for dataset in aggregates)
    return {"protocol": PROTOCOL, "stage": "C1", "status": "PASS_C1" if gate else "PAUSE_COMPATIBILITY_ROOT_CAUSE", "audit": "independent prediction-level metric and coverage reconstruction", "artifact_counts": counts, "aggregate": aggregates, "gates": {"all_90_cells_complete": counts["json"] == 90, "both_accuracy_anchors_within_two_percentage_points": gate, "gcbnet_c2_authorized": gate}}


def _report(result: Mapping[str, Any]) -> str:
    lines = ["# LibEER clean compatibility C1 validation", "", f"Status: **{result['status']}**", "", "No CF-TRE result is included.", "", "| Dataset | Accuracy | Public anchor | Gap | Subject SD | Public SD | Within +/-2 pp |", "|---|---:|---:|---:|---:|---:|---|"]
    for dataset in ("seed", "seediv"):
        row = result["aggregate"][dataset]
        lines.append(f"| {dataset.upper()} | {row['subject_equal_window_accuracy']:.4f} | {row['official_accuracy_anchor']:.4f} | {row['anchor_gap']:+.4f} | {row['unit_population_sd']:.4f} | {row['official_sd_anchor']:.4f} | {'yes' if row['within_two_percentage_points'] else 'no'} |")
    lines.extend(["", "## Gates", "", f"- all_90_cells_complete: {'PASS' if result['gates']['all_90_cells_complete'] else 'FAIL'}", f"- both_accuracy_anchors_within_two_percentage_points: {'PASS' if result['gates']['both_accuracy_anchors_within_two_percentage_points'] else 'FAIL'}", f"- gcbnet_c2_authorized: {'YES' if result['gates']['gcbnet_c2_authorized'] else 'NO'}", "", "The JSON audit is the numeric authority. C2 may start only when the frozen C1 gate passes.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/libeer_clean_compat/c1/formal"))
    parser.add_argument("--split-root", type=Path, default=Path("results/libeer_clean_compat/c1/splits"))
    parser.add_argument("--output", type=Path, default=Path("results/libeer_clean_compat/c1/c1_dgcnn_audit_v2.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/2026-07-15-libeer-clean-compat-c1-validation-v2.md"))
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.split_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with args.report.open("x", encoding="utf-8") as handle:
        handle.write(_report(result))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS_C1" else 2)


if __name__ == "__main__":
    main()
