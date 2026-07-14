"""Independent structural and arithmetic audit for the legacy DGCNN artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_TRAIN_COUNTS = [681, 634, 695]
EXPECTED_TEST_COUNTS = [439, 470, 475]
MEAN_ANCHOR = 0.8948
SD_ANCHOR = 0.0849


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def population_sd(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def audit_payload(payload: dict) -> dict:
    checks: list[str] = []
    require(payload["status"] == "completed", "artifact status completed", checks)
    require(
        payload["purpose"] == "archival_test_selected_protocol_compatibility_audit",
        "artifact purpose matches frozen run",
        checks,
    )
    require(payload["sessions"] == [1, 2], "sessions are exactly 1 and 2", checks)
    require(payload["subjects"] == list(range(1, 16)), "subjects are exactly 1 through 15", checks)
    require(payload["epochs"] == 80, "epoch count is exactly 80", checks)
    require(payload["batch_size"] == 16, "training batch size is exactly 16", checks)
    require(payload["eval_batch_size"] == 512, "evaluation batch size is exactly 512", checks)
    require("intentionally leaky" in payload["test_access"], "test-label leakage is explicit", checks)

    recordings = payload["recordings"]
    require(len(recordings) == 30, "all 30 subject-session units are present", checks)
    expected_pairs = {(session, subject) for session in (1, 2) for subject in range(1, 16)}
    actual_pairs = {(int(row["session"]), int(row["subject"])) for row in recordings}
    require(actual_pairs == expected_pairs, "unit identities match the frozen 2 by 15 design", checks)

    selected_accuracies: list[float] = []
    selected_macro_f1: list[float] = []
    selected_trial_accuracy: list[float] = []
    final_accuracies: list[float] = []
    selected_epochs: list[int] = []
    subject_values: dict[int, list[float]] = {}

    for row in recordings:
        unit = f"session {row['session']} subject {row['subject']}"
        require(row["train_windows"] == 2010, f"{unit}: train window count", checks)
        require(row["test_windows"] == 1384, f"{unit}: test window count", checks)
        require(row["train_class_counts"] == EXPECTED_TRAIN_COUNTS, f"{unit}: train class counts", checks)
        require(row["test_class_counts"] == EXPECTED_TEST_COUNTS, f"{unit}: test class counts", checks)

        history = row["epoch_history"]
        require(len(history) == 80, f"{unit}: 80 selection-time evaluations", checks)
        require([item["epoch"] for item in history] == list(range(1, 81)), f"{unit}: contiguous epochs", checks)
        accuracies = [float(item["metrics"]["window"]["accuracy"]) for item in history]
        maximum = max(accuracies)
        earliest_epoch = accuracies.index(maximum) + 1
        selected = row["selected"]
        require(int(selected["epoch"]) == earliest_epoch, f"{unit}: earliest strict maximum selected", checks)
        require(close(selected["metrics"]["window"]["accuracy"], maximum), f"{unit}: selected maximum accuracy", checks)
        require(selected["metrics"] == selected["selection_time_metrics"], f"{unit}: reload retest exact", checks)
        require(row["final_epoch"] == history[-1], f"{unit}: final epoch retained", checks)
        expected_uplift = maximum - float(history[-1]["metrics"]["window"]["accuracy"])
        require(close(row["selection_uplift_window_accuracy"], expected_uplift), f"{unit}: uplift arithmetic", checks)

        selected_accuracies.append(maximum)
        selected_macro_f1.append(float(selected["metrics"]["window"]["macro_f1"]))
        selected_trial_accuracy.append(float(selected["metrics"]["trial"]["accuracy"]))
        final_accuracies.append(float(history[-1]["metrics"]["window"]["accuracy"]))
        selected_epochs.append(earliest_epoch)
        subject_values.setdefault(int(row["subject"]), []).append(maximum)

    summary = payload["summary"]
    mean_accuracy = sum(selected_accuracies) / 30
    unit_sd = population_sd(selected_accuracies)
    subject_means = [sum(subject_values[key]) / len(subject_values[key]) for key in sorted(subject_values)]
    derived = {
        "selected_window_accuracy_mean_subject_session": mean_accuracy,
        "selected_window_accuracy_sd_subject_session_population": unit_sd,
        "selected_window_macro_f1_mean_subject_session": sum(selected_macro_f1) / 30,
        "selected_trial_accuracy_mean_subject_session": sum(selected_trial_accuracy) / 30,
        "selected_window_accuracy_mean_subject": sum(subject_means) / 15,
        "selected_window_accuracy_sd_subject_population": population_sd(subject_means),
        "final_epoch_window_accuracy_mean_subject_session": sum(final_accuracies) / 30,
        "within_run_selection_uplift_mean": sum(
            selected - final for selected, final in zip(selected_accuracies, final_accuracies)
        )
        / 30,
        "within_run_selection_uplift_min": min(
            selected - final for selected, final in zip(selected_accuracies, final_accuracies)
        ),
        "selected_epoch_mean": sum(selected_epochs) / 30,
        "selected_epoch_median": float(sorted(selected_epochs)[14] + sorted(selected_epochs)[15]) / 2,
        "selected_epoch_min": min(selected_epochs),
        "selected_epoch_max": max(selected_epochs),
        "worst_selected_subject_session_window_accuracy": min(selected_accuracies),
    }
    for key, value in derived.items():
        require(close(summary[key], value), f"summary field {key} independently reproduced", checks)

    mean_difference = abs(mean_accuracy - MEAN_ANCHOR)
    sd_difference = abs(unit_sd - SD_ANCHOR)
    exact = f"{mean_accuracy:.4f}" == "0.8948" and f"{unit_sd:.4f}" == "0.0849"
    tolerant = mean_difference <= 0.02 and sd_difference <= 0.02
    require(
        summary["historical_table_anchor_exact_at_4_decimals"] is exact,
        "exact four-decimal gate independently reproduced",
        checks,
    )
    require(
        summary["historical_pair_compatibility_within_0_02"] is tolerant,
        "tolerant pair gate independently reproduced",
        checks,
    )

    verdict = (
        "PASS_TABLE_ANCHOR_EXACT"
        if exact
        else "PASS_TOLERANT_COMPATIBILITY"
        if tolerant
        else "FAIL_COMPATIBILITY"
    )
    return {
        "status": "PASS",
        "verdict": verdict,
        "checks_passed": len(checks),
        "derived": {
            **derived,
            "historical_mean_absolute_difference": mean_difference,
            "historical_subject_session_sd_absolute_difference": sd_difference,
            "test_evaluations_per_unit": 81,
            "test_evaluations_total": 2430,
        },
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = audit_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
