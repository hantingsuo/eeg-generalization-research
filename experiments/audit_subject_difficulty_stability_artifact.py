"""Independent post-result audit of the frozen Gate-C JSON artifact."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def audit(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    required_top = {"schema_version", "protocol", "configurations", "gate"}
    if not required_top <= set(document):
        raise ValueError("incomplete top-level artifact")
    metric_checks = 0
    correlation_checks = 0
    for aggregate in ("mean_std", "mean"):
        for dataset in ("SEED-IV", "SEED"):
            item = document["configurations"][aggregate][dataset]
            if item["integrity"]["status"] != "PASS":
                raise ValueError(f"integrity failed for {aggregate}/{dataset}")
            metrics_by_session = {}
            for session in ("1", "2", "3"):
                payload = item["sessions"][session]
                if len(payload["folds"]) != 15 or len(payload["subject_metrics"]) != 15:
                    raise ValueError("unexpected LOSO fold/subject count")
                records_by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
                for record in payload["oof_predictions"]:
                    records_by_subject[record["subject"]].append(record)
                expected_subjects = {str(value) for value in range(1, 16)}
                if set(records_by_subject) != expected_subjects:
                    raise ValueError("unexpected OOF subject set")
                metrics = {row["subject"]: row for row in payload["subject_metrics"]}
                for subject, records in records_by_subject.items():
                    expected_trials = 24 if dataset == "SEED-IV" else 15
                    if len(records) != expected_trials:
                        raise ValueError("unexpected OOF trial count")
                    if sorted(int(row["trial"]) for row in records) != list(range(1, expected_trials + 1)):
                        raise ValueError("unexpected trial identifiers")
                    correct = np.asarray([row["y_true"] == row["y_pred"] for row in records])
                    blocks = np.asarray([row["block"] for row in records])
                    stored = metrics[subject]
                    recomputed = (
                        float(correct.mean()),
                        float(correct[blocks == "A"].mean()),
                        float(correct[blocks == "B"].mean()),
                    )
                    expected = (
                        stored["accuracy"], stored["block_a_accuracy"], stored["block_b_accuracy"]
                    )
                    if not all(math.isclose(a, b, abs_tol=1e-15) for a, b in zip(recomputed, expected)):
                        raise ValueError("stored subject metric mismatch")
                    metric_checks += 3
                metrics_by_session[session] = metrics

            subject_ids = [str(value) for value in range(1, 16)]
            overall = {
                session: np.asarray([metrics_by_session[session][sid]["accuracy"] for sid in subject_ids])
                for session in ("1", "2", "3")
            }
            split = {
                f"{block}{session}": np.asarray([
                    metrics_by_session[session][sid][f"block_{block.lower()}_accuracy"]
                    for sid in subject_ids
                ])
                for session in ("1", "2", "3") for block in ("A", "B")
            }
            recomputed_cross = [
                float(spearmanr(overall[left], overall[right]).statistic)
                for left, right in (("1", "2"), ("1", "3"), ("2", "3"))
            ]
            recomputed_split = [
                float(spearmanr(split[f"A{session}"], split[f"B{session}"]).statistic)
                for session in ("1", "2", "3")
            ]
            for key, recomputed in (
                ("cross_session_subject_accuracy", recomputed_cross),
                ("within_session_a_vs_b", recomputed_split),
            ):
                stored_stat = item["stability"][key]
                stored_pairwise = [row["rho"] for row in stored_stat["pairwise"]]
                if not np.allclose(recomputed, stored_pairwise, rtol=0.0, atol=1e-15):
                    raise ValueError("stored pairwise correlation mismatch")
                if not math.isclose(float(np.median(recomputed)), stored_stat["median_rho"], abs_tol=1e-15):
                    raise ValueError("stored median correlation mismatch")
                correlation_checks += 4

    primary = document["configurations"]["mean_std"]["SEED-IV"]["stability"]
    split = primary["within_session_a_vs_b"]
    if not split["median_rho"] < 0.30:
        raise ValueError("manual frozen reliability threshold did not fail")
    gate = document["gate"]
    if gate["decision"] != "FAIL_RELIABILITY" or gate["may_proceed"] is not False:
        raise ValueError("stored Gate-C decision mismatch")
    return {
        "status": "PASS",
        "metric_checks": metric_checks,
        "correlation_checks": correlation_checks,
        "manual_gate": "FAIL_RELIABILITY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_path", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "subject_difficulty_stability.json",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.json_path), sort_keys=True))


if __name__ == "__main__":
    main()
