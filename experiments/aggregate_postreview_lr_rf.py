"""Aggregate post-review LR/RF controls with the frozen G1 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_cf_tre_g1_outer_test import _aggregate  # noqa: E402


METHODS = ("logistic_regression_all555", "random_forest_all555")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("."))
    result.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/cf_tre/postreview_baselines/postreview_lr_rf_aggregate.json"
        ),
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    g1 = json.loads(
        (root / "results/cf_tre/g1/g1_outer_test_result.json").read_text(
            encoding="utf-8"
        )
    )
    result: dict[str, Any] = {
        "protocol": "cf_tre_postreview_lr_rf_v1",
        "status": "PASS",
        "post_review": True,
        "confirmatory_gate_unchanged": True,
        "datasets": {},
    }
    for dataset in ("seed", "seediv"):
        units: list[dict[str, Any]] = []
        for session in range(1, 4):
            for subject in range(1, 16):
                path = (
                    root
                    / "results/cf_tre/postreview_baselines/formal"
                    / dataset
                    / f"s{session:02d}_sub{subject:02d}.npz"
                )
                with np.load(path, allow_pickle=False) as arrays:
                    units.append(
                        {
                            "session": session,
                            "subject": subject,
                            "labels": arrays["test_labels"].astype(np.int64),
                            "trials": arrays["test_trials"].astype(np.int64),
                            "probabilities": {
                                method: arrays[
                                    f"test_probabilities__{method}"
                                ].astype(np.float64)
                                for method in METHODS
                            },
                        }
                    )
        alpha = float(
            g1["datasets"][dataset]["selected_tail_configuration"]["alpha"]
        )
        methods = {method: _aggregate(units, method, alpha) for method in METHODS}
        result["datasets"][dataset] = {
            "environment_count": len(units),
            "methods": methods,
            "frozen_g1_methods": {
                method: {
                    key: g1["datasets"][dataset]["methods"][method][key]
                    for key in (
                        "subject_equal_accuracy",
                        "subject_equal_macro_f1",
                        "subject_equal_balanced_accuracy",
                        "subject_equal_trial_accuracy",
                    )
                }
                for method in (
                    "strongest_single",
                    "uniform",
                    "mean_risk",
                    "tail_risk",
                    "dgcnn",
                    "gcbnet",
                )
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parser().parse_args(argv))
    print(
        json.dumps(
            {
                dataset: {
                    method: {
                        "accuracy": rows["subject_equal_accuracy"],
                        "macro_f1": rows["subject_equal_macro_f1"],
                    }
                    for method, rows in result["datasets"][dataset]["methods"].items()
                }
                for dataset in ("seed", "seediv")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
