"""Aggregate the 30 exact-reproduction learning-curve cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DATASETS = ("seed", "seediv")
FOLDS = range(1, 6)
SEEDS = (2024, 2025, 2026)
METRICS = (
    "train_loss",
    "train_online_window_accuracy",
    "validation_window_accuracy",
    "validation_window_macro_f1",
    "validation_trial_accuracy",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("."))
    result.add_argument(
        "--cell-root",
        type=Path,
        default=Path("results/strict_fivefold/learning_curves/cells"),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/strict_fivefold/learning_curves/learning_curve_aggregate.json"
        ),
    )
    return result


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=0)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def expected_paths(root: Path) -> list[Path]:
    return [
        root / f"{dataset}_fold{fold}_opt{seed}.json"
        for seed in SEEDS
        for fold in FOLDS
        for dataset in DATASETS
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    cell_root = root / args.cell_root
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    paths = expected_paths(cell_root)
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("the complete 30-cell learning-curve matrix is required")
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(cell["status"] != "PASS_REPRODUCED" for cell in cells):
        raise ValueError("every learning-curve cell must pass exact reproduction")

    train_fit = json.loads(
        (
            root
            / "results/strict_fivefold/train_fit_diagnostic/train_fit_diagnostic.json"
        ).read_text(encoding="utf-8")
    )
    result: dict[str, Any] = {
        "protocol": "strict_fivefold_dgcnn_learning_curve_v1",
        "status": "PASS",
        "cell_count": len(cells),
        "test_access": "no test rows loaded; endpoint copied from frozen strict result",
        "datasets": {},
    }
    for dataset in DATASETS:
        selected = [cell for cell in cells if cell["dataset"] == dataset]
        epoch_rows = []
        for epoch_index in range(150):
            row: dict[str, Any] = {"epoch": epoch_index + 1}
            for metric in METRICS:
                row[metric] = summarize(
                    np.asarray(
                        [cell["history"][epoch_index][metric] for cell in selected],
                        dtype=np.float64,
                    )
                )
            epoch_rows.append(row)
        best_epochs = np.asarray([cell["best_epoch"] for cell in selected], dtype=np.float64)
        fit = train_fit["datasets"][dataset]["descriptive_aggregate"]
        result["datasets"][dataset] = {
            "cell_count": len(selected),
            "epochs": epoch_rows,
            "best_epoch": summarize(best_epochs),
            "selected_checkpoint_train_window_accuracy": fit[
                "train_window_subject_equal_accuracy"
            ]["mean_over_five_fold_means"],
            "selected_checkpoint_train_trial_accuracy": fit[
                "train_trial_subject_equal_accuracy"
            ]["mean_over_five_fold_means"],
            "frozen_strict_test_trial_accuracy": fit[
                "test_trial_subject_equal_accuracy"
            ]["mean_over_five_fold_means"],
            "train_minus_test_trial_accuracy": fit["train_minus_test_trial_accuracy"],
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
                "status": result["status"],
                "cell_count": result["cell_count"],
                "datasets": {
                    key: {
                        "best_epoch": value["best_epoch"],
                        "train_trial": value[
                            "selected_checkpoint_train_trial_accuracy"
                        ],
                        "test_trial": value["frozen_strict_test_trial_accuracy"],
                    }
                    for key, value in result["datasets"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
