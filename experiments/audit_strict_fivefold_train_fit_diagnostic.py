"""Independent structural and numerical audit of the train-fit diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = ("seed", "seediv")
EXPECTED_FOLDS = tuple(range(1, 6))
EXPECTED_SEEDS = (2024, 2025, 2026)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute_dataset(cells: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["dataset"] == dataset]
    fold_means: list[dict[str, float]] = []
    for fold in EXPECTED_FOLDS:
        fold_cells = [cell for cell in selected if int(cell["fold"]) == fold]
        if len(fold_cells) != 3:
            raise AssertionError(f"{dataset} fold {fold} does not contain three seeds")
        fold_means.append(
            {
                "train_window_subject_equal_accuracy": float(
                    np.mean(
                        [
                            cell["train"]["window"]["mean_subject_accuracy"]
                            for cell in fold_cells
                        ]
                    )
                ),
                "train_trial_subject_equal_accuracy": float(
                    np.mean(
                        [
                            cell["train"]["trial"]["mean_subject_accuracy"]
                            for cell in fold_cells
                        ]
                    )
                ),
                "validation_window_accuracy": float(
                    np.mean(
                        [cell["validation"]["window"]["accuracy"] for cell in fold_cells]
                    )
                ),
                "test_trial_subject_equal_accuracy": float(
                    np.mean(
                        [
                            cell["test"]["trial"]["mean_subject_accuracy"]
                            for cell in fold_cells
                        ]
                    )
                ),
            }
        )
    recomputed = {
        key: {
            "mean_over_five_fold_means": float(np.mean([row[key] for row in fold_means])),
            "min_fold_mean": float(np.min([row[key] for row in fold_means])),
            "max_fold_mean": float(np.max([row[key] for row in fold_means])),
        }
        for key in fold_means[0]
    }
    recomputed["train_minus_test_trial_accuracy"] = float(
        recomputed["train_trial_subject_equal_accuracy"]["mean_over_five_fold_means"]
        - recomputed["test_trial_subject_equal_accuracy"]["mean_over_five_fold_means"]
    )
    return recomputed


def close(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cells = payload["cells"]

    expected_keys = {
        (dataset, fold, seed)
        for dataset in EXPECTED_DATASETS
        for fold in EXPECTED_FOLDS
        for seed in EXPECTED_SEEDS
    }
    observed_keys = {
        (cell["dataset"], int(cell["fold"]), int(cell["optimization_seed"]))
        for cell in cells
    }
    checks: dict[str, Any] = {
        "status_completed": payload["status"] == "completed",
        "cell_count_30": len(cells) == 30,
        "cell_identity_complete": observed_keys == expected_keys,
        "no_prediction_npz_paths": not any(
            ".npz" in json.dumps(cell, sort_keys=True).lower() for cell in cells
        ),
        "nine_train_subjects_each": all(
            len(cell["train_subjects"]) == 9 and len(set(cell["train_subjects"])) == 9
            for cell in cells
        ),
        "checkpoint_hashes_match": True,
        "source_json_hashes_match": True,
        "metrics_in_unit_interval": True,
        "dataset_aggregates_match": True,
    }

    for cell in cells:
        checkpoint = ROOT / cell["checkpoint_path"]
        source_json = ROOT / cell["json_path"]
        checks["checkpoint_hashes_match"] &= (
            checkpoint.is_file() and sha256(checkpoint) == cell["checkpoint_sha256"]
        )
        checks["source_json_hashes_match"] &= (
            source_json.is_file() and sha256(source_json) == cell["json_sha256"]
        )
        for partition in ("train", "test"):
            for unit in ("window", "trial"):
                for metric in ("accuracy", "macro_f1", "mean_subject_accuracy"):
                    value = float(cell[partition][unit][metric])
                    checks["metrics_in_unit_interval"] &= 0.0 <= value <= 1.0

    recomputed: dict[str, Any] = {}
    for dataset in EXPECTED_DATASETS:
        recomputed[dataset] = recompute_dataset(cells, dataset)
        stored = payload["datasets"][dataset]["descriptive_aggregate"]
        for key, value in recomputed[dataset].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    checks["dataset_aggregates_match"] &= close(
                        subvalue,
                        stored[key][subkey],
                    )
            else:
                checks["dataset_aggregates_match"] &= close(value, stored[key])

    passed = all(bool(value) for value in checks.values())
    audit = {
        "status": "pass" if passed else "fail",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256(args.input),
        },
        "checks": checks,
        "recomputed_dataset_aggregates": recomputed,
        "claim_boundary": (
            "This audit verifies artifact identity, coverage, metric bounds, and "
            "aggregate arithmetic. It does not establish statistical independence "
            "of overlapping training folds or prove the absence of every possible "
            "external file access."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
