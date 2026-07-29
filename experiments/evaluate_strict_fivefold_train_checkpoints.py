"""Evaluate existing strict-fivefold DGCNN checkpoints on their training subjects.

This is a read-only selected-checkpoint fitting diagnostic.  It never opens the
stored test-prediction NPZ files and never recomputes predictions for test
subjects.  Existing scalar validation/test summaries are copied from the frozen
cell JSON files only for descriptive comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_seed_dgcnn_subject_fold import (
    evaluation_metrics as seed_evaluation_metrics,
    load_partition as load_seed_partition,
    predict_rows as predict_seed_rows,
)
from experiments.run_seediv_dgcnn_subject_fold import (
    concatenate_rows as concatenate_seediv_rows,
    evaluation_metrics as seediv_evaluation_metrics,
    predict_rows as predict_seediv_rows,
)
from pcma.data.libeer_seediv import (
    flatten_seediv_subject_windows,
    load_seediv_de_lds_subject,
)


EXPECTED_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"
EXPECTED_FOLDS = tuple(range(1, 6))
EXPECTED_OPTIMIZATION_SEEDS = (2024, 2025, 2026)
MANIFEST = ROOT / "plans" / "2026-07-24-strict-fivefold-train-fit-diagnostic-manifest.md"
SOURCE_ARTIFACT_MANIFEST = (
    ROOT / "results" / "strict_fivefold" / "audits" / "e2_artifact_manifest.json"
)
SOURCE_ARTIFACT_MANIFEST_SHA256 = (
    "72771c181b0222533082e853022337308a98e74aa7d1f9a87862b2c4218fa312"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--libeer-root", type=Path, required=True)
    value.add_argument("--seed-feature-root", type=Path, required=True)
    value.add_argument("--seediv-dataset-root", type=Path, required=True)
    value.add_argument("--seediv-cache-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    value.add_argument("--batch-size", type=int, default=512)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_dgcnn_class(libeer_code: Path) -> type[torch.nn.Module]:
    """Load only DGCNN.py, avoiding LibEER's optional all-model imports."""

    source = libeer_code / "models" / "DGCNN.py"
    spec = importlib.util.spec_from_file_location("pinned_libeer_dgcnn", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DGCNN


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "mean_subject_accuracy": float(metrics["mean_subject_acc"]),
        "worst_subject_accuracy": float(metrics["worst_subject_acc"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
    }


def frozen_validation_summary(cell: dict[str, Any]) -> dict[str, Any]:
    if "best_validation" in cell:
        return {
            "window": scalar_metrics(cell["best_validation"]["window"]),
            "trial": scalar_metrics(cell["best_validation"]["trial"]),
        }
    matches = [
        row for row in cell["history"] if int(row["epoch"]) == int(cell["best_epoch"])
    ]
    if len(matches) != 1:
        raise RuntimeError("could not identify the frozen SEED-IV validation epoch")
    row = matches[0]
    return {
        "window": {
            "accuracy": float(row["validation_accuracy"]),
            "macro_f1": float(row["validation_macro_f1"]),
        },
        "trial": None,
    }


def frozen_test_summary(cell: dict[str, Any]) -> dict[str, Any]:
    """Copy scalar summaries only; never open the checkpoint's NPZ predictions."""

    return {
        "window": scalar_metrics(cell["test"]["window"]),
        "trial": scalar_metrics(cell["test"]["trial"]),
    }


def expected_cells() -> list[tuple[str, int, int, Path, Path]]:
    cells: list[tuple[str, int, int, Path, Path]] = []
    for dataset in ("seed", "seediv"):
        for fold in EXPECTED_FOLDS:
            for optimization_seed in EXPECTED_OPTIMIZATION_SEEDS:
                stem = f"{dataset}_fold{fold}_opt{optimization_seed}"
                directory = ROOT / "results" / "strict_fivefold" / dataset
                cells.append(
                    (
                        dataset,
                        fold,
                        optimization_seed,
                        directory / f"{stem}.json",
                        directory / f"{stem}.pt",
                    )
                )
    return cells


def load_and_validate_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for dataset, fold, optimization_seed, json_path, checkpoint_path in expected_cells():
        if not json_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing frozen cell artifact: {json_path} / {checkpoint_path}")
        cell = json.loads(json_path.read_text(encoding="utf-8"))
        if cell["status"] != "completed":
            raise RuntimeError(f"cell is not completed: {json_path}")
        if cell["dataset"] != dataset:
            raise RuntimeError(f"dataset mismatch: {json_path}")
        if int(cell["fold"]) != fold or int(cell["optimization_seed"]) != optimization_seed:
            raise RuntimeError(f"fold/seed mismatch: {json_path}")
        if int(cell["partition_seed"]) != 2024 or int(cell["session"]) != 1:
            raise RuntimeError(f"partition/session mismatch: {json_path}")
        train_subjects = tuple(int(item) for item in cell["split_one_based"]["train"])
        validation_subjects = tuple(
            int(item) for item in cell["split_one_based"]["validation"]
        )
        test_subjects = tuple(int(item) for item in cell["split_one_based"]["test"])
        if len(train_subjects) != 9 or len(validation_subjects) != 3 or len(test_subjects) != 3:
            raise RuntimeError(f"unexpected split size: {json_path}")
        if set(train_subjects) & (set(validation_subjects) | set(test_subjects)):
            raise RuntimeError(f"overlapping split: {json_path}")
        if set(train_subjects) | set(validation_subjects) | set(test_subjects) != set(
            range(1, 16)
        ):
            raise RuntimeError(f"incomplete split: {json_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        for key, expected in (
            ("fold", fold),
            ("partition_seed", 2024),
            ("optimization_seed", optimization_seed),
            ("best_epoch", int(cell["best_epoch"])),
        ):
            if int(checkpoint[key]) != int(expected):
                raise RuntimeError(f"checkpoint {key} mismatch: {checkpoint_path}")
        checkpoint_train = tuple(int(item) for item in checkpoint["split_one_based"]["train"])
        if checkpoint_train != train_subjects:
            raise RuntimeError(f"checkpoint train split mismatch: {checkpoint_path}")
        cells.append(
            {
                "dataset": dataset,
                "fold": fold,
                "optimization_seed": optimization_seed,
                "json_path": json_path,
                "checkpoint_path": checkpoint_path,
                "json_sha256": sha256(json_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "cell": cell,
                "checkpoint": checkpoint,
                "train_subjects": train_subjects,
            }
        )
    if len(cells) != 30:
        raise RuntimeError(f"expected 30 cells, found {len(cells)}")
    return cells


def summarize_cells(cells: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    selected = [cell for cell in cells if cell["dataset"] == dataset]
    fold_rows: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        fold_cells = [cell for cell in selected if int(cell["fold"]) == fold]
        fold_rows.append(
            {
                "fold": fold,
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
    keys = (
        "train_window_subject_equal_accuracy",
        "train_trial_subject_equal_accuracy",
        "validation_window_accuracy",
        "test_trial_subject_equal_accuracy",
    )
    aggregate = {
        key: {
            "mean_over_five_fold_means": float(np.mean([row[key] for row in fold_rows])),
            "min_fold_mean": float(np.min([row[key] for row in fold_rows])),
            "max_fold_mean": float(np.max([row[key] for row in fold_rows])),
        }
        for key in keys
    }
    aggregate["train_minus_test_trial_accuracy"] = float(
        aggregate["train_trial_subject_equal_accuracy"]["mean_over_five_fold_means"]
        - aggregate["test_trial_subject_equal_accuracy"]["mean_over_five_fold_means"]
    )
    return {
        "n_cells": len(selected),
        "fold_means": fold_rows,
        "descriptive_aggregate": aggregate,
        "interpretation": (
            "Training subjects overlap across folds; fold means are descriptive and "
            "do not define independent sampling units or a population confidence interval."
        ),
    }


def write_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "fold",
        "optimization_seed",
        "best_epoch",
        "train_window_subject_equal_accuracy",
        "train_window_macro_f1",
        "train_trial_subject_equal_accuracy",
        "train_trial_macro_f1",
        "validation_window_accuracy",
        "validation_window_macro_f1",
        "test_trial_subject_equal_accuracy",
        "test_trial_macro_f1",
        "train_minus_test_trial_accuracy",
        "checkpoint_sha256",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "dataset": cell["dataset"],
                    "fold": cell["fold"],
                    "optimization_seed": cell["optimization_seed"],
                    "best_epoch": cell["best_epoch"],
                    "train_window_subject_equal_accuracy": cell["train"]["window"][
                        "mean_subject_accuracy"
                    ],
                    "train_window_macro_f1": cell["train"]["window"]["macro_f1"],
                    "train_trial_subject_equal_accuracy": cell["train"]["trial"][
                        "mean_subject_accuracy"
                    ],
                    "train_trial_macro_f1": cell["train"]["trial"]["macro_f1"],
                    "validation_window_accuracy": cell["validation"]["window"]["accuracy"],
                    "validation_window_macro_f1": cell["validation"]["window"]["macro_f1"],
                    "test_trial_subject_equal_accuracy": cell["test"]["trial"][
                        "mean_subject_accuracy"
                    ],
                    "test_trial_macro_f1": cell["test"]["trial"]["macro_f1"],
                    "train_minus_test_trial_accuracy": (
                        cell["train"]["trial"]["mean_subject_accuracy"]
                        - cell["test"]["trial"]["mean_subject_accuracy"]
                    ),
                    "checkpoint_sha256": cell["checkpoint_sha256"],
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    if sha256(SOURCE_ARTIFACT_MANIFEST) != SOURCE_ARTIFACT_MANIFEST_SHA256:
        raise RuntimeError("source artifact manifest hash does not match the frozen value")
    commit = git_commit(args.libeer_root.resolve())
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"expected LibEER {EXPECTED_COMMIT}, found {commit}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cells = load_and_validate_cells()
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    libeer_code = args.libeer_root.resolve() / "LibEER"
    original_cwd = Path.cwd()
    sys.path.insert(0, str(libeer_code))
    os.chdir(libeer_code)
    completed_cells: list[dict[str, Any]] = []
    try:
        DGCNN = load_dgcnn_class(libeer_code)

        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            grouped[(cell["dataset"], int(cell["fold"]))].append(cell)

        for (dataset, fold), fold_cells in sorted(grouped.items()):
            train_subjects = fold_cells[0]["train_subjects"]
            if any(cell["train_subjects"] != train_subjects for cell in fold_cells):
                raise RuntimeError(f"optimization seeds use different train subjects: {dataset} fold {fold}")

            if dataset == "seed":
                train_rows = load_seed_partition(
                    args.seed_feature_root.resolve(),
                    train_subjects,
                ).rows
                num_classes = 3
            else:
                rows = []
                for subject in train_subjects:
                    loaded = load_seediv_de_lds_subject(
                        args.seediv_dataset_root.resolve(),
                        args.libeer_root.resolve(),
                        1,
                        subject,
                        cache_root=args.seediv_cache_root.resolve(),
                        expected_commit="39dc27e",
                    )
                    rows.append(
                        flatten_seediv_subject_windows(
                            loaded,
                            subject=subject,
                            session=1,
                        )
                    )
                train_rows = concatenate_seediv_rows(rows)
                num_classes = 4

            for cell in sorted(fold_cells, key=lambda item: item["optimization_seed"]):
                model = DGCNN(
                    num_electrodes=62,
                    in_channels=5,
                    num_classes=num_classes,
                ).to(device)
                model.load_state_dict(cell["checkpoint"]["model"])
                if dataset == "seed":
                    prediction_rows = predict_seed_rows(
                        model,
                        train_rows,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    metrics = seed_evaluation_metrics(prediction_rows)
                else:
                    prediction_rows = predict_seediv_rows(
                        model,
                        train_rows,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    metrics = seediv_evaluation_metrics(prediction_rows)

                result = {
                    "dataset": dataset,
                    "fold": fold,
                    "optimization_seed": int(cell["optimization_seed"]),
                    "best_epoch": int(cell["cell"]["best_epoch"]),
                    "train_subjects": list(train_subjects),
                    "window_count": int(len(train_rows.labels)),
                    "json_path": str(cell["json_path"].relative_to(ROOT)),
                    "checkpoint_path": str(cell["checkpoint_path"].relative_to(ROOT)),
                    "json_sha256": cell["json_sha256"],
                    "checkpoint_sha256": cell["checkpoint_sha256"],
                    "train": {
                        "window": scalar_metrics(metrics["window"]),
                        "trial": scalar_metrics(metrics["trial"]),
                        "per_subject": json_safe(metrics["per_subject"]),
                    },
                    "validation": frozen_validation_summary(cell["cell"]),
                    "test": frozen_test_summary(cell["cell"]),
                }
                completed_cells.append(result)
                print(
                    json.dumps(
                        {
                            "status": "cell_completed",
                            "dataset": dataset,
                            "fold": fold,
                            "optimization_seed": cell["optimization_seed"],
                            "best_epoch": result["best_epoch"],
                            "train_window_accuracy": result["train"]["window"]["accuracy"],
                            "train_trial_accuracy": result["train"]["trial"]["accuracy"],
                            "frozen_test_trial_accuracy": result["test"]["trial"][
                                "mean_subject_accuracy"
                            ],
                        }
                    ),
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(str(libeer_code))
        except ValueError:
            pass

    if len(completed_cells) != 30:
        raise RuntimeError(f"expected 30 completed cells, found {len(completed_cells)}")

    payload = {
        "status": "completed",
        "purpose": "selected_checkpoint_train_fit_diagnostic",
        "claim_boundary": (
            "Read-only inference on original training partitions; no test prediction "
            "arrays opened and no test predictions recomputed. Existing scalar test "
            "summaries were copied from frozen cell JSON files."
        ),
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": sha256(MANIFEST),
        },
        "source_artifact_manifest": {
            "path": str(SOURCE_ARTIFACT_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(SOURCE_ARTIFACT_MANIFEST),
        },
        "libeer": {
            "root": str(args.libeer_root.resolve()),
            "commit": commit,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "cells": completed_cells,
        "datasets": {
            dataset: summarize_cells(completed_cells, dataset)
            for dataset in ("seed", "seediv")
        },
        "elapsed_seconds": time.time() - started,
    }

    json_path = args.output_root / "train_fit_diagnostic.json"
    csv_path = args.output_root / "train_fit_cells.csv"
    json_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(csv_path, completed_cells)
    print(
        json.dumps(
            {
                "status": "completed",
                "json": str(json_path),
                "csv": str(csv_path),
                "elapsed_seconds": payload["elapsed_seconds"],
                "dataset_summaries": payload["datasets"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
