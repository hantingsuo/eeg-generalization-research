"""Strict all-subject SEED DGCNN fold runner.

Each invocation runs one of five frozen 9-train/3-validation/3-test subject
folds on SEED session 1.  The partition RNG and optimization RNG are explicit
and independent.  Checkpoint selection is fixed to pooled validation-window
macro-F1.  Test-subject files are not loaded until fitting and validation-only
selection have finished, and the selected checkpoint is evaluated on test data
exactly once.

This runner intentionally imports the historical LibEER DGCNN implementation.
It does not modify or replace ``run_seed_dgcnn_subject_split.py``, whose fold-1
seed-2024 result remains the regression reference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from pcma.data.libeer_seed import (
    SeedWindowRows,
    flatten_seed_subject_windows,
    load_seed_de_lds_subject,
    one_hot,
)
from pcma.data.seed_folds import DEFAULT_PARTITION_SEED, get_subject_fold
from pcma.eval.metrics import summarize
from pcma.model.seed_da import aggregate_scores_by_group, trial_group_keys


DEFAULT_EXPECTED_COMMIT = "39dc27e"
SESSION = 1
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OPTIMIZATION_SEED = 2024
VALIDATION_METRIC = "macro_f1"
REFERENCE_FOLD_ONE_SPLIT = {
    "train": (15, 11, 7, 4, 5, 10, 12, 3, 8),
    "validation": (14, 13, 2),
    "test": (9, 1, 6),
}


@dataclass(frozen=True)
class EvaluationRows:
    labels: np.ndarray
    logits: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


@dataclass(frozen=True)
class LoadedPartition:
    rows: SeedWindowRows
    source_files: dict[str, str]
    source_sha256: dict[str, str]
    label_file: str
    label_sha256: str


@dataclass
class FitResult:
    model: nn.Module
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_value: float
    best_validation_evaluation: dict[str, object]
    history: list[dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libeer-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=DEFAULT_EXPECTED_COMMIT)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument("--optimization-seed", type=int, default=DEFAULT_OPTIMIZATION_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = args.output.with_suffix(".pt")
    if args.predictions is None:
        args.predictions = args.output.with_suffix(".npz")
    return args


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path_for(output: Path, checkpoint: Path | None) -> Path:
    return checkpoint if checkpoint is not None else output.with_suffix(".pt")


def prediction_path_for(output: Path, predictions: Path | None) -> Path:
    return predictions if predictions is not None else output.with_suffix(".npz")


def require_new_artifacts(output: Path, checkpoint: Path, predictions: Path) -> None:
    paths = (output, checkpoint, predictions)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("output, checkpoint, and predictions paths must be distinct")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def save_prediction_rows(path: Path, rows: EvaluationRows) -> None:
    """Persist the already-computed single test evaluation without re-running the model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            logits=rows.logits,
            labels=rows.labels,
            subject=rows.subject,
            session=rows.session,
            trial=rows.trial,
        )


def subject_fold(fold: int, partition_seed: int) -> dict[str, tuple[int, ...]]:
    split = get_subject_fold(fold, partition_seed=partition_seed)
    validate_subject_split(split)
    return split


def validate_subject_split(split: dict[str, tuple[int, ...]]) -> None:
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("split must contain train, validation, and test")
    if (len(split["train"]), len(split["validation"]), len(split["test"])) != (9, 3, 3):
        raise ValueError("split must have 9 train, 3 validation, and 3 test subjects")
    partitions = [set(split[name]) for name in ("train", "validation", "test")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("subject partitions must be disjoint")
    if set.union(*partitions) != set(range(1, 16)):
        raise ValueError("subject partitions must cover subjects 1..15 exactly")


def split_zero_based(split: dict[str, tuple[int, ...]]) -> dict[str, list[int]]:
    return {name: [subject - 1 for subject in subjects] for name, subjects in split.items()}


def is_reference_fold_one_configuration(args: argparse.Namespace, split: dict[str, tuple[int, ...]]) -> bool:
    return bool(
        args.fold == 1
        and args.partition_seed == DEFAULT_PARTITION_SEED
        and args.optimization_seed == DEFAULT_OPTIMIZATION_SEED
        and args.epochs == DEFAULT_EPOCHS
        and args.batch_size == DEFAULT_BATCH_SIZE
        and args.eval_batch_size == DEFAULT_EVAL_BATCH_SIZE
        and args.lr == DEFAULT_LEARNING_RATE
        and split == REFERENCE_FOLD_ONE_SPLIT
    )


def concatenate_rows(rows: list[SeedWindowRows]) -> SeedWindowRows:
    if not rows:
        raise ValueError("cannot concatenate an empty row list")
    return SeedWindowRows(
        features=np.concatenate([row.features for row in rows]),
        labels=np.concatenate([row.labels for row in rows]),
        subject=np.concatenate([row.subject for row in rows]),
        session=np.concatenate([row.session for row in rows]),
        trial=np.concatenate([row.trial for row in rows]),
    )


def load_partition(feature_root: Path, subjects: tuple[int, ...]) -> LoadedPartition:
    """Load one partition; callers control when target-subject data enter memory."""

    rows: list[SeedWindowRows] = []
    source_files: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    label_file: Path | None = None
    for subject in subjects:
        loaded = load_seed_de_lds_subject(feature_root, session=SESSION, subject=subject)
        rows.append(flatten_seed_subject_windows(loaded, subject=subject, session=SESSION))
        source_files[str(subject)] = str(loaded.source_file)
        source_hashes[str(subject)] = file_sha256(loaded.source_file)
        if label_file is None:
            label_file = loaded.label_file
        elif loaded.label_file.resolve() != label_file.resolve():
            raise RuntimeError("partition subjects resolved to different SEED label files")
    if label_file is None:
        raise ValueError("partition subjects must not be empty")
    return LoadedPartition(
        rows=concatenate_rows(rows),
        source_files=source_files,
        source_sha256=source_hashes,
        label_file=str(label_file),
        label_sha256=file_sha256(label_file),
    )


@torch.no_grad()
def predict_rows(
    model: nn.Module,
    rows: SeedWindowRows,
    *,
    device: torch.device,
    batch_size: int,
) -> EvaluationRows:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(rows.features), torch.from_numpy(rows.labels)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for features, targets in loader:
        logits.append(model(features.to(device)).cpu().numpy())
        labels.append(targets.numpy())
    return EvaluationRows(
        labels=np.concatenate(labels),
        logits=np.concatenate(logits),
        subject=rows.subject,
        session=rows.session,
        trial=rows.trial,
    )


def evaluation_metrics(rows: EvaluationRows) -> dict[str, object]:
    predictions = rows.logits.argmax(axis=1)
    window = summarize(rows.labels, predictions, rows.subject)

    groups = trial_group_keys(rows.subject, rows.session, rows.trial)
    trial_y, trial_pred, trial_groups = aggregate_scores_by_group(
        rows.logits,
        np.arange(rows.logits.shape[1]),
        groups,
        rows.labels,
    )
    group_to_subject = {group: int(str(group).split("_s", 1)[0]) for group in trial_groups}
    trial_subject = np.asarray([group_to_subject[group] for group in trial_groups])
    trial = summarize(trial_y, trial_pred, trial_subject)

    per_subject: dict[str, object] = {}
    for subject in np.unique(rows.subject):
        mask = rows.subject == subject
        trial_mask = trial_subject == subject
        per_subject[str(int(subject))] = {
            "window": summarize(rows.labels[mask], predictions[mask], rows.subject[mask]),
            "trial": summarize(trial_y[trial_mask], trial_pred[trial_mask], trial_subject[trial_mask]),
            "window_count": int(mask.sum()),
            "trial_count": int(trial_mask.sum()),
        }
    return {"window": window, "trial": trial, "per_subject": per_subject}


def validation_score(window_metrics: dict[str, object]) -> float:
    """The checkpoint objective is deliberately not configurable."""

    return float(window_metrics[VALIDATION_METRIC])


def fit_model(
    *,
    train_rows: SeedWindowRows,
    validation_rows: SeedWindowRows,
    model_factory: Callable[..., nn.Module],
    regularizer_factory: Callable[[float], nn.Module],
    setup_seed_fn: Callable[[int], None],
    device: torch.device,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    optimization_seed: int,
) -> FitResult:
    """Fit and select using train/validation only; no test argument exists."""

    setup_seed_fn(optimization_seed)
    train_dataset = TensorDataset(
        torch.from_numpy(train_rows.features),
        torch.from_numpy(one_hot(train_rows.labels)),
    )
    generator = torch.Generator().manual_seed(optimization_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = model_factory(num_electrodes=62, in_channels=5, num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-4)
    criterion = nn.CrossEntropyLoss()
    regularizer = regularizer_factory(0.01).to(device)

    history: list[dict[str, object]] = []
    best_value = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_evaluation: dict[str, object] | None = None
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets) + regularizer(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch}: non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_evaluation = evaluation_metrics(
            predict_rows(model, validation_rows, device=device, batch_size=eval_batch_size)
        )
        validation_window = validation_evaluation["window"]
        chosen = validation_score(validation_window)
        if chosen > best_value:
            best_value = chosen
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            best_validation_evaluation = copy.deepcopy(validation_evaluation)

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_window": validation_window,
            "checkpoint_metric": VALIDATION_METRIC,
            "checkpoint_value": chosen,
            "checkpoint_updated": bool(epoch == best_epoch),
            "seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    if best_state is None or best_validation_evaluation is None:
        raise RuntimeError("no validation checkpoint was selected")
    return FitResult(
        model=model,
        best_state=best_state,
        best_epoch=best_epoch,
        best_validation_value=best_value,
        best_validation_evaluation=best_validation_evaluation,
        history=history,
    )


def validate_numeric_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_numeric_args(args)
    checkpoint = checkpoint_path_for(args.output, args.checkpoint)
    predictions = prediction_path_for(args.output, args.predictions)
    require_new_artifacts(args.output, checkpoint, predictions)
    split = subject_fold(args.fold, args.partition_seed)

    started = time.time()
    libeer_root = args.libeer_root.resolve()
    libeer_code = libeer_root / "LibEER"
    feature_root = args.feature_root.resolve()
    commit = git_commit(libeer_root)
    if not commit.startswith(args.expected_commit):
        raise RuntimeError(f"expected LibEER {args.expected_commit}, found {commit}")

    # Strict target isolation: only train and validation files enter memory here.
    train_partition = load_partition(feature_root, split["train"])
    validation_partition = load_partition(feature_root, split["validation"])

    original_cwd = Path.cwd()
    sys.path.insert(0, str(libeer_code))
    os.chdir(libeer_code)
    try:
        from models.DGCNN import DGCNN, NewSparseL2Regularization
        from utils.utils import setup_seed

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        fitted = fit_model(
            train_rows=train_partition.rows,
            validation_rows=validation_partition.rows,
            model_factory=DGCNN,
            regularizer_factory=NewSparseL2Regularization,
            setup_seed_fn=setup_seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.lr,
            optimization_seed=args.optimization_seed,
        )
        fitted.model.load_state_dict(fitted.best_state)

        # Test files are loaded only after the validation checkpoint is fixed.
        test_partition = load_partition(feature_root, split["test"])
        test_prediction_rows = predict_rows(
            fitted.model,
            test_partition.rows,
            device=device,
            batch_size=args.eval_batch_size,
        )
        test_evaluation = evaluation_metrics(test_prediction_rows)
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(str(libeer_code))
        except ValueError:
            pass

    runner_path = Path(__file__).resolve()
    loader_path = runner_path.parents[1] / "pcma" / "data" / "libeer_seed.py"
    folds_path = runner_path.parents[1] / "pcma" / "data" / "seed_folds.py"
    dgcnn_path = libeer_code / "models" / "DGCNN.py"
    dgcnn_config_path = libeer_code / "config" / "model_param" / "DGCNN.yaml"
    provenance = {
        "runner_sha256": file_sha256(runner_path),
        "loader_sha256": file_sha256(loader_path),
        "fold_definition_sha256": file_sha256(folds_path),
        "libeer_dgcnn_sha256": file_sha256(dgcnn_path),
        "libeer_dgcnn_config_sha256": file_sha256(dgcnn_config_path),
        "source_files": {
            **train_partition.source_files,
            **validation_partition.source_files,
            **test_partition.source_files,
        },
        "source_file_sha256": {
            **train_partition.source_sha256,
            **validation_partition.source_sha256,
            **test_partition.source_sha256,
        },
        "label_file": train_partition.label_file,
        "label_sha256": train_partition.label_sha256,
    }
    if not (
        train_partition.label_sha256
        == validation_partition.label_sha256
        == test_partition.label_sha256
    ):
        raise RuntimeError("partition label digests differ")

    checkpoint_payload = {
        "model": fitted.best_state,
        "libeer_commit": commit,
        "fold": args.fold,
        "partition_seed": args.partition_seed,
        "optimization_seed": args.optimization_seed,
        "session": SESSION,
        "split_one_based": split,
        "best_epoch": fitted.best_epoch,
        "validation_metric": VALIDATION_METRIC,
        "validation_value": fitted.best_validation_value,
        "provenance": provenance,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as handle:
        torch.save(checkpoint_payload, handle)
    save_prediction_rows(predictions, test_prediction_rows)

    payload = {
        "status": "completed",
        "dataset": "seed",
        "purpose": "strict_all_subject_seed_fold_baseline",
        "protocol": "SEED session 1; 9 train / 3 validation / 3 test subjects",
        "target_access": (
            "test-subject data are not loaded during fitting or validation-only checkpoint selection; "
            "the fixed checkpoint is evaluated on test exactly once"
        ),
        "test_evaluation_count": 1,
        "prediction_units": ["one_second_window", "trial_mean_logit"],
        "libeer_commit": commit,
        "dgcnn_historical_implementation": "Chebyshev T0 is an all-ones tensor at this commit",
        "session": SESSION,
        "fold": args.fold,
        "partition_seed": args.partition_seed,
        "optimization_seed": args.optimization_seed,
        "split_one_based": split,
        "split_zero_based": split_zero_based(split),
        "fold1_original_split_regression_compatible": is_reference_fold_one_configuration(args, split),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "validation_metric": "pooled_window_macro_f1",
        "checkpoint_tie_rule": "strict greater-than; earliest epoch retained on ties",
        "best_epoch": fitted.best_epoch,
        "best_validation_value": fitted.best_validation_value,
        "best_validation": fitted.best_validation_evaluation,
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sample_counts": {
            "train": int(len(train_partition.rows.labels)),
            "validation": int(len(validation_partition.rows.labels)),
            "test": int(len(test_partition.rows.labels)),
        },
        "subject_sample_counts": {
            str(subject): int(np.sum(partition.rows.subject == subject))
            for partition in (train_partition, validation_partition, test_partition)
            for subject in np.unique(partition.rows.subject)
        },
        "provenance": provenance,
        "checkpoint": str(checkpoint.resolve()),
        "predictions": str(predictions.resolve()),
        "history": fitted.history,
        "test": test_evaluation,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
