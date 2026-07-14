"""Protocol-audited LibEER DGCNN baseline on subject-disjoint SEED-IV.

This runner matches the table-bearing LibEER SEED-IV configuration while
keeping the three validation subjects separate from the three test subjects.
Only validation macro-F1 may select a checkpoint; test subjects are evaluated
once, after training and checkpoint selection have finished.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from pcma.data.libeer_seediv import (
    flatten_seediv_subject_windows,
    load_seediv_de_lds_subject,
)
from pcma.eval.metrics import summarize
from pcma.model.seed_da import aggregate_scores_by_group, trial_group_keys


DEFAULT_EXPECTED_COMMIT = "39dc27e"
NUM_SUBJECTS = 15
NUM_CLASSES = 4


@dataclass(frozen=True)
class WindowRows:
    features: np.ndarray
    labels: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


@dataclass(frozen=True)
class EvaluationRows:
    labels: np.ndarray
    logits: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libeer-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=DEFAULT_EXPECTED_COMMIT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--session", type=int, default=1, choices=(1,))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--metric-choose", default="macro_f1", choices=("macro_f1",))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def output_paths(output: Path, checkpoint: Path | None) -> tuple[Path, Path]:
    output = Path(output).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint else output.with_suffix(".pt")
    if output == checkpoint:
        raise ValueError("output JSON and checkpoint must use different paths")
    return output, checkpoint


def ensure_outputs_absent(output: Path, checkpoint: Path) -> None:
    """Refuse to overwrite either inferential result artifact."""

    collisions = [path for path in (output, checkpoint) if path.exists()]
    if collisions:
        joined = ", ".join(str(path) for path in collisions)
        raise FileExistsError(f"refusing to overwrite existing artifact(s): {joined}")


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def libeer_subject_split(seed: int) -> dict[str, list[int]]:
    """Match LibEER's seeded 20%/20%/60% split of 15 subjects."""

    indexes = list(range(NUM_SUBJECTS))
    random.Random(seed).shuffle(indexes)
    return {
        "test": indexes[:3],
        "validation": indexes[3:6],
        "train": indexes[6:],
    }


def concatenate_rows(rows: Sequence[object]) -> WindowRows:
    if not rows:
        raise ValueError("cannot concatenate an empty row list")
    return WindowRows(
        features=np.concatenate([row.features for row in rows]),
        labels=np.concatenate([row.labels for row in rows]),
        subject=np.concatenate([row.subject for row in rows]),
        session=np.concatenate([row.session for row in rows]),
        trial=np.concatenate([row.trial for row in rows]),
    )


def one_hot_four_class(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError("labels outside SEED-IV class range 0..3")
    return np.eye(NUM_CLASSES, dtype=np.float32)[labels]


@torch.no_grad()
def predict_rows(
    model: nn.Module,
    rows: WindowRows,
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
    if rows.logits.ndim != 2 or rows.logits.shape[1] != NUM_CLASSES:
        raise ValueError("SEED-IV evaluation requires four model logits per row")
    observed = set(np.unique(rows.labels).tolist())
    if observed != set(range(NUM_CLASSES)):
        raise ValueError(f"SEED-IV evaluation requires all four labels, observed {sorted(observed)}")

    predictions = rows.logits.argmax(axis=1)
    window = summarize(rows.labels, predictions, rows.subject)

    groups = trial_group_keys(rows.subject, rows.session, rows.trial)
    trial_y, trial_pred, trial_groups = aggregate_scores_by_group(
        rows.logits,
        np.arange(NUM_CLASSES),
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
    return {
        "window": window,
        "trial": trial,
        "per_subject": per_subject,
        "worst": {
            "window_subject_accuracy": float(window["worst_subject_acc"]),
            "trial_subject_accuracy": float(trial["worst_subject_acc"]),
        },
    }


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def subject_preprocessing_record(loaded: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source_file": str(Path(loaded.source_file).resolve()),
        "loaded_from_cache": bool(getattr(loaded, "loaded_from_cache", False)),
        "cache_file": _json_safe(getattr(loaded, "cache_file", None)),
        "cache_manifest_file": _json_safe(getattr(loaded, "cache_manifest_file", None)),
        "manifest": _json_safe(getattr(loaded, "manifest", getattr(loaded, "provenance", {}))),
        "trial_count": int(len(loaded.trials)),
        "trial_window_counts": [int(len(trial)) for trial in loaded.trials],
    }
    return record


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    output, checkpoint = output_paths(args.output, args.checkpoint)
    ensure_outputs_absent(output, checkpoint)
    started = time.time()

    libeer_root = args.libeer_root.resolve()
    libeer_code = libeer_root / "LibEER"
    commit = git_commit(libeer_root)
    if commit != args.expected_commit:
        raise RuntimeError(f"expected LibEER {args.expected_commit}, found {commit}")

    split = libeer_subject_split(args.seed)
    all_subjects: dict[int, WindowRows] = {}
    preprocessing_subjects: dict[str, object] = {}
    for subject_index in range(NUM_SUBJECTS):
        subject = subject_index + 1
        loaded = load_seediv_de_lds_subject(
            args.dataset_root,
            libeer_root,
            args.session,
            subject,
            cache_root=args.cache_root,
            expected_commit=args.expected_commit,
        )
        all_subjects[subject_index] = flatten_seediv_subject_windows(
            loaded,
            subject=subject,
            session=args.session,
        )
        preprocessing_subjects[str(subject)] = subject_preprocessing_record(loaded)
        print(
            json.dumps(
                {
                    "stage": "preprocessing",
                    "subject": subject,
                    "windows": int(len(all_subjects[subject_index].labels)),
                    "loaded_from_cache": bool(loaded.loaded_from_cache),
                }
            ),
            flush=True,
        )

    partitions = {
        name: concatenate_rows([all_subjects[index] for index in indexes])
        for name, indexes in split.items()
    }

    original_cwd = Path.cwd()
    sys.path.insert(0, str(libeer_code))
    os.chdir(libeer_code)
    try:
        from models.DGCNN import DGCNN, NewSparseL2Regularization
        from utils.utils import setup_seed

        setup_seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        train_rows = partitions["train"]
        train_dataset = TensorDataset(
            torch.from_numpy(train_rows.features),
            torch.from_numpy(one_hot_four_class(train_rows.labels)),
        )
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        model = DGCNN(num_electrodes=62, in_channels=5, num_classes=NUM_CLASSES).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
        criterion = nn.CrossEntropyLoss()
        regularizer = NewSparseL2Regularization(0.01).to(device)

        history: list[dict[str, float]] = []
        best_value = float("-inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, args.epochs + 1):
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

            validation_rows = predict_rows(
                model,
                partitions["validation"],
                device=device,
                batch_size=args.eval_batch_size,
            )
            validation = evaluation_metrics(validation_rows)["window"]
            chosen = float(validation["macro_f1"])
            if chosen > best_value:
                best_value = chosen
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )

            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_macro_f1": float(validation["macro_f1"]),
                "seconds": time.time() - epoch_started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

        if best_state is None:
            raise RuntimeError("no checkpoint was selected")
        model.load_state_dict(best_state)
        test_evaluation = evaluation_metrics(
            predict_rows(
                model,
                partitions["test"],
                device=device,
                batch_size=args.eval_batch_size,
            )
        )
    finally:
        os.chdir(original_cwd)

    ensure_outputs_absent(output, checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "libeer_commit": commit,
            "best_epoch": best_epoch,
            "validation_metric": "macro_f1",
            "validation_value": best_value,
            "split": split,
        },
        checkpoint,
    )

    payload = {
        "status": "completed",
        "purpose": "protocol_matched_target_free_baseline",
        "protocol": "SEED-IV session 1 subject-disjoint 9 train / 3 validation / 3 test",
        "target_access": (
            "fixed per-trial feature extraction sees test inputs but no test labels or "
            "cross-subject target statistics; fitting and checkpoint selection use only "
            "train and validation subjects"
        ),
        "test_evaluation_count": 1,
        "prediction_units": ["one_second_window", "trial_mean_logit"],
        "libeer_commit": commit,
        "dgcnn_historical_implementation": "Chebyshev T0 is an all-ones tensor at this commit",
        "preprocessing": {
            "dataset": "seediv_raw",
            "time_window_seconds": 1,
            "feature_type": "de_lds",
            "normalization": "none in the DGCNN path",
            "cache_root": str(args.cache_root.resolve()) if args.cache_root else None,
            "subjects": preprocessing_subjects,
        },
        "session": args.session,
        "seed": args.seed,
        "num_classes": NUM_CLASSES,
        "split_zero_based": split,
        "split_one_based": {name: [index + 1 for index in indexes] for name, indexes in split.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "metric_choose": "macro_f1",
        "best_epoch": best_epoch,
        "best_validation_value": best_value,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sample_counts": {name: int(len(rows.labels)) for name, rows in partitions.items()},
        "subject_sample_counts": {
            str(subject + 1): int(len(rows.labels)) for subject, rows in all_subjects.items()
        },
        "checkpoint": str(checkpoint),
        "history": history,
        "test": test_evaluation,
        "elapsed_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
