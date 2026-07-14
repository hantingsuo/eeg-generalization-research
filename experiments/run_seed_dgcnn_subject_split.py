"""Protocol-audited LibEER DGCNN baseline on subject-disjoint SEED.

The runner reproduces the intended LibEER 9/3/3 subject split while fixing the
released runner's reproducibility hazards:

* test subjects are never reused as validation subjects;
* the best checkpoint is selected only on the three validation subjects;
* official LDS-DE variables are loaded by name and one subject at a time;
* window, trial, and subject-level metrics are all retained.

It still uses the requested historical LibEER DGCNN implementation and loss.
The historical commit matters because the DGCNN graph convolution changed in
2026; results from the two implementations must not be pooled.
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
from pcma.eval.metrics import summarize
from pcma.model.seed_da import aggregate_scores_by_group, trial_group_keys


DEFAULT_EXPECTED_COMMIT = "39dc27e"


@dataclass(frozen=True)
class EvaluationRows:
    labels: np.ndarray
    logits: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libeer-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=DEFAULT_EXPECTED_COMMIT)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--metric-choose", default="macro_f1", choices=("accuracy", "macro_f1"))
    return parser.parse_args()


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def libeer_subject_split(seed: int) -> dict[str, list[int]]:
    """Match LibEER's Python-random 20%/20%/60% subject split exactly."""

    indexes = list(range(15))
    random.Random(seed).shuffle(indexes)
    return {
        "test": indexes[:3],
        "validation": indexes[3:6],
        "train": indexes[6:],
    }


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


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    started = time.time()
    libeer_root = args.libeer_root.resolve()
    libeer_code = libeer_root / "LibEER"
    commit = git_commit(libeer_root)
    if commit != args.expected_commit:
        raise RuntimeError(f"expected LibEER {args.expected_commit}, found {commit}")

    split = libeer_subject_split(args.seed)
    all_subjects: dict[int, SeedWindowRows] = {}
    source_files: dict[str, str] = {}
    for subject_index in range(15):
        loaded = load_seed_de_lds_subject(
            args.feature_root,
            session=args.session,
            subject=subject_index + 1,
        )
        all_subjects[subject_index] = flatten_seed_subject_windows(
            loaded,
            subject=subject_index + 1,
            session=args.session,
        )
        source_files[str(subject_index + 1)] = str(loaded.source_file)

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
            torch.from_numpy(one_hot(train_rows.labels)),
        )
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        model = DGCNN(num_electrodes=62, in_channels=5, num_classes=3).to(device)
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
            chosen = float(validation[args.metric_choose])
            if chosen > best_value:
                best_value = chosen
                best_epoch = epoch
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

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

    checkpoint = args.checkpoint or args.output.with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "libeer_commit": commit,
            "best_epoch": best_epoch,
            "validation_metric": args.metric_choose,
            "validation_value": best_value,
            "split": split,
        },
        checkpoint,
    )

    payload = {
        "status": "completed",
        "purpose": "protocol_matched_target_free_baseline",
        "protocol": "SEED session-wise subject-disjoint 9 train / 3 validation / 3 test",
        "target_access": "none during fitting, normalization, checkpoint selection, or hyperparameter selection",
        "prediction_units": ["one_second_window", "trial_mean_logit"],
        "libeer_commit": commit,
        "dgcnn_historical_implementation": "Chebyshev T0 is an all-ones tensor at this commit",
        "session": args.session,
        "seed": args.seed,
        "split_zero_based": split,
        "split_one_based": {name: [index + 1 for index in indexes] for name, indexes in split.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "metric_choose": args.metric_choose,
        "best_epoch": best_epoch,
        "best_validation_value": best_value,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sample_counts": {name: int(len(rows.labels)) for name, rows in partitions.items()},
        "subject_sample_counts": {
            str(subject + 1): int(len(rows.labels)) for subject, rows in all_subjects.items()
        },
        "source_files": source_files,
        "checkpoint": str(checkpoint.resolve()),
        "history": history,
        "test": test_evaluation,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
