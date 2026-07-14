"""Clean fixed-epoch SEED subject-dependent DGCNN compatibility benchmark.

This runner retains the conventional first-nine/last-six SEED trial protocol
but removes the historical LibEER runner's test-as-validation checkpoint
selection.  Each subject-session model trains for a predeclared number of
epochs and the test trials are evaluated exactly once at the final epoch.

The result is a compatibility/high-accuracy diagnostic.  It does not support a
cross-subject generalization claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from pcma.data.libeer_seed import load_seed_de_lds_subject, make_fixed_front_back_split, one_hot
from pcma.model.seed_da import aggregate_scores_by_group


DEFAULT_EXPECTED_COMMIT = "39dc27e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libeer-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=DEFAULT_EXPECTED_COMMIT)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    logits: list[np.ndarray] = []
    for (batch,) in loader:
        logits.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(logits)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    classes = np.arange(3)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "recall_per_class": {
            str(int(label)): float(value)
            for label, value in zip(
                classes,
                recall_score(labels, predictions, labels=classes, average=None, zero_division=0),
            )
        },
    }


def evaluate_recording(
    labels: np.ndarray,
    logits: np.ndarray,
    trial_ids: np.ndarray,
) -> dict[str, object]:
    window_predictions = logits.argmax(axis=1)
    window = classification_metrics(labels, window_predictions)
    trial_y, trial_predictions, trial_groups = aggregate_scores_by_group(
        logits,
        np.arange(3),
        trial_ids.astype(str),
        labels,
    )
    return {
        "window": window,
        "trial": classification_metrics(trial_y, trial_predictions),
        "window_count": int(len(labels)),
        "trial_count": int(len(trial_groups)),
    }


def aggregate_recordings(recordings: list[dict[str, object]]) -> dict[str, object]:
    metrics = [row["metrics"] for row in recordings]
    window_accuracy = np.asarray([row["window"]["accuracy"] for row in metrics], dtype=float)
    window_macro_f1 = np.asarray([row["window"]["macro_f1"] for row in metrics], dtype=float)
    trial_accuracy = np.asarray([row["trial"]["accuracy"] for row in metrics], dtype=float)
    trial_macro_f1 = np.asarray([row["trial"]["macro_f1"] for row in metrics], dtype=float)
    return {
        "num_subject_sessions": len(recordings),
        "window_accuracy_mean": float(window_accuracy.mean()),
        "window_accuracy_sd_population": float(window_accuracy.std(ddof=0)),
        "window_macro_f1_mean": float(window_macro_f1.mean()),
        "window_macro_f1_sd_population": float(window_macro_f1.std(ddof=0)),
        "trial_accuracy_mean": float(trial_accuracy.mean()),
        "trial_accuracy_sd_population": float(trial_accuracy.std(ddof=0)),
        "trial_macro_f1_mean": float(trial_macro_f1.mean()),
        "trial_macro_f1_sd_population": float(trial_macro_f1.std(ddof=0)),
        "worst_subject_session_window_accuracy": float(window_accuracy.min()),
        "worst_subject_session_trial_accuracy": float(trial_accuracy.min()),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    sessions = list(dict.fromkeys(args.sessions))
    subjects = list(dict.fromkeys(args.subjects or list(range(1, 16))))
    if not sessions or min(sessions) < 1 or max(sessions) > 3:
        raise ValueError("sessions must be in 1..3")
    if not subjects or min(subjects) < 1 or max(subjects) > 15:
        raise ValueError("subjects must be in 1..15")

    started = time.time()
    libeer_root = args.libeer_root.resolve()
    libeer_code = libeer_root / "LibEER"
    commit = git_commit(libeer_root)
    if commit != args.expected_commit:
        raise RuntimeError(f"expected LibEER {args.expected_commit}, found {commit}")

    original_cwd = Path.cwd()
    sys.path.insert(0, str(libeer_code))
    os.chdir(libeer_code)
    try:
        from models.DGCNN import DGCNN, NewSparseL2Regularization
        from utils.utils import setup_seed

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        recordings: list[dict[str, object]] = []
        for session in sessions:
            for subject in subjects:
                unit_started = time.time()
                loaded = load_seed_de_lds_subject(
                    args.feature_root,
                    session=session,
                    subject=subject,
                )
                split = make_fixed_front_back_split(loaded)
                train_features, train_labels, train_trials = split["train"]
                test_features, test_labels, test_trials = split["test"]
                if set(np.unique(train_trials)) != set(range(1, 10)):
                    raise RuntimeError("unexpected training trial provenance")
                if set(np.unique(test_trials)) != set(range(10, 16)):
                    raise RuntimeError("unexpected test trial provenance")

                setup_seed(args.seed)
                generator = torch.Generator().manual_seed(args.seed)
                train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_features),
                        torch.from_numpy(one_hot(train_labels)),
                    ),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    generator=generator,
                )
                model = DGCNN(num_electrodes=62, in_channels=5, num_classes=3).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
                criterion = nn.CrossEntropyLoss()
                regularizer = NewSparseL2Regularization(0.01).to(device)
                final_train_loss = float("nan")
                for _ in range(args.epochs):
                    model.train()
                    epoch_losses: list[float] = []
                    for features, targets in train_loader:
                        features = features.to(device)
                        targets = targets.to(device)
                        optimizer.zero_grad()
                        logits = model(features)
                        loss = criterion(logits, targets) + regularizer(model)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"session {session}, subject {subject}: non-finite loss")
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(float(loss.detach().cpu()))
                    final_train_loss = float(np.mean(epoch_losses))

                test_logits = predict_logits(
                    model,
                    test_features,
                    device=device,
                    batch_size=args.eval_batch_size,
                )
                metrics = evaluate_recording(test_labels, test_logits, test_trials)
                row = {
                    "session": session,
                    "subject": subject,
                    "source_file": str(loaded.source_file),
                    "train_windows": int(len(train_labels)),
                    "test_windows": int(len(test_labels)),
                    "train_class_counts": np.bincount(train_labels, minlength=3).astype(int).tolist(),
                    "test_class_counts": np.bincount(test_labels, minlength=3).astype(int).tolist(),
                    "final_train_loss": final_train_loss,
                    "metrics": metrics,
                    "elapsed_seconds": time.time() - unit_started,
                }
                recordings.append(row)
                print(json.dumps(row), flush=True)
    finally:
        os.chdir(original_cwd)

    payload = {
        "status": "completed",
        "purpose": "clean_fixed_epoch_subject_dependent_compatibility",
        "protocol": "SEED subject-dependent first 9 trials train / last 6 trials test",
        "test_access": "test partition evaluated once after the predeclared final epoch; no checkpoint selection",
        "claim_boundary": "high-accuracy compatibility only; no cross-subject generalization claim",
        "libeer_commit": commit,
        "dgcnn_historical_implementation": "Chebyshev T0 is an all-ones tensor at this commit",
        "sessions": sessions,
        "subjects": subjects,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "recordings": recordings,
        "summary": aggregate_recordings(recordings),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
