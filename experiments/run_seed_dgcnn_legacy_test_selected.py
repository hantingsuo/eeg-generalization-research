"""Audit the archival LibEER SEED high-score checkpoint protocol.

This runner intentionally reproduces the *selection semantics* of LibEER's
``seed_sub_dependent_front_back_setting``: the last-six-trial test partition is
evaluated after every epoch and the earliest epoch with the highest test-window
accuracy is retained.  That is direct test-label model selection.  The output
is a protocol-compatibility artifact, never a leakage-free performance claim.

The implementation keeps LibEER's model, optimizer, loss, split, seed, and
global-RNG RandomSampler behavior.  Evaluation batching is enlarged because
the archival DGCNN has no batch-dependent evaluation layers; this does not
change the selected estimand and materially reduces audit runtime on Windows.
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
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pcma.data.libeer_seed import load_seed_de_lds_subject, make_fixed_front_back_split, one_hot
from pcma.model.seed_da import aggregate_scores_by_group


DEFAULT_EXPECTED_COMMIT = "39dc27e"
HISTORICAL_ACCURACY_ANCHOR = 0.8948
HISTORICAL_SD_ANCHOR = 0.0849


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


def require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Evaluate without shuffling; construction mirrors LibEER evaluation RNG contact."""

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
    trial_y, trial_predictions, trial_groups = aggregate_scores_by_group(
        logits,
        np.arange(3),
        trial_ids.astype(str),
        labels,
    )
    return {
        "window": classification_metrics(labels, window_predictions),
        "trial": classification_metrics(trial_y, trial_predictions),
        "window_count": int(len(labels)),
        "trial_count": int(len(trial_groups)),
    }


def earliest_strict_best(history: list[dict[str, object]]) -> dict[str, object]:
    """Match LibEER's zero-initialized strict-``>`` checkpoint update rule."""

    best: dict[str, object] | None = None
    best_accuracy = 0.0
    for row in history:
        accuracy = float(row["metrics"]["window"]["accuracy"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = row
    if best is None:
        raise RuntimeError("LibEER's strict-best rule selected no checkpoint")
    return best


def aggregate_recordings(recordings: list[dict[str, object]]) -> dict[str, object]:
    if not recordings:
        raise ValueError("recordings must not be empty")

    selected_accuracy = np.asarray(
        [row["selected"]["metrics"]["window"]["accuracy"] for row in recordings], dtype=float
    )
    selected_macro_f1 = np.asarray(
        [row["selected"]["metrics"]["window"]["macro_f1"] for row in recordings], dtype=float
    )
    selected_trial_accuracy = np.asarray(
        [row["selected"]["metrics"]["trial"]["accuracy"] for row in recordings], dtype=float
    )
    final_accuracy = np.asarray(
        [row["final_epoch"]["metrics"]["window"]["accuracy"] for row in recordings], dtype=float
    )
    selected_epochs = np.asarray([row["selected"]["epoch"] for row in recordings], dtype=int)

    by_subject: dict[int, list[float]] = {}
    for row in recordings:
        by_subject.setdefault(int(row["subject"]), []).append(
            float(row["selected"]["metrics"]["window"]["accuracy"])
        )
    subject_means = np.asarray([np.mean(by_subject[key]) for key in sorted(by_subject)], dtype=float)

    return {
        "num_subject_sessions": int(len(recordings)),
        "num_subjects": int(len(by_subject)),
        "selected_window_accuracy_mean_subject_session": float(selected_accuracy.mean()),
        "selected_window_accuracy_sd_subject_session_population": float(selected_accuracy.std(ddof=0)),
        "selected_window_macro_f1_mean_subject_session": float(selected_macro_f1.mean()),
        "selected_trial_accuracy_mean_subject_session": float(selected_trial_accuracy.mean()),
        "selected_window_accuracy_mean_subject": float(subject_means.mean()),
        "selected_window_accuracy_sd_subject_population": float(subject_means.std(ddof=0)),
        "final_epoch_window_accuracy_mean_subject_session": float(final_accuracy.mean()),
        "within_run_selection_uplift_mean": float((selected_accuracy - final_accuracy).mean()),
        "within_run_selection_uplift_min": float((selected_accuracy - final_accuracy).min()),
        "selected_epoch_mean": float(selected_epochs.mean()),
        "selected_epoch_median": float(np.median(selected_epochs)),
        "selected_epoch_min": int(selected_epochs.min()),
        "selected_epoch_max": int(selected_epochs.max()),
        "worst_selected_subject_session_window_accuracy": float(selected_accuracy.min()),
        "historical_mean_anchor": HISTORICAL_ACCURACY_ANCHOR,
        "historical_subject_session_sd_anchor": HISTORICAL_SD_ANCHOR,
        "historical_mean_absolute_difference": float(
            abs(selected_accuracy.mean() - HISTORICAL_ACCURACY_ANCHOR)
        ),
        "historical_subject_session_sd_absolute_difference": float(
            abs(selected_accuracy.std(ddof=0) - HISTORICAL_SD_ANCHOR)
        ),
        "historical_mean_compatibility_within_0_02": bool(
            abs(selected_accuracy.mean() - HISTORICAL_ACCURACY_ANCHOR) <= 0.02
        ),
        "historical_subject_session_sd_compatibility_within_0_02": bool(
            abs(selected_accuracy.std(ddof=0) - HISTORICAL_SD_ANCHOR) <= 0.02
        ),
        "historical_pair_compatibility_within_0_02": bool(
            abs(selected_accuracy.mean() - HISTORICAL_ACCURACY_ANCHOR) <= 0.02
            and abs(selected_accuracy.std(ddof=0) - HISTORICAL_SD_ANCHOR) <= 0.02
        ),
        "historical_table_anchor_exact_at_4_decimals": bool(
            f"{selected_accuracy.mean():.4f}" == f"{HISTORICAL_ACCURACY_ANCHOR:.4f}"
            and f"{selected_accuracy.std(ddof=0):.4f}" == f"{HISTORICAL_SD_ANCHOR:.4f}"
        ),
    }


def main() -> None:
    args = parse_args()
    require_new_output(args.output)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
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
                train_dataset = TensorDataset(
                    torch.from_numpy(train_features),
                    torch.from_numpy(one_hot(train_labels)),
                )
                train_loader = DataLoader(
                    train_dataset,
                    sampler=RandomSampler(train_dataset),
                    batch_size=args.batch_size,
                    num_workers=0,
                )
                model = DGCNN(num_electrodes=62, in_channels=5, num_classes=3).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
                criterion = nn.CrossEntropyLoss()
                regularizer = NewSparseL2Regularization(0.01).to(device)

                history: list[dict[str, object]] = []
                best_accuracy = 0.0
                best_state: dict[str, torch.Tensor] | None = None
                for epoch_index in range(args.epochs):
                    model.train()
                    epoch_losses: list[float] = []
                    for features, targets in train_loader:
                        features = features.to(device)
                        targets = targets.to(device)
                        optimizer.zero_grad()
                        logits = model(features)
                        loss = criterion(logits, targets) + regularizer(model)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"session {session}, subject {subject}: non-finite loss"
                            )
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(float(loss.detach().cpu()))

                    test_logits = predict_logits(
                        model,
                        test_features,
                        device=device,
                        batch_size=args.eval_batch_size,
                    )
                    epoch_row = {
                        "epoch": epoch_index + 1,
                        "train_loss": float(np.mean(epoch_losses)),
                        "metrics": evaluate_recording(test_labels, test_logits, test_trials),
                    }
                    history.append(epoch_row)

                    epoch_accuracy = float(epoch_row["metrics"]["window"]["accuracy"])
                    if epoch_accuracy > best_accuracy:
                        best_accuracy = epoch_accuracy
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }

                selected_at_selection = earliest_strict_best(history)
                if best_state is None:
                    raise RuntimeError("LibEER's strict-best rule selected no checkpoint")
                model.load_state_dict(best_state)
                selected_logits = predict_logits(
                    model,
                    test_features,
                    device=device,
                    batch_size=args.eval_batch_size,
                )
                selected_retest_metrics = evaluate_recording(
                    test_labels,
                    selected_logits,
                    test_trials,
                )
                if (
                    selected_retest_metrics["window"]["accuracy"]
                    != selected_at_selection["metrics"]["window"]["accuracy"]
                ):
                    raise RuntimeError("selected checkpoint retest did not reproduce selection accuracy")
                selected = {
                    "epoch": selected_at_selection["epoch"],
                    "train_loss": selected_at_selection["train_loss"],
                    "metrics": selected_retest_metrics,
                    "selection_time_metrics": selected_at_selection["metrics"],
                }
                row = {
                    "session": session,
                    "subject": subject,
                    "source_file": str(loaded.source_file),
                    "train_windows": int(len(train_labels)),
                    "test_windows": int(len(test_labels)),
                    "train_class_counts": np.bincount(train_labels, minlength=3).astype(int).tolist(),
                    "test_class_counts": np.bincount(test_labels, minlength=3).astype(int).tolist(),
                    "selected": selected,
                    "final_epoch": history[-1],
                    "selection_uplift_window_accuracy": float(
                        selected["metrics"]["window"]["accuracy"]
                        - history[-1]["metrics"]["window"]["accuracy"]
                    ),
                    "epoch_history": history,
                    "elapsed_seconds": time.time() - unit_started,
                }
                recordings.append(row)
                print(json.dumps({key: value for key, value in row.items() if key != "epoch_history"}), flush=True)
    finally:
        os.chdir(original_cwd)

    payload = {
        "status": "completed",
        "purpose": "archival_test_selected_protocol_compatibility_audit",
        "protocol": "SEED subject-dependent first 9 trials train / last 6 trials test",
        "selection_rule": (
            "test-window labels evaluated after every epoch; earliest strict maximum test accuracy retained"
        ),
        "test_access": (
            "direct repeated test-label checkpoint selection (80 epoch contacts plus one selected-checkpoint "
            "retest per subject-session); intentionally leaky"
        ),
        "claim_boundary": (
            "historical implementation/protocol audit only; not leakage-free, not cross-subject, not method evidence"
        ),
        "implementation_note": (
            "LibEER model/optimizer/loss/split/global-RNG RandomSampler semantics retained; "
            "evaluation uses an equivalent larger deterministic batch and zero workers"
        ),
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
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
