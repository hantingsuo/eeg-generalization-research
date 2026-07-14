"""Run a leakage-free, one-recording DGCNN compatibility smoke test.

This is a plumbing check, not a benchmark.  It uses the DGCNN model, loss, and
training routine from a pinned LibEER checkout, but avoids two upstream issues:

1. the official runner loads the complete SEED dataset before applying ``pr``;
2. its front/back preset aliases the test set as validation and selects epochs
   on test labels.

The smoke test instead loads one subject/session, reserves one of the first
nine trials as validation, and executes a small number of optimizer steps.
Trials 10--15 remain test-only.  It intentionally does not call LibEER's
checkpoint trainer because a zero validation score can leave no checkpoint and
crash an otherwise successful plumbing test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score

from pcma.data.libeer_seed import load_seed_de_lds_subject, make_clean_front_back_split, one_hot


DEFAULT_EXPECTED_COMMIT = "39dc27e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libeer-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default=DEFAULT_EXPECTED_COMMIT)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--validation-trial", type=int, default=9)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TensorDataset, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for features, one_hot_targets in loader:
        logits = model(features.to(device))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(one_hot_targets.argmax(dim=1).cpu().tolist())
    return {
        "acc": float(accuracy_score(targets, predictions)),
        "macro-f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    libeer_root = args.libeer_root.resolve()
    libeer_code = libeer_root / "LibEER"
    if not (libeer_code / "models" / "DGCNN.py").is_file():
        raise FileNotFoundError("LibEER DGCNN.py not found")
    commit = git_commit(libeer_root)
    if commit != args.expected_commit:
        raise RuntimeError(f"expected LibEER {args.expected_commit}, found {commit}")

    subject = load_seed_de_lds_subject(
        args.feature_root,
        session=args.session,
        subject=args.subject,
    )
    split = make_clean_front_back_split(subject, validation_trial=args.validation_trial)

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

        datasets: dict[str, TensorDataset] = {}
        for part, (features, labels) in split.items():
            datasets[part] = TensorDataset(
                torch.from_numpy(features),
                torch.from_numpy(one_hot(labels)),
            )

        model = DGCNN(num_electrodes=62, in_channels=5, num_classes=3)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
        criterion = nn.CrossEntropyLoss()
        regularizer = NewSparseL2Regularization(0.01).to(device)
        model = model.to(device)
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        iterator = iter(loader)
        losses: list[float] = []
        model.train()
        for _ in range(args.steps):
            try:
                features, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                features, targets = next(iterator)
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets) + regularizer(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {loss.item()}")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = {
            "validation": evaluate(model, datasets["validation"], device, args.batch_size),
            "test": evaluate(model, datasets["test"], device, args.batch_size),
        }
    finally:
        os.chdir(original_cwd)

    payload = {
        "status": "completed",
        "purpose": "plumbing_smoke_not_benchmark",
        "protocol": "subject-dependent clean 8-trial train / 1-trial validation / 6-trial test",
        "target_access": "test labels used once for diagnostic evaluation and never for selection",
        "libeer_commit": commit,
        "session": args.session,
        "subject": args.subject,
        "validation_trial": args.validation_trial,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "source_file": str(subject.source_file),
        "source_sha256": sha256(subject.source_file),
        "label_file": str(subject.label_file),
        "label_sha256": sha256(subject.label_file),
        "sample_counts": {name: int(len(values[1])) for name, values in split.items()},
        "class_counts": {
            name: np.bincount(values[1], minlength=3).astype(int).tolist()
            for name, values in split.items()
        },
        "training_losses": losses,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
