"""Reproduce one strict DGCNN cell while recording a per-epoch train curve.

The runner loads only the frozen train and validation participants. It never
loads the test partition. The newly selected state and the original 150-epoch
history must reproduce the frozen cell exactly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_seed_dgcnn_subject_fold as seed_runner  # noqa: E402
from experiments import run_seediv_dgcnn_subject_fold as seediv_runner  # noqa: E402
from pcma.data.libeer_seediv import (  # noqa: E402
    flatten_seediv_subject_windows,
    load_seediv_de_lds_subject,
)
from pcma.data.seed_folds import get_subject_fold  # noqa: E402


PROTOCOL = "strict_fivefold_dgcnn_learning_curve_v1"
PARTITION_SEED = 2024
EPOCHS = 150
SETTINGS = {
    "seed": {"classes": 3, "batch_size": 16, "learning_rate": 0.001},
    "seediv": {"classes": 4, "batch_size": 32, "learning_rate": 0.0015},
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("seed", "seediv"), required=True)
    result.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    result.add_argument("--optimization-seed", type=int, choices=(2024, 2025, 2026), required=True)
    result.add_argument("--libeer-root", type=Path, required=True)
    result.add_argument("--seed-feature-root", type=Path, required=True)
    result.add_argument("--seediv-dataset-root", type=Path, required=True)
    result.add_argument("--seediv-cache-root", type=Path, required=True)
    result.add_argument("--reference-root", type=Path, default=Path("results/strict_fivefold"))
    result.add_argument(
        "--train-fit-result",
        type=Path,
        default=Path("results/strict_fivefold/train_fit_diagnostic/train_fit_diagnostic.json"),
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--eval-batch-size", type=int, default=512)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    # Codex executes child processes under a restricted Windows account while
    # the pinned worktree belongs to the desktop user.  Scope Git's ownership
    # exception to this read-only command instead of changing global config.
    safe_directory = repo.resolve().as_posix()
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={safe_directory}",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_dgcnn_module(libeer_code: Path) -> Any:
    """Load only pinned DGCNN.py, avoiding unrelated optional model imports."""

    source = libeer_code / "models" / "DGCNN.py"
    spec = importlib.util.spec_from_file_location(
        "_strict_learning_curve_pinned_dgcnn",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned DGCNN module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "DGCNN") or not hasattr(module, "NewSparseL2Regularization"):
        raise AttributeError("pinned DGCNN module lacks required classes")
    return module


def reference_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stem = f"{args.dataset}_fold{args.fold}_opt{args.optimization_seed}"
    base = args.reference_root.resolve() / args.dataset / stem
    return base.with_suffix(".json"), base.with_suffix(".pt")


def load_train_fit_cell(path: Path, *, dataset: str, fold: int, seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        cell
        for cell in payload["cells"]
        if cell["dataset"] == dataset
        and int(cell["fold"]) == fold
        and int(cell["optimization_seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError("train-fit diagnostic does not contain exactly one matching cell")
    return matches[0]


def load_partitions(args: argparse.Namespace) -> tuple[dict[str, tuple[int, ...]], Any, Any]:
    split = get_subject_fold(args.fold, partition_seed=PARTITION_SEED)
    if args.dataset == "seed":
        train = seed_runner.load_partition(args.seed_feature_root.resolve(), split["train"]).rows
        validation = seed_runner.load_partition(
            args.seed_feature_root.resolve(), split["validation"]
        ).rows
        return split, train, validation

    def load(subjects: tuple[int, ...]) -> Any:
        rows = []
        for subject in subjects:
            loaded = load_seediv_de_lds_subject(
                args.seediv_dataset_root.resolve(),
                args.libeer_root.resolve(),
                1,
                subject,
                cache_root=args.seediv_cache_root.resolve(),
                expected_commit="39dc27e504e14138767b87ce8bce485380fd4f5a",
            )
            rows.append(flatten_seediv_subject_windows(loaded, subject=subject, session=1))
        return seediv_runner.concatenate_rows(rows)

    return split, load(split["train"]), load(split["validation"])


def history_reference_values(dataset: str, row: dict[str, Any]) -> tuple[float, float, float]:
    if dataset == "seed":
        return (
            float(row["train_loss"]),
            float(row["validation_window"]["accuracy"]),
            float(row["validation_window"]["macro_f1"]),
        )
    return (
        float(row["train_loss"]),
        float(row["validation_accuracy"]),
        float(row["validation_macro_f1"]),
    )


def states_equal(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if set(actual) != set(expected):
        issues.append("state keys differ")
        return False, issues
    for key in sorted(actual):
        left, right = actual[key].cpu(), expected[key].cpu()
        if left.dtype != right.dtype or left.shape != right.shape:
            issues.append(f"{key}: dtype or shape differs")
        elif not torch.equal(left, right):
            issues.append(f"{key}: tensor values differ")
    return not issues, issues


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reference_json_path, reference_checkpoint_path = reference_paths(args)
    reference = json.loads(reference_json_path.read_text(encoding="utf-8"))
    reference_checkpoint = torch.load(
        reference_checkpoint_path, map_location="cpu", weights_only=False
    )
    train_fit_cell = load_train_fit_cell(
        args.train_fit_result.resolve(),
        dataset=args.dataset,
        fold=args.fold,
        seed=args.optimization_seed,
    )
    settings = SETTINGS[args.dataset]
    split, train_rows, validation_rows = load_partitions(args)
    if tuple(split["train"]) != tuple(train_fit_cell["train_subjects"]):
        raise ValueError("learning-curve split and train-fit split differ")

    commit = git_commit(args.libeer_root.resolve())
    if commit != "39dc27e504e14138767b87ce8bce485380fd4f5a":
        raise RuntimeError(f"unexpected LibEER commit {commit}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    libeer_code = args.libeer_root.resolve() / "LibEER"
    original_cwd = Path.cwd()
    sys.path.insert(0, str(libeer_code))
    os.chdir(libeer_code)
    started = time.time()
    try:
        dgcnn_module = load_dgcnn_module(libeer_code)
        DGCNN = dgcnn_module.DGCNN
        NewSparseL2Regularization = dgcnn_module.NewSparseL2Regularization
        from utils.utils import setup_seed

        setup_seed(args.optimization_seed)
        if args.dataset == "seed":
            target_rows = seed_runner.one_hot(train_rows.labels)
            evaluate = seed_runner.evaluation_metrics
            predict = seed_runner.predict_rows
        else:
            target_rows = seediv_runner.one_hot_four_class(train_rows.labels)
            evaluate = seediv_runner.evaluation_metrics
            predict = seediv_runner.predict_rows

        generator = torch.Generator().manual_seed(args.optimization_seed)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(train_rows.features),
                torch.from_numpy(target_rows),
            ),
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        model = DGCNN(
            num_electrodes=62,
            in_channels=5,
            num_classes=int(settings["classes"]),
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=1e-4,
            eps=1e-4,
        )
        criterion = nn.CrossEntropyLoss()
        regularizer = NewSparseL2Regularization(0.01).to(device)
        history: list[dict[str, Any]] = []
        best_value = float("-inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.time()
            model.train()
            losses: list[float] = []
            correct = torch.zeros((), dtype=torch.long, device=device)
            count = 0
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits = model(features)
                true_labels = targets.argmax(dim=1)
                correct += (logits.argmax(dim=1) == true_labels).sum()
                count += int(len(features))
                loss = criterion(logits, targets) + regularizer(model)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"epoch {epoch}: non-finite loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            validation = evaluate(
                predict(
                    model,
                    validation_rows,
                    device=device,
                    batch_size=args.eval_batch_size,
                )
            )
            checkpoint_value = float(validation["window"]["macro_f1"])
            updated = checkpoint_value > best_value
            if updated:
                best_value = checkpoint_value
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_online_window_accuracy": float(correct.item() / count),
                "validation_window_accuracy": float(validation["window"]["accuracy"]),
                "validation_window_macro_f1": float(validation["window"]["macro_f1"]),
                "validation_trial_accuracy": float(validation["trial"]["accuracy"]),
                "checkpoint_updated": bool(updated),
                "seconds": time.time() - epoch_started,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(str(libeer_code))
        except ValueError:
            pass

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    reference_history = reference["history"]
    history_issues: list[str] = []
    if len(reference_history) != EPOCHS:
        history_issues.append("reference history does not contain 150 epochs")
    else:
        for new, old in zip(history, reference_history):
            old_loss, old_acc, old_f1 = history_reference_values(args.dataset, old)
            comparisons = {
                "train_loss": (new["train_loss"], old_loss),
                "validation_window_accuracy": (new["validation_window_accuracy"], old_acc),
                "validation_window_macro_f1": (new["validation_window_macro_f1"], old_f1),
            }
            for name, (actual, expected) in comparisons.items():
                if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                    history_issues.append(
                        f"epoch {new['epoch']} {name}: {actual} != {expected}"
                    )
                    break
            if history_issues:
                break
    state_ok, state_issues = states_equal(best_state, reference_checkpoint["model"])
    metadata_ok = bool(
        best_epoch == int(reference["best_epoch"])
        and np.isclose(
            best_value,
            float(reference["best_validation_value"]),
            rtol=0.0,
            atol=1e-12,
        )
    )
    reproduced = bool(not history_issues and state_ok and metadata_ok)
    result = {
        "protocol": PROTOCOL,
        "status": "PASS_REPRODUCED" if reproduced else "FAIL_REPRODUCTION",
        "dataset": args.dataset,
        "fold": args.fold,
        "optimization_seed": args.optimization_seed,
        "partition_seed": PARTITION_SEED,
        "split_one_based": split,
        "epochs": EPOCHS,
        "batch_size": int(settings["batch_size"]),
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": float(settings["learning_rate"]),
        "validation_selection": "pooled_window_macro_f1",
        "test_rows_loaded": 0,
        "test_evaluation_count": 0,
        "training_curve_definition": (
            "accuracy accumulated from shuffled minibatch predictions within each epoch"
        ),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_value": best_value,
        "selected_checkpoint": {
            "train_window_accuracy": float(train_fit_cell["train"]["window"]["accuracy"]),
            "train_trial_accuracy": float(train_fit_cell["train"]["trial"]["accuracy"]),
            "frozen_test_window_accuracy": float(train_fit_cell["test"]["window"]["accuracy"]),
            "frozen_test_trial_accuracy": float(train_fit_cell["test"]["trial"]["accuracy"]),
        },
        "exact_reproduction": {
            "history_match": not history_issues,
            "history_issues": history_issues,
            "best_checkpoint_metadata_match": metadata_ok,
            "state_dict_match": state_ok,
            "state_dict_issues": state_issues,
        },
        "reference": {
            "json": str(reference_json_path.resolve()),
            "json_sha256": sha256(reference_json_path),
            "checkpoint": str(reference_checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(reference_checkpoint_path),
            "train_fit_result": str(args.train_fit_result.resolve()),
            "train_fit_result_sha256": sha256(args.train_fit_result.resolve()),
        },
        "environment": {
            "libeer_commit": commit,
            "torch": torch.__version__,
            "device": str(device),
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output)}), flush=True)
    return result, 0 if reproduced else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _, return_code = run(args)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
