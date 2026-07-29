"""Pinned LibEER DGCNN/GCBNet runner for one frozen Track-A recording."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from pcma.data.libeer_seed import load_seed_de_lds_subject
from pcma.data.libeer_seediv import load_seediv_de_lds_subject
from pcma.model.cf_tre_baselines import (
    load_track_a_unit,
    normalize_probabilities,
    probability_metrics,
    trial_mean_probability_metrics,
)


EXPECTED_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"
DEFAULT_SEED_ROOT = Path("data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s")
DEFAULT_SEEDIV_ROOT = Path("data/SEED/SEED_IV")
DEFAULT_LIBEER_ROOT = Path(
    os.environ.get("LIBEER_CODE_ROOT", "third_party/libeer/LibEER")
)
DEFAULT_SEEDIV_CACHE = Path("results/libeer_gate_b/cache_seediv_1s_de_lds_39dc27e")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", choices=("seed", "seediv"), required=True)
    result.add_argument("--model", choices=("dgcnn", "gcbnet"), required=True)
    result.add_argument("--session", type=int, required=True)
    result.add_argument("--subject", type=int, required=True)
    result.add_argument("--split-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--predictions", type=Path)
    result.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    result.add_argument("--seediv-root", type=Path, default=DEFAULT_SEEDIV_ROOT)
    result.add_argument("--libeer-root", type=Path, default=DEFAULT_LIBEER_ROOT)
    result.add_argument("--seediv-cache", type=Path, default=DEFAULT_SEEDIV_CACHE)
    result.add_argument("--optimization-seed", type=int, default=2024)
    result.add_argument("--protocol", default="cf_tre_seed_family_v1")
    result.add_argument("--stage", default="G0-B")
    result.add_argument("--epochs", type=int, default=150)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--eval-batch-size", type=int, default=512)
    result.add_argument("--lr", type=float)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--upstream-sampler", action="store_true")
    result.add_argument("--validation-only", action="store_true")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(libeer_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(libeer_root.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _load_model_module(libeer_root: Path, model_name: str) -> Any:
    filename = "DGCNN.py" if model_name == "dgcnn" else "GCBNet.py"
    module_path = libeer_root / "models" / filename
    spec = importlib.util.spec_from_file_location(f"_cf_tre_g0b_{model_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned model file {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_recording(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.dataset == "seed":
        loaded = load_seed_de_lds_subject(
            args.seed_root, session=args.session, subject=args.subject
        )
        return loaded, {
            "source_file": str(loaded.source_file.resolve()),
            "source_sha256": _sha256(loaded.source_file),
            "label_file": str(loaded.label_file.resolve()),
            "label_sha256": _sha256(loaded.label_file),
            "representation": "official_de_LDS_1s",
        }
    loaded = load_seediv_de_lds_subject(
        args.seediv_root,
        args.libeer_root,
        args.session,
        args.subject,
        cache_root=args.seediv_cache,
        expected_commit=EXPECTED_COMMIT,
    )
    provenance = dict(loaded.provenance)
    provenance.update(
        {
            "source_file": str(loaded.source_file.resolve()),
            "cache_file": str(loaded.cache_file.resolve()) if loaded.cache_file else None,
            "loaded_from_cache": bool(loaded.loaded_from_cache),
            "representation": "pinned_libeer_raw_to_de_lds_1s",
        }
    )
    return loaded, provenance


def _partition(recording: Any, trial_numbers: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    trials: list[np.ndarray] = []
    for value in trial_numbers:
        trial_number = int(value)
        if not 1 <= trial_number <= len(recording.trials):
            raise ValueError(f"invalid trial number {trial_number}")
        trial = recording.trials[trial_number - 1]
        features.append(trial)
        labels.append(
            np.full(len(trial), recording.trial_labels[trial_number - 1], dtype=np.int64)
        )
        trials.append(np.full(len(trial), trial_number, dtype=np.int16))
    return (
        np.ascontiguousarray(np.concatenate(features), dtype=np.float32),
        np.concatenate(labels),
        np.concatenate(trials),
    )


@torch.no_grad()
def _predict(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for (batch,) in loader:
        chunks.append(model(batch.to(device)).detach().cpu().numpy())
    logits = np.concatenate(chunks)
    probabilities = normalize_probabilities(torch.softmax(torch.from_numpy(logits), dim=1).numpy())
    return logits, probabilities


def _regularization(module: Any, model_name: str, model: nn.Module) -> nn.Module:
    if model_name == "dgcnn":
        return module.NewSparseL2Regularization(0.01)(model)
    return module.SparseL2Regularization(0.001)(model.original_fc.weight)


def _model_class(module: Any, model_name: str) -> type[nn.Module]:
    return module.DGCNN if model_name == "dgcnn" else module.GCBNet


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not 1 <= args.session <= 3 or not 1 <= args.subject <= 15:
        raise ValueError("session must be 1..3 and subject must be 1..15")
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    learning_rate = args.lr if args.lr is not None else (0.001 if args.dataset == "seed" else 0.0015)
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    output = args.output.resolve()
    checkpoint = (args.checkpoint or args.output.with_suffix(".pt")).resolve()
    predictions = (args.predictions or args.output.with_suffix(".npz")).resolve()
    paths = (output, checkpoint, predictions)
    if len(set(paths)) != 3:
        raise ValueError("JSON, checkpoint, and prediction paths must differ")
    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite artifacts: {collisions}")

    started = time.time()
    split_path = args.split_manifest.resolve()
    unit = load_track_a_unit(
        split_path,
        dataset=args.dataset,
        session=args.session,
        subject=args.subject,
        expected_protocol=args.protocol,
    )
    recording, provenance = _load_recording(args)
    train_x, train_y, train_trials = _partition(recording, unit["train_trials"])
    validation_x, validation_y, validation_trials = _partition(
        recording, unit["validation_trials"]
    )

    libeer_root = args.libeer_root.resolve()
    commit = _git_commit(libeer_root)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"LibEER commit mismatch: expected {EXPECTED_COMMIT}, found {commit}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    old_cwd = Path.cwd()
    sys.path.insert(0, str(libeer_root))
    try:
        os.chdir(libeer_root)
        module = _load_model_module(libeer_root, args.model)
        from utils.utils import setup_seed

        setup_seed(args.optimization_seed)
        num_classes = 3 if args.dataset == "seed" else 4
        model = _model_class(module, args.model)(
            num_electrodes=62, in_channels=5, num_classes=num_classes
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        loader_kwargs: dict[str, Any] = {}
        if not args.upstream_sampler:
            loader_kwargs["generator"] = torch.Generator().manual_seed(
                args.optimization_seed
            )
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            **loader_kwargs,
        )

        best_value = float("-inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        best_validation: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            model.train()
            losses: list[float] = []
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x.to(device))
                loss = criterion(logits, batch_y.to(device)) + _regularization(
                    module, args.model, model
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"epoch {epoch}: non-finite loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_logits, validation_probabilities = _predict(
                model,
                validation_x,
                device=device,
                batch_size=args.eval_batch_size,
            )
            validation_metrics = probability_metrics(validation_y, validation_probabilities)
            chosen = float(validation_metrics["macro_f1"])
            updated = chosen > best_value
            if updated:
                best_value = chosen
                best_epoch = epoch
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
                best_validation = {
                    "window": validation_metrics,
                    "trial_mean_probability": trial_mean_probability_metrics(
                        validation_y, validation_probabilities, validation_trials
                    ),
                }
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "checkpoint_updated": updated,
                "seconds": time.time() - epoch_started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)

        if best_state is None or best_validation is None:
            raise RuntimeError("no validation checkpoint selected")
        model.load_state_dict(best_state)
        validation_logits, validation_probabilities = _predict(
            model, validation_x, device=device, batch_size=args.eval_batch_size
        )

        prediction_payload: dict[str, np.ndarray] = {
            "validation_logits": validation_logits,
            "validation_probabilities": validation_probabilities,
            "validation_labels": validation_y,
            "validation_trials": validation_trials,
        }
        test_contacted = not args.validation_only
        test_metrics: dict[str, Any] | None = None
        if test_contacted:
            test_x, test_y, test_trials = _partition(recording, unit["test_trials"])
            test_logits, test_probabilities = _predict(
                model, test_x, device=device, batch_size=args.eval_batch_size
            )
            prediction_payload.update(
                {
                    "test_logits": test_logits,
                    "test_probabilities": test_probabilities,
                    "test_labels": test_y,
                    "test_trials": test_trials,
                }
            )
            test_metrics = {
                "window": probability_metrics(test_y, test_probabilities),
                "trial_mean_probability": trial_mean_probability_metrics(
                    test_y, test_probabilities, test_trials
                ),
            }
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == str(libeer_root):
            sys.path.pop(0)

    checkpoint_payload = {
        "model": best_state,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "model_name": args.model,
        "session": args.session,
        "subject": args.subject,
        "optimization_seed": args.optimization_seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_value,
        "libeer_commit": commit,
    }
    result = {
        "protocol": args.protocol,
        "stage": args.stage,
        "runner": "pinned_libeer_deep_subject_session_v1",
        "dataset": args.dataset,
        "model": args.model,
        "session": args.session,
        "subject": args.subject,
        "unit_id": unit["unit_id"],
        "optimization_seed": args.optimization_seed,
        "sampler_semantics": "upstream_global_rng" if args.upstream_sampler else "dedicated_generator",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "validation_only": bool(args.validation_only),
        "test_contacted": test_contacted,
        "selection": "higher outer-validation window macro-F1; strict greater-than; earliest epoch retained on ties",
        "fit_boundary": "outer train only; validation not refit; test partition materialized for the model only after checkpoint selection",
        "normalization": "none, matching the pinned LibEER DGCNN/GCBNet path",
        "split_manifest": str(split_path),
        "split_manifest_sha256": _sha256(split_path),
        "trial_split": {
            "train": unit["train_trials"],
            "validation": unit["validation_trials"],
            "test": unit["test_trials"],
        },
        "row_counts": {
            "train": int(len(train_y)),
            "validation": int(len(validation_y)),
            "test": int(len(prediction_payload.get("test_labels", []))),
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_value,
        "best_validation": best_validation,
        "test": test_metrics,
        "history": history,
        "provenance": provenance,
        "libeer_root": str(libeer_root),
        "libeer_commit": commit,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "predictions": str(predictions),
    }

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("xb") as handle:
        torch.save(checkpoint_payload, handle)
    with predictions.open("xb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
