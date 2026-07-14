"""Audit exact fold-1 equivalence against the original SEED-family runs.

The strict five-fold runners deliberately preserve fold 1 / optimization seed
2024 as an implementation-regression cell.  This auditor compares that new
cell with the original subject-split artifact without accepting numerical
tolerances.  Epoch timing and newly added provenance fields are intentionally
excluded; the frozen optimization trajectory, selected model, and test
evaluation must otherwise be identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pcma.eval.metrics import summarize
from pcma.model.seed_da import aggregate_scores_by_group, trial_group_keys


FOLD_ONE_SPLIT = {
    "train": [15, 11, 7, 4, 5, 10, 12, 3, 8],
    "validation": [14, 13, 2],
    "test": [9, 1, 6],
}
FOLD_ONE_SPLIT_ZERO_BASED = {
    name: [subject - 1 for subject in subjects]
    for name, subjects in FOLD_ONE_SPLIT.items()
}
LIBEER_COMMIT_PREFIX = "39dc27e"
REQUIRED_PREDICTION_ARRAYS = {"logits", "labels", "subject", "session", "trial"}
MISSING = object()


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    batch_size: int
    learning_rate: float
    reference_best_epoch: int
    reference_best_validation_value: float
    candidate_purpose: str


DATASET_SPECS = {
    "seed": DatasetSpec(
        num_classes=3,
        batch_size=16,
        learning_rate=0.001,
        reference_best_epoch=7,
        reference_best_validation_value=0.7214388283939682,
        candidate_purpose="strict_all_subject_seed_fold_baseline",
    ),
    "seediv": DatasetSpec(
        num_classes=4,
        batch_size=32,
        learning_rate=0.0015,
        reference_best_epoch=3,
        reference_best_validation_value=0.37400058245287876,
        candidate_purpose="strict_all_subject_cross_validation_fold",
    ),
}


@dataclass
class AuditRecorder:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        check_id: str,
        passed: bool,
        *,
        expected: Any = MISSING,
        actual: Any = MISSING,
        detail: str | None = None,
    ) -> None:
        item: dict[str, Any] = {"id": check_id, "passed": bool(passed)}
        if expected is not MISSING:
            item["expected"] = json_safe(expected)
        if actual is not MISSING:
            item["actual"] = json_safe(actual)
        if detail:
            item["detail"] = detail
        self.checks.append(item)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(item["passed"] for item in self.checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--reference-pt", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--candidate-pt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def json_safe(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = sha256_file(resolved)
    return record


def _same_scalar(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    numeric = (int, float, np.integer, np.floating)
    if isinstance(expected, numeric) and isinstance(actual, numeric):
        try:
            if not math.isfinite(float(expected)) or not math.isfinite(float(actual)):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def compare_exact(recorder: AuditRecorder, prefix: str, expected: Any, actual: Any) -> None:
    """Recursively record exact key, length, and scalar equality checks."""

    expected = json_safe(expected)
    actual = json_safe(actual)
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = sorted(expected)
        actual_keys = sorted(actual)
        recorder.add(
            f"{prefix}.keys",
            expected_keys == actual_keys,
            expected=expected_keys,
            actual=actual_keys,
        )
        for key in sorted(set(expected).intersection(actual)):
            compare_exact(recorder, f"{prefix}.{key}", expected[key], actual[key])
        return
    if isinstance(expected, list) and isinstance(actual, list):
        recorder.add(
            f"{prefix}.length",
            len(expected) == len(actual),
            expected=len(expected),
            actual=len(actual),
        )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            compare_exact(recorder, f"{prefix}[{index}]", expected_item, actual_item)
        return
    recorder.add(
        prefix,
        _same_scalar(expected, actual),
        expected=expected,
        actual=actual,
    )


def _value(mapping: Mapping[str, Any], key: str) -> Any:
    return mapping[key] if key in mapping else MISSING


def _load_json(path: Path, recorder: AuditRecorder, label: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("JSON root is not an object")
    except Exception as error:  # report malformed/missing artifacts instead of aborting
        recorder.add(
            f"load.{label}_json",
            False,
            detail=f"{type(error).__name__}: {error}",
        )
        return None
    recorder.add(f"load.{label}_json", True)
    return payload


def _load_checkpoint(path: Path, recorder: AuditRecorder, label: str) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("checkpoint root is not a mapping")
    except Exception as error:  # report malformed/missing artifacts instead of aborting
        recorder.add(
            f"load.{label}_checkpoint",
            False,
            detail=f"{type(error).__name__}: {error}",
        )
        return None
    recorder.add(f"load.{label}_checkpoint", True)
    return payload


def _resolve_recorded_path(raw: Any, json_path: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = json_path.resolve().parent / path
    return path.resolve()


def _check_recorded_path(
    recorder: AuditRecorder,
    check_id: str,
    raw: Any,
    json_path: Path,
    expected_path: Path,
) -> None:
    recorded = _resolve_recorded_path(raw, json_path)
    recorder.add(
        check_id,
        recorded == expected_path.resolve(),
        expected=str(expected_path.resolve()),
        actual=str(recorded) if recorded is not None else "<missing-or-invalid>",
    )


def _check_commit(recorder: AuditRecorder, check_id: str, value: Any) -> None:
    passed = isinstance(value, str) and value.startswith(LIBEER_COMMIT_PREFIX)
    recorder.add(
        check_id,
        passed,
        expected=f"prefix:{LIBEER_COMMIT_PREFIX}",
        actual=value,
    )


def _check_frozen_json(
    recorder: AuditRecorder,
    dataset: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_json_path: Path,
    reference_pt_path: Path,
    candidate_json_path: Path,
    candidate_pt_path: Path,
) -> None:
    spec = DATASET_SPECS[dataset]
    frozen_candidate = {
        "status": "completed",
        "purpose": spec.candidate_purpose,
        "test_evaluation_count": 1,
        "prediction_units": ["one_second_window", "trial_mean_logit"],
        "session": 1,
        "fold": 1,
        "partition_seed": 2024,
        "optimization_seed": 2024,
        "split_one_based": FOLD_ONE_SPLIT,
        "split_zero_based": FOLD_ONE_SPLIT_ZERO_BASED,
        "epochs": 150,
        "batch_size": spec.batch_size,
        "eval_batch_size": 512,
        "learning_rate": spec.learning_rate,
        "best_epoch": spec.reference_best_epoch,
        "best_validation_value": spec.reference_best_validation_value,
    }
    for key, expected in frozen_candidate.items():
        compare_exact(recorder, f"candidate.config.{key}", expected, _value(candidate, key))

    validation_key = "validation_metric" if dataset == "seed" else "validation_selection"
    compare_exact(
        recorder,
        f"candidate.config.{validation_key}",
        "pooled_window_macro_f1",
        _value(candidate, validation_key),
    )
    if dataset == "seed":
        compare_exact(
            recorder,
            "candidate.config.fold1_original_split_regression_compatible",
            True,
            _value(candidate, "fold1_original_split_regression_compatible"),
        )
    else:
        compare_exact(
            recorder,
            "candidate.config.num_classes",
            spec.num_classes,
            _value(candidate, "num_classes"),
        )

    frozen_reference = {
        "status": "completed",
        "purpose": "protocol_matched_target_free_baseline",
        "session": 1,
        "seed": 2024,
        "split_one_based": FOLD_ONE_SPLIT,
        "split_zero_based": FOLD_ONE_SPLIT_ZERO_BASED,
        "epochs": 150,
        "batch_size": spec.batch_size,
        "eval_batch_size": 512,
        "learning_rate": spec.learning_rate,
        "metric_choose": "macro_f1",
        "best_epoch": spec.reference_best_epoch,
        "best_validation_value": spec.reference_best_validation_value,
    }
    for key, expected in frozen_reference.items():
        compare_exact(recorder, f"reference.config.{key}", expected, _value(reference, key))

    _check_commit(recorder, "reference.config.libeer_commit", reference.get("libeer_commit"))
    _check_commit(recorder, "candidate.config.libeer_commit", candidate.get("libeer_commit"))
    recorder.add(
        "config.libeer_commit_compatible",
        isinstance(reference.get("libeer_commit"), str)
        and isinstance(candidate.get("libeer_commit"), str)
        and reference["libeer_commit"][:7] == candidate["libeer_commit"][:7],
        expected=reference.get("libeer_commit"),
        actual=candidate.get("libeer_commit"),
    )

    for key in (
        "session",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "sample_counts",
        "subject_sample_counts",
    ):
        compare_exact(
            recorder,
            f"reference_candidate.{key}",
            _value(reference, key),
            _value(candidate, key),
        )

    _check_recorded_path(
        recorder,
        "reference.artifact_pair.checkpoint_path",
        reference.get("checkpoint"),
        reference_json_path,
        reference_pt_path,
    )
    _check_recorded_path(
        recorder,
        "candidate.artifact_pair.checkpoint_path",
        candidate.get("checkpoint"),
        candidate_json_path,
        candidate_pt_path,
    )


def _canonical_history_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("history row is not an object")
    if "validation_window" in row:
        validation = row["validation_window"]
        if not isinstance(validation, Mapping):
            raise TypeError("validation_window is not an object")
        accuracy = validation["accuracy"]
        macro_f1 = validation["macro_f1"]
    else:
        accuracy = row["validation_accuracy"]
        macro_f1 = row["validation_macro_f1"]
    return {
        "epoch": row["epoch"],
        "train_loss": row["train_loss"],
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
    }


def _canonical_history(
    payload: Mapping[str, Any], recorder: AuditRecorder, label: str
) -> list[dict[str, Any]] | None:
    history = payload.get("history")
    if not isinstance(history, list):
        recorder.add(f"{label}.history.load", False, detail="history is not a list")
        return None
    recorder.add(
        f"{label}.history.length_frozen",
        len(history) == 150,
        expected=150,
        actual=len(history),
    )
    try:
        canonical = [_canonical_history_row(row) for row in history]
    except Exception as error:
        recorder.add(
            f"{label}.history.load",
            False,
            detail=f"{type(error).__name__}: {error}",
        )
        return None
    recorder.add(f"{label}.history.load", True)
    return canonical


def _check_history(
    recorder: AuditRecorder,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    reference_history = _canonical_history(reference, recorder, "reference")
    candidate_history = _canonical_history(candidate, recorder, "candidate")
    if reference_history is None or candidate_history is None:
        return
    compare_exact(recorder, "history.reference_candidate", reference_history, candidate_history)

    for label, payload, history in (
        ("reference", reference, reference_history),
        ("candidate", candidate, candidate_history),
    ):
        if not history:
            recorder.add(f"{label}.history.selection_rule", False, detail="empty history")
            continue
        best_value = max(row["validation_macro_f1"] for row in history)
        best_epoch = next(
            row["epoch"] for row in history if row["validation_macro_f1"] == best_value
        )
        compare_exact(
            recorder,
            f"{label}.history.selected_epoch",
            best_epoch,
            payload.get("best_epoch", MISSING),
        )
        compare_exact(
            recorder,
            f"{label}.history.selected_value",
            best_value,
            payload.get("best_validation_value", MISSING),
        )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    try:
        return contiguous.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def _check_model_tensors(
    recorder: AuditRecorder,
    reference_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
) -> None:
    reference_model = reference_checkpoint.get("model")
    candidate_model = candidate_checkpoint.get("model")
    if not isinstance(reference_model, Mapping) or not isinstance(candidate_model, Mapping):
        recorder.add(
            "checkpoint.model_mappings",
            False,
            detail="both checkpoints must contain a model state mapping",
        )
        return
    reference_keys = sorted(str(key) for key in reference_model)
    candidate_keys = sorted(str(key) for key in candidate_model)
    recorder.add(
        "checkpoint.model.keys",
        reference_keys == candidate_keys,
        expected=reference_keys,
        actual=candidate_keys,
    )
    for key in sorted(set(reference_model).intersection(candidate_model), key=str):
        reference_tensor = reference_model[key]
        candidate_tensor = candidate_model[key]
        tensors = isinstance(reference_tensor, torch.Tensor) and isinstance(
            candidate_tensor, torch.Tensor
        )
        recorder.add(f"checkpoint.model.{key}.tensor_type", tensors)
        if not tensors:
            continue
        recorder.add(
            f"checkpoint.model.{key}.dtype",
            reference_tensor.dtype == candidate_tensor.dtype,
            expected=str(reference_tensor.dtype),
            actual=str(candidate_tensor.dtype),
        )
        recorder.add(
            f"checkpoint.model.{key}.shape",
            tuple(reference_tensor.shape) == tuple(candidate_tensor.shape),
            expected=list(reference_tensor.shape),
            actual=list(candidate_tensor.shape),
        )
        reference_bytes = _tensor_bytes(reference_tensor)
        candidate_bytes = _tensor_bytes(candidate_tensor)
        recorder.add(
            f"checkpoint.model.{key}.bitwise_value",
            reference_bytes == candidate_bytes,
        )
        # Put the tensor digests in detail without serializing tensor values.
        recorder.checks[-1]["expected_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        recorder.checks[-1]["actual_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()


def _check_checkpoint_metadata(
    recorder: AuditRecorder,
    dataset: str,
    reference_json: Mapping[str, Any],
    candidate_json: Mapping[str, Any],
    reference_checkpoint: Mapping[str, Any],
    candidate_checkpoint: Mapping[str, Any],
) -> None:
    spec = DATASET_SPECS[dataset]
    for label, checkpoint, payload in (
        ("reference", reference_checkpoint, reference_json),
        ("candidate", candidate_checkpoint, candidate_json),
    ):
        compare_exact(
            recorder,
            f"checkpoint.{label}.best_epoch_matches_json",
            payload.get("best_epoch", MISSING),
            checkpoint.get("best_epoch", MISSING),
        )
        compare_exact(
            recorder,
            f"checkpoint.{label}.validation_value_matches_json",
            payload.get("best_validation_value", MISSING),
            checkpoint.get("validation_value", MISSING),
        )
        _check_commit(
            recorder,
            f"checkpoint.{label}.libeer_commit",
            checkpoint.get("libeer_commit"),
        )
        compare_exact(
            recorder,
            f"checkpoint.{label}.libeer_commit_matches_json",
            payload.get("libeer_commit", MISSING),
            checkpoint.get("libeer_commit", MISSING),
        )

    compare_exact(
        recorder,
        "checkpoint.reference.validation_metric",
        "macro_f1",
        reference_checkpoint.get("validation_metric", MISSING),
    )
    compare_exact(
        recorder,
        "checkpoint.reference.split_zero_based",
        FOLD_ONE_SPLIT_ZERO_BASED,
        reference_checkpoint.get("split", MISSING),
    )
    for key, expected in {
        "fold": 1,
        "partition_seed": 2024,
        "optimization_seed": 2024,
        "best_epoch": spec.reference_best_epoch,
        "validation_metric": "macro_f1" if dataset == "seed" else "pooled_window_macro_f1",
        "validation_value": spec.reference_best_validation_value,
        "split_one_based": FOLD_ONE_SPLIT,
    }.items():
        compare_exact(
            recorder,
            f"checkpoint.candidate.{key}",
            expected,
            candidate_checkpoint.get(key, MISSING),
        )


def recompute_test_metrics(arrays: Mapping[str, np.ndarray], num_classes: int) -> dict[str, Any]:
    logits = np.asarray(arrays["logits"])
    labels = np.asarray(arrays["labels"])
    subject = np.asarray(arrays["subject"])
    session = np.asarray(arrays["session"])
    trial = np.asarray(arrays["trial"])
    predictions = logits.argmax(axis=1)
    window = summarize(labels, predictions, subject)

    groups = trial_group_keys(subject, session, trial)
    trial_y, trial_predictions, trial_groups = aggregate_scores_by_group(
        logits,
        np.arange(num_classes),
        groups,
        labels,
    )
    group_to_subject = {
        group: int(str(group).split("_s", 1)[0]) for group in trial_groups
    }
    trial_subject = np.asarray([group_to_subject[group] for group in trial_groups])
    trial_metrics = summarize(trial_y, trial_predictions, trial_subject)

    per_subject: dict[str, Any] = {}
    for subject_id in np.unique(subject):
        window_mask = subject == subject_id
        trial_mask = trial_subject == subject_id
        per_subject[str(int(subject_id))] = {
            "window": summarize(
                labels[window_mask], predictions[window_mask], subject[window_mask]
            ),
            "trial": summarize(
                trial_y[trial_mask],
                trial_predictions[trial_mask],
                trial_subject[trial_mask],
            ),
            "window_count": int(window_mask.sum()),
            "trial_count": int(trial_mask.sum()),
        }
    return {"window": window, "trial": trial_metrics, "per_subject": per_subject}


def _load_and_check_predictions(
    recorder: AuditRecorder,
    dataset: str,
    candidate: Mapping[str, Any],
    candidate_json_path: Path,
) -> tuple[Path | None, dict[str, np.ndarray] | None, dict[str, Any] | None]:
    raw_path = candidate.get("predictions")
    predictions_path = _resolve_recorded_path(raw_path, candidate_json_path)
    recorder.add(
        "candidate.predictions.path_recorded",
        predictions_path is not None,
        actual=raw_path if raw_path is not None else "<missing>",
    )
    if predictions_path is None:
        return None, None, None
    recorder.add(
        "candidate.predictions.exists",
        predictions_path.is_file(),
        expected=True,
        actual=predictions_path.is_file(),
        detail=str(predictions_path),
    )
    if not predictions_path.is_file():
        return predictions_path, None, None
    try:
        with np.load(predictions_path, allow_pickle=False) as archive:
            recorder.add(
                "candidate.predictions.array_keys",
                set(archive.files) == REQUIRED_PREDICTION_ARRAYS,
                expected=sorted(REQUIRED_PREDICTION_ARRAYS),
                actual=sorted(archive.files),
            )
            arrays = {name: np.asarray(archive[name]).copy() for name in REQUIRED_PREDICTION_ARRAYS}
    except Exception as error:
        recorder.add(
            "candidate.predictions.load",
            False,
            detail=f"{type(error).__name__}: {error}",
        )
        return predictions_path, None, None
    recorder.add("candidate.predictions.load", True)

    spec = DATASET_SPECS[dataset]
    lengths = {name: int(len(value)) for name, value in arrays.items()}
    recorder.add(
        "candidate.predictions.row_count_consistent",
        len(set(lengths.values())) == 1,
        actual=lengths,
    )
    recorder.add(
        "candidate.predictions.logits_shape",
        arrays["logits"].ndim == 2 and arrays["logits"].shape[1] == spec.num_classes,
        expected=["N", spec.num_classes],
        actual=list(arrays["logits"].shape),
    )
    recorder.add(
        "candidate.predictions.logits_finite",
        bool(np.isfinite(arrays["logits"]).all()),
    )
    for name in ("labels", "subject", "session", "trial"):
        recorder.add(
            f"candidate.predictions.{name}_one_dimensional",
            arrays[name].ndim == 1,
            expected=1,
            actual=arrays[name].ndim,
        )
    recorder.add(
        "candidate.predictions.labels_frozen_range",
        set(np.unique(arrays["labels"]).tolist()) == set(range(spec.num_classes)),
        expected=list(range(spec.num_classes)),
        actual=sorted(np.unique(arrays["labels"]).tolist()),
    )
    recorder.add(
        "candidate.predictions.test_subjects",
        set(np.unique(arrays["subject"]).tolist()) == set(FOLD_ONE_SPLIT["test"]),
        expected=sorted(FOLD_ONE_SPLIT["test"]),
        actual=sorted(np.unique(arrays["subject"]).tolist()),
    )
    recorder.add(
        "candidate.predictions.session",
        set(np.unique(arrays["session"]).tolist()) == {1},
        expected=[1],
        actual=sorted(np.unique(arrays["session"]).tolist()),
    )
    try:
        recomputed = recompute_test_metrics(arrays, spec.num_classes)
    except Exception as error:
        recorder.add(
            "candidate.predictions.recompute_metrics",
            False,
            detail=f"{type(error).__name__}: {error}",
        )
        return predictions_path, arrays, None
    recorder.add("candidate.predictions.recompute_metrics", True)
    return predictions_path, arrays, recomputed


def _check_test_metrics(
    recorder: AuditRecorder,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    recomputed: Mapping[str, Any] | None,
) -> None:
    reference_test = reference.get("test")
    candidate_test = candidate.get("test")
    if not isinstance(reference_test, Mapping) or not isinstance(candidate_test, Mapping):
        recorder.add(
            "test.metric_objects",
            False,
            detail="both JSON artifacts must contain test metric objects",
        )
        return
    for section in ("window", "trial", "per_subject"):
        compare_exact(
            recorder,
            f"test.reference_candidate.{section}",
            reference_test.get(section, MISSING),
            candidate_test.get(section, MISSING),
        )
        if recomputed is not None:
            compare_exact(
                recorder,
                f"test.candidate_npz_recomputed.{section}",
                candidate_test.get(section, MISSING),
                recomputed.get(section, MISSING),
            )


def audit(
    *,
    dataset: str,
    reference_json_path: Path,
    reference_pt_path: Path,
    candidate_json_path: Path,
    candidate_pt_path: Path,
) -> dict[str, Any]:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {dataset}")
    recorder = AuditRecorder()
    named_paths = {
        "reference_json": reference_json_path,
        "reference_pt": reference_pt_path,
        "candidate_json": candidate_json_path,
        "candidate_pt": candidate_pt_path,
    }
    artifacts: dict[str, Any] = {}
    for label, path in named_paths.items():
        try:
            artifacts[label] = file_record(path)
            recorder.add(f"artifact.{label}.exists", artifacts[label]["exists"])
        except Exception as error:
            artifacts[label] = {"path": str(path.resolve()), "exists": False}
            recorder.add(
                f"artifact.{label}.exists",
                False,
                detail=f"{type(error).__name__}: {error}",
            )

    reference_json = _load_json(reference_json_path, recorder, "reference")
    candidate_json = _load_json(candidate_json_path, recorder, "candidate")
    reference_checkpoint = _load_checkpoint(reference_pt_path, recorder, "reference")
    candidate_checkpoint = _load_checkpoint(candidate_pt_path, recorder, "candidate")

    predictions_path: Path | None = None
    recomputed: dict[str, Any] | None = None
    if reference_json is not None and candidate_json is not None:
        _check_frozen_json(
            recorder,
            dataset,
            reference_json,
            candidate_json,
            reference_json_path,
            reference_pt_path,
            candidate_json_path,
            candidate_pt_path,
        )
        _check_history(recorder, reference_json, candidate_json)
        predictions_path, _, recomputed = _load_and_check_predictions(
            recorder,
            dataset,
            candidate_json,
            candidate_json_path,
        )
        _check_test_metrics(recorder, reference_json, candidate_json, recomputed)

    if reference_checkpoint is not None and candidate_checkpoint is not None:
        _check_model_tensors(recorder, reference_checkpoint, candidate_checkpoint)
        if reference_json is not None and candidate_json is not None:
            _check_checkpoint_metadata(
                recorder,
                dataset,
                reference_json,
                candidate_json,
                reference_checkpoint,
                candidate_checkpoint,
            )

    if predictions_path is not None:
        try:
            artifacts["candidate_predictions"] = file_record(predictions_path)
        except Exception as error:
            artifacts["candidate_predictions"] = {
                "path": str(predictions_path),
                "exists": False,
                "hash_error": f"{type(error).__name__}: {error}",
            }

    passed_count = sum(item["passed"] for item in recorder.checks)
    failed_checks = [item["id"] for item in recorder.checks if not item["passed"]]
    return {
        "status": "completed",
        "verdict": "PASS" if recorder.passed else "FAIL",
        "audit_scope": "fold1_optimization_seed2024_exact_new_old_regression",
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_policy": {
            "numeric_tolerance": 0.0,
            "history_fields": [
                "epoch",
                "train_loss",
                "validation_accuracy",
                "validation_macro_f1",
            ],
            "ignored_history_fields": ["seconds"],
            "checkpoint_model_values": "raw tensor bytes must match",
            "test_sections": ["window", "trial", "per_subject"],
            "candidate_npz_metrics_recomputed": True,
        },
        "artifacts": artifacts,
        "summary": {
            "checks_total": len(recorder.checks),
            "checks_passed": passed_count,
            "checks_failed": len(recorder.checks) - passed_count,
            "failed_check_ids": failed_checks,
        },
        "checks": recorder.checks,
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def _fatal_report(dataset: str, error: Exception) -> dict[str, Any]:
    return {
        "status": "completed_with_audit_error",
        "verdict": "FAIL",
        "audit_scope": "fold1_optimization_seed2024_exact_new_old_regression",
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "checks_total": 1,
            "checks_passed": 0,
            "checks_failed": 1,
            "failed_check_ids": ["audit.fatal_error"],
        },
        "checks": [
            {
                "id": "audit.fatal_error",
                "passed": False,
                "detail": f"{type(error).__name__}: {error}",
            }
        ],
    }


def write_report_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite existing audit report: {args.output}", file=sys.stderr)
        return 2
    try:
        report = audit(
            dataset=args.dataset,
            reference_json_path=args.reference_json,
            reference_pt_path=args.reference_pt,
            candidate_json_path=args.candidate_json,
            candidate_pt_path=args.candidate_pt,
        )
    except Exception as error:  # make a best-effort FAIL artifact for unexpected errors
        report = _fatal_report(args.dataset, error)
    try:
        write_report_exclusive(args.output, report)
    except Exception as error:
        print(f"failed to write audit report: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(json_safe(report["summary"]), indent=2))
    print(f"verdict={report['verdict']} report={args.output.resolve()}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
