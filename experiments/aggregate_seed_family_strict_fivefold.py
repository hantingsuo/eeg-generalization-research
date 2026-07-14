"""Aggregate the frozen strict five-fold SEED-family DGCNN experiment.

This module deliberately refuses partial grids.  Every input cell is audited
against the frozen 2024 subject partition and its companion prediction NPZ is
used to reconstruct the reported test metrics before any cross-fold summary is
computed.  Optimization seeds are paired repeated fits of the same subjects;
they are never treated as additional independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from pcma.data.seed_folds import DEFAULT_PARTITION_SEED, get_subject_fold
from pcma.eval.metrics import summarize
from pcma.model.seed_da import aggregate_scores_by_group, trial_group_keys


OPTIMIZATION_SEEDS = (2024, 2025, 2026)
FOLDS = (1, 2, 3, 4, 5)
SUBJECTS = tuple(range(1, 16))
SESSION = 1
EPOCHS = 150
EVAL_BATCH_SIZE = 512
VALIDATION_SELECTION = "pooled_window_macro_f1"
LIBEER_COMMIT_PREFIX = "39dc27e"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    num_classes: int
    trials_per_subject: int
    trials_per_class: int
    batch_size: int
    learning_rate: float
    purpose: str
    target_access_fragments: tuple[str, ...]


DATASET_SPECS = {
    "seed": DatasetSpec(
        key="seed",
        display_name="SEED",
        num_classes=3,
        trials_per_subject=15,
        trials_per_class=5,
        batch_size=16,
        learning_rate=0.001,
        purpose="strict_all_subject_seed_fold_baseline",
        target_access_fragments=(
            "not loaded during fitting",
            "validation-only checkpoint selection",
            "test exactly once",
        ),
    ),
    "seediv": DatasetSpec(
        key="seediv",
        display_name="SEED-IV",
        num_classes=4,
        trials_per_subject=24,
        trials_per_class=6,
        batch_size=32,
        learning_rate=0.0015,
        purpose="strict_all_subject_cross_validation_fold",
        target_access_fragments=(
            "uses no labels or cross-subject statistics",
            "fitting uses train subjects",
            "checkpoint selection uses only pooled validation-window labels",
            "single final evaluation",
        ),
    ),
}


SEED_TARGET_ACCESS = (
    "test-subject data are not loaded during fitting or validation-only checkpoint "
    "selection; the fixed checkpoint is evaluated on test exactly once"
)
SEEDIV_TARGET_ACCESS = (
    "fixed per-trial raw-to-DE+LDS extraction uses no labels or cross-subject "
    "statistics; fitting uses train subjects, checkpoint selection uses only "
    "pooled validation-window labels, and test labels are used only in the single "
    "final evaluation"
)


@dataclass(frozen=True)
class PredictionRows:
    logits: np.ndarray
    labels: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


@dataclass(frozen=True)
class ValidatedCell:
    json_path: Path
    prediction_path: Path
    fold: int
    optimization_seed: int
    split: dict[str, tuple[int, ...]]
    rows: PredictionRows
    metrics: dict[str, object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and aggregate all 15 strict five-fold SEED-family cells."
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SPECS))
    parser.add_argument("--inputs", required=True, nargs=15, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_713)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _exact_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer_array(value: np.ndarray, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"prediction NPZ {field} must be one-dimensional")
    if array.dtype.kind not in "iu":
        raise ValueError(f"prediction NPZ {field} must use an integer dtype")
    return array.astype(np.int64, copy=False)


def _canonical_split(value: object, field: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or set(value) != {"train", "validation", "test"}:
        raise ValueError(f"{field} must contain exactly train, validation, and test")
    split: dict[str, tuple[int, ...]] = {}
    for name in ("train", "validation", "test"):
        raw = value[name]
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"{field}.{name} must be a list")
        split[name] = tuple(_exact_int(item, f"{field}.{name}") for item in raw)
    return split


def _infer_payload_dataset(payload: Mapping[str, object]) -> str:
    explicit = payload.get("dataset")
    if explicit is not None:
        key = str(explicit).strip().lower().replace("-", "")
        if key in DATASET_SPECS:
            return key
        raise ValueError(f"unsupported JSON dataset field: {explicit!r}")

    protocol = str(payload.get("protocol", "")).upper()
    preprocessing = payload.get("preprocessing")
    preprocessing_name = ""
    if isinstance(preprocessing, Mapping):
        preprocessing_name = str(preprocessing.get("dataset", "")).lower()
    if "SEED-IV" in protocol or preprocessing_name.startswith("seediv"):
        return "seediv"
    if "SEED" in protocol:
        return "seed"
    raise ValueError("cannot infer dataset from JSON protocol/preprocessing fields")


def _resolve_prediction_path(payload: Mapping[str, object], json_path: Path) -> Path:
    raw = payload.get("predictions")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{json_path}: missing non-empty predictions path")
    path = Path(raw)
    if not path.is_absolute():
        path = json_path.parent / path
    path = path.resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError(f"{json_path}: companion predictions must be an NPZ file")
    if not path.is_file():
        raise FileNotFoundError(f"{json_path}: companion prediction NPZ not found: {path}")
    return path


def _load_prediction_rows(path: Path, spec: DatasetSpec) -> PredictionRows:
    required = {"logits", "labels", "subject", "session", "trial"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: prediction NPZ missing keys {sorted(missing)}")
        logits = np.asarray(archive["logits"])
        labels = _integer_array(archive["labels"], "labels")
        subject = _integer_array(archive["subject"], "subject")
        session = _integer_array(archive["session"], "session")
        trial = _integer_array(archive["trial"], "trial")

    if logits.ndim != 2 or logits.shape[1] != spec.num_classes:
        raise ValueError(
            f"{path}: logits must have shape (n, {spec.num_classes}), found {logits.shape}"
        )
    count = len(labels)
    if count == 0 or any(len(array) != count for array in (logits, subject, session, trial)):
        raise ValueError(f"{path}: prediction arrays must have one common positive length")
    if logits.dtype.kind not in "fiu" or not np.all(np.isfinite(logits)):
        raise ValueError(f"{path}: logits must be finite numeric values")
    if labels.min() < 0 or labels.max() >= spec.num_classes:
        raise ValueError(f"{path}: labels fall outside 0..{spec.num_classes - 1}")
    if set(np.unique(labels).tolist()) != set(range(spec.num_classes)):
        raise ValueError(f"{path}: all {spec.num_classes} classes must occur")
    if set(np.unique(session).tolist()) != {SESSION}:
        raise ValueError(f"{path}: predictions must contain session 1 only")
    return PredictionRows(
        logits=logits.astype(np.float64, copy=False),
        labels=labels,
        subject=subject,
        session=session,
        trial=trial,
    )


def recompute_test_metrics(rows: PredictionRows, num_classes: int) -> dict[str, object]:
    """Reconstruct the runners' window and trial-mean-logit summaries."""

    window_prediction = rows.logits.argmax(axis=1)
    window = summarize(rows.labels, window_prediction, rows.subject)

    groups = trial_group_keys(rows.subject, rows.session, rows.trial)
    trial_y, trial_prediction, trial_groups = aggregate_scores_by_group(
        rows.logits,
        np.arange(num_classes),
        groups,
        rows.labels,
    )
    group_to_subject: dict[str, int] = {}
    for group, subject in zip(groups.tolist(), rows.subject.tolist()):
        group_to_subject.setdefault(str(group), int(subject))
    trial_subject = np.asarray([group_to_subject[str(group)] for group in trial_groups])
    trial = summarize(trial_y, trial_prediction, trial_subject)

    per_subject: dict[str, object] = {}
    for subject in sorted(np.unique(rows.subject).tolist()):
        window_mask = rows.subject == subject
        trial_mask = trial_subject == subject
        per_subject[str(int(subject))] = {
            "window": summarize(
                rows.labels[window_mask],
                window_prediction[window_mask],
                rows.subject[window_mask],
            ),
            "trial": summarize(
                trial_y[trial_mask],
                trial_prediction[trial_mask],
                trial_subject[trial_mask],
            ),
            "window_count": int(window_mask.sum()),
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


def _require_numeric_match(expected: object, actual: object, field: str) -> None:
    left = _finite_float(expected, field)
    right = _finite_float(actual, field)
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} disagrees with prediction NPZ: JSON={left}, NPZ={right}")


def _require_summary_match(
    expected: object,
    actual: Mapping[str, object],
    field: str,
    num_classes: int,
) -> None:
    if not isinstance(expected, Mapping):
        raise ValueError(f"{field} must be an object")
    scalar_fields = (
        "accuracy",
        "macro_f1",
        "worst_subject_acc",
        "mean_subject_acc",
        "balanced_accuracy",
    )
    for name in scalar_fields:
        if name not in expected:
            raise ValueError(f"{field}.{name} is required")
        _require_numeric_match(expected[name], actual[name], f"{field}.{name}")

    expected_recall = expected.get("recall_per_class")
    if not isinstance(expected_recall, Mapping):
        raise ValueError(f"{field}.recall_per_class is required")
    normalized_recall = {str(key): value for key, value in expected_recall.items()}
    if set(normalized_recall) != {str(label) for label in range(num_classes)}:
        raise ValueError(f"{field}.recall_per_class must cover every class")
    actual_recall = {str(key): value for key, value in actual["recall_per_class"].items()}
    for label in range(num_classes):
        _require_numeric_match(
            normalized_recall[str(label)],
            actual_recall[str(label)],
            f"{field}.recall_per_class.{label}",
        )

    expected_confusion = np.asarray(expected.get("confusion"))
    actual_confusion = np.asarray(actual["confusion"])
    if expected_confusion.shape != (num_classes, num_classes):
        raise ValueError(f"{field}.confusion must be {num_classes}x{num_classes}")
    if not np.array_equal(expected_confusion, actual_confusion):
        raise ValueError(f"{field}.confusion disagrees with prediction NPZ")


def _require_test_metrics_match(
    expected: object,
    actual: Mapping[str, object],
    expected_subjects: tuple[int, ...],
    spec: DatasetSpec,
) -> None:
    if not isinstance(expected, Mapping):
        raise ValueError("JSON test field must be an object")
    for unit in ("window", "trial"):
        if unit not in expected:
            raise ValueError(f"JSON test.{unit} is required")
        _require_summary_match(
            expected[unit], actual[unit], f"test.{unit}", spec.num_classes
        )

    per_subject = expected.get("per_subject")
    if not isinstance(per_subject, Mapping):
        raise ValueError("JSON test.per_subject is required")
    if set(map(str, per_subject)) != {str(subject) for subject in expected_subjects}:
        raise ValueError("JSON test.per_subject does not exactly cover fold test subjects")
    for subject in expected_subjects:
        key = str(subject)
        expected_record = per_subject[key]
        actual_record = actual["per_subject"][key]
        if not isinstance(expected_record, Mapping):
            raise ValueError(f"test.per_subject.{key} must be an object")
        for unit in ("window", "trial"):
            _require_summary_match(
                expected_record.get(unit),
                actual_record[unit],
                f"test.per_subject.{key}.{unit}",
                spec.num_classes,
            )
        for count_name in ("window_count", "trial_count"):
            if _exact_int(
                expected_record.get(count_name),
                f"test.per_subject.{key}.{count_name}",
            ) != int(actual_record[count_name]):
                raise ValueError(
                    f"test.per_subject.{key}.{count_name} disagrees with prediction NPZ"
                )

    if "worst" in expected:
        worst = expected["worst"]
        if not isinstance(worst, Mapping):
            raise ValueError("test.worst must be an object")
        for name in ("window_subject_accuracy", "trial_subject_accuracy"):
            _require_numeric_match(worst.get(name), actual["worst"][name], f"test.worst.{name}")


def _validate_trial_content(
    rows: PredictionRows,
    expected_subjects: tuple[int, ...],
    spec: DatasetSpec,
    path: Path,
) -> None:
    observed_subjects = tuple(sorted(np.unique(rows.subject).tolist()))
    if set(observed_subjects) != set(expected_subjects):
        raise ValueError(
            f"{path}: prediction subjects {observed_subjects} do not match fold test "
            f"subjects {expected_subjects}"
        )
    groups = trial_group_keys(rows.subject, rows.session, rows.trial)
    for subject in expected_subjects:
        mask = rows.subject == subject
        subject_groups = groups[mask]
        unique_groups = list(dict.fromkeys(subject_groups.tolist()))
        if len(unique_groups) != spec.trials_per_subject:
            raise ValueError(
                f"{path}: subject {subject} has {len(unique_groups)} trials, expected "
                f"{spec.trials_per_subject}"
            )
        trial_labels: list[int] = []
        for group in unique_groups:
            labels = np.unique(rows.labels[groups == group])
            if len(labels) != 1:
                raise ValueError(f"{path}: trial group {group} has inconsistent labels")
            trial_labels.append(int(labels[0]))
        counts = np.bincount(trial_labels, minlength=spec.num_classes)
        expected_counts = np.full(spec.num_classes, spec.trials_per_class)
        if not np.array_equal(counts, expected_counts):
            raise ValueError(
                f"{path}: subject {subject} trial labels are not class-balanced: "
                f"{counts.tolist()}"
            )


def validate_cell(json_path: Path, dataset: str) -> ValidatedCell:
    spec = DATASET_SPECS[dataset]
    json_path = Path(json_path).resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"input JSON not found: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{json_path}: invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{json_path}: top-level JSON must be an object")

    if _infer_payload_dataset(payload) != dataset:
        raise ValueError(f"{json_path}: JSON dataset does not match --dataset {dataset}")
    if payload.get("status") != "completed":
        raise ValueError(f"{json_path}: status must be completed")
    if payload.get("purpose") != spec.purpose:
        raise ValueError(f"{json_path}: unexpected purpose field")
    if _exact_int(payload.get("session"), "session") != SESSION:
        raise ValueError(f"{json_path}: session must be 1")
    fold = _exact_int(payload.get("fold"), "fold")
    if fold not in FOLDS:
        raise ValueError(f"{json_path}: fold must be in 1..5")
    partition_seed = _exact_int(payload.get("partition_seed"), "partition_seed")
    if partition_seed != DEFAULT_PARTITION_SEED:
        raise ValueError(f"{json_path}: partition_seed must be {DEFAULT_PARTITION_SEED}")
    optimization_seed = _exact_int(payload.get("optimization_seed"), "optimization_seed")
    if optimization_seed not in OPTIMIZATION_SEEDS:
        raise ValueError(f"{json_path}: unexpected optimization seed {optimization_seed}")

    expected_split = get_subject_fold(fold, partition_seed=DEFAULT_PARTITION_SEED)
    split = _canonical_split(payload.get("split_one_based"), "split_one_based")
    if split != expected_split:
        raise ValueError(f"{json_path}: split_one_based does not match frozen fold {fold}")
    if "split_zero_based" in payload:
        expected_zero = {
            name: tuple(subject - 1 for subject in members)
            for name, members in expected_split.items()
        }
        observed_zero = _canonical_split(payload["split_zero_based"], "split_zero_based")
        if observed_zero != expected_zero:
            raise ValueError(f"{json_path}: split_zero_based disagrees with frozen fold")

    if _exact_int(payload.get("test_evaluation_count"), "test_evaluation_count") != 1:
        raise ValueError(f"{json_path}: test_evaluation_count must equal 1")
    if _exact_int(payload.get("epochs"), "epochs") != EPOCHS:
        raise ValueError(f"{json_path}: epochs must equal {EPOCHS}")
    if _exact_int(payload.get("batch_size"), "batch_size") != spec.batch_size:
        raise ValueError(f"{json_path}: unexpected batch_size")
    if _exact_int(payload.get("eval_batch_size"), "eval_batch_size") != EVAL_BATCH_SIZE:
        raise ValueError(f"{json_path}: unexpected eval_batch_size")
    learning_rate = _finite_float(payload.get("learning_rate"), "learning_rate")
    if not math.isclose(learning_rate, spec.learning_rate, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{json_path}: unexpected learning_rate")
    best_epoch = _exact_int(payload.get("best_epoch"), "best_epoch")
    if not 1 <= best_epoch <= EPOCHS:
        raise ValueError(f"{json_path}: best_epoch falls outside 1..{EPOCHS}")
    best_value = _finite_float(payload.get("best_validation_value"), "best_validation_value")
    if not 0.0 <= best_value <= 1.0:
        raise ValueError(f"{json_path}: best_validation_value must be in [0, 1]")

    validation_fields = [
        payload[name]
        for name in ("validation_metric", "validation_selection")
        if name in payload
    ]
    if not validation_fields or any(value != VALIDATION_SELECTION for value in validation_fields):
        raise ValueError(
            f"{json_path}: validation selection must be {VALIDATION_SELECTION}"
        )
    units = payload.get("prediction_units")
    if not isinstance(units, list) or set(units) != {"one_second_window", "trial_mean_logit"}:
        raise ValueError(f"{json_path}: unexpected prediction_units")
    commit = str(payload.get("libeer_commit", ""))
    if not commit.startswith(LIBEER_COMMIT_PREFIX):
        raise ValueError(f"{json_path}: unexpected LibEER commit {commit!r}")
    target_access = payload.get("target_access")
    if not isinstance(target_access, str):
        raise ValueError(f"{json_path}: target_access must be a string")
    target_access_lower = target_access.lower()
    missing_fragments = [
        fragment for fragment in spec.target_access_fragments if fragment not in target_access_lower
    ]
    if missing_fragments:
        raise ValueError(
            f"{json_path}: target_access is missing frozen declaration fragment(s): "
            f"{missing_fragments}"
        )

    prediction_path = _resolve_prediction_path(payload, json_path)
    rows = _load_prediction_rows(prediction_path, spec)
    _validate_trial_content(rows, expected_split["test"], spec, prediction_path)
    sample_counts = payload.get("sample_counts")
    if not isinstance(sample_counts, Mapping):
        raise ValueError(f"{json_path}: sample_counts is required")
    if _exact_int(sample_counts.get("test"), "sample_counts.test") != len(rows.labels):
        raise ValueError(f"{json_path}: sample_counts.test disagrees with prediction NPZ")
    subject_counts = payload.get("subject_sample_counts")
    if not isinstance(subject_counts, Mapping):
        raise ValueError(f"{json_path}: subject_sample_counts is required")
    for subject in expected_split["test"]:
        expected_count = int(np.sum(rows.subject == subject))
        if _exact_int(
            subject_counts.get(str(subject)), f"subject_sample_counts.{subject}"
        ) != expected_count:
            raise ValueError(
                f"{json_path}: subject_sample_counts.{subject} disagrees with prediction NPZ"
            )

    metrics = recompute_test_metrics(rows, spec.num_classes)
    _require_test_metrics_match(payload.get("test"), metrics, expected_split["test"], spec)
    return ValidatedCell(
        json_path=json_path,
        prediction_path=prediction_path,
        fold=fold,
        optimization_seed=optimization_seed,
        split=split,
        rows=rows,
        metrics=metrics,
    )


def bca_mean_interval(
    subject_values: Sequence[float],
    *,
    n_resamples: int = 20_000,
    seed: int = 20_260_713,
) -> dict[str, object]:
    """BCa interval for the mean of already paired, seed-averaged subject rows."""

    values = np.asarray(subject_values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("subject_values must contain at least two finite values")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    observed = float(values.mean())
    if np.all(values == values[0]):
        return {
            "method": "BCa subject-row bootstrap",
            "confidence_level": 0.95,
            "n_subjects": int(len(values)),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "status": "DEGENERATE_CONSTANT",
            "observed": observed,
            "z0": 0.0,
            "acceleration": 0.0,
            "adjusted_quantiles": [0.025, 0.975],
            "low": observed,
            "high": observed,
        }

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    bootstrap_values = values[indices].mean(axis=1)
    less = int(np.sum(bootstrap_values < observed))
    equal = int(np.sum(bootstrap_values == observed))
    proportion = (less + 0.5 * equal) / n_resamples
    boundary = 0.5 / n_resamples
    proportion = float(np.clip(proportion, boundary, 1.0 - boundary))
    z0 = float(norm.ppf(proportion))

    jackknife = np.asarray(
        [np.delete(values, omitted).mean() for omitted in range(len(values))],
        dtype=float,
    )
    delta = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(delta**2) ** 1.5)
    acceleration = 0.0 if denominator == 0.0 else float(np.sum(delta**3) / denominator)

    adjusted: list[float] = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        divisor = 1.0 - acceleration * (z0 + z_alpha)
        if divisor == 0.0:
            raise ValueError("BCa adjusted quantile is undefined")
        quantile = float(norm.cdf(z0 + (z0 + z_alpha) / divisor))
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("BCa adjusted quantile falls outside [0, 1]")
        adjusted.append(quantile)
    low, high = np.quantile(bootstrap_values, adjusted)
    return {
        "method": "BCa subject-row bootstrap",
        "confidence_level": 0.95,
        "n_subjects": int(len(values)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "status": "OK",
        "observed": observed,
        "z0": z0,
        "acceleration": acceleration,
        "adjusted_quantiles": adjusted,
        "low": float(low),
        "high": float(high),
    }


def _metric_record(
    subject_seed_metrics: Mapping[int, Mapping[int, Mapping[str, float]]],
    metric: str,
) -> dict[str, object]:
    per_seed = {
        str(seed): float(
            np.mean([subject_seed_metrics[subject][seed][metric] for subject in SUBJECTS])
        )
        for seed in OPTIMIZATION_SEEDS
    }
    per_subject = {
        subject: float(
            np.mean(
                [subject_seed_metrics[subject][seed][metric] for seed in OPTIMIZATION_SEEDS]
            )
        )
        for subject in SUBJECTS
    }
    seed_values = np.asarray(list(per_seed.values()), dtype=float)
    return {
        "estimate": float(np.mean(list(per_subject.values()))),
        "aggregation": "average three paired optimization seeds within subject, then equal-weight subjects",
        "per_seed_subject_balanced_mean": per_seed,
        "optimization_seed_population_sd": float(seed_values.std(ddof=0)),
        "optimization_seed_range": [float(seed_values.min()), float(seed_values.max())],
        "per_subject_seed_averaged": {str(key): value for key, value in per_subject.items()},
    }


def aggregate_files(
    dataset: str,
    inputs: Sequence[Path],
    *,
    bootstrap: int = 20_000,
    bootstrap_seed: int = 20_260_713,
) -> dict[str, object]:
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if len(inputs) != 15:
        raise ValueError("exactly 15 JSON inputs are required")
    resolved_inputs = [Path(path).resolve() for path in inputs]
    if len(set(resolved_inputs)) != 15:
        raise ValueError("input JSON paths must be distinct")
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")

    cells = [validate_cell(path, dataset) for path in resolved_inputs]
    expected_cells = {(fold, seed) for fold in FOLDS for seed in OPTIMIZATION_SEEDS}
    observed_cells = {(cell.fold, cell.optimization_seed) for cell in cells}
    if len(observed_cells) != len(cells):
        raise ValueError("duplicate fold/optimization-seed cell")
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(f"incomplete cell grid; missing={missing}, extra={extra}")
    prediction_paths = [cell.prediction_path for cell in cells]
    if len(set(prediction_paths)) != len(prediction_paths):
        raise ValueError("each cell must use a distinct companion prediction NPZ")

    by_cell = {(cell.fold, cell.optimization_seed): cell for cell in cells}
    for seed in OPTIMIZATION_SEEDS:
        coverage = [
            subject
            for fold in FOLDS
            for subject in by_cell[(fold, seed)].split["test"]
        ]
        if sorted(coverage) != list(SUBJECTS) or len(set(coverage)) != len(coverage):
            raise ValueError(f"optimization seed {seed} does not test every subject exactly once")

    metric_paths = {
        "trial_accuracy": ("trial", "accuracy"),
        "trial_macro_f1": ("trial", "macro_f1"),
        "trial_balanced_accuracy": ("trial", "balanced_accuracy"),
        "window_accuracy": ("window", "accuracy"),
        "window_macro_f1": ("window", "macro_f1"),
        "window_balanced_accuracy": ("window", "balanced_accuracy"),
    }
    subject_seed_metrics: dict[int, dict[int, dict[str, float]]] = {
        subject: {} for subject in SUBJECTS
    }
    subject_fold: dict[int, int] = {}
    trial_recall: dict[int, dict[int, dict[int, float]]] = {
        subject: {} for subject in SUBJECTS
    }
    pooled_rows: dict[int, list[PredictionRows]] = {seed: [] for seed in OPTIMIZATION_SEEDS}
    for cell in cells:
        pooled_rows[cell.optimization_seed].append(cell.rows)
        for subject in cell.split["test"]:
            record = cell.metrics["per_subject"][str(subject)]
            subject_seed_metrics[subject][cell.optimization_seed] = {
                metric: float(record[unit][name])
                for metric, (unit, name) in metric_paths.items()
            }
            recall = {
                int(label): float(value)
                for label, value in record["trial"]["recall_per_class"].items()
            }
            trial_recall[subject][cell.optimization_seed] = recall
            subject_fold.setdefault(subject, cell.fold)
            if subject_fold[subject] != cell.fold:
                raise ValueError(f"subject {subject} changes test fold across seeds")

    for subject in SUBJECTS:
        if set(subject_seed_metrics[subject]) != set(OPTIMIZATION_SEEDS):
            raise ValueError(f"subject {subject} lacks all three paired optimization seeds")

    primary = _metric_record(subject_seed_metrics, "trial_accuracy")
    primary_values = [primary["per_subject_seed_averaged"][str(subject)] for subject in SUBJECTS]
    primary.update(
        {
            "metric": "subject-balanced trial mean-logit accuracy",
            "statistical_unit": "subject",
            "n_subjects": 15,
            "confidence_interval": bca_mean_interval(
                primary_values,
                n_resamples=bootstrap,
                seed=bootstrap_seed,
            ),
        }
    )

    secondary = {
        metric: _metric_record(subject_seed_metrics, metric)
        for metric in metric_paths
        if metric != "trial_accuracy"
    }
    primary_array = np.asarray(primary_values, dtype=float)
    minimum = float(primary_array.min())
    secondary["subject_tail"] = {
        "basis": "per-subject trial accuracy after paired-seed averaging",
        "worst_accuracy": minimum,
        "worst_subjects": [
            subject
            for subject, value in zip(SUBJECTS, primary_array)
            if math.isclose(float(value), minimum, rel_tol=0.0, abs_tol=1e-15)
        ],
        "tenth_percentile": float(np.quantile(primary_array, 0.10)),
        "interpretation": "descriptive; not an additional inferential sample size",
    }

    spec = DATASET_SPECS[dataset]
    subject_seed_recall: dict[str, object] = {}
    class_estimates: dict[str, float] = {}
    for label in range(spec.num_classes):
        subject_means = {
            subject: float(
                np.mean(
                    [trial_recall[subject][seed][label] for seed in OPTIMIZATION_SEEDS]
                )
            )
            for subject in SUBJECTS
        }
        subject_seed_recall[str(label)] = {
            "estimate": float(np.mean(list(subject_means.values()))),
            "per_subject_seed_averaged": {
                str(subject): value for subject, value in subject_means.items()
            },
        }
        class_estimates[str(label)] = float(np.mean(list(subject_means.values())))
    secondary["trial_recall_per_class"] = {
        "aggregation": "average paired seeds within subject, then equal-weight subjects",
        "estimates": class_estimates,
        "detail": subject_seed_recall,
    }

    pooled_compatibility: dict[str, object] = {}
    for seed in OPTIMIZATION_SEEDS:
        seed_rows = pooled_rows[seed]
        labels = np.concatenate([rows.labels for rows in seed_rows])
        prediction = np.concatenate([rows.logits.argmax(axis=1) for rows in seed_rows])
        subjects = np.concatenate([rows.subject for rows in seed_rows])
        summary = summarize(labels, prediction, subjects)
        pooled_compatibility[str(seed)] = {
            "window_count": int(len(labels)),
            "accuracy": float(summary["accuracy"]),
            "macro_f1": float(summary["macro_f1"]),
            "balanced_accuracy": float(summary["balanced_accuracy"]),
        }
    secondary["pooled_window_compatibility"] = {
        "role": "non-inferential LibEER compatibility summary; longer trials receive more weight",
        "per_seed": pooled_compatibility,
    }

    per_subject: dict[str, object] = {}
    for subject in SUBJECTS:
        per_subject[str(subject)] = {
            "test_fold": subject_fold[subject],
            "per_seed": {
                str(seed): subject_seed_metrics[subject][seed]
                for seed in OPTIMIZATION_SEEDS
            },
            "paired_seed_mean": {
                metric: float(
                    np.mean(
                        [subject_seed_metrics[subject][seed][metric] for seed in OPTIMIZATION_SEEDS]
                    )
                )
                for metric in metric_paths
            },
        }

    cell_records = [
        {
            "fold": cell.fold,
            "optimization_seed": cell.optimization_seed,
            "json": str(cell.json_path),
            "predictions": str(cell.prediction_path),
            "test_subjects": list(cell.split["test"]),
            "prediction_rows": int(len(cell.rows.labels)),
            "json_test_metrics_recomputed": True,
        }
        for cell in sorted(cells, key=lambda item: (item.fold, item.optimization_seed))
    ]
    return {
        "status": "completed",
        "analysis": "strict_seed_family_fivefold_aggregate",
        "dataset": dataset,
        "dataset_display_name": spec.display_name,
        "contract": {
            "partition_seed": DEFAULT_PARTITION_SEED,
            "optimization_seeds": list(OPTIMIZATION_SEEDS),
            "folds": list(FOLDS),
            "session": SESSION,
            "epochs": EPOCHS,
            "validation_selection": VALIDATION_SELECTION,
            "test_evaluations_per_cell": 1,
            "primary_aggregation": (
                "trial mean-logit prediction; metric within subject and seed; "
                "paired-seed average within subject; equal-weight mean over 15 subjects"
            ),
            "known_result_disclosure": (
                "fold 1 / optimization seed 2024 existed before this all-cell completion contract"
            ),
        },
        "validation": {
            "input_json_count": 15,
            "prediction_npz_count": 15,
            "expected_cell_grid_complete": True,
            "every_subject_tested_once_per_seed": True,
            "json_test_metrics_recomputed_from_predictions": True,
            "test_windows_are_not_inferential_units": True,
            "optimization_seeds_are_paired_repeated_fits": True,
        },
        "cells": cell_records,
        "primary": primary,
        "secondary": secondary,
        "per_subject": per_subject,
        "claim_boundary": (
            "Pinned archival DGCNN, first-session fixed five-fold 9/3/3 subject-disjoint "
            "evaluation on the 15 benchmark subjects. This is not LOSO, cross-session, "
            "unseen-stimulus, clinical, SOTA, or architecture-general evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing aggregate: {output}")
    result = aggregate_files(
        args.dataset,
        args.inputs,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
