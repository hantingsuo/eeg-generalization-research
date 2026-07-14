"""Fail-closed baseline helpers for the frozen CF-TRE G0-B matrix."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.data.cf_tre_features import concatenate_feature_views, extract_cf_tre_feature_views


PROTOCOL = "cf_tre_seed_family_v1"


@dataclass(frozen=True)
class ComponentSpec:
    family: str
    views: tuple[str, ...]
    candidates: tuple[Mapping[str, Any], ...]

    @property
    def dimension(self) -> int:
        dimensions = {"de310": 310, "asymmetry45": 45, "regional50": 50, "trial_context150": 150}
        return sum(dimensions[name] for name in self.views)


_LINEAR = tuple({"C": value} for value in (0.1, 1.0, 10.0))
_RBF = tuple(
    {"C": c_value, "gamma": gamma}
    for c_value in (1.0, 10.0)
    for gamma in ("scale", 0.001)
)
_XGB = (
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 2, "learning_rate": 0.03},
)
_LGB = (
    {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05},
    {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.03},
)
_ALL_VIEWS = ("de310", "asymmetry45", "regional50", "trial_context150")

COMPONENT_SPECS: dict[str, ComponentSpec] = {
    "linear_de310": ComponentSpec("linear_svm", ("de310",), _LINEAR),
    "linear_structural245": ComponentSpec(
        "linear_svm", ("asymmetry45", "regional50", "trial_context150"), _LINEAR
    ),
    "linear_all555": ComponentSpec("linear_svm", _ALL_VIEWS, _LINEAR),
    "rbf_de310": ComponentSpec("rbf_svm", ("de310",), _RBF),
    "rbf_all555": ComponentSpec("rbf_svm", _ALL_VIEWS, _RBF),
    "xgboost_all555": ComponentSpec("xgboost", _ALL_VIEWS, _XGB),
    "lightgbm_all555": ComponentSpec("lightgbm", _ALL_VIEWS, _LGB),
}


def load_track_a_unit(
    manifest_path: str | Path,
    *,
    dataset: str,
    session: int,
    subject: int,
) -> dict[str, Any]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol mismatch: {payload.get('protocol')!r}")
    if payload.get("dataset") != dataset:
        raise ValueError(f"dataset mismatch: expected {dataset!r}, found {payload.get('dataset')!r}")
    matches = [
        unit
        for unit in payload.get("units", [])
        if int(unit.get("session", -1)) == session and int(unit.get("subject", -1)) == subject
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one split unit for session={session}, subject={subject}; found {len(matches)}")
    unit = dict(matches[0])
    expected_id = f"{dataset}:s{session:02d}:sub{subject:02d}"
    if unit.get("unit_id") != expected_id:
        raise ValueError(f"unit id mismatch: expected {expected_id}, found {unit.get('unit_id')!r}")
    return unit


def partition_trial_indices(
    row_trials: np.ndarray,
    *,
    train_trials: Sequence[int],
    validation_trials: Sequence[int],
    test_trials: Sequence[int],
) -> dict[str, np.ndarray]:
    trials = np.asarray(row_trials)
    declared = {
        "train": {int(value) for value in train_trials},
        "validation": {int(value) for value in validation_trials},
        "test": {int(value) for value in test_trials},
    }
    if any(not values for values in declared.values()):
        raise ValueError("all outer partitions must contain trials")
    if declared["train"] & declared["validation"] or declared["train"] & declared["test"] or declared["validation"] & declared["test"]:
        raise ValueError("outer trial partitions overlap")
    observed = {int(value) for value in np.unique(trials)}
    if set().union(*declared.values()) != observed:
        raise ValueError("outer trial partitions do not exactly cover the recording")
    indices = {
        name: np.flatnonzero(np.isin(trials, sorted(values)))
        for name, values in declared.items()
    }
    if any(len(value) == 0 for value in indices.values()):
        raise ValueError("outer partition produced zero windows")
    return indices


def _stable_order(values: Sequence[int], *, seed: int, label: int) -> list[int]:
    return sorted(
        (int(value) for value in values),
        key=lambda value: hashlib.sha256(
            f"{PROTOCOL}|calibration_trials|{seed}|{label}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def calibration_cv_by_trial(
    labels: np.ndarray,
    trials: np.ndarray,
    *,
    seed: int,
    n_splits: int = 3,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create explicit class-stratified, complete-trial calibration folds."""

    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(trials, dtype=np.int64)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or len(y) == 0:
        raise ValueError("labels and trials must be equal non-empty vectors")
    trial_labels: dict[int, int] = {}
    for trial in np.unique(groups):
        values = np.unique(y[groups == trial])
        if len(values) != 1:
            raise ValueError(f"trial {int(trial)} has inconsistent labels")
        trial_labels[int(trial)] = int(values[0])
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(set(trial_labels.values())):
        label_trials = [trial for trial, value in trial_labels.items() if value == label]
        if len(label_trials) < n_splits:
            raise ValueError(f"class {label} has fewer than {n_splits} calibration groups")
        for index, trial in enumerate(_stable_order(label_trials, seed=seed, label=label)):
            buckets[index % n_splits].append(trial)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    scored: list[int] = []
    for score_trials in buckets:
        score_set = set(score_trials)
        score = np.flatnonzero(np.isin(groups, sorted(score_set)))
        fit = np.flatnonzero(~np.isin(groups, sorted(score_set)))
        if set(groups[fit].tolist()) & set(groups[score].tolist()):
            raise AssertionError("trial leakage in calibration fold")
        if set(np.unique(y[fit]).tolist()) != set(np.unique(y).tolist()):
            raise ValueError("calibration fit fold is missing a class")
        if set(np.unique(y[score]).tolist()) != set(np.unique(y).tolist()):
            raise ValueError("calibration score fold is missing a class")
        folds.append((fit, score))
        scored.extend(score.tolist())
    if sorted(scored) != list(range(len(y))):
        raise AssertionError("calibration folds do not score each row exactly once")
    return tuple(folds)


def feature_rows_for_recording(
    trials: Sequence[np.ndarray],
    trial_labels: np.ndarray,
    *,
    trial_numbers: Sequence[int] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    labels = np.asarray(trial_labels, dtype=np.int64)
    if len(trials) != len(labels):
        raise ValueError("trial feature and label counts differ")
    numbers = list(range(1, len(trials) + 1)) if trial_numbers is None else [int(value) for value in trial_numbers]
    if len(numbers) != len(trials) or len(set(numbers)) != len(numbers) or any(value <= 0 for value in numbers):
        raise ValueError("trial_numbers must be unique positive identifiers")
    by_view: dict[str, list[np.ndarray]] = {name: [] for name in _ALL_VIEWS}
    row_labels: list[np.ndarray] = []
    row_trials: list[np.ndarray] = []
    for index, (trial, trial_number) in enumerate(zip(trials, numbers), start=1):
        views = extract_cf_tre_feature_views(trial)
        for name in by_view:
            by_view[name].append(views[name])
        row_labels.append(np.full(len(trial), labels[index - 1], dtype=np.int64))
        row_trials.append(np.full(len(trial), trial_number, dtype=np.int16))
    return (
        {name: np.concatenate(parts, axis=0) for name, parts in by_view.items()},
        np.concatenate(row_labels),
        np.concatenate(row_trials),
    )


def component_matrix(views: Mapping[str, np.ndarray], component: str) -> np.ndarray:
    if component not in COMPONENT_SPECS:
        raise KeyError(f"unknown component {component!r}")
    return concatenate_feature_views(views, list(COMPONENT_SPECS[component].views))


def build_estimator(
    component: str,
    candidate: Mapping[str, Any],
    *,
    train_labels: np.ndarray,
    train_trials: np.ndarray,
    seed: int,
) -> Any:
    spec = COMPONENT_SPECS[component]
    params = dict(candidate)
    if spec.family in {"linear_svm", "rbf_svm"}:
        kernel = "linear" if spec.family == "linear_svm" else "rbf"
        svc = SVC(
            kernel=kernel,
            probability=False,
            class_weight="balanced",
            random_state=seed,
            **params,
        )
        pipeline = Pipeline([("scale", StandardScaler()), ("svm", svc)])
        cv = calibration_cv_by_trial(train_labels, train_trials, seed=seed)
        return CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=cv, n_jobs=1, ensemble=True)
    if spec.family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=int(len(np.unique(train_labels))),
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            **params,
        )
    if spec.family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            verbosity=-1,
            **params,
        )
    raise AssertionError(f"unhandled family {spec.family}")


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probs = normalize_probabilities(probabilities)
    if len(probs) != len(y):
        raise ValueError("probability and label row counts differ")
    predictions = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "log_loss": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
    }


def normalize_probabilities(probabilities: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2 or not np.all(np.isfinite(probs)):
        raise ValueError("invalid probability matrix")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    clipped = np.clip(probs, epsilon, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def trial_mean_probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    trials: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probs = normalize_probabilities(probabilities)
    groups = np.asarray(trials, dtype=np.int64)
    trial_y: list[int] = []
    trial_probs: list[np.ndarray] = []
    for trial in sorted(np.unique(groups).tolist()):
        mask = groups == trial
        values = np.unique(y[mask])
        if len(values) != 1:
            raise ValueError(f"trial {trial} has inconsistent labels")
        trial_y.append(int(values[0]))
        trial_probs.append(probs[mask].mean(axis=0))
    return probability_metrics(np.asarray(trial_y), np.stack(trial_probs))


def select_validation_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate table cannot be empty")
    required = {"candidate_index", "accuracy", "macro_f1"}
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"candidate row missing fields: {required - set(row)}")
    return dict(
        min(
            rows,
            key=lambda row: (
                -float(row["accuracy"]),
                -float(row["macro_f1"]),
                int(row["candidate_index"]),
            ),
        )
    )
