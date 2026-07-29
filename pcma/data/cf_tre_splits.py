"""Frozen split and cross-fitting contracts for CF-TRE.

This module contains no model code and never loads feature matrices.  It turns
trial labels and row/group identifiers into explicit, auditable assignments for
protocol ``cf_tre_seed_family_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "cf_tre_seed_family_v1"
DEFAULT_SPLIT_SEED = 2024
TRACK_A_COUNTS: dict[str, tuple[int, int, int]] = {
    "seed": (3, 1, 1),
    "seediv": (4, 1, 1),
}
TRACK_A_CLASSES: dict[str, tuple[int, ...]] = {
    "seed": (0, 1, 2),
    "seediv": (0, 1, 2, 3),
}


@dataclass(frozen=True)
class TrialSplit:
    """One subject-session's one-based trial assignment."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    derivation_strings: tuple[str, ...]
    derived_seeds: tuple[int, ...]

    def assignment_for(self, trial: int) -> str:
        if trial in self.train:
            return "train"
        if trial in self.validation:
            return "validation"
        if trial in self.test:
            return "test"
        raise KeyError(f"trial {trial} is not assigned")


@dataclass(frozen=True)
class CrossFitFold:
    """One group-disjoint inner cross-fitting fold."""

    fold: int
    fit_row_ids: tuple[str, ...]
    score_row_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    score_group_ids: tuple[str, ...]


def normalize_dataset_name(dataset: str) -> str:
    """Normalize the two frozen public dataset names."""

    normalized = dataset.strip().lower().replace("-", "").replace("_", "")
    if normalized not in TRACK_A_COUNTS:
        raise ValueError(f"dataset must be SEED or SEED-IV, got {dataset!r}")
    return normalized


def split_seed_material(
    dataset: str,
    *,
    session: int,
    subject: int,
    class_label: int,
    root_seed: int = DEFAULT_SPLIT_SEED,
    suffix: str = "",
) -> str:
    """Return the exact frozen string hashed for one class shuffle."""

    dataset_name = normalize_dataset_name(dataset)
    if session <= 0 or subject <= 0:
        raise ValueError("session and subject must be positive")
    if suffix and not suffix.startswith("|"):
        raise ValueError("suffix must be empty or start with '|'")
    return (
        f"{PROTOCOL}|{dataset_name}|{session}|{subject}|"
        f"{int(class_label)}|{int(root_seed)}{suffix}"
    )


def seed_from_material(material: str) -> int:
    """Interpret the first eight SHA-256 bytes as an unsigned big-endian seed."""

    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical UTF-8 JSON, excluding a top-level stored hash field."""

    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def class_balanced_outer_split(
    labels: Sequence[int] | np.ndarray,
    *,
    dataset: str,
    session: int,
    subject: int,
    root_seed: int = DEFAULT_SPLIT_SEED,
) -> TrialSplit:
    """Build the frozen class-balanced Track A split for one recording."""

    dataset_name = normalize_dataset_name(dataset)
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")

    expected_classes = TRACK_A_CLASSES[dataset_name]
    if tuple(sorted(np.unique(label_array).tolist())) != expected_classes:
        raise ValueError(
            f"{dataset_name} labels must contain exactly {expected_classes}, "
            f"got {tuple(sorted(np.unique(label_array).tolist()))}"
        )

    train_per_class, validation_per_class, test_per_class = TRACK_A_COUNTS[dataset_name]
    required_per_class = train_per_class + validation_per_class + test_per_class
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    materials: list[str] = []
    derived_seeds: list[int] = []

    for class_label in expected_classes:
        trial_ids = (np.flatnonzero(label_array == class_label) + 1).tolist()
        if len(trial_ids) != required_per_class:
            raise ValueError(
                f"{dataset_name} class {class_label} must have "
                f"{required_per_class} trials, got {len(trial_ids)}"
            )
        material = split_seed_material(
            dataset_name,
            session=session,
            subject=subject,
            class_label=class_label,
            root_seed=root_seed,
        )
        derived_seed = seed_from_material(material)
        random.Random(derived_seed).shuffle(trial_ids)
        train.extend(trial_ids[:train_per_class])
        validation.extend(
            trial_ids[train_per_class : train_per_class + validation_per_class]
        )
        test.extend(trial_ids[-test_per_class:])
        materials.append(material)
        derived_seeds.append(derived_seed)

    split = TrialSplit(
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
        test=tuple(sorted(test)),
        derivation_strings=tuple(materials),
        derived_seeds=tuple(derived_seeds),
    )
    all_trials = set(range(1, label_array.size + 1))
    assigned = set(split.train) | set(split.validation) | set(split.test)
    if assigned != all_trials:
        raise AssertionError("internal error: split does not cover every trial")
    if set(split.train) & set(split.validation) or set(split.train) & set(split.test):
        raise AssertionError("internal error: split overlap")
    if set(split.validation) & set(split.test):
        raise AssertionError("internal error: split overlap")
    return split


def build_track_a_split_manifest(
    dataset: str,
    labels_by_session: Sequence[Sequence[int] | np.ndarray],
    *,
    root_seed: int = DEFAULT_SPLIT_SEED,
    num_subjects: int = 15,
) -> dict[str, Any]:
    """Build the complete explicit Track A split artifact for one dataset."""

    dataset_name = normalize_dataset_name(dataset)
    if len(labels_by_session) != 3:
        raise ValueError("Track A requires exactly three session label sequences")
    if num_subjects <= 0:
        raise ValueError("num_subjects must be positive")

    units: list[dict[str, Any]] = []
    for session, labels in enumerate(labels_by_session, start=1):
        label_array = np.asarray(labels, dtype=np.int64)
        for subject in range(1, num_subjects + 1):
            split = class_balanced_outer_split(
                label_array,
                dataset=dataset_name,
                session=session,
                subject=subject,
                root_seed=root_seed,
            )
            units.append(
                {
                    "unit_id": f"{dataset_name}:s{session:02d}:sub{subject:02d}",
                    "session": session,
                    "subject": subject,
                    "label_sequence": label_array.tolist(),
                    "train_trials": list(split.train),
                    "validation_trials": list(split.validation),
                    "test_trials": list(split.test),
                    "derivation_strings": list(split.derivation_strings),
                    "derived_seeds": list(split.derived_seeds),
                }
            )

    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "dataset": dataset_name,
        "root_seed": int(root_seed),
        "split_unit": "complete_stimulus_trial",
        "num_subjects": int(num_subjects),
        "num_sessions": 3,
        "units": units,
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return payload


def audit_track_a_split_manifest(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail closed on structural or deterministic split disagreement."""

    errors: list[str] = []
    try:
        dataset = normalize_dataset_name(str(payload["dataset"]))
        root_seed = int(payload["root_seed"])
        num_subjects = int(payload["num_subjects"])
        units = list(payload["units"])
    except (KeyError, TypeError, ValueError) as exc:
        return (f"invalid manifest header: {exc}",)

    stored_hash = payload.get("canonical_sha256")
    actual_hash = canonical_json_sha256(payload)
    if stored_hash != actual_hash:
        errors.append("canonical_sha256 mismatch")
    if payload.get("protocol") != PROTOCOL:
        errors.append("protocol mismatch")
    if payload.get("split_unit") != "complete_stimulus_trial":
        errors.append("split unit mismatch")
    if len(units) != num_subjects * 3:
        errors.append(f"expected {num_subjects * 3} units, got {len(units)}")

    seen_units: set[str] = set()
    for unit in units:
        try:
            unit_id = str(unit["unit_id"])
            session = int(unit["session"])
            subject = int(unit["subject"])
            labels = np.asarray(unit["label_sequence"], dtype=np.int64)
            expected = class_balanced_outer_split(
                labels,
                dataset=dataset,
                session=session,
                subject=subject,
                root_seed=root_seed,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid unit: {exc}")
            continue
        expected_unit_id = f"{dataset}:s{session:02d}:sub{subject:02d}"
        if unit_id != expected_unit_id:
            errors.append(f"unit_id mismatch: {unit_id}")
        if unit_id in seen_units:
            errors.append(f"duplicate unit_id: {unit_id}")
        seen_units.add(unit_id)
        comparisons = {
            "train_trials": list(expected.train),
            "validation_trials": list(expected.validation),
            "test_trials": list(expected.test),
            "derivation_strings": list(expected.derivation_strings),
            "derived_seeds": list(expected.derived_seeds),
        }
        for key, expected_value in comparisons.items():
            if unit.get(key) != expected_value:
                errors.append(f"{unit_id} {key} mismatch")
    return tuple(errors)


def make_group_cross_fit_plan(
    row_ids: Sequence[str],
    group_ids: Sequence[str],
    *,
    n_splits: int = 3,
    seed: int = DEFAULT_SPLIT_SEED,
    group_labels: Mapping[str, int] | None = None,
) -> tuple[CrossFitFold, ...]:
    """Assign complete groups to deterministic inner score folds.

    If ``group_labels`` is supplied, groups are shuffled and distributed
    round-robin within label, providing group-level stratification.
    """

    if len(row_ids) != len(group_ids) or not row_ids:
        raise ValueError("row_ids and group_ids must be non-empty and equal length")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")

    group_to_rows: dict[str, list[str]] = {}
    for row_id, group_id in zip(row_ids, group_ids):
        group_to_rows.setdefault(str(group_id), []).append(str(row_id))
    groups = sorted(group_to_rows)
    if len(groups) < n_splits:
        raise ValueError("number of groups must be at least n_splits")

    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    if group_labels is None:
        shuffled = groups.copy()
        material = f"{PROTOCOL}|inner_groups|{int(seed)}"
        random.Random(seed_from_material(material)).shuffle(shuffled)
        for index, group in enumerate(shuffled):
            buckets[index % n_splits].append(group)
    else:
        missing = set(groups) - set(group_labels)
        extra = set(group_labels) - set(groups)
        if missing or extra:
            raise ValueError(f"group_labels mismatch; missing={missing}, extra={extra}")
        by_label: dict[int, list[str]] = {}
        for group in groups:
            by_label.setdefault(int(group_labels[group]), []).append(group)
        for label in sorted(by_label):
            shuffled = sorted(by_label[label])
            material = f"{PROTOCOL}|inner_groups|{int(seed)}|{label}"
            random.Random(seed_from_material(material)).shuffle(shuffled)
            for index, group in enumerate(shuffled):
                buckets[index % n_splits].append(group)

    if any(not bucket for bucket in buckets):
        raise ValueError("cross-fitting produced an empty score fold")

    all_groups = set(groups)
    folds: list[CrossFitFold] = []
    for fold_index, score_groups_list in enumerate(buckets, start=1):
        score_groups = tuple(sorted(score_groups_list))
        fit_groups = tuple(sorted(all_groups - set(score_groups)))
        score_rows = tuple(
            row
            for group in score_groups
            for row in sorted(group_to_rows[group])
        )
        fit_rows = tuple(
            row
            for group in fit_groups
            for row in sorted(group_to_rows[group])
        )
        folds.append(
            CrossFitFold(
                fold=fold_index,
                fit_row_ids=fit_rows,
                score_row_ids=score_rows,
                fit_group_ids=fit_groups,
                score_group_ids=score_groups,
            )
        )
    audit_group_cross_fit_plan(folds, row_ids=row_ids, group_ids=group_ids)
    return tuple(folds)


def audit_group_cross_fit_plan(
    folds: Iterable[CrossFitFold],
    *,
    row_ids: Sequence[str],
    group_ids: Sequence[str],
) -> None:
    """Raise on incomplete coverage, row overlap, or group leakage."""

    if len(row_ids) != len(group_ids):
        raise ValueError("row_ids and group_ids length mismatch")
    row_to_group = {str(row): str(group) for row, group in zip(row_ids, group_ids)}
    if len(row_to_group) != len(row_ids):
        raise ValueError("row_ids must be unique")
    expected_rows = set(row_to_group)
    score_counts = {row: 0 for row in expected_rows}
    fold_numbers: set[int] = set()

    for fold in folds:
        if fold.fold in fold_numbers:
            raise ValueError(f"duplicate fold number {fold.fold}")
        fold_numbers.add(fold.fold)
        fit_rows = set(fold.fit_row_ids)
        score_rows = set(fold.score_row_ids)
        fit_groups = set(fold.fit_group_ids)
        score_groups = set(fold.score_group_ids)
        if not fit_rows or not score_rows:
            raise ValueError(f"fold {fold.fold} has an empty row partition")
        if fit_rows & score_rows:
            raise ValueError(f"fold {fold.fold} has row leakage")
        if fit_groups & score_groups:
            raise ValueError(f"fold {fold.fold} has group leakage")
        if fit_rows | score_rows != expected_rows:
            raise ValueError(f"fold {fold.fold} does not partition all rows")
        if {row_to_group[row] for row in fit_rows} != fit_groups:
            raise ValueError(f"fold {fold.fold} fit group ledger mismatch")
        if {row_to_group[row] for row in score_rows} != score_groups:
            raise ValueError(f"fold {fold.fold} score group ledger mismatch")
        for row in score_rows:
            score_counts[row] += 1

    bad = [row for row, count in score_counts.items() if count != 1]
    if bad:
        raise ValueError(f"rows must be scored exactly once: {bad[:5]}")
