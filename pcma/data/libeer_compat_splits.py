"""Exact clean subject-dependent split contract from pinned LibEER 39dc27e."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL = "libeer_39dc27e_clean_table_compat_v1"
SPLIT_SEED = 2024


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    encoded = json.dumps(
        clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def upstream_train_val_test_split(
    labels: Sequence[int] | np.ndarray,
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, list[int]]:
    """Reproduce pinned ``get_split_index`` and return one-based trial IDs.

    Group insertion order and partition order are scientifically material here.
    They are intentionally not sorted.
    """

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty vector")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(values.tolist(), start=1):
        groups.setdefault(int(label), []).append(index)
    if len(groups) < 2:
        raise ValueError("at least two classes are required")

    rng = random.Random(int(seed))
    split = {"test_trials": [], "validation_trials": [], "train_trials": []}
    others: list[int] = []
    for indexes in groups.values():
        rng.shuffle(indexes)
        total = len(indexes)
        test_num = int(0.2 * total)
        validation_num = int(0.2 * total)
        train_num = int(0.6 * total)
        split["test_trials"].extend(indexes[:test_num])
        split["validation_trials"].extend(
            indexes[test_num : test_num + validation_num]
        )
        split["train_trials"].extend(
            indexes[
                test_num + validation_num : test_num + validation_num + train_num
            ]
        )
        others.extend(indexes[test_num + validation_num + train_num :])

    if others:
        rng.shuffle(others)
        expected_test = int(values.size * 0.2)
        expected_validation = int(values.size * 0.2)
        add_test = expected_test - len(split["test_trials"])
        add_validation = expected_validation - len(split["validation_trials"])
        split["test_trials"].extend(others[:add_test])
        split["validation_trials"].extend(
            others[add_test : add_test + add_validation]
        )
        split["train_trials"].extend(others[add_test + add_validation :])

    partitions = [set(split[key]) for key in ("train_trials", "validation_trials", "test_trials")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("split overlap")
    if set().union(*partitions) != set(range(1, values.size + 1)):
        raise AssertionError("split does not cover all trials")
    return split


def build_compat_manifest(
    dataset: str,
    labels_by_session: Sequence[Sequence[int] | np.ndarray],
    *,
    num_subjects: int = 15,
) -> dict[str, Any]:
    normalized = dataset.lower().replace("-", "").replace("_", "")
    if normalized not in {"seed", "seediv"}:
        raise ValueError("dataset must be SEED or SEED-IV")
    if len(labels_by_session) != 3 or num_subjects <= 0:
        raise ValueError("three sessions and positive num_subjects are required")
    units: list[dict[str, Any]] = []
    for session, labels in enumerate(labels_by_session, start=1):
        label_list = np.asarray(labels, dtype=np.int64).tolist()
        split = upstream_train_val_test_split(label_list)
        for subject in range(1, num_subjects + 1):
            units.append(
                {
                    "unit_id": f"{normalized}:s{session:02d}:sub{subject:02d}",
                    "session": session,
                    "subject": subject,
                    "label_sequence": label_list,
                    **split,
                }
            )
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "dataset": normalized,
        "split_seed": SPLIT_SEED,
        "split_implementation": "LibEER 39dc27e data_utils/split.py:get_split_index",
        "split_unit": "complete_stimulus_trial",
        "num_subjects": num_subjects,
        "num_sessions": 3,
        "units": units,
    }
    payload["canonical_sha256"] = canonical_json_sha256(payload)
    return payload
