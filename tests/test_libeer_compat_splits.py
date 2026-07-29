import random

import numpy as np

from pcma.data.libeer_compat_splits import build_compat_manifest, upstream_train_val_test_split


def _reference(labels):
    groups = {}
    for index, label in enumerate(labels, start=1):
        groups.setdefault(label, []).append(index)
    random.seed(2024)
    test, validation, train, others = [], [], [], []
    for indexes in groups.values():
        random.shuffle(indexes)
        n = len(indexes)
        nt, nv, nr = int(0.2 * n), int(0.2 * n), int(0.6 * n)
        test.extend(indexes[:nt])
        validation.extend(indexes[nt : nt + nv])
        train.extend(indexes[nt + nv : nt + nv + nr])
        others.extend(indexes[nt + nv + nr :])
    random.shuffle(others)
    nt, nv = int(len(labels) * 0.2) - len(test), int(len(labels) * 0.2) - len(validation)
    test.extend(others[:nt])
    validation.extend(others[nt : nt + nv])
    train.extend(others[nt + nv :])
    return {"test_trials": test, "validation_trials": validation, "train_trials": train}


def test_exact_upstream_split_seed_and_seediv():
    for labels in (
        [1, 0, 2, 1, 0] * 3,
        [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    ):
        assert upstream_train_val_test_split(labels) == _reference(labels)


def test_manifest_restarts_seed_for_each_subject():
    labels = np.array([0, 1, 2] * 5)
    payload = build_compat_manifest("seed", [labels] * 3, num_subjects=2)
    assert len(payload["units"]) == 6
    first, second = payload["units"][:2]
    assert first["train_trials"] == second["train_trials"]
    assert first["validation_trials"] == second["validation_trials"]
    assert first["test_trials"] == second["test_trials"]
    assert payload["canonical_sha256"]
