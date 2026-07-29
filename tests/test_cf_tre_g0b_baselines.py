from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pcma.model.cf_tre_baselines import (
    COMPONENT_SPECS,
    calibration_cv_by_trial,
    load_track_a_unit,
    normalize_probabilities,
    partition_trial_indices,
    select_validation_candidate,
)


def test_frozen_component_matrix_has_exactly_seven_entries() -> None:
    assert tuple(COMPONENT_SPECS) == (
        "linear_de310",
        "linear_structural245",
        "linear_all555",
        "rbf_de310",
        "rbf_all555",
        "xgboost_all555",
        "lightgbm_all555",
    )
    assert COMPONENT_SPECS["linear_structural245"].dimension == 245
    assert COMPONENT_SPECS["linear_all555"].dimension == 555


def test_calibration_folds_are_trial_disjoint_and_class_complete() -> None:
    labels = np.repeat(np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2]), 2)
    trials = np.repeat(np.arange(1, 10), 2)
    folds = calibration_cv_by_trial(labels, trials, seed=2024)
    assert len(folds) == 3
    scored_trials: list[int] = []
    for fit, score in folds:
        fit_trials = set(trials[fit].tolist())
        score_trials = set(trials[score].tolist())
        assert fit_trials.isdisjoint(score_trials)
        assert set(labels[fit].tolist()) == {0, 1, 2}
        assert set(labels[score].tolist()) == {0, 1, 2}
        scored_trials.extend(sorted(score_trials))
    assert sorted(scored_trials) == list(range(1, 10))


def test_candidate_selection_uses_accuracy_f1_then_frozen_order() -> None:
    rows = [
        {"candidate_index": 0, "accuracy": 0.7, "macro_f1": 0.6},
        {"candidate_index": 1, "accuracy": 0.7, "macro_f1": 0.7},
        {"candidate_index": 2, "accuracy": 0.8, "macro_f1": 0.5},
        {"candidate_index": 3, "accuracy": 0.8, "macro_f1": 0.5},
    ]
    assert select_validation_candidate(rows)["candidate_index"] == 2


def test_partition_indices_are_complete_and_nonoverlapping() -> None:
    row_trials = np.asarray([1, 1, 2, 3, 4, 4, 5])
    parts = partition_trial_indices(
        row_trials,
        train_trials=[1, 3],
        validation_trials=[2, 5],
        test_trials=[4],
    )
    assert parts["train"].tolist() == [0, 1, 3]
    assert parts["validation"].tolist() == [2, 6]
    assert parts["test"].tolist() == [4, 5]
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(7))


def test_track_a_unit_lookup_rejects_manifest_dataset_mismatch(tmp_path: Path) -> None:
    manifest = {
        "protocol": "cf_tre_seed_family_v1",
        "dataset": "seed",
        "units": [
            {
                "unit_id": "seed:s01:sub01",
                "session": 1,
                "subject": 1,
                "train_trials": [1, 2, 3],
                "validation_trials": [4, 5, 6],
                "test_trials": [7, 8, 9],
            }
        ],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    unit = load_track_a_unit(path, dataset="seed", session=1, subject=1)
    assert unit["unit_id"] == "seed:s01:sub01"
    try:
        load_track_a_unit(path, dataset="seediv", session=1, subject=1)
    except ValueError as exc:
        assert "dataset mismatch" in str(exc)
    else:
        raise AssertionError("dataset mismatch was accepted")


def test_probability_normalization_clips_and_restores_row_sums() -> None:
    raw = np.asarray([[0.2, 0.3, 0.5000002], [0.0, 0.4, 0.6]])
    normalized = normalize_probabilities(raw)
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.all(normalized > 0)


def test_inner_svm_calibration_can_use_two_complete_trial_folds() -> None:
    from pcma.model.cf_tre_baselines import build_estimator

    labels = np.repeat(np.asarray([0, 1, 2, 0, 1, 2]), 3)
    trials = np.repeat(np.arange(1, 7), 3)
    estimator = build_estimator(
        "linear_de310",
        {"C": 1.0},
        train_labels=labels,
        train_trials=trials,
        seed=2024,
        calibration_splits=2,
    )
    assert len(estimator.cv) == 2
