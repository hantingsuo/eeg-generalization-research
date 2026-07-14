from pathlib import Path

import numpy as np
import pytest

from pcma.data.libeer_seed import (
    SeedSubjectWindows,
    flatten_seed_subject_windows,
    make_fixed_front_back_split,
    make_clean_front_back_split,
    one_hot,
    resolve_seed_subject_file,
    seed_de_lds_variable_names,
)


def test_seed_de_lds_variable_names_are_explicit_and_ordered():
    names = seed_de_lds_variable_names()
    assert names[0] == "de_LDS1"
    assert names[-1] == "de_LDS15"
    assert len(names) == 15


def test_resolve_seed_subject_file_uses_official_session_map():
    root = Path("features")
    assert resolve_seed_subject_file(root, 1, 1) == root / "1_20131027.mat"
    assert resolve_seed_subject_file(root, 3, 15) == root / "15_20131105.mat"
    with pytest.raises(ValueError):
        resolve_seed_subject_file(root, 4, 1)


def test_clean_front_back_split_keeps_last_six_trials_test_only(tmp_path):
    trials = tuple(np.full((i + 1, 62, 5), i, dtype=np.float32) for i in range(15))
    subject = SeedSubjectWindows(
        trials=trials,
        trial_labels=np.arange(15) % 3,
        source_file=tmp_path / "subject.mat",
        label_file=tmp_path / "label.mat",
    )
    split = make_clean_front_back_split(subject, validation_trial=9)

    assert len(split["train"][0]) == sum(range(1, 9))
    assert len(split["validation"][0]) == 9
    assert len(split["test"][0]) == sum(range(10, 16))
    assert set(np.unique(split["test"][0][:, 0, 0])) == set(range(9, 15))
    assert 8 not in set(np.unique(split["train"][0][:, 0, 0]))


def test_one_hot_rejects_out_of_range_labels():
    encoded = one_hot(np.array([0, 2, 1]))
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(3))
    with pytest.raises(ValueError):
        one_hot(np.array([3]))


def test_flatten_seed_subject_windows_retains_trial_provenance(tmp_path):
    trials = tuple(np.full((2, 62, 5), i, dtype=np.float32) for i in range(15))
    subject_data = SeedSubjectWindows(
        trials=trials,
        trial_labels=np.arange(15) % 3,
        source_file=tmp_path / "subject.mat",
        label_file=tmp_path / "label.mat",
    )
    rows = flatten_seed_subject_windows(subject_data, subject=4, session=2)
    assert rows.features.shape == (30, 62, 5)
    assert set(rows.subject) == {4}
    assert set(rows.session) == {2}
    assert rows.trial[:2].tolist() == [1, 1]
    assert rows.trial[-2:].tolist() == [15, 15]
    assert rows.labels[:4].tolist() == [0, 0, 1, 1]


def test_fixed_front_back_split_has_no_validation_or_overlap(tmp_path):
    trials = tuple(np.full((2, 62, 5), i, dtype=np.float32) for i in range(15))
    subject_data = SeedSubjectWindows(
        trials=trials,
        trial_labels=np.arange(15) % 3,
        source_file=tmp_path / "subject.mat",
        label_file=tmp_path / "label.mat",
    )
    split = make_fixed_front_back_split(subject_data)
    assert set(split) == {"train", "test"}
    assert set(split["train"][2]) == set(range(1, 10))
    assert set(split["test"][2]) == set(range(10, 16))
    assert set(split["train"][2]).isdisjoint(split["test"][2])
