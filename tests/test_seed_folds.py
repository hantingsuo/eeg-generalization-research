from collections import Counter

import pytest

from pcma.data.seed_folds import five_fold_subject_partitions, get_subject_fold, subject_groups


def test_seed_2024_groups_recover_original_libeer_test_and_validation_groups():
    assert subject_groups(2024) == (
        (9, 1, 6),
        (14, 13, 2),
        (15, 11, 7),
        (4, 5, 10),
        (12, 3, 8),
    )
    fold_one = get_subject_fold(1)
    assert fold_one["test"] == (9, 1, 6)
    assert fold_one["validation"] == (14, 13, 2)


def test_five_folds_are_balanced_and_disjoint():
    folds = five_fold_subject_partitions()
    assert len(folds) == 5
    for split in folds:
        assert len(split["train"]) == 9
        assert len(split["validation"]) == 3
        assert len(split["test"]) == 3
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["train"]).isdisjoint(split["test"])
        assert set(split["validation"]).isdisjoint(split["test"])
        assert set(split["train"] + split["validation"] + split["test"]) == set(range(1, 16))

    assert Counter(subject for split in folds for subject in split["test"]) == Counter(range(1, 16))
    assert Counter(subject for split in folds for subject in split["validation"]) == Counter(range(1, 16))
    assert Counter(subject for split in folds for subject in split["train"]) == Counter(
        {subject: 3 for subject in range(1, 16)}
    )


def test_fold_lookup_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="fold must be in"):
        get_subject_fold(0)
    with pytest.raises(ValueError, match="fold must be in"):
        get_subject_fold(6)
