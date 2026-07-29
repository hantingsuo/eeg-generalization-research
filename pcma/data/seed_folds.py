"""Frozen all-subject cross-validation folds for SEED and SEED-IV."""

from __future__ import annotations

import random


NUM_SUBJECTS = 15
NUM_FOLDS = 5
GROUP_SIZE = 3
DEFAULT_PARTITION_SEED = 2024


def subject_groups(partition_seed: int = DEFAULT_PARTITION_SEED) -> tuple[tuple[int, ...], ...]:
    """Return five one-based three-subject groups from the frozen shuffle."""

    subjects = list(range(1, NUM_SUBJECTS + 1))
    random.Random(partition_seed).shuffle(subjects)
    return tuple(
        tuple(subjects[start : start + GROUP_SIZE])
        for start in range(0, NUM_SUBJECTS, GROUP_SIZE)
    )


def five_fold_subject_partitions(
    partition_seed: int = DEFAULT_PARTITION_SEED,
) -> tuple[dict[str, tuple[int, ...]], ...]:
    """Build balanced 9-train/3-validation/3-test folds.

    Fold ``i`` tests group ``i`` and validates on the next group cyclically.
    Consequently every subject is tested once, validates once, and trains in
    the remaining three folds. Subject identifiers are one-based.
    """

    groups = subject_groups(partition_seed)
    folds: list[dict[str, tuple[int, ...]]] = []
    for index in range(NUM_FOLDS):
        test = groups[index]
        validation = groups[(index + 1) % NUM_FOLDS]
        excluded = set(test) | set(validation)
        train = tuple(subject for group in groups for subject in group if subject not in excluded)
        folds.append({"train": train, "validation": validation, "test": test})
    return tuple(folds)


def get_subject_fold(
    fold: int,
    *,
    partition_seed: int = DEFAULT_PARTITION_SEED,
) -> dict[str, tuple[int, ...]]:
    """Return one frozen fold by its one-based identifier."""

    if not 1 <= fold <= NUM_FOLDS:
        raise ValueError(f"fold must be in 1..{NUM_FOLDS}")
    return five_fold_subject_partitions(partition_seed)[fold - 1]
