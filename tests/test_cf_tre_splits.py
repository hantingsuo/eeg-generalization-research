import copy

import numpy as np
import pytest

from pcma.data.cf_tre_splits import (
    CrossFitFold,
    audit_group_cross_fit_plan,
    audit_track_a_split_manifest,
    build_track_a_split_manifest,
    class_balanced_outer_split,
    make_group_cross_fit_plan,
    seed_from_material,
    split_seed_material,
)


SEED_LABELS = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
SEEDIV_LABELS = np.array([0, 1, 2, 3] * 6)


def test_outer_splits_are_deterministic_class_balanced_and_subject_specific():
    first = class_balanced_outer_split(
        SEED_LABELS, dataset="SEED", session=1, subject=1
    )
    repeated = class_balanced_outer_split(
        SEED_LABELS, dataset="seed", session=1, subject=1
    )
    another_subject = class_balanced_outer_split(
        SEED_LABELS, dataset="SEED", session=1, subject=2
    )
    assert first == repeated
    assert (len(first.train), len(first.validation), len(first.test)) == (9, 3, 3)
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    assert set(first.train + first.validation + first.test) == set(range(1, 16))
    assert first != another_subject
    for assignment in (first.train, first.validation, first.test):
        counts = np.bincount(SEED_LABELS[np.array(assignment) - 1], minlength=3)
        assert len(set(counts.tolist())) == 1


def test_seed_material_and_hash_conversion_are_frozen():
    material = split_seed_material(
        "SEED-IV", session=3, subject=15, class_label=2, root_seed=2024
    )
    assert material == "cf_tre_seed_family_v1|seediv|3|15|2|2024"
    assert seed_from_material(material) == seed_from_material(material)
    assert 0 <= seed_from_material(material) < 2**64


def test_seediv_split_has_frozen_16_4_4_counts():
    split = class_balanced_outer_split(
        SEEDIV_LABELS, dataset="SEED-IV", session=2, subject=7
    )
    assert (len(split.train), len(split.validation), len(split.test)) == (16, 4, 4)
    for assignment, expected in (
        (split.train, 4),
        (split.validation, 1),
        (split.test, 1),
    ):
        counts = np.bincount(SEEDIV_LABELS[np.array(assignment) - 1], minlength=4)
        assert counts.tolist() == [expected] * 4


def test_complete_manifest_audits_and_tampering_fails_closed():
    manifest = build_track_a_split_manifest("SEED", [SEED_LABELS] * 3)
    assert len(manifest["units"]) == 45
    assert len(manifest["canonical_sha256"]) == 64
    assert audit_track_a_split_manifest(manifest) == ()

    tampered = copy.deepcopy(manifest)
    tampered["units"][0]["test_trials"][0] = tampered["units"][0]["train_trials"][0]
    errors = audit_track_a_split_manifest(tampered)
    assert "canonical_sha256 mismatch" in errors
    assert any("test_trials mismatch" in error for error in errors)


def test_group_cross_fit_scores_each_row_once_and_excludes_complete_groups():
    groups = [f"g{index}" for index in range(9)]
    row_ids = [f"{group}:r{row}" for group in groups for row in range(2)]
    group_ids = [group for group in groups for _ in range(2)]
    labels = {group: index % 3 for index, group in enumerate(groups)}
    plan = make_group_cross_fit_plan(
        row_ids,
        group_ids,
        n_splits=3,
        seed=2024,
        group_labels=labels,
    )
    assert len(plan) == 3
    assert {len(fold.score_group_ids) for fold in plan} == {3}
    audit_group_cross_fit_plan(plan, row_ids=row_ids, group_ids=group_ids)

    first = plan[0]
    contaminated = (
        CrossFitFold(
            fold=first.fold,
            fit_row_ids=first.fit_row_ids,
            score_row_ids=first.score_row_ids,
            fit_group_ids=first.fit_group_ids + (first.score_group_ids[0],),
            score_group_ids=first.score_group_ids,
        ),
        *plan[1:],
    )
    with pytest.raises(ValueError, match="group leakage"):
        audit_group_cross_fit_plan(contaminated, row_ids=row_ids, group_ids=group_ids)


def test_split_rejects_wrong_class_counts():
    with pytest.raises(ValueError, match="must have"):
        class_balanced_outer_split(
            np.array([0] * 6 + [1] * 4 + [2] * 5),
            dataset="seed",
            session=1,
            subject=1,
        )

