import numpy as np
import pytest

from pcma.data.cf_tre_features import (
    ASYMMETRY_PAIRS,
    REGION_CHANNELS,
    SEED_CHANNELS,
    VIEW_DIMENSIONS,
    audit_feature_schema,
    concatenate_feature_views,
    extract_cf_tre_feature_views,
)


def test_frozen_channel_regions_and_pairs_are_complete():
    audit_feature_schema()
    assert len(SEED_CHANNELS) == 62
    assert len(ASYMMETRY_PAIRS) == 9
    flattened = [channel for channels in REGION_CHANNELS.values() for channel in channels]
    assert len(flattened) == len(set(flattened)) == 62
    assert set(flattened) == set(SEED_CHANNELS)


def test_feature_views_have_frozen_dimensions_and_finite_constant_context():
    trial = np.ones((4, 62, 5), dtype=np.float32)
    views = extract_cf_tre_feature_views(trial)
    assert set(views) == set(VIEW_DIMENSIONS)
    for name, dimension in VIEW_DIMENSIONS.items():
        assert views[name].shape == (4, dimension)
        assert views[name].dtype == np.float32
        assert np.all(np.isfinite(views[name]))
    np.testing.assert_array_equal(views["asymmetry45"], 0.0)
    np.testing.assert_array_equal(views["trial_context150"][:, 25:], 0.0)


def test_asymmetry_uses_left_minus_right_in_frozen_order():
    trial = np.zeros((2, 62, 5), dtype=np.float32)
    trial[:, SEED_CHANNELS.index("FP1"), 0] = 2.0
    trial[:, SEED_CHANNELS.index("FP2"), 0] = 0.5
    asymmetry = extract_cf_tre_feature_views(trial)["asymmetry45"]
    np.testing.assert_allclose(asymmetry[:, 0], 1.5)
    np.testing.assert_array_equal(asymmetry[:, 1:], 0.0)


def test_trial_context_is_broadcast_only_within_the_supplied_trial():
    rng = np.random.default_rng(7)
    first = extract_cf_tre_feature_views(rng.normal(size=(5, 62, 5)))[
        "trial_context150"
    ]
    second = extract_cf_tre_feature_views(rng.normal(size=(5, 62, 5)) + 4.0)[
        "trial_context150"
    ]
    np.testing.assert_array_equal(first, np.broadcast_to(first[0], first.shape))
    np.testing.assert_array_equal(second, np.broadcast_to(second[0], second.shape))
    assert not np.allclose(first[0], second[0])


def test_concatenate_views_checks_names_dimensions_and_rows():
    views = extract_cf_tre_feature_views(np.zeros((3, 62, 5), dtype=np.float32))
    combined = concatenate_feature_views(
        views, ["asymmetry45", "regional50", "trial_context150"]
    )
    assert combined.shape == (3, 245)
    with pytest.raises(KeyError):
        concatenate_feature_views(views, ["unknown"])
    bad = dict(views)
    bad["de310"] = bad["de310"][:, :-1]
    with pytest.raises(ValueError):
        concatenate_feature_views(bad, ["de310"])


def test_feature_extractor_rejects_cross_trial_or_nonfinite_shapes():
    with pytest.raises(ValueError, match="shape"):
        extract_cf_tre_feature_views(np.zeros((2, 310), dtype=np.float32))
    trial = np.zeros((2, 62, 5), dtype=np.float32)
    trial[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_cf_tre_feature_views(trial)

