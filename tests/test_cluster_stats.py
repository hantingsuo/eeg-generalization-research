import numpy as np
import pytest

from pcma.eval.stats import (
    bootstrap_two_sample_mean_difference,
    permutation_two_sample_mean_difference,
)


def test_subject_level_bootstrap_detects_separated_groups():
    a = np.array([0.20, 0.25, 0.30, 0.35])
    b = np.array([-0.10, -0.05, 0.00, 0.05])
    result = bootstrap_two_sample_mean_difference(a, b, n_boot=2000, seed=7)

    assert result["mean_diff"] == pytest.approx(0.30)
    assert result["ci_low"] > 0
    assert result["n_a"] == 4
    assert result["n_b"] == 4


def test_subject_level_permutation_is_deterministic_and_directional():
    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = np.array([0.0, 0.1, 0.2, 0.3])
    first = permutation_two_sample_mean_difference(a, b, n_perm=2000, seed=11, alternative="greater")
    second = permutation_two_sample_mean_difference(a, b, n_perm=2000, seed=11, alternative="greater")

    assert first == second
    assert first["mean_diff"] == pytest.approx(0.5)
    assert first["p_value"] < 0.05


def test_subject_level_helpers_reject_segment_matrix_input():
    with pytest.raises(ValueError):
        bootstrap_two_sample_mean_difference(np.ones((2, 2)), np.ones(4))
