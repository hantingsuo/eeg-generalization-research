# tests/conftest.py
import numpy as np
import pytest


def _rand_spd(n, rng, cond=5.0):
    A = rng.standard_normal((n, n))
    S = A @ A.T + cond * np.eye(n)
    return (S + S.T) / 2


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def spd_batch(rng):
    """20 SPD matrices, 8x8."""
    return np.stack([_rand_spd(8, rng) for _ in range(20)])
