import numpy as np

from experiments.run_cross_population_gate_a import (
    balanced_target_schedule,
    extract_rich_features_batched,
)
from pcma.data.features import extract_rich_features


def test_balanced_target_schedule_equalizes_exposure():
    subjects = np.array([f"S{i}" for i in range(8)])
    groups = balanced_target_schedule(subjects, target_n=2, repeats=8, rng=np.random.default_rng(3))
    counts = {s: 0 for s in subjects}
    for group in groups:
        for subject in group:
            counts[subject] += 1
    assert set(counts.values()) == {2}


def test_batched_feature_extraction_matches_original_definition():
    X = np.random.default_rng(5).normal(size=(4, 30, 500)).astype(np.float32)
    expected = extract_rich_features(X, sfreq=250.0)
    observed = extract_rich_features_batched(X, sfreq=250.0, batch_size=2)
    assert np.allclose(observed, expected)
