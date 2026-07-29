from pathlib import Path

import numpy as np

from experiments.audit_strict_fivefold_train_fit_diagnostic import (
    recompute_dataset,
)
from experiments.evaluate_strict_fivefold_train_checkpoints import (
    EXPECTED_FOLDS,
    EXPECTED_OPTIMIZATION_SEEDS,
    expected_cells,
    scalar_metrics,
)


def test_expected_train_fit_cells_are_complete_and_do_not_name_npz():
    cells = expected_cells()
    assert len(cells) == 30
    identities = {(dataset, fold, seed) for dataset, fold, seed, _, _ in cells}
    assert len(identities) == 30
    assert {fold for _, fold, _, _, _ in cells} == set(EXPECTED_FOLDS)
    assert {seed for _, _, seed, _, _ in cells} == set(
        EXPECTED_OPTIMIZATION_SEEDS
    )
    assert all(json_path.suffix == ".json" for *_, json_path, _ in cells)
    assert all(checkpoint_path.suffix == ".pt" for *_, checkpoint_path in cells)


def test_scalar_metrics_preserves_subject_equal_accuracy():
    result = scalar_metrics(
        {
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "mean_subject_acc": 0.75,
            "worst_subject_acc": 0.5,
            "balanced_accuracy": 0.72,
        }
    )
    assert result["accuracy"] == 0.8
    assert result["mean_subject_accuracy"] == 0.75
    assert result["worst_subject_accuracy"] == 0.5


def test_independent_recompute_averages_seeds_within_fold_then_folds():
    cells = []
    for fold in EXPECTED_FOLDS:
        for offset, seed in enumerate(EXPECTED_OPTIMIZATION_SEEDS):
            train_trial = 0.80 + 0.01 * fold + 0.001 * offset
            test_trial = 0.40 + 0.01 * fold + 0.001 * offset
            cells.append(
                {
                    "dataset": "seed",
                    "fold": fold,
                    "optimization_seed": seed,
                    "train": {
                        "window": {"mean_subject_accuracy": train_trial + 0.02},
                        "trial": {"mean_subject_accuracy": train_trial},
                    },
                    "validation": {
                        "window": {"accuracy": 0.60 + 0.01 * fold}
                    },
                    "test": {
                        "trial": {"mean_subject_accuracy": test_trial}
                    },
                }
            )
    result = recompute_dataset(cells, "seed")
    assert np.isclose(
        result["train_trial_subject_equal_accuracy"][
            "mean_over_five_fold_means"
        ],
        np.mean([0.80 + 0.01 * fold + 0.001 for fold in EXPECTED_FOLDS]),
    )
    assert np.isclose(result["train_minus_test_trial_accuracy"], 0.4)
