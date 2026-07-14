from dataclasses import replace

import numpy as np
import pytest

from pcma.model.cf_tre import (
    OOFPredictionRecord,
    audit_oof_prediction_ledger,
    empirical_cvar,
    fit_cf_tre_weights,
    mixture_probabilities,
    select_validation_configuration,
    validate_base_probabilities,
)


def _binary_environment_probabilities():
    labels = []
    environments = []
    rows = []
    for environment in range(9):
        true_prob_model0 = 0.99 if environment < 8 else 0.10
        true_prob_model1 = 0.75
        for label in (0, 1):
            labels.append(label)
            environments.append(f"e{environment}")
            if label == 0:
                rows.append(
                    [
                        [true_prob_model0, 1.0 - true_prob_model0],
                        [true_prob_model1, 1.0 - true_prob_model1],
                    ]
                )
            else:
                rows.append(
                    [
                        [1.0 - true_prob_model0, true_prob_model0],
                        [1.0 - true_prob_model1, true_prob_model1],
                    ]
                )
    return np.asarray(rows), np.asarray(labels), np.asarray(environments)


def test_probability_validation_and_mixture_are_fail_closed():
    probabilities, _, _ = _binary_environment_probabilities()
    validated = validate_base_probabilities(probabilities)
    mixed = mixture_probabilities(validated, [0.25, 0.75])
    assert mixed.shape == (18, 2)
    np.testing.assert_allclose(mixed.sum(axis=1), 1.0)

    malformed = probabilities.copy()
    malformed[0, 0] = [0.3, 0.3]
    with pytest.raises(ValueError, match="sum to one"):
        validate_base_probabilities(malformed)
    with pytest.raises(ValueError, match="sum to one"):
        mixture_probabilities(probabilities, [0.3, 0.3])


def test_empirical_cvar_matches_simple_upper_tail_cases():
    losses = [0.0, 0.0, 1.0, 1.0]
    assert empirical_cvar(losses, 0.5) == pytest.approx(1.0)
    assert empirical_cvar(losses, 0.0) == pytest.approx(0.5)


def test_tail_risk_moves_weight_away_from_average_winner_and_respects_guard():
    probabilities, labels, environments = _binary_environment_probabilities()
    mean_result = fit_cf_tre_weights(
        probabilities,
        labels,
        environments,
        lambda_tail=0.0,
        alpha=0.8,
        delta=0.05,
    )
    tail_result = fit_cf_tre_weights(
        probabilities,
        labels,
        environments,
        lambda_tail=0.75,
        alpha=0.8,
        delta=0.05,
    )
    assert mean_result.weights[0] > mean_result.weights[1]
    assert tail_result.weights[1] > mean_result.weights[1]
    assert tail_result.cvar_loss < mean_result.cvar_loss
    assert tail_result.mean_loss <= tail_result.mean_loss_cap + 1e-8


def test_optimizer_is_deterministic_and_infeasibility_is_explicit():
    probabilities, labels, environments = _binary_environment_probabilities()
    first = fit_cf_tre_weights(
        probabilities,
        labels,
        environments,
        lambda_tail=0.5,
        alpha=0.8,
        delta=0.05,
    )
    second = fit_cf_tre_weights(
        probabilities,
        labels,
        environments,
        lambda_tail=0.5,
        alpha=0.8,
        delta=0.05,
    )
    np.testing.assert_allclose(first.weights, second.weights, rtol=0.0, atol=1e-10)
    assert first.objective == pytest.approx(second.objective, abs=1e-12)
    with pytest.raises(RuntimeError, match="optimization failed"):
        fit_cf_tre_weights(
            probabilities,
            labels,
            environments,
            lambda_tail=0.5,
            alpha=0.8,
            delta=0.0,
            mean_loss_cap=0.0,
        )


def _valid_ledger():
    return [
        OOFPredictionRecord(
            row_id=row,
            group_id=group,
            environment_id=group,
            outer_split="train",
            inner_fold=fold,
            base_model=model,
            fit_group_ids=("g2",) if group == "g1" else ("g1",),
            probabilities=(0.7, 0.3),
        )
        for row, group, fold in (("r1", "g1", 1), ("r2", "g2", 2))
        for model in ("linear", "rbf")
    ]


def test_oof_ledger_requires_complete_unique_train_only_group_exclusion():
    ledger = _valid_ledger()
    audit_oof_prediction_ledger(
        ledger,
        expected_row_ids=("r1", "r2"),
        expected_base_models=("linear", "rbf"),
        require_environment_exclusion=True,
    )
    with pytest.raises(ValueError, match="duplicate"):
        audit_oof_prediction_ledger(
            ledger + [ledger[0]],
            expected_row_ids=("r1", "r2"),
            expected_base_models=("linear", "rbf"),
        )
    with pytest.raises(ValueError, match="non-train"):
        audit_oof_prediction_ledger(
            [replace(ledger[0], outer_split="test"), *ledger[1:]],
            expected_row_ids=("r1", "r2"),
            expected_base_models=("linear", "rbf"),
        )
    with pytest.raises(ValueError, match="leaked"):
        audit_oof_prediction_ledger(
            [replace(ledger[0], fit_group_ids=("g1", "g2")), *ledger[1:]],
            expected_row_ids=("r1", "r2"),
            expected_base_models=("linear", "rbf"),
        )
    with pytest.raises(ValueError, match="does not sum"):
        audit_oof_prediction_ledger(
            [replace(ledger[0], probabilities=(0.4, 0.4)), *ledger[1:]],
            expected_row_ids=("r1", "r2"),
            expected_base_models=("linear", "rbf"),
        )


def test_validation_selection_uses_frozen_lexicographic_order():
    candidates = [
        {
            "name": "larger_tail",
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "cvar_log_loss": 0.5,
            "lambda_tail": 0.5,
            "alpha": 0.8,
            "delta": 0.01,
        },
        {
            "name": "simpler_tie",
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "cvar_log_loss": 0.5,
            "lambda_tail": 0.25,
            "alpha": 0.67,
            "delta": 0.0,
        },
        {
            "name": "lower_accuracy",
            "accuracy": 0.79,
            "macro_f1": 0.9,
            "cvar_log_loss": 0.1,
            "lambda_tail": 0.0,
            "alpha": 0.67,
            "delta": 0.0,
        },
    ]
    assert select_validation_configuration(candidates)["name"] == "simpler_tie"

