import pytest

from experiments.run_cf_tre_g1_outer_test import bca_mean_interval, evaluate_gates


def test_bca_constant_paired_subject_rows_is_exact():
    result = bca_mean_interval([0.02] * 15, n_resamples=100, seed=7)
    assert result["status"] == "DEGENERATE_CONSTANT"
    assert result["low"] == pytest.approx(0.02)
    assert result["high"] == pytest.approx(0.02)


def test_g1_gate_requires_every_frozen_condition():
    datasets = {}
    for dataset, accuracy in (("seed", 0.85), ("seediv", 0.56)):
        datasets[dataset] = {
            "methods": {
                "tail_risk": {"subject_equal_accuracy": accuracy, "subject_equal_macro_f1": 0.70, "environment_cvar_log_loss": 0.90},
                "dgcnn": {"subject_equal_accuracy": accuracy - 0.02, "subject_equal_macro_f1": 0.70},
                "mean_risk": {"environment_cvar_log_loss": 0.91},
            },
            "paired_tail_minus_dgcnn_accuracy_bca": {"low": 0.001},
        }
    gates, _ = evaluate_gates(datasets, integrity_pass=True)
    assert all(gates.values())
    datasets["seediv"]["methods"]["tail_risk"]["subject_equal_accuracy"] = 0.54
    gates, _ = evaluate_gates(datasets, integrity_pass=True)
    assert gates["absolute_accuracy"] is False
