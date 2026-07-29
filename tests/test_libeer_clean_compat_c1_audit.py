import numpy as np

from experiments.audit_libeer_clean_compat_c1 import _metric


def test_metric_reconstructs_accuracy_and_macro_f1():
    labels = np.array([0, 1, 2, 2])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]])
    result = _metric(labels, probabilities)
    assert result["accuracy"] == 0.75
    assert 0.0 < result["macro_f1"] < 1.0


def test_metric_rejects_non_simplex_probabilities():
    labels = np.array([0, 1])
    probabilities = np.array([[0.8, 0.8], [0.2, 0.8]])
    try:
        _metric(labels, probabilities)
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("expected simplex failure")
