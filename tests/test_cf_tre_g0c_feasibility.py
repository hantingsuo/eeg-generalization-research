import numpy as np

from experiments.run_cf_tre_g0c_feasibility import _validation_metrics


def test_validation_metrics_are_subject_equal_and_environment_based():
    units = []
    for subject in range(1, 16):
        for session in range(1, 4):
            labels = np.array([0, 1])
            probabilities = np.array([[[0.9, 0.1]], [[0.1, 0.9]]])
            units.append({"subject": subject, "session": session, "environment": f"s{session}_sub{subject}", "labels": labels, "trials": np.array([1, 2]), "probabilities": probabilities})
    result = _validation_metrics(units, np.array([1.0]), 0.8)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert len(result["environment_losses"]) == 45
