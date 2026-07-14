import importlib.util
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seed_dgcnn_fixed_epoch_subject_dependent.py"
    spec = importlib.util.spec_from_file_location("run_seed_dgcnn_fixed_epoch_subject_dependent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_recordings_uses_subject_session_units():
    runner = _load_runner()
    rows = [
        {"metrics": {"window": {"accuracy": 0.8, "macro_f1": 0.7}, "trial": {"accuracy": 0.9, "macro_f1": 0.8}}},
        {"metrics": {"window": {"accuracy": 0.6, "macro_f1": 0.5}, "trial": {"accuracy": 0.7, "macro_f1": 0.6}}},
    ]
    result = runner.aggregate_recordings(rows)
    assert result["num_subject_sessions"] == 2
    assert result["window_accuracy_mean"] == pytest.approx(0.7)
    assert result["window_accuracy_sd_population"] == pytest.approx(0.1)
    assert result["worst_subject_session_trial_accuracy"] == pytest.approx(0.7)
