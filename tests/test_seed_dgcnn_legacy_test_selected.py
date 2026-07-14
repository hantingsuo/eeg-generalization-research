import importlib.util
from pathlib import Path

import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seed_dgcnn_legacy_test_selected.py"
    spec = importlib.util.spec_from_file_location("run_seed_dgcnn_legacy_test_selected", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _metrics(accuracy, macro_f1=0.5, trial_accuracy=0.5):
    return {
        "window": {"accuracy": accuracy, "macro_f1": macro_f1},
        "trial": {"accuracy": trial_accuracy, "macro_f1": 0.5},
    }


def test_earliest_strict_best_matches_libeer_tie_rule():
    runner = _load_runner()
    history = [
        {"epoch": 1, "metrics": _metrics(0.7)},
        {"epoch": 2, "metrics": _metrics(0.8)},
        {"epoch": 3, "metrics": _metrics(0.8)},
        {"epoch": 4, "metrics": _metrics(0.75)},
    ]
    assert runner.earliest_strict_best(history)["epoch"] == 2


def test_earliest_strict_best_preserves_libeer_zero_initialization_failure():
    runner = _load_runner()
    with pytest.raises(RuntimeError, match="selected no checkpoint"):
        runner.earliest_strict_best([{"epoch": 1, "metrics": _metrics(0.0)}])


def test_aggregate_reports_same_subject_weighting_as_libeer():
    runner = _load_runner()
    rows = [
        {
            "subject": 1,
            "selected": {"epoch": 2, "metrics": _metrics(0.9, 0.8, 1.0)},
            "final_epoch": {"epoch": 80, "metrics": _metrics(0.7)},
        },
        {
            "subject": 1,
            "selected": {"epoch": 3, "metrics": _metrics(0.7, 0.6, 0.5)},
            "final_epoch": {"epoch": 80, "metrics": _metrics(0.6)},
        },
        {
            "subject": 2,
            "selected": {"epoch": 4, "metrics": _metrics(0.8, 0.7, 0.5)},
            "final_epoch": {"epoch": 80, "metrics": _metrics(0.5)},
        },
        {
            "subject": 2,
            "selected": {"epoch": 5, "metrics": _metrics(1.0, 0.9, 1.0)},
            "final_epoch": {"epoch": 80, "metrics": _metrics(0.8)},
        },
    ]
    summary = runner.aggregate_recordings(rows)
    assert summary["selected_window_accuracy_mean_subject_session"] == pytest.approx(0.85)
    assert summary["selected_window_accuracy_mean_subject"] == pytest.approx(0.85)
    assert summary["within_run_selection_uplift_mean"] == pytest.approx(0.2)
    assert summary["selected_epoch_median"] == pytest.approx(3.5)


def test_historical_compatibility_gate_is_inclusive():
    runner = _load_runner()
    rows = []
    for subject in range(1, 16):
        for _session in (1, 2):
            rows.append(
                {
                    "subject": subject,
                    "selected": {"epoch": 1, "metrics": _metrics(0.8748)},
                    "final_epoch": {"epoch": 80, "metrics": _metrics(0.7)},
                }
            )
    summary = runner.aggregate_recordings(rows)
    assert summary["historical_mean_compatibility_within_0_02"] is True
    assert summary["historical_subject_session_sd_compatibility_within_0_02"] is False
    assert summary["historical_pair_compatibility_within_0_02"] is False
    assert summary["historical_table_anchor_exact_at_4_decimals"] is False


def test_exact_four_decimal_table_anchor_uses_30_unit_population_sd():
    runner = _load_runner()
    values = [0.8948 - 0.0849] * 15 + [0.8948 + 0.0849] * 15
    rows = [
        {
            "subject": index % 15 + 1,
            "selected": {"epoch": 1, "metrics": _metrics(value)},
            "final_epoch": {"epoch": 80, "metrics": _metrics(0.7)},
        }
        for index, value in enumerate(values)
    ]
    summary = runner.aggregate_recordings(rows)
    assert summary["selected_window_accuracy_mean_subject_session"] == pytest.approx(0.8948)
    assert summary["selected_window_accuracy_sd_subject_session_population"] == pytest.approx(0.0849)
    assert summary["historical_table_anchor_exact_at_4_decimals"] is True


def test_output_guard_refuses_overwrite(tmp_path):
    runner = _load_runner()
    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.require_new_output(output)
