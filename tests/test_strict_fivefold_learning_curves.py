import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.aggregate_strict_fivefold_learning_curves import summarize
from experiments.run_strict_fivefold_learning_curve_batch import schedule
from experiments.run_strict_fivefold_learning_curve_cell import (
    git_commit,
    history_reference_values,
    load_dgcnn_module,
    states_equal,
)


def test_learning_curve_schedule_is_complete_and_unique():
    cells = schedule()
    assert len(cells) == 30
    assert len({cell["cell_id"] for cell in cells}) == 30
    assert {
        (cell["dataset"], cell["fold"], cell["optimization_seed"]) for cell in cells
    } == {
        (dataset, fold, seed)
        for dataset in ("seed", "seediv")
        for fold in range(1, 6)
        for seed in (2024, 2025, 2026)
    }


def test_history_reference_values_handles_both_runner_schemas():
    assert history_reference_values(
        "seed",
        {
            "train_loss": 1.0,
            "validation_window": {"accuracy": 0.5, "macro_f1": 0.4},
        },
    ) == (1.0, 0.5, 0.4)
    assert history_reference_values(
        "seediv",
        {
            "train_loss": 2.0,
            "validation_accuracy": 0.3,
            "validation_macro_f1": 0.2,
        },
    ) == (2.0, 0.3, 0.2)


def test_state_equality_is_exact():
    left = {"a": torch.tensor([1.0, 2.0])}
    right = {"a": torch.tensor([1.0, 2.0])}
    assert states_equal(left, right) == (True, [])
    ok, issues = states_equal(left, {"a": torch.tensor([1.0, 2.1])})
    assert not ok
    assert issues == ["a: tensor values differ"]


def test_curve_summary_is_descriptive():
    row = summarize(np.asarray([0.1, 0.2, 0.3]))
    assert row["mean"] == pytest.approx(0.2)
    assert row["median"] == pytest.approx(0.2)
    assert row["min"] == 0.1
    assert row["max"] == 0.3


def test_git_commit_scopes_safe_directory_to_read_only_command(monkeypatch):
    observed = {}
    repo = Path("C:/tmp/pinned-libeer")

    class Completed:
        stdout = "39dc27e\n"

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(
        "experiments.run_strict_fivefold_learning_curve_cell.subprocess.run",
        fake_run,
    )
    assert git_commit(repo) == "39dc27e"
    assert observed["command"][:2] == ["git", "-c"]
    assert observed["command"][2] == f"safe.directory={repo.resolve().as_posix()}"
    assert observed["command"][-3:] == [str(repo), "rev-parse", "HEAD"]
    assert observed["kwargs"]["check"] is True


def test_dgcnn_loader_does_not_import_models_package(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "__init__.py").write_text(
        "raise RuntimeError('aggregate package must not be imported')\n",
        encoding="utf-8",
    )
    (model_dir / "DGCNN.py").write_text(
        "class DGCNN: pass\nclass NewSparseL2Regularization: pass\n",
        encoding="utf-8",
    )
    module = load_dgcnn_module(tmp_path)
    assert module.DGCNN.__name__ == "DGCNN"
    assert module.NewSparseL2Regularization.__name__ == "NewSparseL2Regularization"
