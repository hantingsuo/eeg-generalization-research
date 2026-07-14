from __future__ import annotations

from pathlib import Path

from experiments.run_cf_tre_g0b_batch import build_cells, cell_command


def test_batch_cell_counts_and_ids_are_complete() -> None:
    classical = build_cells("classical", Path("out"), Path("logs"))
    deep = build_cells("deep", Path("out"), Path("logs"))
    assert len(classical) == 90
    assert len(deep) == 180
    assert len({cell.cell_id for cell in classical}) == 90
    assert len({cell.cell_id for cell in deep}) == 180
    assert classical[0].cell_id == "classical_seed_s01_sub01"
    assert classical[-1].cell_id == "classical_seediv_s03_sub15"
    assert deep[0].cell_id == "deep_dgcnn_seed_s01_sub01"
    assert deep[-1].cell_id == "deep_gcbnet_seediv_s03_sub15"


def test_deep_batch_command_is_fixed_to_150_epochs_and_test_enabled() -> None:
    cell = build_cells("deep", Path("out"), Path("logs"))[0]
    command = cell_command(cell)
    assert command[command.index("--epochs") + 1] == "150"
    assert "--validation-only" not in command
    assert command[command.index("--optimization-seed") + 1] == "2024"
