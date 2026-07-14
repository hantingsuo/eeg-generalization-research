import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_batch():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "run_seed_family_strict_fivefold_batch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_seed_family_strict_fivefold_batch", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python=Path(sys.executable).resolve(),
        libeer_root=(tmp_path / "libeer").resolve(),
        expected_commit="39dc27e",
        seed_feature_root=(tmp_path / "seed_features").resolve(),
        seediv_dataset_root=(tmp_path / "seediv").resolve(),
        seediv_cache_root=(tmp_path / "seediv_cache").resolve(),
        output_root=(tmp_path / "results" / "strict_fivefold").resolve(),
        log_root=(tmp_path / "logs" / "strict_fivefold").resolve(),
        stage1_approval=None,
        device="cuda",
        dry_run=False,
    )


def _write_complete_cell(batch, cell, artifacts, *, batch_log=False, code_hashes=None, command=None):
    subjects = batch.FROZEN_SPLITS[cell.fold]["test"]
    classes = batch.DATASET_SETTINGS[cell.dataset]["classes"]
    subject = np.repeat(np.asarray(subjects, dtype=np.int16), 2)
    count = len(subject)
    labels = np.arange(count, dtype=np.int64) % classes
    logits = np.full((count, classes), -1.0, dtype=np.float32)
    logits[np.arange(count), labels] = 1.0

    artifacts.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.log.parent.mkdir(parents=True, exist_ok=True)
    artifacts.checkpoint.write_bytes(b"checkpoint")
    with artifacts.predictions.open("xb") as handle:
        np.savez_compressed(
            handle,
            logits=logits,
            labels=labels,
            subject=subject,
            session=np.ones(count, dtype=np.int8),
            trial=np.tile([1, 2], len(subjects)).astype(np.int16),
        )

    settings = batch.DATASET_SETTINGS[cell.dataset]
    payload = {
        "status": "completed",
        "purpose": settings["purpose"],
        "test_evaluation_count": 1,
        "session": 1,
        "fold": cell.fold,
        "partition_seed": batch.PARTITION_SEED,
        "optimization_seed": cell.optimization_seed,
        "split_one_based": {
            name: list(values) for name, values in batch.FROZEN_SPLITS[cell.fold].items()
        },
        "epochs": settings["epochs"],
        "batch_size": settings["batch_size"],
        "eval_batch_size": settings["eval_batch_size"],
        "learning_rate": settings["learning_rate"],
        settings["validation_field"]: settings["validation_value"],
        "best_epoch": 1,
        "best_validation_value": 0.25,
        "checkpoint": str(artifacts.checkpoint.resolve()),
        "predictions": str(artifacts.predictions.resolve()),
        "history": [{} for _ in range(settings["epochs"])],
        "sample_counts": {"train": 18, "validation": 6, "test": count},
        "subject_sample_counts": {
            str(subject_id): int(np.sum(subject == subject_id)) for subject_id in subjects
        },
        "test": {
            "window": {"accuracy": 1.0, "macro_f1": 1.0},
            "trial": {"accuracy": 1.0, "macro_f1": 1.0},
            "per_subject": {str(subject_id): {} for subject_id in subjects},
        },
    }
    artifacts.output.write_text(json.dumps(payload), encoding="utf-8")
    if not batch_log:
        artifacts.log.write_text("external stage-1 log\n", encoding="utf-8")
    else:
        assert command is not None and code_hashes is not None
        events = [
            {
                "event": "cell_start",
                "protocol_version": batch.PROTOCOL_VERSION,
                "cell": cell.metadata(),
                "command": list(command),
                "code_sha256": dict(code_hashes),
            },
            {"event": "cell_exit", "cell_id": cell.cell_id, "return_code": 0},
        ]
        artifacts.log.write_text(
            "".join(
                batch.BATCH_EVENT_PREFIX + json.dumps(event, sort_keys=True) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )
    return payload


def test_schedule_freezes_all_30_cells_in_four_stages():
    batch = _load_batch()
    cells = batch.frozen_schedule()

    assert len(cells) == 30
    assert [(cell.dataset, cell.fold, cell.optimization_seed, cell.stage) for cell in cells[:2]] == [
        ("seed", 1, 2024, 1),
        ("seediv", 1, 2024, 1),
    ]
    assert [(cell.dataset, cell.fold) for cell in cells[2:10]] == [
        (dataset, fold) for fold in range(2, 6) for dataset in ("seed", "seediv")
    ]
    assert {cell.optimization_seed for cell in cells[2:10]} == {2024}
    assert {cell.optimization_seed for cell in cells[10:20]} == {2025}
    assert {cell.optimization_seed for cell in cells[20:30]} == {2026}
    assert all(cell.stage == 3 for cell in cells[10:20])
    assert all(cell.stage == 4 for cell in cells[20:30])
    assert {
        (cell.dataset, cell.fold, cell.optimization_seed) for cell in cells
    } == {
        (dataset, fold, seed)
        for dataset in batch.DATASETS
        for fold in range(1, 6)
        for seed in batch.OPTIMIZATION_SEEDS
    }


def test_artifact_names_match_frozen_manual_stage1_layout_and_are_unique(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    first = batch.artifact_paths(batch.frozen_schedule()[0], args.output_root, args.log_root)

    assert first.output == args.output_root / "seed" / "seed_fold1_opt2024.json"
    assert first.checkpoint == args.output_root / "seed" / "seed_fold1_opt2024.pt"
    assert first.predictions == args.output_root / "seed" / "seed_fold1_opt2024.npz"
    assert first.log == args.log_root / "seed_fold1_opt2024.log"
    all_paths = [
        path
        for cell in batch.frozen_schedule()
        for path in batch.artifact_paths(cell, args.output_root, args.log_root).all_paths()
    ]
    assert len(all_paths) == 120
    assert len(set(all_paths)) == 120


@pytest.mark.parametrize(
    ("dataset", "expected_data_flag", "batch_size", "learning_rate"),
    [("seed", "--feature-root", "16", "0.001"), ("seediv", "--dataset-root", "32", "0.0015")],
)
def test_command_has_all_explicit_frozen_paths_and_hyperparameters(
    tmp_path, dataset, expected_data_flag, batch_size, learning_rate
):
    batch = _load_batch()
    args = _args(tmp_path)
    cell = batch.Cell(dataset, 3, 2025, 3)
    artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
    command = batch.build_command(cell, artifacts, args, Path(__file__).parents[1])

    assert command[0] == str(args.python)
    for flag in (
        "--libeer-root",
        expected_data_flag,
        "--output",
        "--checkpoint",
        "--predictions",
        "--fold",
        "--partition-seed",
        "--optimization-seed",
        "--epochs",
        "--batch-size",
        "--eval-batch-size",
        "--lr",
        "--device",
    ):
        assert flag in command
    assert command[command.index("--fold") + 1] == "3"
    assert command[command.index("--partition-seed") + 1] == "2024"
    assert command[command.index("--optimization-seed") + 1] == "2025"
    assert command[command.index("--epochs") + 1] == "150"
    assert command[command.index("--batch-size") + 1] == batch_size
    assert command[command.index("--eval-batch-size") + 1] == "512"
    assert command[command.index("--lr") + 1] == learning_rate
    if dataset == "seediv":
        assert command[command.index("--cache-root") + 1] == str(args.seediv_cache_root)


def test_partial_cell_stops_and_requires_manual_quarantine(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    cell = batch.frozen_schedule()[0]
    artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
    command = batch.build_command(cell, artifacts, args, Path(__file__).parents[1])
    artifacts.output.parent.mkdir(parents=True)
    artifacts.checkpoint.write_bytes(b"partial")

    with pytest.raises(batch.PartialCellError, match="manually quarantine"):
        batch.inspect_cell(cell, artifacts, command, {"code": "hash"})


def test_complete_external_stage1_cell_is_validated_and_resume_safe(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    cell = batch.frozen_schedule()[0]
    artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
    command = batch.build_command(cell, artifacts, args, Path(__file__).parents[1])
    _write_complete_cell(batch, cell, artifacts)

    assert batch.inspect_cell(cell, artifacts, command, {"code": "hash"}) == "complete"


def test_invalid_split_or_npz_subjects_never_resume_skip(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    cell = batch.frozen_schedule()[0]
    artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
    command = batch.build_command(cell, artifacts, args, Path(__file__).parents[1])
    payload = _write_complete_cell(batch, cell, artifacts)

    payload["split_one_based"]["test"] = [1, 2, 3]
    artifacts.output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(batch.BatchIntegrityError, match="split mismatch"):
        batch.inspect_cell(cell, artifacts, command, {"code": "hash"})

    payload["split_one_based"] = {
        name: list(values) for name, values in batch.FROZEN_SPLITS[cell.fold].items()
    }
    artifacts.output.write_text(json.dumps(payload), encoding="utf-8")
    with np.load(artifacts.predictions) as old:
        arrays = {name: old[name] for name in old.files}
    arrays["subject"] = np.full_like(arrays["subject"], 1)
    artifacts.predictions.unlink()
    np.savez_compressed(artifacts.predictions, **arrays)
    with pytest.raises(batch.BatchIntegrityError, match="held-out subjects mismatch"):
        batch.inspect_cell(cell, artifacts, command, {"code": "hash"})


def test_non_stage1_complete_cell_requires_batch_provenance_events(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    cell = batch.Cell("seed", 2, 2024, 2)
    artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
    command = batch.build_command(cell, artifacts, args, Path(__file__).parents[1])
    _write_complete_cell(batch, cell, artifacts)

    with pytest.raises(batch.BatchIntegrityError, match="missing batch provenance"):
        batch.inspect_cell(cell, artifacts, command, {"code": "hash"})


def test_batch_log_binds_command_code_hashes_and_zero_return_code(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    cell = batch.Cell("seediv", 2, 2024, 2)
    artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
    command = batch.build_command(cell, artifacts, args, Path(__file__).parents[1])
    hashes = {"runner": "abc"}
    _write_complete_cell(
        batch,
        cell,
        artifacts,
        batch_log=True,
        code_hashes=hashes,
        command=command,
    )

    assert batch.inspect_cell(cell, artifacts, command, hashes) == "complete"
    with pytest.raises(batch.BatchIntegrityError, match="code hash drift"):
        batch.inspect_cell(cell, artifacts, command, {"runner": "changed"})


def test_tee_preserves_child_return_code_and_merges_stdout_stderr(tmp_path):
    batch = _load_batch()
    cell = batch.Cell("seed", 2, 2024, 2)
    log = tmp_path / "child.log"
    command = [
        sys.executable,
        "-c",
        "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr); sys.exit(7)",
    ]

    returncode = batch.tee_command(cell, command, log, {"code": "hash"}, cwd=tmp_path)
    stored = log.read_text(encoding="utf-8")
    assert returncode == 7
    assert "stdout-line" in stored
    assert "stderr-line" in stored
    assert '"return_code": 7' in stored


def test_stage1_approval_binds_exact_two_artifact_triplets(tmp_path):
    batch = _load_batch()
    args = _args(tmp_path)
    pairs = []
    for cell in batch.frozen_schedule()[:2]:
        artifacts = batch.artifact_paths(cell, args.output_root, args.log_root)
        _write_complete_cell(batch, cell, artifacts)
        pairs.append((cell, artifacts))
    approval = batch.stage1_approval_template(pairs)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    batch.validate_stage1_approval(approval_path, pairs, batch.FROZEN_CODE_SHA256)
    pairs[0][1].checkpoint.write_bytes(b"mutated")
    with pytest.raises(batch.BatchIntegrityError, match="artifact hash mismatch"):
        batch.validate_stage1_approval(approval_path, pairs, batch.FROZEN_CODE_SHA256)


def test_dry_run_lists_all_cells_without_creating_artifacts(tmp_path, monkeypatch, capsys):
    batch = _load_batch()
    args = _args(tmp_path)
    args.dry_run = True
    fake_hashes = {name: "a" * 64 for name in batch.FROZEN_CODE_SHA256}
    monkeypatch.setattr(batch, "verify_frozen_code", lambda _root: fake_hashes)

    assert batch.run_batch(args) == 0
    output = capsys.readouterr().out
    assert output.count("DRY-RUN") == 30
    assert output.count("stage1-gated") == 28
    assert not args.output_root.exists()
    assert not args.log_root.exists()


def test_frozen_split_constants_match_shared_fold_helper():
    batch = _load_batch()
    from pcma.data.seed_folds import get_subject_fold

    for fold in range(1, 6):
        assert batch.FROZEN_SPLITS[fold] == get_subject_fold(fold, partition_seed=2024)
