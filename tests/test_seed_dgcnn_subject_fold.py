import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seed_dgcnn_subject_fold.py"
    return _load(path, "run_seed_dgcnn_subject_fold")


def _load_legacy_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seed_dgcnn_subject_split.py"
    return _load(path, "run_seed_dgcnn_subject_split_reference")


def _required_cli(tmp_path: Path) -> list[str]:
    return [
        "--libeer-root",
        str(tmp_path / "libeer"),
        "--feature-root",
        str(tmp_path / "features"),
        "--output",
        str(tmp_path / "result.json"),
        "--fold",
        "1",
    ]


def test_fold_one_seed_2024_is_split_compatible_with_original_runner():
    runner = _load_runner()
    legacy = _load_legacy_runner()
    split = runner.subject_fold(1, 2024)

    assert split == runner.REFERENCE_FOLD_ONE_SPLIT
    assert runner.split_zero_based(split) == legacy.libeer_subject_split(2024)


def test_cli_freezes_session_and_separates_partition_and_optimization_seeds(tmp_path):
    runner = _load_runner()
    args = runner.parse_args(_required_cli(tmp_path))

    assert runner.SESSION == 1
    assert not hasattr(args, "session")
    assert args.fold == 1
    assert args.partition_seed == 2024
    assert args.optimization_seed == 2024
    assert args.epochs == 150
    assert args.batch_size == 16
    assert args.eval_batch_size == 512
    assert args.lr == pytest.approx(0.001)
    assert args.checkpoint == args.output.with_suffix(".pt")
    assert args.predictions == args.output.with_suffix(".npz")
    assert not hasattr(args, "metric_choose")

    changed = runner.parse_args(
        _required_cli(tmp_path)
        + ["--partition-seed", "7", "--optimization-seed", "11"]
    )
    assert changed.partition_seed == 7
    assert changed.optimization_seed == 11


def test_cli_rejects_out_of_range_fold_and_configurable_metric(tmp_path):
    runner = _load_runner()
    with pytest.raises(SystemExit):
        runner.parse_args(_required_cli(tmp_path)[:-1] + ["6"])
    with pytest.raises(SystemExit):
        runner.parse_args(_required_cli(tmp_path) + ["--metric-choose", "accuracy"])


def test_validation_checkpoint_metric_is_fixed_to_pooled_window_macro_f1():
    runner = _load_runner()
    assert runner.VALIDATION_METRIC == "macro_f1"
    assert runner.validation_score({"accuracy": 0.99, "macro_f1": 0.41}) == pytest.approx(0.41)


def test_fit_model_cannot_receive_test_rows():
    runner = _load_runner()
    parameters = inspect.signature(runner.fit_model).parameters
    assert "train_rows" in parameters
    assert "validation_rows" in parameters
    assert all("test" not in name for name in parameters)


def test_split_validation_rejects_overlap_and_missing_subjects():
    runner = _load_runner()
    overlap = {
        "train": tuple(range(1, 10)),
        "validation": (9, 10, 11),
        "test": (12, 13, 14),
    }
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_subject_split(overlap)

    missing = {
        "train": tuple(range(1, 10)),
        "validation": (10, 11, 12),
        "test": (13, 14, 16),
    }
    with pytest.raises(ValueError, match="cover subjects"):
        runner.validate_subject_split(missing)


def test_artifact_guards_refuse_overwrite_or_pairwise_alias(tmp_path):
    runner = _load_runner()
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "result.pt"
    predictions = tmp_path / "result.npz"
    runner.require_new_artifacts(output, checkpoint, predictions)

    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.require_new_artifacts(output, checkpoint, predictions)

    occupied_predictions = tmp_path / "occupied.npz"
    occupied_predictions.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.require_new_artifacts(
            tmp_path / "fresh.json",
            tmp_path / "fresh.pt",
            occupied_predictions,
        )

    with pytest.raises(ValueError, match="must be distinct"):
        runner.require_new_artifacts(checkpoint, checkpoint, predictions)
    with pytest.raises(ValueError, match="must be distinct"):
        runner.require_new_artifacts(output, checkpoint, output)
    with pytest.raises(ValueError, match="must be distinct"):
        runner.require_new_artifacts(output, predictions, predictions)


def test_prediction_npz_round_trip_and_exclusive_creation(tmp_path):
    runner = _load_runner()
    rows = runner.EvaluationRows(
        labels=np.array([0, 2]),
        logits=np.array([[3.0, 1.0, 0.0], [0.0, 1.0, 4.0]]),
        subject=np.array([9, 9]),
        session=np.array([1, 1]),
        trial=np.array([4, 5]),
    )
    path = tmp_path / "predictions.npz"

    runner.save_prediction_rows(path, rows)
    with np.load(path) as stored:
        assert set(stored.files) == {"logits", "labels", "subject", "session", "trial"}
        for field in stored.files:
            np.testing.assert_array_equal(stored[field], getattr(rows, field))

    with pytest.raises(FileExistsError):
        runner.save_prediction_rows(path, rows)


def test_evaluation_retains_window_and_trial_metrics_per_subject():
    runner = _load_runner()
    rows = runner.EvaluationRows(
        labels=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        logits=np.array(
            [
                [4.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 3.0],
                [2.0, 0.0], [3.0, 0.0], [0.0, 4.0], [2.0, 0.0],
            ]
        ),
        subject=np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        session=np.ones(8, dtype=int),
        trial=np.array([1, 1, 2, 2, 1, 1, 2, 2]),
    )
    metrics = runner.evaluation_metrics(rows)

    assert set(metrics) == {"window", "trial", "per_subject"}
    assert set(metrics["per_subject"]) == {"1", "2"}
    assert metrics["per_subject"]["1"]["window_count"] == 4
    assert metrics["per_subject"]["2"]["trial_count"] == 2
    assert "macro_f1" in metrics["per_subject"]["1"]["window"]
    assert "accuracy" in metrics["per_subject"]["2"]["trial"]


def test_reference_compatibility_requires_all_frozen_defaults(tmp_path):
    runner = _load_runner()
    args = runner.parse_args(_required_cli(tmp_path))
    split = runner.subject_fold(args.fold, args.partition_seed)
    assert runner.is_reference_fold_one_configuration(args, split) is True

    args.optimization_seed = 99
    assert runner.is_reference_fold_one_configuration(args, split) is False
