import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seediv_dgcnn_subject_fold.py"
    spec = importlib.util.spec_from_file_location("run_seediv_dgcnn_subject_fold", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rows(runner, *, subject, labels, feature_offset=0.0):
    labels = np.asarray(labels, dtype=np.int64)
    count = len(labels)
    return runner.WindowRows(
        features=np.arange(count * 4, dtype=np.float32).reshape(count, 2, 2) + feature_offset,
        labels=labels,
        subject=np.full(count, subject, dtype=np.int16),
        session=np.ones(count, dtype=np.int8),
        trial=np.arange(1, count + 1, dtype=np.int16),
    )


def test_fold_one_is_regression_compatible_with_original_seed_2024_split():
    runner = _load_runner()
    split = runner.get_subject_fold(1, partition_seed=2024)
    assert split == {
        "test": (9, 1, 6),
        "validation": (14, 13, 2),
        "train": (15, 11, 7, 4, 5, 10, 12, 3, 8),
    }
    assert {
        name: [subject - 1 for subject in subjects] for name, subjects in split.items()
    } == {
        "test": [8, 0, 5],
        "validation": [13, 12, 1],
        "train": [14, 10, 6, 3, 4, 9, 11, 2, 7],
    }


def test_five_folds_cover_each_subject_once_as_test_and_validation():
    runner = _load_runner()
    folds = [runner.get_subject_fold(fold, partition_seed=2024) for fold in range(1, 6)]
    assert sorted(subject for split in folds for subject in split["test"]) == list(range(1, 16))
    assert sorted(subject for split in folds for subject in split["validation"]) == list(range(1, 16))
    for split in folds:
        assert len(split["train"]) == 9
        assert len(split["validation"]) == 3
        assert len(split["test"]) == 3
        assert set(split["train"]).isdisjoint(split["validation"])
        assert set(split["train"]).isdisjoint(split["test"])
        assert set(split["validation"]).isdisjoint(split["test"])


def test_parser_separates_partition_and_optimization_seeds_and_freezes_defaults():
    runner = _load_runner()
    args = runner.parse_args(
        [
            "--libeer-root", "libeer",
            "--dataset-root", "seediv",
            "--output", "result.json",
            "--fold", "3",
            "--partition-seed", "11",
            "--optimization-seed", "22",
        ]
    )
    assert args.fold == 3
    assert args.partition_seed == 11
    assert args.optimization_seed == 22
    assert not hasattr(args, "seed")
    assert args.session == 1
    assert args.epochs == 150
    assert args.batch_size == 32
    assert args.eval_batch_size == 512
    assert args.lr == pytest.approx(0.0015)
    assert not hasattr(args, "metric_choose")


def test_validation_checkpoint_score_uses_only_pooled_window_macro_f1():
    runner = _load_runner()
    assert runner.VALIDATION_SELECTION == "pooled_window_macro_f1"
    assert runner.validation_checkpoint_score({"accuracy": 0.99, "macro_f1": 0.25}) == 0.25
    assert runner.validation_checkpoint_score({"accuracy": 0.10, "macro_f1": 0.75}) == 0.75


def test_test_partition_is_materialized_only_after_checkpoint_selection():
    runner = _load_runner()
    source = inspect.getsource(runner.main)
    checkpoint_fixed = source.index("model.load_state_dict(best_state)")
    test_loaded = source.index('partitions["test"] = load_partition("test")')

    assert test_loaded > checkpoint_fixed
    assert '"train": load_partition("train")' in source[:checkpoint_fixed]
    assert '"validation": load_partition("validation")' in source[:checkpoint_fixed]
    assert 'load_partition("test")' not in source[:checkpoint_fixed]


def test_data_and_cache_roots_are_resolved_before_working_directory_changes():
    runner = _load_runner()
    source = inspect.getsource(runner.main)
    cwd_change = source.index("os.chdir(libeer_code)")

    assert source.index("dataset_root = args.dataset_root.resolve()") < cwd_change
    assert source.index("cache_root = args.cache_root.resolve()") < cwd_change
    assert "load_seediv_de_lds_subject(\n                dataset_root," in source
    assert "cache_root=cache_root" in source


def test_concatenate_rows_preserves_subject_and_trial_order():
    runner = _load_runner()
    first = _rows(runner, subject=9, labels=[0, 1])
    second = _rows(runner, subject=1, labels=[2, 3], feature_offset=100.0)
    joined = runner.concatenate_rows([first, second])
    assert joined.features.shape == (4, 2, 2)
    assert joined.labels.tolist() == [0, 1, 2, 3]
    assert joined.subject.tolist() == [9, 9, 1, 1]
    assert joined.trial.tolist() == [1, 2, 1, 2]


def test_four_class_evaluation_retains_per_subject_and_worst_metrics():
    runner = _load_runner()
    labels = np.tile(np.arange(4, dtype=np.int64), 2)
    logits = np.full((8, 4), -2.0, dtype=np.float32)
    logits[np.arange(8), labels] = 2.0
    rows = runner.EvaluationRows(
        labels=labels,
        logits=logits,
        subject=np.repeat([1, 2], 4).astype(np.int16),
        session=np.ones(8, dtype=np.int8),
        trial=np.tile(np.arange(1, 5), 2).astype(np.int16),
    )
    metrics = runner.evaluation_metrics(rows)
    assert metrics["window"]["accuracy"] == 1.0
    assert metrics["window"]["macro_f1"] == 1.0
    assert metrics["trial"]["accuracy"] == 1.0
    assert metrics["worst"] == {
        "window_subject_accuracy": 1.0,
        "trial_subject_accuracy": 1.0,
    }
    assert set(metrics["per_subject"]) == {"1", "2"}


def test_output_paths_default_to_three_distinct_artifacts(tmp_path):
    runner = _load_runner()
    output, checkpoint, predictions = runner.output_paths(tmp_path / "result.json", None, None)
    assert output == (tmp_path / "result.json").resolve()
    assert checkpoint == (tmp_path / "result.pt").resolve()
    assert predictions == (tmp_path / "result.npz").resolve()


@pytest.mark.parametrize(
    ("output_name", "checkpoint_name", "predictions_name"),
    [
        ("same.json", "same.json", "predictions.npz"),
        ("result.json", "same.bin", "same.bin"),
        ("same.npz", "checkpoint.pt", "same.npz"),
    ],
)
def test_output_checkpoint_and_predictions_paths_must_be_distinct(
    tmp_path,
    output_name,
    checkpoint_name,
    predictions_name,
):
    runner = _load_runner()
    with pytest.raises(ValueError, match="must use distinct paths"):
        runner.output_paths(
            tmp_path / output_name,
            tmp_path / checkpoint_name,
            tmp_path / predictions_name,
        )


@pytest.mark.parametrize("existing_index", [0, 1, 2])
def test_existing_json_checkpoint_or_predictions_is_never_overwritten(tmp_path, existing_index):
    runner = _load_runner()
    output, checkpoint, predictions = runner.output_paths(tmp_path / "result.json", None, None)
    paths = (output, checkpoint, predictions)
    runner.ensure_outputs_absent(*paths)

    paths[existing_index].write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.ensure_outputs_absent(*paths)
