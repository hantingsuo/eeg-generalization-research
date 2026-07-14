import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "experiments" / "run_seediv_dgcnn_subject_split.py"
    spec = importlib.util.spec_from_file_location("run_seediv_dgcnn_subject_split", path)
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


def test_libeer_subject_split_matches_seed_2024_reference():
    runner = _load_runner()
    split = runner.libeer_subject_split(2024)
    assert split["test"] == [8, 0, 5]
    assert split["validation"] == [13, 12, 1]
    assert split["train"] == [14, 10, 6, 3, 4, 9, 11, 2, 7]
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert set().union(*map(set, split.values())) == set(range(15))


def test_concatenate_rows_preserves_provenance_order():
    runner = _load_runner()
    first = _rows(runner, subject=1, labels=[0, 1])
    second = _rows(runner, subject=2, labels=[2, 3], feature_offset=100.0)
    joined = runner.concatenate_rows([first, second])
    assert joined.features.shape == (4, 2, 2)
    assert joined.labels.tolist() == [0, 1, 2, 3]
    assert joined.subject.tolist() == [1, 1, 2, 2]
    assert joined.session.tolist() == [1, 1, 1, 1]
    assert joined.trial.tolist() == [1, 2, 1, 2]


def test_four_class_window_trial_and_worst_subject_evaluation():
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
    assert metrics["window"]["confusion"] == (2 * np.eye(4, dtype=int)).tolist()
    assert metrics["worst"] == {
        "window_subject_accuracy": 1.0,
        "trial_subject_accuracy": 1.0,
    }
    assert set(metrics["per_subject"]) == {"1", "2"}


def test_protocol_defaults_are_frozen_to_libeer_seediv_table():
    runner = _load_runner()
    args = runner.parse_args(
        ["--libeer-root", "libeer", "--dataset-root", "seediv", "--output", "result.json"]
    )
    assert args.expected_commit == "39dc27e"
    assert args.session == 1
    assert args.epochs == 150
    assert args.batch_size == 32
    assert args.lr == pytest.approx(0.0015)
    assert args.seed == 2024
    assert args.metric_choose == "macro_f1"


def test_existing_output_or_checkpoint_is_never_overwritten(tmp_path):
    runner = _load_runner()
    output, checkpoint = runner.output_paths(tmp_path / "result.json", None)
    runner.ensure_outputs_absent(output, checkpoint)

    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.ensure_outputs_absent(output, checkpoint)

    output.unlink()
    checkpoint.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.ensure_outputs_absent(output, checkpoint)
