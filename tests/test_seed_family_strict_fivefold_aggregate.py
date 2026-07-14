import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_aggregator():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "aggregate_seed_family_strict_fivefold.py"
    )
    name = "aggregate_seed_family_strict_fivefold"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_rows(aggregator, dataset, test_subjects, optimization_seed):
    spec = aggregator.DATASET_SPECS[dataset]
    seed_index = aggregator.OPTIMIZATION_SEEDS.index(optimization_seed)
    logits = []
    labels = []
    subjects = []
    sessions = []
    trials = []
    for subject in test_subjects:
        errors = (subject + seed_index) % 5
        for trial in range(1, spec.trials_per_subject + 1):
            label = (trial - 1) % spec.num_classes
            prediction = (label + 1) % spec.num_classes if trial <= errors else label
            for _ in range(2):
                score = np.full(spec.num_classes, -2.0, dtype=np.float32)
                score[prediction] = 2.0
                logits.append(score)
                labels.append(label)
                subjects.append(subject)
                sessions.append(1)
                trials.append(trial)
    return aggregator.PredictionRows(
        logits=np.asarray(logits, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int16),
        subject=np.asarray(subjects, dtype=np.int16),
        session=np.asarray(sessions, dtype=np.int8),
        trial=np.asarray(trials, dtype=np.int16),
    )


def _write_grid(tmp_path, dataset="seed"):
    aggregator = _load_aggregator()
    spec = aggregator.DATASET_SPECS[dataset]
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold in aggregator.FOLDS:
        split = aggregator.get_subject_fold(fold, partition_seed=2024)
        for optimization_seed in aggregator.OPTIMIZATION_SEEDS:
            rows = _prediction_rows(
                aggregator,
                dataset,
                split["test"],
                optimization_seed,
            )
            stem = f"{dataset}_fold{fold}_seed{optimization_seed}"
            prediction_path = tmp_path / f"{stem}.npz"
            np.savez_compressed(
                prediction_path,
                logits=rows.logits,
                labels=rows.labels,
                subject=rows.subject,
                session=rows.session,
                trial=rows.trial,
            )
            metrics = aggregator.recompute_test_metrics(rows, spec.num_classes)
            payload = {
                "status": "completed",
                "purpose": spec.purpose,
                "protocol": (
                    "SEED-IV session 1 frozen five-fold subject CV"
                    if dataset == "seediv"
                    else "SEED session 1; 9 train / 3 validation / 3 test subjects"
                ),
                "target_access": (
                    aggregator.SEEDIV_TARGET_ACCESS
                    if dataset == "seediv"
                    else aggregator.SEED_TARGET_ACCESS
                ),
                "test_evaluation_count": 1,
                "prediction_units": ["one_second_window", "trial_mean_logit"],
                "libeer_commit": "39dc27e504e14138767b87ce8bce485380fd4f5a",
                "session": 1,
                "fold": fold,
                "partition_seed": 2024,
                "optimization_seed": optimization_seed,
                "num_classes": spec.num_classes,
                "split_one_based": {name: list(value) for name, value in split.items()},
                "split_zero_based": {
                    name: [subject - 1 for subject in value]
                    for name, value in split.items()
                },
                "epochs": 150,
                "batch_size": spec.batch_size,
                "eval_batch_size": 512,
                "learning_rate": spec.learning_rate,
                "best_epoch": 2,
                "best_validation_value": 0.5,
                (
                    "validation_selection" if dataset == "seediv" else "validation_metric"
                ): "pooled_window_macro_f1",
                "sample_counts": {
                    "train": 1,
                    "validation": 1,
                    "test": int(len(rows.labels)),
                },
                "subject_sample_counts": {
                    str(subject): int(np.sum(rows.subject == subject))
                    for subject in split["test"]
                },
                "predictions": str(prediction_path),
                "test": metrics,
            }
            if dataset == "seediv":
                payload["preprocessing"] = {"dataset": "seediv_raw"}
            json_path = tmp_path / f"{stem}.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            paths.append(json_path)
    return aggregator, paths


@pytest.mark.parametrize("dataset", ["seed", "seediv"])
def test_valid_grid_pairs_seeds_within_subject_before_equal_subject_mean(tmp_path, dataset):
    aggregator, paths = _write_grid(tmp_path, dataset)
    result = aggregator.aggregate_files(
        dataset,
        paths,
        bootstrap=300,
        bootstrap_seed=17,
    )
    spec = aggregator.DATASET_SPECS[dataset]
    expected_subject_means = []
    for subject in aggregator.SUBJECTS:
        accuracies = [
            1.0 - ((subject + seed_index) % 5) / spec.trials_per_subject
            for seed_index in range(3)
        ]
        expected_subject_means.append(float(np.mean(accuracies)))

    assert result["primary"]["estimate"] == pytest.approx(np.mean(expected_subject_means))
    assert result["primary"]["n_subjects"] == 15
    assert result["primary"]["confidence_interval"]["status"] == "OK"
    assert result["validation"]["expected_cell_grid_complete"] is True
    assert result["validation"]["json_test_metrics_recomputed_from_predictions"] is True
    assert set(result["per_subject"]["1"]["per_seed"]) == {"2024", "2025", "2026"}
    assert result["per_subject"]["1"]["paired_seed_mean"]["trial_accuracy"] == pytest.approx(
        expected_subject_means[0]
    )
    assert len(result["cells"]) == 15


def test_incomplete_or_duplicate_input_grid_is_rejected(tmp_path):
    aggregator, paths = _write_grid(tmp_path, "seed")
    with pytest.raises(ValueError, match="exactly 15"):
        aggregator.aggregate_files("seed", paths[:-1], bootstrap=20)
    with pytest.raises(ValueError, match="paths must be distinct"):
        aggregator.aggregate_files("seed", paths[:-1] + [paths[0]], bootstrap=20)


def test_frozen_split_and_single_test_evaluation_are_enforced(tmp_path):
    aggregator, paths = _write_grid(tmp_path, "seed")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["split_one_based"]["test"] = list(reversed(payload["split_one_based"]["test"]))
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="split_one_based"):
        aggregator.aggregate_files("seed", paths, bootstrap=20)

    aggregator, paths = _write_grid(tmp_path / "second", "seed")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["test_evaluation_count"] = 2
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test_evaluation_count"):
        aggregator.aggregate_files("seed", paths, bootstrap=20)


def test_json_test_metrics_must_equal_npz_recomputation(tmp_path):
    aggregator, paths = _write_grid(tmp_path, "seed")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["test"]["trial"]["accuracy"] += 0.01
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"test\.trial\.accuracy disagrees"):
        aggregator.aggregate_files("seed", paths, bootstrap=20)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 149, "epochs"),
        ("learning_rate", 0.01, "learning_rate"),
        ("target_access", "target test labels choose the checkpoint", "target_access"),
    ],
)
def test_protocol_hyperparameters_and_target_access_are_frozen(
    tmp_path, field, value, message
):
    aggregator, paths = _write_grid(tmp_path, "seed")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload[field] = value
    paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        aggregator.aggregate_files("seed", paths, bootstrap=20)


def test_metadata_must_contain_each_fold_seed_cell_exactly_once(tmp_path):
    aggregator, paths = _write_grid(tmp_path, "seed")
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["optimization_seed"] = 2025
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate fold/optimization-seed cell"):
        aggregator.aggregate_files("seed", paths, bootstrap=20)


def test_prediction_npz_must_exist_and_cover_only_fold_test_subjects(tmp_path):
    aggregator, paths = _write_grid(tmp_path, "seed")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    Path(payload["predictions"]).unlink()
    with pytest.raises(FileNotFoundError, match="prediction NPZ not found"):
        aggregator.aggregate_files("seed", paths, bootstrap=20)

    aggregator, paths = _write_grid(tmp_path / "second", "seed")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    prediction_path = Path(payload["predictions"])
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["subject"] = arrays["subject"].copy()
    arrays["subject"][0] = 15
    np.savez_compressed(prediction_path, **arrays)
    with pytest.raises(ValueError, match="prediction subjects"):
        aggregator.aggregate_files("seed", paths, bootstrap=20)


def test_bca_uses_subject_rows_and_is_reproducible():
    aggregator = _load_aggregator()
    values = np.linspace(0.4, 0.9, 15)
    first = aggregator.bca_mean_interval(values, n_resamples=500, seed=31)
    second = aggregator.bca_mean_interval(values, n_resamples=500, seed=31)
    assert first == second
    assert first["method"] == "BCa subject-row bootstrap"
    assert first["n_subjects"] == 15
    assert first["low"] < np.mean(values) < first["high"]


def test_cli_refuses_existing_output_before_reading_inputs(tmp_path):
    aggregator = _load_aggregator()
    output = tmp_path / "aggregate.json"
    output.write_text("occupied", encoding="utf-8")
    missing_inputs = [str(tmp_path / f"missing_{index}.json") for index in range(15)]
    argv = [
        "--dataset",
        "seed",
        "--inputs",
        *missing_inputs,
        "--output",
        str(output),
        "--bootstrap",
        "20",
    ]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        aggregator.main(argv)
