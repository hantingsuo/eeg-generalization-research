import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_auditor():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "audit_seed_family_fold1_regression.py"
    )
    name = "audit_seed_family_fold1_regression"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_arrays(auditor, dataset):
    num_classes = auditor.DATASET_SPECS[dataset].num_classes
    logits = []
    labels = []
    subjects = []
    sessions = []
    trials = []
    for subject in auditor.FOLD_ONE_SPLIT["test"]:
        for trial in range(1, num_classes + 1):
            label = trial - 1
            for _ in range(2):
                score = np.full(num_classes, -2.0, dtype=np.float32)
                score[label] = 3.0
                logits.append(score)
                labels.append(label)
                subjects.append(subject)
                sessions.append(1)
                trials.append(trial)
    return {
        "logits": np.asarray(logits, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int16),
        "subject": np.asarray(subjects, dtype=np.int16),
        "session": np.asarray(sessions, dtype=np.int8),
        "trial": np.asarray(trials, dtype=np.int16),
    }


def _histories(auditor, dataset):
    spec = auditor.DATASET_SPECS[dataset]
    reference = []
    candidate = []
    for epoch in range(1, 151):
        validation_f1 = (
            spec.reference_best_validation_value
            if epoch == spec.reference_best_epoch
            else 0.1 - epoch * 1e-5
        )
        validation_accuracy = 0.2 + epoch * 1e-4
        common = {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
        }
        reference.append({**common, "seconds": float(epoch)})
        if dataset == "seed":
            candidate.append(
                {
                    "epoch": epoch,
                    "train_loss": common["train_loss"],
                    "validation_window": {
                        "accuracy": validation_accuracy,
                        "macro_f1": validation_f1,
                    },
                    "checkpoint_metric": "macro_f1",
                    "checkpoint_value": validation_f1,
                    "checkpoint_updated": epoch == spec.reference_best_epoch,
                    "seconds": float(epoch) + 1000.0,
                }
            )
        else:
            candidate.append({**common, "seconds": float(epoch) + 1000.0})
    return reference, candidate


def _write_pair(tmp_path, dataset):
    auditor = _load_auditor()
    spec = auditor.DATASET_SPECS[dataset]
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference_json_path = tmp_path / "reference.json"
    reference_pt_path = tmp_path / "reference.pt"
    candidate_json_path = tmp_path / "candidate.json"
    candidate_pt_path = tmp_path / "candidate.pt"
    predictions_path = tmp_path / "candidate.npz"

    arrays = _prediction_arrays(auditor, dataset)
    np.savez_compressed(predictions_path, **arrays)
    metrics = auditor.recompute_test_metrics(arrays, spec.num_classes)
    reference_history, candidate_history = _histories(auditor, dataset)
    sample_counts = {"train": 90, "validation": 30, "test": len(arrays["labels"])}
    subject_counts = {
        str(subject): int(np.sum(arrays["subject"] == subject))
        for subject in auditor.FOLD_ONE_SPLIT["test"]
    }

    reference = {
        "status": "completed",
        "purpose": "protocol_matched_target_free_baseline",
        "session": 1,
        "seed": 2024,
        "libeer_commit": "39dc27e",
        "prediction_units": ["one_second_window", "trial_mean_logit"],
        "split_one_based": auditor.FOLD_ONE_SPLIT,
        "split_zero_based": auditor.FOLD_ONE_SPLIT_ZERO_BASED,
        "epochs": 150,
        "batch_size": spec.batch_size,
        "eval_batch_size": 512,
        "learning_rate": spec.learning_rate,
        "metric_choose": "macro_f1",
        "best_epoch": spec.reference_best_epoch,
        "best_validation_value": spec.reference_best_validation_value,
        "sample_counts": sample_counts,
        "subject_sample_counts": subject_counts,
        "checkpoint": str(reference_pt_path.resolve()),
        "history": reference_history,
        "test": metrics,
    }
    candidate = {
        "status": "completed",
        "purpose": spec.candidate_purpose,
        "test_evaluation_count": 1,
        "prediction_units": ["one_second_window", "trial_mean_logit"],
        "session": 1,
        "fold": 1,
        "partition_seed": 2024,
        "optimization_seed": 2024,
        "libeer_commit": "39dc27e",
        "split_one_based": auditor.FOLD_ONE_SPLIT,
        "split_zero_based": auditor.FOLD_ONE_SPLIT_ZERO_BASED,
        "epochs": 150,
        "batch_size": spec.batch_size,
        "eval_batch_size": 512,
        "learning_rate": spec.learning_rate,
        "best_epoch": spec.reference_best_epoch,
        "best_validation_value": spec.reference_best_validation_value,
        "sample_counts": sample_counts,
        "subject_sample_counts": subject_counts,
        "checkpoint": str(candidate_pt_path.resolve()),
        "predictions": str(predictions_path.resolve()),
        "history": candidate_history,
        "test": deepcopy(metrics),
    }
    if dataset == "seed":
        candidate["validation_metric"] = "pooled_window_macro_f1"
        candidate["fold1_original_split_regression_compatible"] = True
    else:
        candidate["validation_selection"] = "pooled_window_macro_f1"
        candidate["num_classes"] = spec.num_classes
        candidate["test"]["worst"] = {
            "window_subject_accuracy": 1.0,
            "trial_subject_accuracy": 1.0,
        }

    model = {
        "weight": torch.tensor([[1.0, -0.0], [2.5, 4.0]], dtype=torch.float32),
        "bias": torch.tensor([0.5, -1.0], dtype=torch.float32),
    }
    torch.save(
        {
            "model": deepcopy(model),
            "libeer_commit": "39dc27e",
            "best_epoch": spec.reference_best_epoch,
            "validation_metric": "macro_f1",
            "validation_value": spec.reference_best_validation_value,
            "split": auditor.FOLD_ONE_SPLIT_ZERO_BASED,
        },
        reference_pt_path,
    )
    torch.save(
        {
            "model": deepcopy(model),
            "libeer_commit": "39dc27e",
            "fold": 1,
            "partition_seed": 2024,
            "optimization_seed": 2024,
            "best_epoch": spec.reference_best_epoch,
            "validation_metric": (
                "macro_f1" if dataset == "seed" else "pooled_window_macro_f1"
            ),
            "validation_value": spec.reference_best_validation_value,
            "split_one_based": auditor.FOLD_ONE_SPLIT,
        },
        candidate_pt_path,
    )
    reference_json_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    candidate_json_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    return auditor, {
        "reference_json": reference_json_path,
        "reference_pt": reference_pt_path,
        "candidate_json": candidate_json_path,
        "candidate_pt": candidate_pt_path,
        "predictions": predictions_path,
    }


def _arguments(dataset, paths, output):
    return [
        "--dataset",
        dataset,
        "--reference-json",
        str(paths["reference_json"]),
        "--reference-pt",
        str(paths["reference_pt"]),
        "--candidate-json",
        str(paths["candidate_json"]),
        "--candidate-pt",
        str(paths["candidate_pt"]),
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("dataset", ["seed", "seediv"])
def test_exact_synthetic_pair_passes_and_records_hashes(tmp_path, dataset):
    auditor, paths = _write_pair(tmp_path / dataset, dataset)
    output = tmp_path / f"{dataset}_audit.json"

    assert auditor.main(_arguments(dataset, paths, output)) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["verdict"] == "PASS"
    assert report["summary"]["checks_failed"] == 0
    assert report["summary"]["checks_total"] > 650
    assert report["artifacts"]["candidate_predictions"]["sha256"]
    assert report["artifacts"]["reference_pt"]["sha256"]
    assert report["auditor"]["sha256"]
    assert any(
        item["id"].endswith("bitwise_value") and item["passed"]
        for item in report["checks"]
    )


def test_tampered_tensor_fails_nonzero_but_writes_report(tmp_path):
    auditor, paths = _write_pair(tmp_path / "tampered_tensor", "seed")
    checkpoint = torch.load(paths["candidate_pt"], map_location="cpu", weights_only=False)
    checkpoint["model"]["weight"][0, 0] += 1.0
    torch.save(checkpoint, paths["candidate_pt"])
    output = tmp_path / "tampered_tensor_audit.json"

    assert auditor.main(_arguments("seed", paths, output)) == 1
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["verdict"] == "FAIL"
    assert "checkpoint.model.weight.bitwise_value" in report["summary"]["failed_check_ids"]
    assert report["artifacts"]["candidate_pt"]["sha256"]


def test_tampered_json_metric_fails_npz_recomputation_and_refuses_overwrite(tmp_path):
    auditor, paths = _write_pair(tmp_path / "tampered_metric", "seediv")
    payload = json.loads(paths["candidate_json"].read_text(encoding="utf-8"))
    payload["test"]["window"]["accuracy"] = 0.125
    paths["candidate_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output = tmp_path / "tampered_metric_audit.json"

    assert auditor.main(_arguments("seediv", paths, output)) == 1
    report_before = output.read_bytes()
    report = json.loads(report_before)

    assert report["verdict"] == "FAIL"
    assert "test.reference_candidate.window.accuracy" in report["summary"]["failed_check_ids"]
    assert "test.candidate_npz_recomputed.window.accuracy" in report["summary"]["failed_check_ids"]
    assert auditor.main(_arguments("seediv", paths, output)) == 2
    assert output.read_bytes() == report_before
