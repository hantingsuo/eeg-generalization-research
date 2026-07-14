import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import bootstrap as scipy_bootstrap


def _load_auditor():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "audit_seed_family_strict_fivefold_aggregate.py"
    )
    name = "audit_seed_family_strict_fivefold_aggregate"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prediction_rows(auditor, test_subjects, optimization_seed):
    logits = []
    labels = []
    subjects = []
    sessions = []
    trials = []
    seed_index = auditor.OPTIMIZATION_SEEDS.index(optimization_seed)
    for subject in test_subjects:
        errors = (subject + seed_index) % 5
        for trial in range(1, 16):
            label = auditor.SEED_TRIAL_LABELS[trial - 1]
            prediction = (label + 1) % 3 if trial <= errors else label
            for _ in range(auditor.SEED_TRIAL_WINDOW_COUNTS[trial - 1]):
                score = np.full(3, -2.0, dtype=np.float32)
                score[prediction] = 2.0
                logits.append(score)
                labels.append(label)
                subjects.append(subject)
                sessions.append(1)
                trials.append(trial)
    return auditor.PredictionRows(
        logits=np.asarray(logits, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        trial=np.asarray(trials, dtype=np.int64),
    )


def _history():
    rows = []
    for epoch in range(1, 151):
        value = 0.2 + epoch / 1000.0
        rows.append(
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "validation_window": {
                    "accuracy": value,
                    "macro_f1": value,
                    "worst_subject_acc": value,
                    "mean_subject_acc": value,
                    "balanced_accuracy": value,
                    "recall_per_class": {"0": value, "1": value, "2": value},
                    "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                },
                "checkpoint_metric": "macro_f1",
                "checkpoint_value": value,
                "checkpoint_updated": True,
                "seconds": epoch / 100.0,
            }
        )
    return rows


def _provenance(auditor, source_root):
    source_root.mkdir(parents=True, exist_ok=True)
    source_files = {}
    source_hashes = {}
    for subject in range(1, 16):
        path = (source_root / f"subject_{subject}.mat").resolve()
        path.write_bytes(f"synthetic subject {subject}".encode("utf-8"))
        source_files[str(subject)] = str(path)
        source_hashes[str(subject)] = auditor.sha256_file(path)
    label_path = (source_root / "label.mat").resolve()
    label_path.write_bytes(b"synthetic labels")
    return {
        "runner_sha256": auditor.FROZEN_SOURCE_SHA256["seed_runner"],
        "loader_sha256": auditor.FROZEN_SOURCE_SHA256["seed_loader"],
        "fold_definition_sha256": auditor.FROZEN_SOURCE_SHA256["fold_definition"],
        "libeer_dgcnn_sha256": auditor.FROZEN_SOURCE_SHA256["libeer_dgcnn"],
        "libeer_dgcnn_config_sha256": auditor.FROZEN_SOURCE_SHA256[
            "libeer_dgcnn_config"
        ],
        "source_files": source_files,
        "source_file_sha256": source_hashes,
        "label_file": str(label_path),
        "label_sha256": auditor.sha256_file(label_path),
    }


def _write_grid(tmp_path):
    auditor = _load_auditor()
    cell_root = tmp_path / "seed"
    log_root = tmp_path / "logs"
    cell_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    provenance = _provenance(auditor, tmp_path / "source_data")
    paths = []
    for fold in auditor.FOLDS:
        split = auditor.FROZEN_FOLDS[fold]
        for optimization_seed in auditor.OPTIMIZATION_SEEDS:
            stem = f"seed_fold{fold}_opt{optimization_seed}"
            json_path = (cell_root / f"{stem}.json").resolve()
            checkpoint_path = json_path.with_suffix(".pt")
            prediction_path = json_path.with_suffix(".npz")
            log_path = log_root / f"{stem}.log"
            rows = _prediction_rows(auditor, split["test"], optimization_seed)
            np.savez_compressed(
                prediction_path,
                logits=rows.logits.astype(np.float32),
                labels=rows.labels.astype(np.int16),
                subject=rows.subject.astype(np.int16),
                session=rows.session.astype(np.int8),
                trial=rows.trial.astype(np.int16),
            )
            history = _history()
            best_value = history[-1]["validation_window"]["macro_f1"]
            checkpoint = {
                "model": {"weight": torch.tensor([float(fold), float(optimization_seed)])},
                "libeer_commit": auditor.FULL_LIBEER_COMMIT,
                "fold": fold,
                "partition_seed": 2024,
                "optimization_seed": optimization_seed,
                "session": 1,
                "split_one_based": split,
                "best_epoch": 150,
                "validation_metric": "macro_f1",
                "validation_value": best_value,
                "provenance": provenance,
            }
            torch.save(checkpoint, checkpoint_path)
            metrics = auditor.recompute_test_metrics(rows, auditor.DATASET_SPECS["seed"])
            subject_counts = {str(subject): 1 for subject in range(1, 16)}
            for subject in split["test"]:
                subject_counts[str(subject)] = int(np.sum(rows.subject == subject))
            payload = {
                "status": "completed",
                "dataset": "seed",
                "purpose": "strict_all_subject_seed_fold_baseline",
                "protocol": "SEED session 1; 9 train / 3 validation / 3 test subjects",
                "target_access": auditor.DATASET_SPECS["seed"].target_access,
                "test_evaluation_count": 1,
                "prediction_units": ["one_second_window", "trial_mean_logit"],
                "libeer_commit": auditor.FULL_LIBEER_COMMIT,
                "session": 1,
                "fold": fold,
                "partition_seed": 2024,
                "optimization_seed": optimization_seed,
                "split_one_based": split,
                "split_zero_based": {
                    name: [subject - 1 for subject in members]
                    for name, members in split.items()
                },
                "fold1_original_split_regression_compatible": (
                    fold == 1 and optimization_seed == 2024
                ),
                "epochs": 150,
                "batch_size": 16,
                "eval_batch_size": 512,
                "learning_rate": 0.001,
                "validation_metric": "pooled_window_macro_f1",
                "checkpoint_tie_rule": "strict greater-than; earliest epoch retained on ties",
                "best_epoch": 150,
                "best_validation_value": best_value,
                "best_validation": {"window": history[-1]["validation_window"]},
                "sample_counts": {
                    "train": sum(subject_counts[str(s)] for s in split["train"]),
                    "validation": sum(
                        subject_counts[str(s)] for s in split["validation"]
                    ),
                    "test": int(len(rows.labels)),
                },
                "subject_sample_counts": subject_counts,
                "provenance": provenance,
                "checkpoint": str(checkpoint_path),
                "predictions": str(prediction_path),
                "history": history,
                "test": metrics,
                "elapsed_seconds": 1.0,
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            log_lines = [json.dumps(row) for row in history]
            log_lines.append(json.dumps({"status": "completed", "elapsed_seconds": 1.0}))
            if not (fold == 1 and optimization_seed == 2024):
                code_hashes = {
                    name: auditor.FROZEN_SOURCE_SHA256[name]
                    for name in (
                        "seed_runner",
                        "seediv_runner",
                        "fold_definition",
                        "seed_loader",
                        "seediv_loader",
                        "aggregator",
                    )
                }
                cell = {
                    "dataset": "seed",
                    "fold": fold,
                    "optimization_seed": optimization_seed,
                    "partition_seed": 2024,
                    "stage": {2024: 2, 2025: 3, 2026: 4}[optimization_seed],
                    "cell_id": stem,
                }
                start_event = {
                    "event": "cell_start",
                    "protocol_version": "seed_family_dgcnn_strict_5fold_v1",
                    "cell": cell,
                    "command": auditor._expected_batch_command(
                        payload,
                        "seed",
                        json_path,
                        checkpoint_path,
                        prediction_path,
                    ),
                    "code_sha256": code_hashes,
                }
                exit_event = {
                    "event": "cell_exit",
                    "cell_id": stem,
                    "return_code": 0,
                }
                log_lines.insert(
                    0, auditor.BATCH_EVENT_PREFIX + json.dumps(start_event)
                )
                log_lines.append(
                    auditor.BATCH_EVENT_PREFIX + json.dumps(exit_event)
                )
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            paths.append(json_path)

    _, aggregate = auditor.validate_cells_and_recompute(
        "seed",
        paths,
        log_root,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    seediv_root = tmp_path / "seediv"
    seediv_root.mkdir()
    seediv_fold1 = (seediv_root / "seediv_fold1_opt2024.json").resolve()
    seediv_fold1.write_text("synthetic seediv fold1 JSON", encoding="utf-8")
    seediv_fold1.with_suffix(".pt").write_bytes(b"synthetic seediv checkpoint")
    seediv_fold1.with_suffix(".npz").write_bytes(b"synthetic seediv predictions")

    audit_root = tmp_path / "audits"
    audit_root.mkdir()
    audit_records = {}
    candidate_paths = {
        "seed": next(path for path in paths if path.stem == "seed_fold1_opt2024"),
        "seediv": seediv_fold1,
    }
    fold1_auditor = Path(auditor.__file__).with_name(
        "audit_seed_family_fold1_regression.py"
    ).resolve()
    for dataset, checks_passed in auditor.STAGE1_AUDIT_COUNTS.items():
        stem = f"{dataset}_fold1_opt2024"
        report_path = (audit_root / f"{stem}_regression_audit.json").resolve()
        reference_json = (tmp_path / f"{dataset}_reference.json").resolve()
        reference_pt = (tmp_path / f"{dataset}_reference.pt").resolve()
        reference_json.write_text(f"{dataset} reference JSON", encoding="utf-8")
        reference_pt.write_bytes(f"{dataset} reference checkpoint".encode("utf-8"))
        candidate = candidate_paths[dataset]
        report_payload = {
            "status": "completed",
            "verdict": "PASS",
            "dataset": dataset,
            "audit_scope": "fold1_optimization_seed2024_exact_new_old_regression",
            "summary": {
                "checks_total": checks_passed,
                "checks_passed": checks_passed,
                "checks_failed": 0,
                "failed_check_ids": [],
            },
            "checks": [
                {"id": f"synthetic.{index}", "passed": True}
                for index in range(checks_passed)
            ],
            "auditor": {
                "path": str(fold1_auditor),
                "sha256": auditor.FROZEN_FOLD1_AUDITOR_SHA256,
            },
            "artifacts": {
                "reference_json": auditor.file_record(reference_json),
                "reference_pt": auditor.file_record(reference_pt),
                "candidate_json": auditor.file_record(candidate),
                "candidate_pt": auditor.file_record(candidate.with_suffix(".pt")),
                "candidate_predictions": auditor.file_record(
                    candidate.with_suffix(".npz")
                ),
            },
        }
        report_path.write_text(json.dumps(report_payload), encoding="utf-8")
        audit_records[stem] = {
            "verdict": "PASS",
            "checks_passed": checks_passed,
            "checks_failed": 0,
            "report": str(report_path),
            "report_sha256": auditor.sha256_file(report_path),
        }
    seed_fold1 = candidate_paths["seed"]
    stage1_approval = (tmp_path / "stage1_approval.json").resolve()
    stage1_payload = {
        "protocol_version": "seed_family_dgcnn_strict_5fold_v1",
        "verdict": "PASS",
        "partition_seed": 2024,
        "optimization_seed": 2024,
        "code_sha256": {
            name: auditor.FROZEN_SOURCE_SHA256[name]
            for name in (
                "seed_runner",
                "seediv_runner",
                "fold_definition",
                "seed_loader",
                "seediv_loader",
                "aggregator",
            )
        },
        "cells": {
            "seed_fold1_opt2024": {
                "json_sha256": auditor.sha256_file(seed_fold1),
                "checkpoint_sha256": auditor.sha256_file(
                    seed_fold1.with_suffix(".pt")
                ),
                "predictions_sha256": auditor.sha256_file(
                    seed_fold1.with_suffix(".npz")
                ),
            },
            "seediv_fold1_opt2024": {
                "json_sha256": auditor.sha256_file(seediv_fold1),
                "checkpoint_sha256": auditor.sha256_file(
                    seediv_fold1.with_suffix(".pt")
                ),
                "predictions_sha256": auditor.sha256_file(
                    seediv_fold1.with_suffix(".npz")
                ),
            },
        },
        "audits": audit_records,
    }
    stage1_approval.write_text(json.dumps(stage1_payload), encoding="utf-8")
    auditor.FROZEN_STAGE1_APPROVAL_SHA256 = auditor.sha256_file(stage1_approval)
    return auditor, paths, log_root, aggregate_path, stage1_approval


def test_valid_synthetic_grid_passes_and_binds_all_required_files(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    assert report["status"] == "PASS"
    assert report["summary"]["validated_cells"] == 15
    assert report["summary"]["failed"] == 0
    assert report["contract"]["minimum_core_files_bound"] == 74
    assert report["stage1_binding"]["approval"]["sha256"] == auditor.sha256_file(
        stage1_approval
    )
    assert set(report["stage1_binding"]["fold1_audits"]) == {"seed", "seediv"}
    assert len(report["artifact_binding"]) == 15
    for cell in report["artifact_binding"]:
        for kind in ("json", "checkpoint", "predictions", "log"):
            assert len(cell[kind]["sha256"]) == 64
    source = Path(auditor.__file__).read_text(encoding="utf-8")
    assert "from experiments.aggregate_seed_family_strict_fivefold" not in source
    assert "pcma.eval.metrics" not in source
    assert "pcma.model.seed_da" not in source


def test_aggregate_primary_tampering_produces_fail(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["primary"]["estimate"] += 0.01
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    assert report["status"] == "FAIL"
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "aggregate.section.primary" in failed


def test_stage1_approval_or_fold1_audit_tampering_produces_fail(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    approval = json.loads(stage1_approval.read_text(encoding="utf-8"))
    approval["audits"]["seed_fold1_opt2024"]["report_sha256"] = "f" * 64
    stage1_approval.write_text(json.dumps(approval), encoding="utf-8")
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    assert report["status"] == "FAIL"
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "stage1.release_bundle" in failed


def test_noncanonical_bootstrap_parameters_cannot_pass(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=1,
        bootstrap_seed=17,
    )
    assert report["status"] == "FAIL"
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert {"bootstrap.frozen_resamples", "bootstrap.frozen_seed"} <= failed


def test_deep_stage1_report_tampering_fails_even_if_approval_hash_is_updated(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    report_path = tmp_path / "audits" / "seed_fold1_opt2024_regression_audit.json"
    fold1_report = json.loads(report_path.read_text(encoding="utf-8"))
    fold1_report["checks"][0]["passed"] = False
    report_path.write_text(json.dumps(fold1_report), encoding="utf-8")
    approval = json.loads(stage1_approval.read_text(encoding="utf-8"))
    approval["audits"]["seed_fold1_opt2024"]["report_sha256"] = (
        auditor.sha256_file(report_path)
    )
    stage1_approval.write_text(json.dumps(approval), encoding="utf-8")
    auditor.FROZEN_STAGE1_APPROVAL_SHA256 = auditor.sha256_file(stage1_approval)
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert report["status"] == "FAIL"
    assert "stage1.release_bundle" in failed


def test_extra_aggregate_key_and_checkpoint_schema_are_rejected(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["leaderboard_claim"] = "misleading extra field"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    checkpoint_path = paths[-1].with_suffix(".pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model"]["unexpected_tensor"] = torch.tensor([1.0])
    torch.save(checkpoint, checkpoint_path)

    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert report["status"] == "FAIL"
    assert "aggregate.top_level_schema" in failed
    assert "checkpoint.all_cells_match_stage1_model_schema" in failed


def test_batch_event_code_hash_tampering_is_rejected(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    event_log = log_root / f"{paths[1].stem}.log"
    lines = event_log.read_text(encoding="utf-8").splitlines()
    start = json.loads(lines[0][len(auditor.BATCH_EVENT_PREFIX) :])
    start["code_sha256"]["seed_runner"] = "0" * 64
    lines[0] = auditor.BATCH_EVENT_PREFIX + json.dumps(start)
    event_log.write_text("\n".join(lines), encoding="utf-8")
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert report["status"] == "FAIL"
    assert f"cell.{paths[1].stem}" in failed


def test_class_balanced_trial_label_permutation_is_rejected(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    json_path = paths[3]
    prediction_path = json_path.with_suffix(".npz")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    subject = int(np.unique(arrays["subject"])[0])
    mask4 = (arrays["subject"] == subject) & (arrays["trial"] == 4)
    mask14 = (arrays["subject"] == subject) & (arrays["trial"] == 14)
    label4 = int(arrays["labels"][mask4][0])
    label14 = int(arrays["labels"][mask14][0])
    arrays["labels"][mask4] = label14
    arrays["labels"][mask14] = label4
    np.savez_compressed(prediction_path, **arrays)
    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    check = next(
        item for item in report["checks"] if item["id"] == f"cell.{json_path.stem}"
    )
    assert report["status"] == "FAIL"
    assert "label does not match frozen provenance" in check["detail"]


def test_bca_interval_matches_scipy_reference_for_fixed_subject_vector():
    auditor = _load_auditor()
    values = np.linspace(0.2, 0.9, 15)
    observed = auditor.bca_mean_interval(
        values,
        n_resamples=auditor.BOOTSTRAP_RESAMPLES,
        seed=auditor.BOOTSTRAP_SEED,
    )
    reference = scipy_bootstrap(
        (values,),
        np.mean,
        method="BCa",
        confidence_level=0.95,
        n_resamples=auditor.BOOTSTRAP_RESAMPLES,
        random_state=np.random.default_rng(auditor.BOOTSTRAP_SEED),
    )
    assert observed["low"] == pytest.approx(reference.confidence_interval.low, abs=0.0)
    assert observed["high"] == pytest.approx(reference.confidence_interval.high, abs=0.0)


def test_earliest_max_tie_rule_is_recomputed_from_history(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    json_path = paths[0]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    prior_value = payload["history"][-2]["validation_window"]["macro_f1"]
    payload["history"][-1]["validation_window"]["macro_f1"] = prior_value
    payload["history"][-1]["checkpoint_value"] = prior_value
    payload["history"][-1]["checkpoint_updated"] = False
    payload["best_validation_value"] = prior_value
    payload["best_validation"]["window"] = payload["history"][-1]["validation_window"]
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = torch.load(json_path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    checkpoint["validation_value"] = prior_value
    torch.save(checkpoint, json_path.with_suffix(".pt"))
    log_path = log_root / f"{json_path.stem}.log"
    lines = [json.dumps(row) for row in payload["history"]]
    lines.append(json.dumps({"status": "completed", "elapsed_seconds": 1.0}))
    log_path.write_text("\n".join(lines), encoding="utf-8")

    report = auditor.audit_dataset(
        "seed",
        paths,
        aggregate_path,
        log_root,
        stage1_approval,
        bootstrap=auditor.BOOTSTRAP_RESAMPLES,
        bootstrap_seed=auditor.BOOTSTRAP_SEED,
    )
    assert report["status"] == "FAIL"
    check = next(item for item in report["checks"] if item["id"] == f"cell.{json_path.stem}")
    assert "earliest maximum" in check["detail"]


def test_missing_log_writes_fail_report_and_cli_refuses_overwrite(tmp_path):
    auditor, paths, log_root, aggregate_path, stage1_approval = _write_grid(tmp_path)
    (log_root / f"{paths[0].stem}.log").unlink()
    output = tmp_path / "audit.json"
    argv = [
        "--dataset",
        "seed",
        "--aggregate",
        str(aggregate_path),
        "--inputs",
        *map(str, paths),
        "--log-root",
        str(log_root),
        "--stage1-approval",
        str(stage1_approval),
        "--output",
        str(output),
        "--bootstrap",
        str(auditor.BOOTSTRAP_RESAMPLES),
        "--bootstrap-seed",
        str(auditor.BOOTSTRAP_SEED),
    ]
    assert auditor.main(argv) == 1
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["status"] == "FAIL"
    assert output.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        auditor.main(argv)
