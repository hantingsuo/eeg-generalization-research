import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_builder():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "build_e2_strict_fivefold_artifact_manifest.py"
    )
    name = "build_e2_strict_fivefold_artifact_manifest"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write(path, content=b"artifact"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _synthetic_repo(tmp_path):
    builder = _load_builder()
    result_root = tmp_path / "results" / "strict_fivefold"
    audit_root = result_root / "audits"
    log_root = tmp_path / "logs" / "runs" / "strict_fivefold"

    for dataset in builder.DATASETS:
        for fold in builder.FOLDS:
            for seed in builder.OPTIMIZATION_SEEDS:
                stem = f"{dataset}_fold{fold}_opt{seed}"
                for suffix in ("json", "pt", "npz"):
                    _write(result_root / dataset / f"{stem}.{suffix}")
                _write(log_root / f"{stem}.log")

    for path in (
        "results/strict_fivefold/stage1_approval.json",
        "results/strict_fivefold/audits/seed_fold1_opt2024_regression_audit.json",
        "results/strict_fivefold/audits/seediv_fold1_opt2024_regression_audit.json",
        "experiments/run_seed_family_strict_fivefold_batch.py",
        "experiments/aggregate_seed_family_strict_fivefold.py",
        "experiments/audit_seed_family_strict_fivefold_aggregate.py",
        "experiments/close_seed_family_strict_fivefold_e2.ps1",
        "experiments/build_e2_strict_fivefold_artifact_manifest.py",
        "experiments/audit_seed_family_fold1_regression.py",
        "experiments/run_seed_dgcnn_subject_fold.py",
        "experiments/run_seediv_dgcnn_subject_fold.py",
        "pcma/data/seed_folds.py",
        "pcma/data/libeer_seed.py",
        "pcma/data/libeer_seediv.py",
        "plans/2026-07-13-seed-family-strict-fivefold-manifest.md",
        "plans/2026-07-13-seed-family-strict-fivefold-execution-ledger.md",
    ):
        _write(tmp_path / path)

    def bound(path):
        path = path.resolve()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": path.stat().st_size,
            "sha256": builder.sha256_file(path),
        }

    reference_root = tmp_path / "stage1_references"
    for dataset in builder.DATASETS:
        _write(reference_root / f"{dataset}_reference.json", b"reference json")
        _write(reference_root / f"{dataset}_reference.pt", b"reference checkpoint")

    stage1_binding = {
        "approval": bound(result_root / "stage1_approval.json"),
        "fold1_audits": {
            dataset: bound(
                audit_root / f"{dataset}_fold1_opt2024_regression_audit.json"
            )
            for dataset in builder.DATASETS
        },
        "fold1_cells": {
            dataset: {
                "json": bound(result_root / dataset / f"{dataset}_fold1_opt2024.json"),
                "checkpoint": bound(result_root / dataset / f"{dataset}_fold1_opt2024.pt"),
                "predictions": bound(result_root / dataset / f"{dataset}_fold1_opt2024.npz"),
            }
            for dataset in builder.DATASETS
        },
        "fold1_auditor": bound(
            tmp_path / "experiments" / "audit_seed_family_fold1_regression.py"
        ),
        "reference_artifacts": {
            dataset: {
                "reference_json": bound(
                    reference_root / f"{dataset}_reference.json"
                ),
                "reference_pt": bound(reference_root / f"{dataset}_reference.pt"),
            }
            for dataset in builder.DATASETS
        },
    }

    for dataset in builder.DATASETS:
        primary = _write(
            result_root / f"{dataset}_strict_fivefold_aggregate.json",
            b"deterministic aggregate",
        )
        _write(
            result_root / f"{dataset}_strict_fivefold_aggregate_repro.json",
            b"deterministic aggregate",
        )
        source_binding = {
            "fold_definition": bound(tmp_path / "pcma" / "data" / "seed_folds.py"),
            f"{dataset}_runner": bound(
                tmp_path / "experiments" / f"run_{dataset}_dgcnn_subject_fold.py"
            ),
            f"{dataset}_loader": bound(
                tmp_path / "pcma" / "data" / f"libeer_{dataset}.py"
            ),
            "aggregator": bound(
                tmp_path
                / "experiments"
                / "aggregate_seed_family_strict_fivefold.py"
            ),
        }
        artifact_binding = []
        for fold in builder.FOLDS:
            for seed in builder.OPTIMIZATION_SEEDS:
                stem = f"{dataset}_fold{fold}_opt{seed}"
                artifact_binding.append(
                    {
                        "cell": stem,
                        "json": bound(result_root / dataset / f"{stem}.json"),
                        "checkpoint": bound(result_root / dataset / f"{stem}.pt"),
                        "predictions": bound(result_root / dataset / f"{stem}.npz"),
                        "log": bound(log_root / f"{stem}.log"),
                    }
                )
        audit = {
            "status": "PASS",
            "summary": {"failed": 0},
            "aggregate_binding": {"sha256": builder.sha256_file(primary)},
            "artifact_binding": artifact_binding,
            "source_binding": source_binding,
            "auditor_binding": bound(
                tmp_path
                / "experiments"
                / "audit_seed_family_strict_fivefold_aggregate.py"
            ),
            "stage1_binding": stage1_binding,
        }
        _write(
            audit_root / f"{dataset}_strict_fivefold_aggregate_audit.json",
            json.dumps(audit).encode("utf-8"),
        )

    closure_logs = (
        "batch_remaining28.log",
        "aggregate_seed.log",
        "aggregate_seed_repro.log",
        "aggregate_seediv.log",
        "aggregate_seediv_repro.log",
        "audit_seed_strict_fivefold_aggregate.log",
        "audit_seediv_strict_fivefold_aggregate.log",
    )
    for name in closure_logs:
        _write(log_root / name)
    dry_run = "\n".join(f"RESUME-SKIP cell{index}" for index in range(30))
    _write(
        log_root / "postcompletion_dryrun_audit.log",
        dry_run.encode("utf-16"),
    )
    _write(
        log_root / "postcompletion_contract_tests.log",
        "40 passed in 1.0s".encode("utf-16"),
    )
    return builder


def test_complete_closure_builds_deterministic_manifest(tmp_path):
    builder = _synthetic_repo(tmp_path)
    first = builder.build_manifest(tmp_path)
    second = builder.build_manifest(tmp_path)
    assert first == second
    assert first["status"] == "PASS"
    assert first["validation"]["cell_count"] == 30
    assert first["validation"]["cell_artifact_count"] == 120
    assert len(first["cells"]) == 30


def test_aggregate_reproduction_mismatch_is_rejected(tmp_path):
    builder = _synthetic_repo(tmp_path)
    path = (
        tmp_path
        / "results"
        / "strict_fivefold"
        / "seed_strict_fivefold_aggregate_repro.json"
    )
    path.write_bytes(b"tampered aggregate")
    with pytest.raises(ValueError, match="reproduction hash mismatch"):
        builder.build_manifest(tmp_path)


def test_post_audit_cell_mutation_is_rejected(tmp_path):
    builder = _synthetic_repo(tmp_path)
    path = (
        tmp_path
        / "logs"
        / "runs"
        / "strict_fivefold"
        / "seed_fold2_opt2025.log"
    )
    path.write_bytes(b"changed after independent audit")
    with pytest.raises(ValueError, match="no longer matches its audited file binding"):
        builder.build_manifest(tmp_path)


def test_post_audit_stage1_reference_mutation_is_rejected(tmp_path):
    builder = _synthetic_repo(tmp_path)
    path = tmp_path / "stage1_references" / "seediv_reference.pt"
    path.write_bytes(b"changed after independent audit")
    with pytest.raises(ValueError, match="no longer matches its audited file binding"):
        builder.build_manifest(tmp_path)
