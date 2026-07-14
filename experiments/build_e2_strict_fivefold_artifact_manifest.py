"""Build the deterministic E2 strict-five-fold artifact manifest.

This is a closure index, not a metric implementation.  It runs only after the
two production-independent aggregate audits have passed and binds the complete
cell matrix, repeated aggregates, audit reports, closure logs, and executable
sources by SHA-256.  It refuses incomplete grids and existing output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASETS = ("seed", "seediv")
FOLDS = (1, 2, 3, 4, 5)
OPTIMIZATION_SEEDS = (2024, 2025, 2026)
PROTOCOL_VERSION = "seed_family_dgcnn_strict_5fold_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required E2 artifact not found: {path}")
    try:
        display = str(path.relative_to(repo_root.resolve()))
    except ValueError:
        display = str(path)
    return {
        "path": display,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_bound_record(
    observed: Any,
    expected_path: Path,
    repo_root: Path,
    label: str,
) -> None:
    """Recheck an audit file binding against the file that exists right now."""

    if not isinstance(observed, Mapping):
        raise ValueError(f"{label} is not a file binding")
    recorded_path = observed.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError(f"{label} lacks a recorded path")
    resolved_recorded = Path(recorded_path)
    if not resolved_recorded.is_absolute():
        resolved_recorded = repo_root / resolved_recorded
    expected_path = expected_path.resolve()
    if resolved_recorded.resolve() != expected_path:
        raise ValueError(f"{label} points to an unexpected file")
    current = file_record(expected_path, repo_root)
    if (
        observed.get("exists") is not True
        or observed.get("sha256") != current["sha256"]
        or observed.get("size_bytes") != current["size_bytes"]
    ):
        raise ValueError(f"{label} no longer matches its audited file binding")


def _recheck_audit_bindings(
    audit_payload: Mapping[str, Any],
    dataset: str,
    cells_by_name: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> None:
    """Close the narrow audit-to-manifest mutation window for core E2 files."""

    artifact_binding = audit_payload.get("artifact_binding")
    if not isinstance(artifact_binding, list) or len(artifact_binding) != 15:
        raise ValueError(f"{dataset} audit does not bind exactly 15 cell quartets")
    expected_cells = {
        f"{dataset}_fold{fold}_opt{seed}"
        for fold in FOLDS
        for seed in OPTIMIZATION_SEEDS
    }
    observed_cells = {
        record.get("cell")
        for record in artifact_binding
        if isinstance(record, Mapping)
    }
    if observed_cells != expected_cells:
        raise ValueError(f"{dataset} audit cell binding is not the exact 15-cell grid")
    suffixes = {
        "json": ".json",
        "checkpoint": ".pt",
        "predictions": ".npz",
        "log": ".log",
    }
    for record in artifact_binding:
        cell = str(record["cell"])
        fresh = cells_by_name[cell]
        for kind, suffix in suffixes.items():
            if kind == "log":
                expected_path = (
                    repo_root / "logs" / "runs" / "strict_fivefold" / f"{cell}{suffix}"
                )
            else:
                expected_path = (
                    repo_root
                    / "results"
                    / "strict_fivefold"
                    / dataset
                    / f"{cell}{suffix}"
                )
            _require_bound_record(
                record.get(kind), expected_path, repo_root, f"{dataset}.{cell}.{kind}"
            )
            if (
                record[kind].get("sha256") != fresh[kind]["sha256"]
                or record[kind].get("size_bytes") != fresh[kind]["size_bytes"]
            ):
                raise ValueError(f"{dataset}.{cell}.{kind} disagrees with fresh manifest hash")

    source_binding = audit_payload.get("source_binding")
    expected_sources = {
        "fold_definition": repo_root / "pcma" / "data" / "seed_folds.py",
        f"{dataset}_runner": repo_root
        / "experiments"
        / f"run_{dataset}_dgcnn_subject_fold.py",
        f"{dataset}_loader": repo_root / "pcma" / "data" / f"libeer_{dataset}.py",
        "aggregator": repo_root
        / "experiments"
        / "aggregate_seed_family_strict_fivefold.py",
    }
    if not isinstance(source_binding, Mapping) or set(source_binding) != set(expected_sources):
        raise ValueError(f"{dataset} audit source binding has an unexpected schema")
    for name, path in expected_sources.items():
        _require_bound_record(
            source_binding[name], path, repo_root, f"{dataset}.source_binding.{name}"
        )

    _require_bound_record(
        audit_payload.get("auditor_binding"),
        repo_root / "experiments" / "audit_seed_family_strict_fivefold_aggregate.py",
        repo_root,
        f"{dataset}.auditor_binding",
    )

    stage1_binding = audit_payload.get("stage1_binding")
    if not isinstance(stage1_binding, Mapping) or set(stage1_binding) != {
        "approval",
        "fold1_audits",
        "fold1_cells",
        "fold1_auditor",
        "reference_artifacts",
    }:
        raise ValueError(f"{dataset} audit stage1 binding has an unexpected schema")
    _require_bound_record(
        stage1_binding["approval"],
        repo_root / "results" / "strict_fivefold" / "stage1_approval.json",
        repo_root,
        f"{dataset}.stage1.approval",
    )
    fold1_audits = stage1_binding["fold1_audits"]
    fold1_cells = stage1_binding["fold1_cells"]
    if (
        not isinstance(fold1_audits, Mapping)
        or set(fold1_audits) != set(DATASETS)
        or not isinstance(fold1_cells, Mapping)
        or set(fold1_cells) != set(DATASETS)
    ):
        raise ValueError(f"{dataset} audit stage1 dataset binding is incomplete")
    for bound_dataset in DATASETS:
        _require_bound_record(
            fold1_audits[bound_dataset],
            repo_root
            / "results"
            / "strict_fivefold"
            / "audits"
            / f"{bound_dataset}_fold1_opt2024_regression_audit.json",
            repo_root,
            f"{dataset}.stage1.fold1_audits.{bound_dataset}",
        )
        cell_binding = fold1_cells[bound_dataset]
        if not isinstance(cell_binding, Mapping) or set(cell_binding) != {
            "json",
            "checkpoint",
            "predictions",
        }:
            raise ValueError(f"{dataset} audit stage1 cell binding is incomplete")
        for kind, suffix in (("json", "json"), ("checkpoint", "pt"), ("predictions", "npz")):
            _require_bound_record(
                cell_binding[kind],
                repo_root
                / "results"
                / "strict_fivefold"
                / bound_dataset
                / f"{bound_dataset}_fold1_opt2024.{suffix}",
                repo_root,
                f"{dataset}.stage1.fold1_cells.{bound_dataset}.{kind}",
            )
    _require_bound_record(
        stage1_binding["fold1_auditor"],
        repo_root / "experiments" / "audit_seed_family_fold1_regression.py",
        repo_root,
        f"{dataset}.stage1.fold1_auditor",
    )
    reference_artifacts = stage1_binding["reference_artifacts"]
    if not isinstance(reference_artifacts, Mapping) or set(reference_artifacts) != set(DATASETS):
        raise ValueError(f"{dataset} audit stage1 reference binding is incomplete")
    for bound_dataset in DATASETS:
        references = reference_artifacts[bound_dataset]
        if not isinstance(references, Mapping) or set(references) != {
            "reference_json",
            "reference_pt",
        }:
            raise ValueError(f"{dataset} audit stage1 reference schema is incomplete")
        for kind, record in references.items():
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                raise ValueError(f"{dataset} audit stage1 reference record is malformed")
            expected_path = Path(record["path"])
            if not expected_path.is_absolute():
                expected_path = repo_root / expected_path
            _require_bound_record(
                record,
                expected_path,
                repo_root,
                f"{dataset}.stage1.reference_artifacts.{bound_dataset}.{kind}",
            )


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def _cell_records(repo_root: Path) -> list[dict[str, Any]]:
    result_root = repo_root / "results" / "strict_fivefold"
    log_root = repo_root / "logs" / "runs" / "strict_fivefold"
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for fold in FOLDS:
            for seed in OPTIMIZATION_SEEDS:
                stem = f"{dataset}_fold{fold}_opt{seed}"
                records.append(
                    {
                        "cell": stem,
                        "dataset": dataset,
                        "fold": fold,
                        "optimization_seed": seed,
                        "json": file_record(
                            result_root / dataset / f"{stem}.json", repo_root
                        ),
                        "checkpoint": file_record(
                            result_root / dataset / f"{stem}.pt", repo_root
                        ),
                        "predictions": file_record(
                            result_root / dataset / f"{stem}.npz", repo_root
                        ),
                        "log": file_record(log_root / f"{stem}.log", repo_root),
                    }
                )
    return records


def build_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    result_root = repo_root / "results" / "strict_fivefold"
    audit_root = result_root / "audits"
    log_root = repo_root / "logs" / "runs" / "strict_fivefold"

    cells = _cell_records(repo_root)
    if len(cells) != 30 or len({record["cell"] for record in cells}) != 30:
        raise ValueError("cell manifest is not the exact 30-cell grid")
    cells_by_name = {record["cell"]: record for record in cells}

    aggregate_records: dict[str, Any] = {}
    audit_records: dict[str, Any] = {}
    for dataset in DATASETS:
        primary_path = result_root / f"{dataset}_strict_fivefold_aggregate.json"
        repro_path = result_root / f"{dataset}_strict_fivefold_aggregate_repro.json"
        primary = file_record(primary_path, repo_root)
        reproduction = file_record(repro_path, repo_root)
        if primary["sha256"] != reproduction["sha256"]:
            raise ValueError(f"{dataset} aggregate reproduction hash mismatch")
        aggregate_records[dataset] = {
            "primary": primary,
            "reproduction": reproduction,
            "byte_identical": True,
        }

        audit_path = audit_root / f"{dataset}_strict_fivefold_aggregate_audit.json"
        audit_payload = load_json(audit_path)
        summary = audit_payload.get("summary")
        aggregate_binding = audit_payload.get("aggregate_binding")
        if (
            audit_payload.get("status") != "PASS"
            or not isinstance(summary, Mapping)
            or summary.get("failed") != 0
            or not isinstance(aggregate_binding, Mapping)
            or aggregate_binding.get("sha256") != primary["sha256"]
        ):
            raise ValueError(f"{dataset} independent audit is not an aggregate-bound PASS")
        _recheck_audit_bindings(audit_payload, dataset, cells_by_name, repo_root)
        audit_records[dataset] = file_record(audit_path, repo_root)

    closure_log_names = (
        "batch_remaining28.log",
        "postcompletion_dryrun_audit.log",
        "postcompletion_contract_tests.log",
        "aggregate_seed.log",
        "aggregate_seed_repro.log",
        "aggregate_seediv.log",
        "aggregate_seediv_repro.log",
        "audit_seed_strict_fivefold_aggregate.log",
        "audit_seediv_strict_fivefold_aggregate.log",
    )
    closure_logs = {
        name: file_record(log_root / name, repo_root) for name in closure_log_names
    }
    dry_run_text = read_log_text(log_root / "postcompletion_dryrun_audit.log")
    if len(re.findall(r"(?m)^RESUME-SKIP ", dry_run_text)) != 30:
        raise ValueError("post-completion dry-run did not verify exactly 30 cells")
    if re.search(r"(?m)^(?:DRY-RUN|ERROR:)|Traceback", dry_run_text):
        raise ValueError("post-completion dry-run contains a missing cell or error")
    test_text = read_log_text(log_root / "postcompletion_contract_tests.log")
    passed_matches = [int(value) for value in re.findall(r"(\d+) passed", test_text)]
    if not passed_matches or max(passed_matches) < 40:
        raise ValueError("post-completion contract test log lacks the expected PASS count")

    source_paths = (
        "experiments/run_seed_family_strict_fivefold_batch.py",
        "experiments/aggregate_seed_family_strict_fivefold.py",
        "experiments/audit_seed_family_strict_fivefold_aggregate.py",
        "experiments/close_seed_family_strict_fivefold_e2.ps1",
        "experiments/build_e2_strict_fivefold_artifact_manifest.py",
        "plans/2026-07-13-seed-family-strict-fivefold-manifest.md",
    )
    sources = {
        path: file_record(repo_root / path, repo_root) for path in source_paths
    }

    stage1_paths = (
        "results/strict_fivefold/stage1_approval.json",
        "results/strict_fivefold/audits/seed_fold1_opt2024_regression_audit.json",
        "results/strict_fivefold/audits/seediv_fold1_opt2024_regression_audit.json",
    )
    stage1 = {
        path: file_record(repo_root / path, repo_root) for path in stage1_paths
    }

    return {
        "status": "PASS",
        "analysis": "e2_strict_fivefold_artifact_manifest",
        "protocol_version": PROTOCOL_VERSION,
        "validation": {
            "cell_grid_complete": True,
            "cell_count": 30,
            "cell_artifact_count": 120,
            "aggregate_reproductions_byte_identical": True,
            "independent_aggregate_audits_passed": True,
            "audit_bindings_rechecked_at_manifest_build": True,
            "postcompletion_dryrun_verified_cells": 30,
            "targeted_contract_tests_passed_minimum": max(passed_matches),
        },
        "cells": cells,
        "aggregates": aggregate_records,
        "independent_audits": audit_records,
        "stage1": stage1,
        "closure_logs": closure_logs,
        "sources": sources,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing E2 manifest: {output}")
    manifest = build_manifest(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest["validation"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
