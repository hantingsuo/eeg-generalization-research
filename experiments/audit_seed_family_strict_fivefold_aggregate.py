"""Independently audit one frozen SEED-family strict five-fold aggregate.

The production aggregator is deliberately *not* imported here.  This program
starts again from the 15 JSON/PT/NPZ/log cell quartets, reconstructs window and
trial predictions with scikit-learn primitives, re-selects the earliest maximum
validation checkpoint, and independently recomputes every published aggregate
field including the subject-row BCa interval.

The audit report is an immutable SHA-256 binding of the aggregate and all 60
cell artifacts.  A report is PASS only when every check succeeds.  FAIL reports
are still written for diagnosis and the command exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


FULL_LIBEER_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"
SHORT_LIBEER_COMMIT = "39dc27e"
PARTITION_SEED = 2024
OPTIMIZATION_SEEDS = (2024, 2025, 2026)
FOLDS = (1, 2, 3, 4, 5)
SUBJECTS = tuple(range(1, 16))
SESSION = 1
EPOCHS = 150
EVAL_BATCH_SIZE = 512
VALIDATION_SELECTION = "pooled_window_macro_f1"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_713
REQUIRED_NPZ_KEYS = {"logits", "labels", "subject", "session", "trial"}
BATCH_EVENT_PREFIX = "BATCH_EVENT\t"
FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-12
STAGE1_AUDIT_COUNTS = {"seed": 1391, "seediv": 1535}
FROZEN_STAGE1_APPROVAL_SHA256 = (
    "35d7ef3e638d9f34e5eb6e89bbcc22496b25d9b766c7602b0d206696977c778e"
)
FROZEN_FOLD1_AUDITOR_SHA256 = (
    "4ad9d4b1da016be56a23b7a16e2c89ab83769bc023661df82801e50aaa1d4bbe"
)
SEED_TRIAL_LABELS = (2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0)
SEED_TRIAL_WINDOW_COUNTS = (
    235,
    233,
    206,
    238,
    185,
    195,
    237,
    216,
    265,
    237,
    235,
    233,
    235,
    238,
    206,
)


FROZEN_FOLDS = {
    1: {
        "train": (15, 11, 7, 4, 5, 10, 12, 3, 8),
        "validation": (14, 13, 2),
        "test": (9, 1, 6),
    },
    2: {
        "train": (9, 1, 6, 4, 5, 10, 12, 3, 8),
        "validation": (15, 11, 7),
        "test": (14, 13, 2),
    },
    3: {
        "train": (9, 1, 6, 14, 13, 2, 12, 3, 8),
        "validation": (4, 5, 10),
        "test": (15, 11, 7),
    },
    4: {
        "train": (9, 1, 6, 14, 13, 2, 15, 11, 7),
        "validation": (12, 3, 8),
        "test": (4, 5, 10),
    },
    5: {
        "train": (14, 13, 2, 15, 11, 7, 4, 5, 10),
        "validation": (9, 1, 6),
        "test": (12, 3, 8),
    },
}


FROZEN_SOURCE_SHA256 = {
    "fold_definition": "275273dfb227000451b15352ad042b9e5047e6c8b04fbe00edf2bad48557930b",
    "seed_runner": "3de0cc948840a24ec7704c910beb3a429594d5e30835893f33e0705d7a323d95",
    "seediv_runner": "87d73dd92103c61f91c91af3c93463f3677f0c9e4295226113bdb7e3b9a92a2c",
    "seed_loader": "9372e319847cd9d4426e0b4ffa5481c3c9e33d5d5e985b26854889530c86578e",
    "seediv_loader": "3979f5c37c3add4abcfbfd99686bbe5f1c55f2eef926a78143512ab339076c1e",
    "aggregator": "f4e238a1bb24c3397bfc4614cc48b1d49eeb6174ae00a4d4db9240a40fa3502a",
    "libeer_dgcnn": "020e369d54676e78bc3ef42df2fa76d65eb85140f566c22857fd2c6c65b0577d",
    "libeer_dgcnn_config": "0e54afb0986eade332316449217d07acb6af04b6383490228d8a6c05230f0d4a",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    num_classes: int
    trials_per_subject: int
    trials_per_class: int
    batch_size: int
    learning_rate: float
    purpose: str
    protocol_template: str
    target_access: str
    checkpoint_metric: str


DATASET_SPECS = {
    "seed": DatasetSpec(
        key="seed",
        display_name="SEED",
        num_classes=3,
        trials_per_subject=15,
        trials_per_class=5,
        batch_size=16,
        learning_rate=0.001,
        purpose="strict_all_subject_seed_fold_baseline",
        protocol_template="SEED session 1; 9 train / 3 validation / 3 test subjects",
        target_access=(
            "test-subject data are not loaded during fitting or validation-only "
            "checkpoint selection; the fixed checkpoint is evaluated on test exactly once"
        ),
        checkpoint_metric="macro_f1",
    ),
    "seediv": DatasetSpec(
        key="seediv",
        display_name="SEED-IV",
        num_classes=4,
        trials_per_subject=24,
        trials_per_class=6,
        batch_size=32,
        learning_rate=0.0015,
        purpose="strict_all_subject_cross_validation_fold",
        protocol_template="SEED-IV session 1 frozen five-fold subject CV, fold {fold}, 9/3/3",
        target_access=(
            "fixed per-trial raw-to-DE+LDS extraction uses no labels or cross-subject "
            "statistics; fitting uses train subjects, checkpoint selection uses only "
            "pooled validation-window labels, and test labels are used only in the "
            "single final evaluation"
        ),
        checkpoint_metric=VALIDATION_SELECTION,
    ),
}


@dataclass
class AuditRecorder:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        check_id: str,
        passed: bool,
        *,
        expected: Any = None,
        actual: Any = None,
        detail: str | None = None,
        include_values: bool = False,
    ) -> None:
        item: dict[str, Any] = {"id": check_id, "passed": bool(passed)}
        if include_values:
            item["expected"] = _json_safe(expected)
            item["actual"] = _json_safe(actual)
        if detail:
            item["detail"] = detail
        self.checks.append(item)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(item["passed"] for item in self.checks)


@dataclass(frozen=True)
class PredictionRows:
    logits: np.ndarray
    labels: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


@dataclass(frozen=True)
class AuditedCell:
    json_path: Path
    checkpoint_path: Path
    prediction_path: Path
    log_path: Path
    fold: int
    optimization_seed: int
    split: dict[str, tuple[int, ...]]
    rows: PredictionRows
    metrics: dict[str, Any]
    checkpoint_schema: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SPECS))
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--inputs", required=True, nargs=15, type=Path)
    parser.add_argument(
        "--log-root",
        required=True,
        type=Path,
        help="Directory containing <dataset>_fold<F>_opt<S>.log files.",
    )
    parser.add_argument(
        "--stage1-approval",
        required=True,
        type=Path,
        help="Frozen approval JSON binding both fold-1 implementation audits.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = sha256_file(resolved)
    return record


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _exact_int(value: Any, field: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _numeric_close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _tree_mismatches(
    expected: Any,
    actual: Any,
    path: str = "$",
    *,
    limit: int = 40,
) -> list[str]:
    """Return bounded, human-readable structural/numeric mismatches."""

    mismatches: list[str] = []

    def visit(left: Any, right: Any, where: str) -> None:
        if len(mismatches) >= limit:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_map = {str(key): value for key, value in left.items()}
            right_map = {str(key): value for key, value in right.items()}
            if set(left_map) != set(right_map):
                missing = sorted(set(left_map) - set(right_map))
                extra = sorted(set(right_map) - set(left_map))
                mismatches.append(f"{where}: keys missing={missing}, extra={extra}")
            for key in sorted(set(left_map).intersection(right_map)):
                visit(left_map[key], right_map[key], f"{where}.{key}")
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                mismatches.append(f"{where}: length {len(right)} != {len(left)}")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{where}[{index}]")
            return
        numeric = (int, float, np.integer, np.floating)
        if (
            isinstance(left, numeric)
            and not isinstance(left, bool)
            and isinstance(right, numeric)
            and not isinstance(right, bool)
        ):
            if not _numeric_close(left, right):
                mismatches.append(f"{where}: {right!r} != {left!r}")
            return
        if type(left) is not type(right) or left != right:
            mismatches.append(f"{where}: {right!r} != {left!r}")

    visit(expected, actual, path)
    return mismatches


def _require_tree_equal(expected: Any, actual: Any, field: str) -> None:
    mismatches = _tree_mismatches(expected, actual, field)
    if mismatches:
        raise ValueError("; ".join(mismatches))


def _canonical_split(value: Any, field: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or set(value) != {"train", "validation", "test"}:
        raise ValueError(f"{field} must contain exactly train, validation, and test")
    result: dict[str, tuple[int, ...]] = {}
    for name in ("train", "validation", "test"):
        raw = value[name]
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"{field}.{name} must be a list")
        result[name] = tuple(_exact_int(item, f"{field}.{name}") for item in raw)
    return result


def _resolve_recorded_path(value: Any, json_path: Path, suffix: str, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = json_path.parent / path
    path = path.resolve()
    expected = json_path.with_suffix(suffix).resolve()
    if path != expected:
        raise ValueError(f"{field} must bind to {expected}, found {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


def _source_freeze_checks(recorder: AuditRecorder, dataset: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "fold_definition": root / "pcma" / "data" / "seed_folds.py",
        f"{dataset}_runner": root
        / "experiments"
        / f"run_{dataset}_dgcnn_subject_fold.py",
        f"{dataset}_loader": root / "pcma" / "data" / f"libeer_{dataset}.py",
        "aggregator": root
        / "experiments"
        / "aggregate_seed_family_strict_fivefold.py",
    }
    records: dict[str, Any] = {}
    for name, path in source_paths.items():
        record = file_record(path)
        records[name] = record
        expected_key = name
        expected = FROZEN_SOURCE_SHA256[expected_key]
        actual = record.get("sha256")
        recorder.add(
            f"source_freeze.{name}",
            actual == expected,
            expected=expected,
            actual=actual,
            include_values=True,
        )
    return records


def _stage1_bundle_records(approval_path: Path) -> dict[str, Any]:
    approval_path = approval_path.resolve()
    audit_root = approval_path.parent / "audits"
    cell_root = approval_path.parent
    fold1_auditor = Path(__file__).resolve().with_name(
        "audit_seed_family_fold1_regression.py"
    )
    return {
        "approval": file_record(approval_path),
        "fold1_audits": {
            dataset: file_record(
                audit_root / f"{dataset}_fold1_opt2024_regression_audit.json"
            )
            for dataset in DATASET_SPECS
        },
        "fold1_cells": {
            dataset: {
                kind: file_record(
                    cell_root
                    / dataset
                    / f"{dataset}_fold1_opt2024.{suffix}"
                )
                for kind, suffix in (
                    ("json", "json"),
                    ("checkpoint", "pt"),
                    ("predictions", "npz"),
                )
            }
            for dataset in DATASET_SPECS
        },
        "fold1_auditor": file_record(fold1_auditor),
    }


def _validate_stage1_bundle(
    approval_path: Path,
    inputs: Sequence[Path],
    dataset: str,
) -> dict[str, Any]:
    """Validate and hash-bind the frozen stage-1 release evidence.

    Both datasets' fold-1 cells and regression reports are verified on every
    invocation, so either final aggregate audit is a self-contained binding of
    the release decision rather than a summary trusting another report.
    """

    approval_path = approval_path.resolve()
    records = _stage1_bundle_records(approval_path)
    if sha256_file(approval_path) != FROZEN_STAGE1_APPROVAL_SHA256:
        raise ValueError("stage1 approval SHA-256 differs from the released file")
    approval = _load_json(approval_path)
    expected_code = {
        name: FROZEN_SOURCE_SHA256[name]
        for name in (
            "seed_runner",
            "seediv_runner",
            "fold_definition",
            "seed_loader",
            "seediv_loader",
            "aggregator",
        )
    }
    frozen_header = {
        "protocol_version": "seed_family_dgcnn_strict_5fold_v1",
        "verdict": "PASS",
        "partition_seed": PARTITION_SEED,
        "optimization_seed": 2024,
        "code_sha256": expected_code,
    }
    for field, expected in frozen_header.items():
        _require_tree_equal(expected, approval.get(field), f"stage1.{field}")

    cells = approval.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != {
        "seed_fold1_opt2024",
        "seediv_fold1_opt2024",
    }:
        raise ValueError("stage1.cells must bind exactly the two fold-1 cells")
    input_by_stem = {Path(path).stem: Path(path).resolve() for path in inputs}
    current_stem = f"{dataset}_fold1_opt2024"
    expected_current_json = (
        approval_path.parent / dataset / f"{current_stem}.json"
    ).resolve()
    if input_by_stem.get(current_stem) != expected_current_json:
        raise ValueError(f"audit inputs do not bind released stage1 cell {current_stem}")

    for cell_dataset in DATASET_SPECS:
        stem = f"{cell_dataset}_fold1_opt2024"
        base = (approval_path.parent / cell_dataset / stem).resolve()
        expected_cell_hashes = {
            "json_sha256": sha256_file(base.with_suffix(".json")),
            "checkpoint_sha256": sha256_file(base.with_suffix(".pt")),
            "predictions_sha256": sha256_file(base.with_suffix(".npz")),
        }
        _require_tree_equal(
            expected_cell_hashes,
            cells.get(stem),
            f"stage1.cells.{stem}",
        )

    approval_audits = approval.get("audits")
    if not isinstance(approval_audits, Mapping) or set(approval_audits) != {
        "seed_fold1_opt2024",
        "seediv_fold1_opt2024",
    }:
        raise ValueError("stage1.audits must bind exactly the two fold-1 reports")
    for audit_dataset, expected_count in STAGE1_AUDIT_COUNTS.items():
        stem = f"{audit_dataset}_fold1_opt2024"
        audit_path = (
            approval_path.parent
            / "audits"
            / f"{stem}_regression_audit.json"
        ).resolve()
        audit_record = approval_audits.get(stem)
        if not isinstance(audit_record, Mapping):
            raise ValueError(f"stage1.audits.{stem} must be an object")
        expected_record = {
            "verdict": "PASS",
            "checks_passed": expected_count,
            "checks_failed": 0,
            "report": str(audit_path),
            "report_sha256": sha256_file(audit_path),
        }
        _require_tree_equal(expected_record, audit_record, f"stage1.audits.{stem}")
        audit_payload = _load_json(audit_path)
        _require_tree_equal(
            "completed", audit_payload.get("status"), f"{stem}.status"
        )
        _require_tree_equal("PASS", audit_payload.get("verdict"), f"{stem}.verdict")
        _require_tree_equal(
            audit_dataset, audit_payload.get("dataset"), f"{stem}.dataset"
        )
        _require_tree_equal(
            "fold1_optimization_seed2024_exact_new_old_regression",
            audit_payload.get("audit_scope"),
            f"{stem}.audit_scope",
        )
        summary = audit_payload.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{stem}.summary must be an object")
        _require_tree_equal(
            expected_count,
            summary.get("checks_total"),
            f"{stem}.summary.checks_total",
        )
        _require_tree_equal(
            expected_count,
            summary.get("checks_passed"),
            f"{stem}.summary.checks_passed",
        )
        _require_tree_equal(
            0,
            summary.get("checks_failed"),
            f"{stem}.summary.checks_failed",
        )
        _require_tree_equal(
            [],
            summary.get("failed_check_ids"),
            f"{stem}.summary.failed_check_ids",
        )
        checks = audit_payload.get("checks")
        if not isinstance(checks, list) or len(checks) != expected_count:
            raise ValueError(f"{stem}.checks must contain {expected_count} rows")
        if any(not isinstance(check, Mapping) or check.get("passed") is not True for check in checks):
            raise ValueError(f"{stem}.checks contains a failed or malformed row")

        expected_auditor_path = Path(__file__).resolve().with_name(
            "audit_seed_family_fold1_regression.py"
        )
        expected_auditor = {
            "path": str(expected_auditor_path),
            "sha256": FROZEN_FOLD1_AUDITOR_SHA256,
        }
        _require_tree_equal(
            expected_auditor,
            audit_payload.get("auditor"),
            f"{stem}.auditor",
        )
        if sha256_file(expected_auditor_path) != FROZEN_FOLD1_AUDITOR_SHA256:
            raise ValueError("fold1 regression auditor source hash has drifted")

        artifacts = audit_payload.get("artifacts")
        expected_artifact_names = {
            "reference_json",
            "reference_pt",
            "candidate_json",
            "candidate_pt",
            "candidate_predictions",
        }
        if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifact_names:
            raise ValueError(f"{stem}.artifacts does not have the frozen five-file schema")
        for artifact_name in sorted(expected_artifact_names):
            artifact = artifacts[artifact_name]
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
                raise ValueError(f"{stem}.artifacts.{artifact_name} is malformed")
            artifact_path = Path(artifact["path"]).resolve()
            _require_tree_equal(
                file_record(artifact_path),
                artifact,
                f"{stem}.artifacts.{artifact_name}",
            )

        candidate_base = (
            approval_path.parent / audit_dataset / f"{audit_dataset}_fold1_opt2024"
        ).resolve()
        expected_candidate_paths = {
            "candidate_json": candidate_base.with_suffix(".json"),
            "candidate_pt": candidate_base.with_suffix(".pt"),
            "candidate_predictions": candidate_base.with_suffix(".npz"),
        }
        for artifact_name, expected_path in expected_candidate_paths.items():
            if Path(artifacts[artifact_name]["path"]).resolve() != expected_path:
                raise ValueError(f"{stem}.{artifact_name} path is not the released cell")
        records.setdefault("reference_artifacts", {})[audit_dataset] = {
            name: dict(artifacts[name])
            for name in ("reference_json", "reference_pt")
        }
    return records


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _check_seed_provenance(payload: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("SEED provenance must be an object")
    expected_hashes = {
        "runner_sha256": FROZEN_SOURCE_SHA256["seed_runner"],
        "loader_sha256": FROZEN_SOURCE_SHA256["seed_loader"],
        "fold_definition_sha256": FROZEN_SOURCE_SHA256["fold_definition"],
        "libeer_dgcnn_sha256": FROZEN_SOURCE_SHA256["libeer_dgcnn"],
        "libeer_dgcnn_config_sha256": FROZEN_SOURCE_SHA256["libeer_dgcnn_config"],
    }
    for name, expected in expected_hashes.items():
        if provenance.get(name) != expected:
            raise ValueError(f"provenance.{name} is not frozen")
    source_files = provenance.get("source_files")
    source_hashes = provenance.get("source_file_sha256")
    if not isinstance(source_files, Mapping) or set(map(str, source_files)) != set(
        map(str, SUBJECTS)
    ):
        raise ValueError("provenance.source_files must cover all 15 subjects")
    if not isinstance(source_hashes, Mapping) or set(map(str, source_hashes)) != set(
        map(str, SUBJECTS)
    ):
        raise ValueError("provenance.source_file_sha256 must cover all 15 subjects")
    for subject, digest in source_hashes.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid source SHA-256 for subject {subject}")
    label_digest = provenance.get("label_sha256")
    if not isinstance(label_digest, str) or len(label_digest) != 64:
        raise ValueError("provenance.label_sha256 must be a SHA-256")
    _require_tree_equal(provenance, checkpoint.get("provenance"), "checkpoint.provenance")


def _check_seediv_full_commit(payload: Mapping[str, Any]) -> None:
    preprocessing = payload.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("SEED-IV preprocessing provenance is required")
    expected_preprocessing = {
        "dataset": "seediv_raw",
        "time_window_seconds": 1,
        "feature_type": "de_lds",
        "normalization": "none in the DGCNN path",
        "cross_subject_statistics": "none",
    }
    for name, expected in expected_preprocessing.items():
        if preprocessing.get(name) != expected:
            raise ValueError(f"preprocessing.{name} is not frozen")
    records = preprocessing.get("subjects")
    if not isinstance(records, Mapping) or set(map(str, records)) != set(map(str, SUBJECTS)):
        raise ValueError("SEED-IV preprocessing.subjects must cover all 15 subjects")
    for subject in SUBJECTS:
        record = records[str(subject)]
        if not isinstance(record, Mapping):
            raise ValueError(f"preprocessing.subjects.{subject} must be an object")
        if _exact_int(record.get("trial_count"), "trial_count") != 24:
            raise ValueError(f"subject {subject} preprocessing trial_count must be 24")
        manifest = record.get("manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError(f"subject {subject} cache manifest is required")
        if manifest.get("libeer_commit") != FULL_LIBEER_COMMIT:
            raise ValueError(f"subject {subject} cache manifest lacks the frozen full commit")
        if _exact_int(manifest.get("subject"), "manifest.subject") != subject:
            raise ValueError(f"subject {subject} cache manifest subject mismatch")
        if _exact_int(manifest.get("session"), "manifest.session") != SESSION:
            raise ValueError(f"subject {subject} cache manifest session mismatch")
        labels = manifest.get("trial_labels")
        shapes = manifest.get("trial_shapes")
        if not isinstance(labels, list) or len(labels) != 24:
            raise ValueError(f"subject {subject} cache manifest must contain 24 trial labels")
        if np.bincount(np.asarray(labels, dtype=int), minlength=4).tolist() != [6, 6, 6, 6]:
            raise ValueError(f"subject {subject} cache labels are not class-balanced")
        if not isinstance(shapes, list) or len(shapes) != 24:
            raise ValueError(f"subject {subject} cache manifest must contain 24 shapes")


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error
    return value.lower()


def _validate_cross_cell_data_provenance(
    dataset: str, inputs: Sequence[Path]
) -> dict[str, Any]:
    """Require one immutable data identity across all folds and seeds.

    Raw source files are content-hashed once here, after the complete grid is
    available.  Cell-level validation separately binds the exact trial labels
    and window counts carried into each test NPZ.
    """

    payloads = [_load_json(Path(path).resolve()) for path in inputs]
    records: dict[str, Any] = {
        "declared_freeze_digests": {
            "seed_session1_metadata": (
                "142ae024c1deb45f39d955e083168ae3f4f36b034b0f2c455d46f8df9ade123a"
            ),
            "seediv_session1_raw_metadata": (
                "d722f4810dd14de5916ce9506ca9cd6ac04e74db039305ce2489f77ad9ff89bd"
            ),
            "seediv_complete_raw_hash_manifest": (
                "6f0af43c1dbcae4adedcad5056c959c629af38a3cdac060306f5d8ec08efa994"
            ),
        }
    }
    if dataset == "seed":
        canonical = payloads[0].get("provenance")
        if not isinstance(canonical, Mapping):
            raise ValueError("SEED canonical provenance is missing")
        for index, payload in enumerate(payloads[1:], start=2):
            _require_tree_equal(
                canonical,
                payload.get("provenance"),
                f"seed.provenance.cell{index}",
            )
        source_files = canonical.get("source_files")
        source_hashes = canonical.get("source_file_sha256")
        if not isinstance(source_files, Mapping) or not isinstance(source_hashes, Mapping):
            raise ValueError("SEED source-file provenance is incomplete")
        records["raw_subject_files"] = {}
        for subject in SUBJECTS:
            path = Path(str(source_files[str(subject)])).resolve()
            expected = _require_sha256(
                source_hashes[str(subject)], f"source_file_sha256.{subject}"
            )
            actual_record = file_record(path)
            if actual_record.get("sha256") != expected:
                raise ValueError(f"SEED subject {subject} raw feature hash mismatch")
            records["raw_subject_files"][str(subject)] = actual_record
        label_path = Path(str(canonical.get("label_file"))).resolve()
        label_expected = _require_sha256(
            canonical.get("label_sha256"), "label_sha256"
        )
        label_record = file_record(label_path)
        if label_record.get("sha256") != label_expected:
            raise ValueError("SEED label file hash mismatch")
        records["label_file"] = label_record
        return records

    canonical_preprocessing = payloads[0].get("preprocessing")
    if not isinstance(canonical_preprocessing, Mapping):
        raise ValueError("SEED-IV canonical preprocessing provenance is missing")
    canonical_subjects = canonical_preprocessing.get("subjects")
    if not isinstance(canonical_subjects, Mapping):
        raise ValueError("SEED-IV canonical subject provenance is missing")
    canonical_manifests = {
        str(subject): canonical_subjects[str(subject)]["manifest"]
        for subject in SUBJECTS
    }
    for index, payload in enumerate(payloads[1:], start=2):
        preprocessing = payload.get("preprocessing")
        subjects = preprocessing.get("subjects") if isinstance(preprocessing, Mapping) else None
        if not isinstance(subjects, Mapping):
            raise ValueError(f"SEED-IV cell {index} lacks subject provenance")
        observed_manifests = {
            str(subject): subjects[str(subject)].get("manifest")
            for subject in SUBJECTS
        }
        _require_tree_equal(
            canonical_manifests,
            observed_manifests,
            f"seediv.preprocessing.cell{index}",
        )
    records["raw_subject_files"] = {}
    records["cache_manifests"] = {}
    records["cache_files"] = {}
    for subject in SUBJECTS:
        subject_record = canonical_subjects[str(subject)]
        manifest = canonical_manifests[str(subject)]
        if not isinstance(subject_record, Mapping) or not isinstance(manifest, Mapping):
            raise ValueError(f"SEED-IV subject {subject} provenance is malformed")
        raw_path = Path(str(manifest.get("source_file"))).resolve()
        raw_expected = _require_sha256(
            manifest.get("raw_sha256"), f"seediv.{subject}.raw_sha256"
        )
        raw_record = file_record(raw_path)
        if raw_record.get("sha256") != raw_expected:
            raise ValueError(f"SEED-IV subject {subject} raw file hash mismatch")
        if raw_record.get("size_bytes") != _exact_int(
            manifest.get("raw_size_bytes"), f"seediv.{subject}.raw_size_bytes"
        ):
            raise ValueError(f"SEED-IV subject {subject} raw file size mismatch")
        records["raw_subject_files"][str(subject)] = raw_record

        manifest_path = Path(str(subject_record.get("cache_manifest_file"))).resolve()
        manifest_payload = _load_json(manifest_path)
        _require_tree_equal(
            manifest, manifest_payload, f"seediv.{subject}.cache_manifest"
        )
        records["cache_manifests"][str(subject)] = file_record(manifest_path)
        cache_path = Path(str(subject_record.get("cache_file"))).resolve()
        cache_record = file_record(cache_path)
        if not cache_record.get("exists") or cache_record.get("size_bytes", 0) <= 0:
            raise ValueError(f"SEED-IV subject {subject} cache file is missing or empty")
        records["cache_files"][str(subject)] = cache_record
    return records


def _history_score(row: Mapping[str, Any], dataset: str, index: int) -> float:
    if dataset == "seed":
        validation = row.get("validation_window")
        if not isinstance(validation, Mapping):
            raise ValueError(f"history[{index}].validation_window is required")
        score = _finite_float(validation.get("macro_f1"), f"history[{index}].macro_f1")
        if row.get("checkpoint_metric") != "macro_f1":
            raise ValueError(f"history[{index}].checkpoint_metric must be macro_f1")
        if not _numeric_close(row.get("checkpoint_value"), score):
            raise ValueError(f"history[{index}].checkpoint_value mismatch")
    else:
        score = _finite_float(
            row.get("validation_macro_f1"), f"history[{index}].validation_macro_f1"
        )
        _finite_float(row.get("validation_accuracy"), f"history[{index}].validation_accuracy")
    return score


def _validate_history(payload: Mapping[str, Any], dataset: str) -> list[Mapping[str, Any]]:
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise ValueError(f"history must contain exactly {EPOCHS} rows")
    scores: list[float] = []
    running = -math.inf
    for index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"history[{index}] must be an object")
        if _exact_int(row.get("epoch"), f"history[{index}].epoch") != index:
            raise ValueError("history epochs must be exactly 1..150")
        _finite_float(row.get("train_loss"), f"history[{index}].train_loss")
        _finite_float(row.get("seconds"), f"history[{index}].seconds")
        score = _history_score(row, dataset, index)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"history[{index}] validation score outside [0, 1]")
        updated = score > running
        if dataset == "seed" and row.get("checkpoint_updated") is not updated:
            raise ValueError(f"history[{index}].checkpoint_updated violates strict-greater rule")
        if updated:
            running = score
        scores.append(score)
    earliest_index = int(np.argmax(np.asarray(scores)))
    best_epoch = earliest_index + 1
    best_value = scores[earliest_index]
    if _exact_int(payload.get("best_epoch"), "best_epoch") != best_epoch:
        raise ValueError("best_epoch is not the earliest maximum validation epoch")
    if not _numeric_close(payload.get("best_validation_value"), best_value):
        raise ValueError("best_validation_value is not the history maximum")
    if dataset == "seed":
        best_validation = payload.get("best_validation")
        if not isinstance(best_validation, Mapping):
            raise ValueError("SEED best_validation is required")
        _require_tree_equal(
            history[earliest_index]["validation_window"],
            best_validation.get("window"),
            "best_validation.window",
        )
    return history


def _expected_batch_command(
    payload: Mapping[str, Any],
    dataset: str,
    json_path: Path,
    checkpoint_path: Path,
    prediction_path: Path,
) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    spec = DATASET_SPECS[dataset]
    command = [
        os.environ.get("PCMA_PYTHON", sys.executable),
        str(root / "experiments" / f"run_{dataset}_dgcnn_subject_fold.py"),
        "--libeer-root",
        os.environ.get("LIBEER_ROOT", str(root / "external" / "LibEER")),
        "--expected-commit",
        SHORT_LIBEER_COMMIT,
    ]
    if dataset == "seed":
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("SEED provenance is required for command audit")
        feature_root = Path(str(provenance.get("label_file"))).resolve().parent
        command.extend(["--feature-root", str(feature_root)])
    else:
        preprocessing = payload.get("preprocessing")
        subjects = preprocessing.get("subjects") if isinstance(preprocessing, Mapping) else None
        if not isinstance(subjects, Mapping):
            raise ValueError("SEED-IV preprocessing is required for command audit")
        first = subjects["1"]
        if not isinstance(first, Mapping):
            raise ValueError("SEED-IV subject-1 preprocessing is malformed")
        cache_root = Path(str(first.get("cache_file"))).resolve().parent
        command.extend(
            [
                "--dataset-root",
                str(root / "data" / "SEED" / "SEED_IV"),
                "--cache-root",
                str(cache_root),
                "--session",
                str(SESSION),
            ]
        )
    command.extend(
        [
            "--output",
            str(json_path),
            "--checkpoint",
            str(checkpoint_path),
            "--predictions",
            str(prediction_path),
            "--fold",
            str(payload["fold"]),
            "--partition-seed",
            str(PARTITION_SEED),
            "--optimization-seed",
            str(payload["optimization_seed"]),
            "--epochs",
            str(EPOCHS),
            "--batch-size",
            str(spec.batch_size),
            "--eval-batch-size",
            str(EVAL_BATCH_SIZE),
            "--lr",
            str(spec.learning_rate),
            "--device",
            "cuda",
        ]
    )
    return command


def _validate_log(
    log_path: Path,
    history: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    dataset: str,
    json_path: Path,
    checkpoint_path: Path,
    prediction_path: Path,
) -> None:
    if not log_path.is_file() or log_path.stat().st_size == 0:
        raise FileNotFoundError(f"non-empty log not found: {log_path}")
    raw = log_path.read_bytes()
    # Windows PowerShell 5.1 Tee-Object writes UTF-16LE, whereas synthetic and
    # newer PowerShell logs are normally UTF-8.  Decode the on-disk evidence
    # according to its BOM instead of silently replacing every NUL byte.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    error_tokens = (
        "ERROR:",
        "RuntimeError",
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "BatchIntegrityError",
        "PartialCellError",
        "runner exited with return code",
    )
    found = [token for token in error_tokens if token.lower() in text.lower()]
    if found:
        raise ValueError(f"log contains terminal error token(s): {found}")
    logged_history: list[Mapping[str, Any]] = []
    batch_events: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(BATCH_EVENT_PREFIX):
            try:
                event = json.loads(stripped[len(BATCH_EVENT_PREFIX) :])
            except json.JSONDecodeError as error:
                raise ValueError("log contains malformed BATCH_EVENT JSON") from error
            if not isinstance(event, Mapping):
                raise ValueError("log BATCH_EVENT is not an object")
            batch_events.append(event)
            continue
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "epoch" in value and "train_loss" in value:
            logged_history.append(value)
    if len(logged_history) != EPOCHS:
        raise ValueError(f"log contains {len(logged_history)} compact epoch rows, expected 150")
    _require_tree_equal(history, logged_history, "log.history")
    if '"status": "completed"' not in text or '"elapsed_seconds"' not in text:
        raise ValueError("log lacks the final completed payload markers")

    fold = _exact_int(payload.get("fold"), "fold")
    optimization_seed = _exact_int(
        payload.get("optimization_seed"), "optimization_seed"
    )
    is_stage1 = fold == 1 and optimization_seed == 2024
    if is_stage1:
        if batch_events:
            raise ValueError("manually released stage1 log unexpectedly has batch events")
        return
    if len(batch_events) != 2:
        raise ValueError(f"non-stage1 log has {len(batch_events)} batch events, expected 2")
    start, exit_event = batch_events
    expected_cell = {
        "dataset": dataset,
        "fold": fold,
        "optimization_seed": optimization_seed,
        "partition_seed": PARTITION_SEED,
        "stage": {2024: 2, 2025: 3, 2026: 4}[optimization_seed],
        "cell_id": f"{dataset}_fold{fold}_opt{optimization_seed}",
    }
    expected_code = {
        name: FROZEN_SOURCE_SHA256[name]
        for name in (
            "seed_runner",
            "seediv_runner",
            "fold_definition",
            "seed_loader",
            "seediv_loader",
            "aggregator",
        )
    }
    expected_start = {
        "event": "cell_start",
        "protocol_version": "seed_family_dgcnn_strict_5fold_v1",
        "cell": expected_cell,
        "command": _expected_batch_command(
            payload,
            dataset,
            json_path,
            checkpoint_path,
            prediction_path,
        ),
        "code_sha256": expected_code,
    }
    _require_tree_equal(expected_start, start, "log.batch_start")
    _require_tree_equal(
        {
            "event": "cell_exit",
            "cell_id": expected_cell["cell_id"],
            "return_code": 0,
        },
        exit_event,
        "log.batch_exit",
    )


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"cannot load checkpoint on CPU: {type(error).__name__}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    model = value.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("checkpoint model state must be a non-empty mapping")
    for name, tensor in model.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint model.{name} is not a tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"checkpoint model.{name} was not mapped to CPU")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"checkpoint model.{name} contains non-finite values")
    return value


def _checkpoint_model_schema(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    model = checkpoint["model"]
    assert isinstance(model, Mapping)
    return {
        str(name): {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(model.items(), key=lambda item: str(item[0]))
    }


def _integer_array(value: np.ndarray, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"prediction {field} must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


def _load_prediction_rows(path: Path, spec: DatasetSpec) -> PredictionRows:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_NPZ_KEYS:
            raise ValueError(
                f"prediction NPZ keys {sorted(archive.files)} != {sorted(REQUIRED_NPZ_KEYS)}"
            )
        logits = np.asarray(archive["logits"])
        labels = _integer_array(archive["labels"], "labels")
        subject = _integer_array(archive["subject"], "subject")
        session = _integer_array(archive["session"], "session")
        trial = _integer_array(archive["trial"], "trial")
    if logits.ndim != 2 or logits.shape[1] != spec.num_classes:
        raise ValueError(f"logits must have shape (n, {spec.num_classes})")
    count = len(labels)
    if count == 0 or any(len(item) != count for item in (logits, subject, session, trial)):
        raise ValueError("prediction arrays must share one positive row count")
    if logits.dtype.kind not in "fiu" or not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite numeric values")
    if labels.min() < 0 or labels.max() >= spec.num_classes:
        raise ValueError("prediction labels fall outside the frozen class range")
    if set(np.unique(labels).tolist()) != set(range(spec.num_classes)):
        raise ValueError("prediction rows must contain every class")
    if set(np.unique(session).tolist()) != {SESSION}:
        raise ValueError("prediction rows must contain session 1 only")
    return PredictionRows(
        logits=logits.astype(np.float64, copy=False),
        labels=labels,
        subject=subject,
        session=session,
        trial=trial,
    )


def _summary(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    subject_accuracy = [
        float(np.mean(y_true[groups == group] == y_pred[groups == group]))
        for group in np.unique(groups)
    ]
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "worst_subject_acc": float(np.min(subject_accuracy)),
        "mean_subject_acc": float(np.mean(subject_accuracy)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_per_class": {
            str(label): float(value) for label, value in zip(labels, recall)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _trial_rows(rows: PredictionRows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(
        zip(rows.subject.tolist(), rows.session.tolist(), rows.trial.tolist())
    ):
        groups.setdefault(tuple(map(int, key)), []).append(index)
    labels: list[int] = []
    predictions: list[int] = []
    subjects: list[int] = []
    for (subject, _session, _trial), indices in groups.items():
        index_array = np.asarray(indices, dtype=int)
        unique_labels = np.unique(rows.labels[index_array])
        if len(unique_labels) != 1:
            raise ValueError(f"trial {(subject, _session, _trial)} has inconsistent labels")
        labels.append(int(unique_labels[0]))
        predictions.append(int(np.argmax(rows.logits[index_array].mean(axis=0))))
        subjects.append(subject)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
    )


def _validate_trial_design(
    rows: PredictionRows,
    expected_subjects: tuple[int, ...],
    spec: DatasetSpec,
    preprocessing_subjects: Mapping[str, Any] | None = None,
) -> None:
    if set(np.unique(rows.subject).tolist()) != set(expected_subjects):
        raise ValueError("prediction subjects do not exactly match the frozen test fold")
    for subject in expected_subjects:
        subject_mask = rows.subject == subject
        trials = np.unique(rows.trial[subject_mask])
        if set(trials.tolist()) != set(range(1, spec.trials_per_subject + 1)):
            raise ValueError(f"subject {subject} does not contain the frozen trial IDs")
        labels: list[int] = []
        for trial in trials:
            trial_mask = subject_mask & (rows.trial == trial)
            unique = np.unique(rows.labels[trial_mask])
            if len(unique) != 1:
                raise ValueError(f"subject {subject} trial {trial} has inconsistent labels")
            observed_label = int(unique[0])
            observed_windows = int(trial_mask.sum())
            if spec.key == "seed":
                expected_label = SEED_TRIAL_LABELS[int(trial) - 1]
                expected_windows = SEED_TRIAL_WINDOW_COUNTS[int(trial) - 1]
            else:
                if not isinstance(preprocessing_subjects, Mapping):
                    raise ValueError("SEED-IV preprocessing subjects are required")
                record = preprocessing_subjects.get(str(subject))
                if not isinstance(record, Mapping):
                    raise ValueError(f"SEED-IV preprocessing subject {subject} is missing")
                manifest = record.get("manifest")
                if not isinstance(manifest, Mapping):
                    raise ValueError(f"SEED-IV subject {subject} manifest is missing")
                manifest_labels = manifest.get("trial_labels")
                shapes = manifest.get("trial_shapes")
                window_counts = record.get("trial_window_counts")
                if (
                    not isinstance(manifest_labels, list)
                    or len(manifest_labels) != spec.trials_per_subject
                    or not isinstance(shapes, list)
                    or len(shapes) != spec.trials_per_subject
                    or not isinstance(window_counts, list)
                    or len(window_counts) != spec.trials_per_subject
                ):
                    raise ValueError(
                        f"SEED-IV subject {subject} trial provenance is incomplete"
                    )
                expected_label = _exact_int(
                    manifest_labels[int(trial) - 1], "manifest.trial_label"
                )
                shape = shapes[int(trial) - 1]
                if (
                    not isinstance(shape, list)
                    or len(shape) != 3
                    or shape[1:] != [62, 5]
                ):
                    raise ValueError(
                        f"SEED-IV subject {subject} trial {trial} shape is not [W,62,5]"
                    )
                expected_windows = _exact_int(shape[0], "manifest.trial_shape.windows")
                if _exact_int(
                    window_counts[int(trial) - 1], "trial_window_counts"
                ) != expected_windows:
                    raise ValueError(
                        f"SEED-IV subject {subject} trial {trial} count/shape mismatch"
                    )
            if observed_label != expected_label:
                raise ValueError(
                    f"subject {subject} trial {trial} label does not match frozen provenance"
                )
            if observed_windows != expected_windows:
                raise ValueError(
                    f"subject {subject} trial {trial} window count does not match provenance"
                )
            labels.append(observed_label)
        counts = np.bincount(labels, minlength=spec.num_classes)
        if counts.tolist() != [spec.trials_per_class] * spec.num_classes:
            raise ValueError(f"subject {subject} trial labels are not class-balanced")


def recompute_test_metrics(rows: PredictionRows, spec: DatasetSpec) -> dict[str, Any]:
    window_prediction = rows.logits.argmax(axis=1)
    trial_labels, trial_prediction, trial_subject = _trial_rows(rows)
    per_subject: dict[str, Any] = {}
    for subject in sorted(np.unique(rows.subject).tolist()):
        window_mask = rows.subject == subject
        trial_mask = trial_subject == subject
        per_subject[str(subject)] = {
            "window": _summary(
                rows.labels[window_mask], window_prediction[window_mask], rows.subject[window_mask]
            ),
            "trial": _summary(
                trial_labels[trial_mask], trial_prediction[trial_mask], trial_subject[trial_mask]
            ),
            "window_count": int(window_mask.sum()),
            "trial_count": int(trial_mask.sum()),
        }
    window = _summary(rows.labels, window_prediction, rows.subject)
    trial = _summary(trial_labels, trial_prediction, trial_subject)
    return {
        "window": window,
        "trial": trial,
        "per_subject": per_subject,
        "worst": {
            "window_subject_accuracy": float(window["worst_subject_acc"]),
            "trial_subject_accuracy": float(trial["worst_subject_acc"]),
        },
    }


def _validate_cell(json_path: Path, dataset: str, log_root: Path) -> AuditedCell:
    spec = DATASET_SPECS[dataset]
    json_path = json_path.resolve()
    payload = _load_json(json_path)
    if payload.get("status") != "completed" or payload.get("dataset") != dataset:
        raise ValueError("status/dataset does not identify a completed frozen cell")
    if payload.get("purpose") != spec.purpose:
        raise ValueError("purpose is not frozen")
    fold = _exact_int(payload.get("fold"), "fold")
    optimization_seed = _exact_int(payload.get("optimization_seed"), "optimization_seed")
    if fold not in FOLDS or optimization_seed not in OPTIMIZATION_SEEDS:
        raise ValueError("fold/optimization_seed is outside the frozen cell grid")
    expected_stem = f"{dataset}_fold{fold}_opt{optimization_seed}"
    if json_path.stem != expected_stem:
        raise ValueError(f"JSON stem must be {expected_stem}")
    if payload.get("protocol") != spec.protocol_template.format(fold=fold):
        raise ValueError("protocol declaration is not frozen")
    if payload.get("target_access") != spec.target_access:
        raise ValueError("target_access declaration is not frozen")
    frozen_scalars = {
        "session": SESSION,
        "partition_seed": PARTITION_SEED,
        "test_evaluation_count": 1,
        "epochs": EPOCHS,
        "batch_size": spec.batch_size,
        "eval_batch_size": EVAL_BATCH_SIZE,
    }
    for name, expected in frozen_scalars.items():
        if _exact_int(payload.get(name), name) != expected:
            raise ValueError(f"{name} is not frozen")
    if not _numeric_close(payload.get("learning_rate"), spec.learning_rate):
        raise ValueError("learning_rate is not frozen")
    units = payload.get("prediction_units")
    if not isinstance(units, list) or set(units) != {
        "one_second_window",
        "trial_mean_logit",
    }:
        raise ValueError("prediction_units is not frozen")
    validation_fields = [
        payload[name] for name in ("validation_metric", "validation_selection") if name in payload
    ]
    if not validation_fields or any(value != VALIDATION_SELECTION for value in validation_fields):
        raise ValueError("validation selection is not frozen")
    split = _canonical_split(payload.get("split_one_based"), "split_one_based")
    if split != FROZEN_FOLDS[fold]:
        raise ValueError("split_one_based does not match the hardcoded frozen fold")
    zero_based = _canonical_split(payload.get("split_zero_based"), "split_zero_based")
    expected_zero = {
        name: tuple(subject - 1 for subject in members) for name, members in split.items()
    }
    if zero_based != expected_zero:
        raise ValueError("split_zero_based does not match the hardcoded frozen fold")

    if dataset == "seed":
        if payload.get("libeer_commit") != FULL_LIBEER_COMMIT:
            raise ValueError("SEED JSON does not record the frozen full LibEER commit")
        if payload.get("checkpoint_tie_rule") != (
            "strict greater-than; earliest epoch retained on ties"
        ):
            raise ValueError("checkpoint tie rule is not frozen")
        expected_regression_flag = fold == 1 and optimization_seed == 2024
        if payload.get("fold1_original_split_regression_compatible") is not expected_regression_flag:
            raise ValueError("fold-1 regression compatibility flag is inconsistent")
    else:
        if payload.get("libeer_commit") != SHORT_LIBEER_COMMIT:
            raise ValueError("SEED-IV JSON short commit is not frozen")
        if _exact_int(payload.get("num_classes"), "num_classes") != spec.num_classes:
            raise ValueError("SEED-IV num_classes is not frozen")
        _check_seediv_full_commit(payload)

    history = _validate_history(payload, dataset)
    checkpoint_path = _resolve_recorded_path(payload.get("checkpoint"), json_path, ".pt", "checkpoint")
    prediction_path = _resolve_recorded_path(
        payload.get("predictions"), json_path, ".npz", "predictions"
    )
    log_path = (log_root.resolve() / f"{expected_stem}.log").resolve()
    _validate_log(
        log_path,
        history,
        payload,
        dataset,
        json_path,
        checkpoint_path,
        prediction_path,
    )

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_expectations = {
        "fold": fold,
        "partition_seed": PARTITION_SEED,
        "optimization_seed": optimization_seed,
        "best_epoch": payload["best_epoch"],
        "validation_metric": spec.checkpoint_metric,
        "validation_value": payload["best_validation_value"],
        "split_one_based": split,
    }
    if dataset == "seed":
        checkpoint_expectations["session"] = SESSION
        checkpoint_expectations["libeer_commit"] = FULL_LIBEER_COMMIT
    else:
        checkpoint_expectations["libeer_commit"] = SHORT_LIBEER_COMMIT
    for name, expected in checkpoint_expectations.items():
        _require_tree_equal(expected, checkpoint.get(name), f"checkpoint.{name}")
    if dataset == "seed":
        _check_seed_provenance(payload, checkpoint)

    rows = _load_prediction_rows(prediction_path, spec)
    preprocessing = payload.get("preprocessing")
    preprocessing_subjects = (
        preprocessing.get("subjects") if isinstance(preprocessing, Mapping) else None
    )
    _validate_trial_design(rows, split["test"], spec, preprocessing_subjects)
    sample_counts = payload.get("sample_counts")
    if not isinstance(sample_counts, Mapping) or set(sample_counts) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("sample_counts must contain train/validation/test exactly")
    subject_counts = payload.get("subject_sample_counts")
    if not isinstance(subject_counts, Mapping) or set(map(str, subject_counts)) != set(
        map(str, SUBJECTS)
    ):
        raise ValueError("subject_sample_counts must cover all 15 subjects")
    normalized_subject_counts = {
        subject: _exact_int(
            subject_counts.get(str(subject)), f"subject_sample_counts.{subject}"
        )
        for subject in SUBJECTS
    }
    if any(count <= 0 for count in normalized_subject_counts.values()):
        raise ValueError("subject_sample_counts must be positive")
    for partition, members in split.items():
        expected_partition_count = sum(normalized_subject_counts[s] for s in members)
        if _exact_int(
            sample_counts.get(partition), f"sample_counts.{partition}"
        ) != expected_partition_count:
            raise ValueError(f"sample_counts.{partition} does not match subject totals")
    if normalized_subject_counts and sample_counts.get("test") != len(rows.labels):
        raise ValueError("sample_counts.test does not match NPZ")
    for subject in split["test"]:
        count = int(np.sum(rows.subject == subject))
        if normalized_subject_counts[subject] != count:
            raise ValueError(f"subject_sample_counts.{subject} does not match NPZ")
    metrics = recompute_test_metrics(rows, spec)
    reported_test = payload.get("test")
    if not isinstance(reported_test, Mapping):
        raise ValueError("test metrics must be an object")
    # The runners' frozen JSON contract requires window/trial/per-subject
    # summaries; the redundant top-level ``worst`` convenience block was
    # optional in the historical cells.  Audit it when present without making
    # its absence invalidate otherwise complete evidence.
    expected_reported = {
        name: value
        for name, value in metrics.items()
        if name != "worst" or "worst" in reported_test
    }
    _require_tree_equal(expected_reported, reported_test, "test")
    return AuditedCell(
        json_path=json_path,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        log_path=log_path,
        fold=fold,
        optimization_seed=optimization_seed,
        split=split,
        rows=rows,
        metrics=metrics,
        checkpoint_schema=_checkpoint_model_schema(checkpoint),
    )


def bca_mean_interval(
    subject_values: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = np.asarray(subject_values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("BCa values must be a one-dimensional finite subject vector")
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    observed = float(values.mean())
    if np.all(values == values[0]):
        return {
            "method": "BCa subject-row bootstrap",
            "confidence_level": 0.95,
            "n_subjects": int(len(values)),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
            "status": "DEGENERATE_CONSTANT",
            "observed": observed,
            "z0": 0.0,
            "acceleration": 0.0,
            "adjusted_quantiles": [0.025, 0.975],
            "low": observed,
            "high": observed,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    boot = values[indices].mean(axis=1)
    less = int(np.sum(boot < observed))
    equal = int(np.sum(boot == observed))
    proportion = (less + 0.5 * equal) / n_resamples
    boundary = 0.5 / n_resamples
    z0 = float(norm.ppf(float(np.clip(proportion, boundary, 1.0 - boundary))))
    jackknife = np.asarray(
        [np.delete(values, omitted).mean() for omitted in range(len(values))], dtype=float
    )
    delta = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(delta**2) ** 1.5)
    acceleration = 0.0 if denominator == 0.0 else float(np.sum(delta**3) / denominator)
    adjusted: list[float] = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        divisor = 1.0 - acceleration * (z0 + z_alpha)
        if divisor == 0.0:
            raise ValueError("BCa adjusted quantile is undefined")
        quantile = float(norm.cdf(z0 + (z0 + z_alpha) / divisor))
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("BCa adjusted quantile is outside [0, 1]")
        adjusted.append(quantile)
    low, high = np.quantile(boot, adjusted)
    return {
        "method": "BCa subject-row bootstrap",
        "confidence_level": 0.95,
        "n_subjects": int(len(values)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "status": "OK",
        "observed": observed,
        "z0": z0,
        "acceleration": acceleration,
        "adjusted_quantiles": adjusted,
        "low": float(low),
        "high": float(high),
    }


def _metric_record(
    subject_seed_metrics: Mapping[int, Mapping[int, Mapping[str, float]]], metric: str
) -> dict[str, Any]:
    per_seed = {
        str(seed): float(
            np.mean([subject_seed_metrics[subject][seed][metric] for subject in SUBJECTS])
        )
        for seed in OPTIMIZATION_SEEDS
    }
    per_subject = {
        subject: float(
            np.mean(
                [subject_seed_metrics[subject][seed][metric] for seed in OPTIMIZATION_SEEDS]
            )
        )
        for subject in SUBJECTS
    }
    seed_values = np.asarray(list(per_seed.values()), dtype=float)
    return {
        "estimate": float(np.mean(list(per_subject.values()))),
        "aggregation": "average three paired optimization seeds within subject, then equal-weight subjects",
        "per_seed_subject_balanced_mean": per_seed,
        "optimization_seed_population_sd": float(seed_values.std(ddof=0)),
        "optimization_seed_range": [float(seed_values.min()), float(seed_values.max())],
        "per_subject_seed_averaged": {str(key): value for key, value in per_subject.items()},
    }


def recompute_aggregate(
    dataset: str,
    cells: Sequence[AuditedCell],
    *,
    bootstrap: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    by_cell = {(cell.fold, cell.optimization_seed): cell for cell in cells}
    expected_grid = {(fold, seed) for fold in FOLDS for seed in OPTIMIZATION_SEEDS}
    if len(cells) != 15 or set(by_cell) != expected_grid:
        raise ValueError("validated cells do not form the exact 15-cell grid")
    metric_paths = {
        "trial_accuracy": ("trial", "accuracy"),
        "trial_macro_f1": ("trial", "macro_f1"),
        "trial_balanced_accuracy": ("trial", "balanced_accuracy"),
        "window_accuracy": ("window", "accuracy"),
        "window_macro_f1": ("window", "macro_f1"),
        "window_balanced_accuracy": ("window", "balanced_accuracy"),
    }
    subject_seed_metrics: dict[int, dict[int, dict[str, float]]] = {
        subject: {} for subject in SUBJECTS
    }
    subject_fold: dict[int, int] = {}
    trial_recall: dict[int, dict[int, dict[int, float]]] = {
        subject: {} for subject in SUBJECTS
    }
    pooled_rows: dict[int, list[PredictionRows]] = {seed: [] for seed in OPTIMIZATION_SEEDS}
    for seed in OPTIMIZATION_SEEDS:
        coverage = [
            subject for fold in FOLDS for subject in by_cell[(fold, seed)].split["test"]
        ]
        if sorted(coverage) != list(SUBJECTS) or len(set(coverage)) != 15:
            raise ValueError(f"optimization seed {seed} does not test each subject once")
    for cell in cells:
        pooled_rows[cell.optimization_seed].append(cell.rows)
        for subject in cell.split["test"]:
            record = cell.metrics["per_subject"][str(subject)]
            subject_seed_metrics[subject][cell.optimization_seed] = {
                metric: float(record[unit][name])
                for metric, (unit, name) in metric_paths.items()
            }
            trial_recall[subject][cell.optimization_seed] = {
                int(label): float(value)
                for label, value in record["trial"]["recall_per_class"].items()
            }
            subject_fold.setdefault(subject, cell.fold)
            if subject_fold[subject] != cell.fold:
                raise ValueError(f"subject {subject} changes test fold across seeds")
    for subject in SUBJECTS:
        if set(subject_seed_metrics[subject]) != set(OPTIMIZATION_SEEDS):
            raise ValueError(f"subject {subject} lacks three paired seeds")

    primary = _metric_record(subject_seed_metrics, "trial_accuracy")
    primary_values = [primary["per_subject_seed_averaged"][str(s)] for s in SUBJECTS]
    primary.update(
        {
            "metric": "subject-balanced trial mean-logit accuracy",
            "statistical_unit": "subject",
            "n_subjects": 15,
            "confidence_interval": bca_mean_interval(
                primary_values, n_resamples=bootstrap, seed=bootstrap_seed
            ),
        }
    )
    secondary = {
        metric: _metric_record(subject_seed_metrics, metric)
        for metric in metric_paths
        if metric != "trial_accuracy"
    }
    primary_array = np.asarray(primary_values, dtype=float)
    minimum = float(primary_array.min())
    secondary["subject_tail"] = {
        "basis": "per-subject trial accuracy after paired-seed averaging",
        "worst_accuracy": minimum,
        "worst_subjects": [
            subject
            for subject, value in zip(SUBJECTS, primary_array)
            if math.isclose(float(value), minimum, rel_tol=0.0, abs_tol=1e-15)
        ],
        "tenth_percentile": float(np.quantile(primary_array, 0.10)),
        "interpretation": "descriptive; not an additional inferential sample size",
    }
    detail: dict[str, Any] = {}
    class_estimates: dict[str, float] = {}
    for label in range(spec.num_classes):
        subject_means = {
            subject: float(
                np.mean([trial_recall[subject][seed][label] for seed in OPTIMIZATION_SEEDS])
            )
            for subject in SUBJECTS
        }
        detail[str(label)] = {
            "estimate": float(np.mean(list(subject_means.values()))),
            "per_subject_seed_averaged": {
                str(subject): value for subject, value in subject_means.items()
            },
        }
        class_estimates[str(label)] = float(np.mean(list(subject_means.values())))
    secondary["trial_recall_per_class"] = {
        "aggregation": "average paired seeds within subject, then equal-weight subjects",
        "estimates": class_estimates,
        "detail": detail,
    }
    pooled: dict[str, Any] = {}
    for seed in OPTIMIZATION_SEEDS:
        rows = pooled_rows[seed]
        labels = np.concatenate([item.labels for item in rows])
        prediction = np.concatenate([item.logits.argmax(axis=1) for item in rows])
        subjects = np.concatenate([item.subject for item in rows])
        summary = _summary(labels, prediction, subjects)
        pooled[str(seed)] = {
            "window_count": int(len(labels)),
            "accuracy": float(summary["accuracy"]),
            "macro_f1": float(summary["macro_f1"]),
            "balanced_accuracy": float(summary["balanced_accuracy"]),
        }
    secondary["pooled_window_compatibility"] = {
        "role": "non-inferential LibEER compatibility summary; longer trials receive more weight",
        "per_seed": pooled,
    }
    per_subject = {
        str(subject): {
            "test_fold": subject_fold[subject],
            "per_seed": {
                str(seed): subject_seed_metrics[subject][seed] for seed in OPTIMIZATION_SEEDS
            },
            "paired_seed_mean": {
                metric: float(
                    np.mean(
                        [
                            subject_seed_metrics[subject][seed][metric]
                            for seed in OPTIMIZATION_SEEDS
                        ]
                    )
                )
                for metric in metric_paths
            },
        }
        for subject in SUBJECTS
    }
    cell_records = [
        {
            "fold": cell.fold,
            "optimization_seed": cell.optimization_seed,
            "json": str(cell.json_path),
            "predictions": str(cell.prediction_path),
            "test_subjects": list(cell.split["test"]),
            "prediction_rows": int(len(cell.rows.labels)),
            "json_test_metrics_recomputed": True,
        }
        for cell in sorted(cells, key=lambda item: (item.fold, item.optimization_seed))
    ]
    return {
        "status": "completed",
        "analysis": "strict_seed_family_fivefold_aggregate",
        "dataset": dataset,
        "dataset_display_name": spec.display_name,
        "contract": {
            "partition_seed": PARTITION_SEED,
            "optimization_seeds": list(OPTIMIZATION_SEEDS),
            "folds": list(FOLDS),
            "session": SESSION,
            "epochs": EPOCHS,
            "validation_selection": VALIDATION_SELECTION,
            "test_evaluations_per_cell": 1,
            "primary_aggregation": (
                "trial mean-logit prediction; metric within subject and seed; "
                "paired-seed average within subject; equal-weight mean over 15 subjects"
            ),
            "known_result_disclosure": (
                "fold 1 / optimization seed 2024 existed before this all-cell completion contract"
            ),
        },
        "validation": {
            "input_json_count": 15,
            "prediction_npz_count": 15,
            "expected_cell_grid_complete": True,
            "every_subject_tested_once_per_seed": True,
            "json_test_metrics_recomputed_from_predictions": True,
            "test_windows_are_not_inferential_units": True,
            "optimization_seeds_are_paired_repeated_fits": True,
        },
        "cells": cell_records,
        "primary": primary,
        "secondary": secondary,
        "per_subject": per_subject,
        "claim_boundary": (
            "Pinned archival DGCNN, first-session fixed five-fold 9/3/3 subject-disjoint "
            "evaluation on the 15 benchmark subjects. This is not LOSO, cross-session, "
            "unseen-stimulus, clinical, SOTA, or architecture-general evidence."
        ),
    }


def _artifact_records(inputs: Sequence[Path], dataset: str, log_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in inputs:
        json_path = Path(path).resolve()
        stem = json_path.stem
        records.append(
            {
                "cell": stem,
                "json": file_record(json_path),
                "checkpoint": file_record(json_path.with_suffix(".pt")),
                "predictions": file_record(json_path.with_suffix(".npz")),
                "log": file_record(log_root.resolve() / f"{stem}.log"),
            }
        )
    return records


def validate_cells_and_recompute(
    dataset: str,
    inputs: Sequence[Path],
    log_root: Path,
    *,
    bootstrap: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[list[AuditedCell], dict[str, Any]]:
    """Validate all cell quartets and return an independently rebuilt aggregate.

    This helper is intentionally public for deterministic synthetic tests and
    offline verification tooling.  It never imports the production aggregator.
    """

    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if len(inputs) != 15:
        raise ValueError("exactly 15 JSON inputs are required")
    resolved = [Path(path).resolve() for path in inputs]
    if len(set(resolved)) != 15:
        raise ValueError("the 15 input JSON paths must be distinct")
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    cells = [_validate_cell(path, dataset, Path(log_root)) for path in resolved]
    grid = {(cell.fold, cell.optimization_seed) for cell in cells}
    expected = {(fold, seed) for fold in FOLDS for seed in OPTIMIZATION_SEEDS}
    if len(grid) != 15 or grid != expected:
        raise ValueError("input metadata does not form the exact fold/seed grid")
    if len({cell.checkpoint_path for cell in cells}) != 15:
        raise ValueError("checkpoint paths must be unique")
    if len({cell.prediction_path for cell in cells}) != 15:
        raise ValueError("prediction paths must be unique")
    if len({cell.log_path for cell in cells}) != 15:
        raise ValueError("log paths must be unique")
    stage1_cell = next(
        cell
        for cell in cells
        if cell.fold == 1 and cell.optimization_seed == 2024
    )
    for cell in cells:
        _require_tree_equal(
            stage1_cell.checkpoint_schema,
            cell.checkpoint_schema,
            f"checkpoint_schema.{cell.json_path.stem}",
        )
    aggregate = recompute_aggregate(
        dataset, cells, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed
    )
    return cells, aggregate


def audit_dataset(
    dataset: str,
    inputs: Sequence[Path],
    aggregate_path: Path,
    log_root: Path,
    stage1_approval_path: Path,
    *,
    bootstrap: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    recorder = AuditRecorder()
    recorder.add(
        "bootstrap.frozen_resamples",
        bootstrap == BOOTSTRAP_RESAMPLES,
        expected=BOOTSTRAP_RESAMPLES,
        actual=bootstrap,
        include_values=True,
    )
    recorder.add(
        "bootstrap.frozen_seed",
        bootstrap_seed == BOOTSTRAP_SEED,
        expected=BOOTSTRAP_SEED,
        actual=bootstrap_seed,
        include_values=True,
    )
    source_binding = _source_freeze_checks(recorder, dataset)
    resolved_inputs = [Path(path).resolve() for path in inputs]
    data_provenance_binding: dict[str, Any] = {}
    if len(resolved_inputs) == 15 and len(set(resolved_inputs)) == 15:
        try:
            data_provenance_binding = _validate_cross_cell_data_provenance(
                dataset, resolved_inputs
            )
        except Exception as error:
            recorder.add(
                "data_provenance.cross_cell_and_source_binding",
                False,
                detail=f"{type(error).__name__}: {error}",
            )
        else:
            recorder.add("data_provenance.cross_cell_and_source_binding", True)
    stage1_binding = _stage1_bundle_records(Path(stage1_approval_path))
    try:
        stage1_binding = _validate_stage1_bundle(
            Path(stage1_approval_path), resolved_inputs, dataset
        )
    except Exception as error:
        recorder.add(
            "stage1.release_bundle",
            False,
            detail=f"{type(error).__name__}: {error}",
        )
    else:
        recorder.add("stage1.release_bundle", True)
    recorder.add("grid.input_count", len(resolved_inputs) == 15, expected=15, actual=len(resolved_inputs), include_values=True)
    recorder.add("grid.input_paths_unique", len(set(resolved_inputs)) == len(resolved_inputs))
    artifact_binding = _artifact_records(resolved_inputs, dataset, Path(log_root))
    all_four_exist = all(
        all(record[kind]["exists"] for kind in ("json", "checkpoint", "predictions", "log"))
        for record in artifact_binding
    )
    recorder.add("artifacts.all_15_quartets_exist", len(artifact_binding) == 15 and all_four_exist)

    cells: list[AuditedCell] = []
    recomputed: dict[str, Any] | None = None
    if len(resolved_inputs) == 15 and len(set(resolved_inputs)) == 15:
        for path in resolved_inputs:
            try:
                cell = _validate_cell(path, dataset, Path(log_root))
            except Exception as error:
                recorder.add(
                    f"cell.{path.stem}",
                    False,
                    detail=f"{type(error).__name__}: {error}",
                )
            else:
                cells.append(cell)
                recorder.add(f"cell.{path.stem}", True)
        if len(cells) == 15:
            try:
                recomputed = recompute_aggregate(
                    dataset,
                    cells,
                    bootstrap=bootstrap,
                    bootstrap_seed=bootstrap_seed,
                )
            except Exception as error:
                recorder.add(
                    "aggregate.independent_recomputation",
                    False,
                    detail=f"{type(error).__name__}: {error}",
                )
            else:
                recorder.add("aggregate.independent_recomputation", True)
    if len(cells) != 15:
        recorder.add(
            "grid.validated_cell_count",
            False,
            expected=15,
            actual=len(cells),
            include_values=True,
        )
    else:
        recorder.add("grid.validated_cell_count", True)
        stage1_cell = next(
            cell
            for cell in cells
            if cell.fold == 1 and cell.optimization_seed == 2024
        )
        schema_mismatches = [
            cell.json_path.stem
            for cell in cells
            if _tree_mismatches(
                stage1_cell.checkpoint_schema,
                cell.checkpoint_schema,
                "checkpoint_schema",
            )
        ]
        recorder.add(
            "checkpoint.all_cells_match_stage1_model_schema",
            not schema_mismatches,
            detail=(
                "schema mismatch: " + ", ".join(schema_mismatches)
                if schema_mismatches
                else None
            ),
        )

    aggregate_path = Path(aggregate_path).resolve()
    aggregate_binding = file_record(aggregate_path)
    if not aggregate_path.is_file():
        recorder.add("aggregate.file_exists", False, detail=str(aggregate_path))
    else:
        recorder.add("aggregate.file_exists", True)
        try:
            observed = _load_json(aggregate_path)
        except Exception as error:
            recorder.add(
                "aggregate.load",
                False,
                detail=f"{type(error).__name__}: {error}",
            )
        else:
            recorder.add("aggregate.load", True)
            if recomputed is not None:
                expected_keys = sorted(recomputed)
                observed_keys = sorted(observed)
                recorder.add(
                    "aggregate.top_level_schema",
                    observed_keys == expected_keys,
                    expected=expected_keys,
                    actual=observed_keys,
                    include_values=True,
                )
                for section in (
                    "status",
                    "analysis",
                    "dataset",
                    "dataset_display_name",
                    "contract",
                    "validation",
                    "cells",
                    "primary",
                    "secondary",
                    "per_subject",
                    "claim_boundary",
                ):
                    mismatches = _tree_mismatches(
                        recomputed[section], observed.get(section), f"aggregate.{section}"
                    )
                    recorder.add(
                        f"aggregate.section.{section}",
                        not mismatches,
                        detail="; ".join(mismatches) if mismatches else None,
                    )

    auditor_binding = file_record(Path(__file__).resolve())
    passed_count = sum(check["passed"] for check in recorder.checks)
    failed_count = len(recorder.checks) - passed_count
    report = {
        "status": "PASS" if recorder.passed else "FAIL",
        "analysis": "strict_seed_family_fivefold_independent_aggregate_audit",
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "independent_of_production_aggregator_imports": True,
            "artifact_quartets": 15,
            "minimum_core_files_bound": 74,
            "data_source_and_cache_files_are_additionally_bound": True,
            "partition_seed": PARTITION_SEED,
            "optimization_seeds": list(OPTIMIZATION_SEEDS),
            "folds": list(FOLDS),
            "full_libeer_commit": FULL_LIBEER_COMMIT,
            "bootstrap_resamples": int(bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "statistical_unit": "15 paired-seed subject rows",
            "numeric_relative_tolerance": FLOAT_REL_TOL,
            "numeric_absolute_tolerance": FLOAT_ABS_TOL,
        },
        "summary": {
            "checks": len(recorder.checks),
            "passed": passed_count,
            "failed": failed_count,
            "validated_cells": len(cells),
        },
        "source_binding": source_binding,
        "auditor_binding": auditor_binding,
        "aggregate_binding": aggregate_binding,
        "stage1_binding": stage1_binding,
        "data_provenance_binding": data_provenance_binding,
        "artifact_binding": artifact_binding,
        "recomputed": (
            {
                "primary": recomputed["primary"],
                "secondary": recomputed["secondary"],
                "per_subject": recomputed["per_subject"],
            }
            if recomputed is not None
            else None
        ),
        "checks": recorder.checks,
    }
    return _json_safe(report)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit report: {output}")
    report = audit_dataset(
        args.dataset,
        args.inputs,
        args.aggregate,
        args.log_root,
        args.stage1_approval,
        bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
