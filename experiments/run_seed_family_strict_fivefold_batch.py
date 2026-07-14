"""Sequential, fail-closed orchestration for the frozen SEED-family matrix.

The protocol is ``seed_family_dgcnn_strict_5fold_v1``: two datasets, five
subject folds, and optimization seeds 2024/2025/2026 with partition seed 2024.
There is deliberately no performance-dependent stopping or seed replacement.

Recovery is cell-atomic.  A completed cell is skipped only after its JSON,
checkpoint, NPZ, and log pass basic integrity and frozen-metadata checks.  Any
partial or invalid cell stops the batch and must be manually moved to a
quarantine location before the identical cell is rerun; this script never
deletes, overwrites, or resumes a training process mid-cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROTOCOL_VERSION = "seed_family_dgcnn_strict_5fold_v1"
PARTITION_SEED = 2024
OPTIMIZATION_SEEDS = (2024, 2025, 2026)
DATASETS = ("seed", "seediv")
BATCH_EVENT_PREFIX = "BATCH_EVENT\t"

# These values are replaced only when the frozen implementation is deliberately
# amended.  They make a resumed batch fail closed if code drifts between cells.
FROZEN_CODE_SHA256 = {
    "seed_runner": "3de0cc948840a24ec7704c910beb3a429594d5e30835893f33e0705d7a323d95",
    "seediv_runner": "87d73dd92103c61f91c91af3c93463f3677f0c9e4295226113bdb7e3b9a92a2c",
    "fold_definition": "275273dfb227000451b15352ad042b9e5047e6c8b04fbe00edf2bad48557930b",
    "seed_loader": "9372e319847cd9d4426e0b4ffa5481c3c9e33d5d5e985b26854889530c86578e",
    "seediv_loader": "3979f5c37c3add4abcfbfd99686bbe5f1c55f2eef926a78143512ab339076c1e",
    "aggregator": "f4e238a1bb24c3397bfc4614cc48b1d49eeb6174ae00a4d4db9240a40fa3502a",
}

FROZEN_SPLITS = {
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

DATASET_SETTINGS = {
    "seed": {
        "purpose": "strict_all_subject_seed_fold_baseline",
        "classes": 3,
        "epochs": 150,
        "batch_size": 16,
        "eval_batch_size": 512,
        "learning_rate": 0.001,
        "validation_field": "validation_metric",
        "validation_value": "pooled_window_macro_f1",
    },
    "seediv": {
        "purpose": "strict_all_subject_cross_validation_fold",
        "classes": 4,
        "epochs": 150,
        "batch_size": 32,
        "eval_batch_size": 512,
        "learning_rate": 0.0015,
        "validation_field": "validation_selection",
        "validation_value": "pooled_window_macro_f1",
    },
}


class BatchIntegrityError(RuntimeError):
    """The frozen matrix cannot safely continue."""


class PartialCellError(BatchIntegrityError):
    """At least one, but not all, cell artifacts exist."""


class ChildProcessFailed(RuntimeError):
    """A fold runner returned nonzero; its exact code is retained."""

    def __init__(self, cell: "Cell", returncode: int):
        super().__init__(f"{cell.cell_id} runner exited with return code {returncode}")
        self.cell = cell
        self.returncode = int(returncode)


@dataclass(frozen=True)
class Cell:
    dataset: str
    fold: int
    optimization_seed: int
    stage: int

    @property
    def cell_id(self) -> str:
        return f"{self.dataset}_fold{self.fold}_opt{self.optimization_seed}"

    def metadata(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "fold": self.fold,
            "optimization_seed": self.optimization_seed,
            "partition_seed": PARTITION_SEED,
            "stage": self.stage,
            "cell_id": self.cell_id,
        }


@dataclass(frozen=True)
class Artifacts:
    output: Path
    checkpoint: Path
    predictions: Path
    log: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (self.output, self.checkpoint, self.predictions, self.log)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--libeer-root", type=Path, required=True)
    parser.add_argument("--expected-commit", default="39dc27e")
    parser.add_argument("--seed-feature-root", type=Path, required=True)
    parser.add_argument("--seediv-dataset-root", type=Path, required=True)
    parser.add_argument("--seediv-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--stage1-approval", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for field in (
        "python",
        "libeer_root",
        "seed_feature_root",
        "seediv_dataset_root",
        "seediv_cache_root",
        "output_root",
        "log_root",
    ):
        setattr(args, field, Path(getattr(args, field)).resolve())
    if args.stage1_approval is not None:
        args.stage1_approval = args.stage1_approval.resolve()
    return args


def frozen_schedule() -> tuple[Cell, ...]:
    """Return the one and only 30-cell execution order."""

    cells: list[Cell] = []
    for dataset in DATASETS:
        cells.append(Cell(dataset, 1, 2024, 1))
    for fold in range(2, 6):
        for dataset in DATASETS:
            cells.append(Cell(dataset, fold, 2024, 2))
    for stage, optimization_seed in ((3, 2025), (4, 2026)):
        for fold in range(1, 6):
            for dataset in DATASETS:
                cells.append(Cell(dataset, fold, optimization_seed, stage))
    return tuple(cells)


def artifact_paths(cell: Cell, output_root: Path, log_root: Path) -> Artifacts:
    stem = cell.cell_id
    dataset_root = Path(output_root).resolve() / cell.dataset
    return Artifacts(
        output=dataset_root / f"{stem}.json",
        checkpoint=dataset_root / f"{stem}.pt",
        predictions=dataset_root / f"{stem}.npz",
        log=Path(log_root).resolve() / f"{stem}.log",
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def code_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "seed_runner": repo_root / "experiments" / "run_seed_dgcnn_subject_fold.py",
        "seediv_runner": repo_root / "experiments" / "run_seediv_dgcnn_subject_fold.py",
        "fold_definition": repo_root / "pcma" / "data" / "seed_folds.py",
        "seed_loader": repo_root / "pcma" / "data" / "libeer_seed.py",
        "seediv_loader": repo_root / "pcma" / "data" / "libeer_seediv.py",
        "aggregator": repo_root / "experiments" / "aggregate_seed_family_strict_fivefold.py",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_code(repo_root: Path) -> dict[str, str]:
    paths = code_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise BatchIntegrityError(f"frozen code file(s) missing: {', '.join(missing)}")
    actual = {name: file_sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": FROZEN_CODE_SHA256[name], "actual": digest}
        for name, digest in actual.items()
        if digest != FROZEN_CODE_SHA256[name]
    }
    if mismatches:
        raise BatchIntegrityError(
            "frozen code hash mismatch; do not mix implementations in one matrix: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def build_command(
    cell: Cell,
    artifacts: Artifacts,
    args: argparse.Namespace,
    repo_root: Path,
) -> list[str]:
    settings = DATASET_SETTINGS[cell.dataset]
    runner = code_paths(repo_root)[f"{cell.dataset}_runner"]
    command = [
        str(args.python),
        str(runner.resolve()),
        "--libeer-root",
        str(args.libeer_root),
        "--expected-commit",
        str(args.expected_commit),
    ]
    if cell.dataset == "seed":
        command.extend(["--feature-root", str(args.seed_feature_root)])
    else:
        command.extend(
            [
                "--dataset-root",
                str(args.seediv_dataset_root),
                "--cache-root",
                str(args.seediv_cache_root),
                "--session",
                "1",
            ]
        )
    command.extend(
        [
            "--output",
            str(artifacts.output),
            "--checkpoint",
            str(artifacts.checkpoint),
            "--predictions",
            str(artifacts.predictions),
            "--fold",
            str(cell.fold),
            "--partition-seed",
            str(PARTITION_SEED),
            "--optimization-seed",
            str(cell.optimization_seed),
            "--epochs",
            str(settings["epochs"]),
            "--batch-size",
            str(settings["batch_size"]),
            "--eval-batch-size",
            str(settings["eval_batch_size"]),
            "--lr",
            str(settings["learning_rate"]),
            "--device",
            args.device,
        ]
    )
    return command


def _normalized_split(value: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise BatchIntegrityError("split_one_based must be an object")
    if set(value) != {"train", "validation", "test"}:
        raise BatchIntegrityError("split_one_based has the wrong partition names")
    try:
        return {name: tuple(int(item) for item in value[name]) for name in value}
    except (TypeError, ValueError) as exc:
        raise BatchIntegrityError("split_one_based is not an integer subject split") from exc


def _require_equal(payload: Mapping[str, object], field: str, expected: object) -> None:
    if payload.get(field) != expected:
        raise BatchIntegrityError(
            f"JSON field {field!r} mismatch: expected {expected!r}, found {payload.get(field)!r}"
        )


def _require_float(payload: Mapping[str, object], field: str, expected: float) -> None:
    try:
        actual = float(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchIntegrityError(f"JSON field {field!r} is missing or nonnumeric") from exc
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise BatchIntegrityError(
            f"JSON field {field!r} mismatch: expected {expected}, found {actual}"
        )


def validate_result_json(cell: Cell, artifacts: Artifacts) -> dict[str, object]:
    try:
        payload = json.loads(artifacts.output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"invalid result JSON for {cell.cell_id}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BatchIntegrityError(f"result JSON for {cell.cell_id} is not an object")

    settings = DATASET_SETTINGS[cell.dataset]
    for field, expected in (
        ("status", "completed"),
        ("purpose", settings["purpose"]),
        ("test_evaluation_count", 1),
        ("session", 1),
        ("fold", cell.fold),
        ("partition_seed", PARTITION_SEED),
        ("optimization_seed", cell.optimization_seed),
        ("epochs", settings["epochs"]),
        ("batch_size", settings["batch_size"]),
        ("eval_batch_size", settings["eval_batch_size"]),
        (settings["validation_field"], settings["validation_value"]),
    ):
        _require_equal(payload, str(field), expected)
    _require_float(payload, "learning_rate", float(settings["learning_rate"]))

    if _normalized_split(payload.get("split_one_based")) != FROZEN_SPLITS[cell.fold]:
        raise BatchIntegrityError(f"frozen subject split mismatch for {cell.cell_id}")
    if Path(str(payload.get("checkpoint", ""))).resolve() != artifacts.checkpoint.resolve():
        raise BatchIntegrityError(f"checkpoint path mismatch for {cell.cell_id}")
    if Path(str(payload.get("predictions", ""))).resolve() != artifacts.predictions.resolve():
        raise BatchIntegrityError(f"predictions path mismatch for {cell.cell_id}")

    try:
        best_epoch = int(payload["best_epoch"])
        best_value = float(payload["best_validation_value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchIntegrityError(f"invalid best-checkpoint metadata for {cell.cell_id}") from exc
    if not 1 <= best_epoch <= int(settings["epochs"]) or not math.isfinite(best_value):
        raise BatchIntegrityError(f"invalid best-checkpoint values for {cell.cell_id}")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != int(settings["epochs"]):
        raise BatchIntegrityError(f"history length mismatch for {cell.cell_id}")

    test = payload.get("test")
    if not isinstance(test, Mapping) or not all(
        name in test for name in ("window", "trial", "per_subject")
    ):
        raise BatchIntegrityError(f"test metric structure is incomplete for {cell.cell_id}")
    expected_subjects = {str(subject) for subject in FROZEN_SPLITS[cell.fold]["test"]}
    per_subject = test["per_subject"]
    if not isinstance(per_subject, Mapping) or set(per_subject) != expected_subjects:
        raise BatchIntegrityError(f"test per-subject keys mismatch for {cell.cell_id}")
    return payload


def validate_predictions_npz(
    cell: Cell,
    artifacts: Artifacts,
    payload: Mapping[str, object],
) -> None:
    required = {"logits", "labels", "subject", "session", "trial"}
    try:
        with np.load(artifacts.predictions, allow_pickle=False) as stored:
            if set(stored.files) != required:
                raise BatchIntegrityError(
                    f"NPZ schema mismatch for {cell.cell_id}: {sorted(stored.files)}"
                )
            arrays = {name: np.asarray(stored[name]) for name in required}
    except (OSError, ValueError) as exc:
        raise BatchIntegrityError(f"invalid predictions NPZ for {cell.cell_id}: {exc}") from exc

    logits = arrays["logits"]
    labels = arrays["labels"]
    count = labels.shape[0] if labels.ndim == 1 else -1
    classes = int(DATASET_SETTINGS[cell.dataset]["classes"])
    if count <= 0 or logits.shape != (count, classes):
        raise BatchIntegrityError(f"logit/label shape mismatch for {cell.cell_id}")
    for field in ("subject", "session", "trial"):
        if arrays[field].shape != (count,):
            raise BatchIntegrityError(f"NPZ {field} shape mismatch for {cell.cell_id}")
    if not np.isfinite(logits).all():
        raise BatchIntegrityError(f"nonfinite logits for {cell.cell_id}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise BatchIntegrityError(f"noninteger labels for {cell.cell_id}")
    if int(labels.min()) < 0 or int(labels.max()) >= classes:
        raise BatchIntegrityError(f"out-of-range labels for {cell.cell_id}")
    expected_subjects = set(FROZEN_SPLITS[cell.fold]["test"])
    observed_subjects = {int(subject) for subject in arrays["subject"]}
    if observed_subjects != expected_subjects:
        raise BatchIntegrityError(f"NPZ held-out subjects mismatch for {cell.cell_id}")
    if not np.all(arrays["session"] == 1):
        raise BatchIntegrityError(f"NPZ session mismatch for {cell.cell_id}")

    sample_counts = payload.get("sample_counts")
    if not isinstance(sample_counts, Mapping) or int(sample_counts.get("test", -1)) != count:
        raise BatchIntegrityError(f"JSON/NPZ test sample count mismatch for {cell.cell_id}")
    subject_counts = payload.get("subject_sample_counts")
    if not isinstance(subject_counts, Mapping):
        raise BatchIntegrityError(f"missing subject sample counts for {cell.cell_id}")
    for subject in expected_subjects:
        observed = int(np.sum(arrays["subject"] == subject))
        if int(subject_counts.get(str(subject), -1)) != observed:
            raise BatchIntegrityError(
                f"JSON/NPZ subject count mismatch for {cell.cell_id}, subject {subject}"
            )


def _batch_events(log_path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(BATCH_EVENT_PREFIX):
                value = json.loads(line[len(BATCH_EVENT_PREFIX) :])
                if isinstance(value, dict):
                    events.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"invalid batch log {log_path}: {exc}") from exc
    return events


def validate_log(
    cell: Cell,
    artifacts: Artifacts,
    expected_command: Sequence[str],
    code_hashes: Mapping[str, str],
) -> None:
    if artifacts.log.stat().st_size <= 0:
        raise BatchIntegrityError(f"empty log for {cell.cell_id}")
    events = _batch_events(artifacts.log)
    if not events:
        # Stage 1 may be launched manually for external exact-regression audit.
        # Its artifact hashes are subsequently bound by the PASS approval file.
        if cell.stage == 1:
            return
        raise BatchIntegrityError(f"missing batch provenance events in log for {cell.cell_id}")
    starts = [event for event in events if event.get("event") == "cell_start"]
    exits = [event for event in events if event.get("event") == "cell_exit"]
    if len(starts) != 1 or len(exits) != 1:
        raise BatchIntegrityError(f"ambiguous start/exit events for {cell.cell_id}")
    start, exit_event = starts[0], exits[0]
    if start.get("protocol_version") != PROTOCOL_VERSION:
        raise BatchIntegrityError(f"protocol version mismatch in log for {cell.cell_id}")
    if start.get("cell") != cell.metadata():
        raise BatchIntegrityError(f"cell metadata mismatch in log for {cell.cell_id}")
    if start.get("command") != list(expected_command):
        raise BatchIntegrityError(f"command drift in log for {cell.cell_id}")
    if start.get("code_sha256") != dict(code_hashes):
        raise BatchIntegrityError(f"code hash drift in log for {cell.cell_id}")
    if exit_event.get("cell_id") != cell.cell_id or exit_event.get("return_code") != 0:
        raise BatchIntegrityError(f"nonzero or mismatched exit event for {cell.cell_id}")


def inspect_cell(
    cell: Cell,
    artifacts: Artifacts,
    expected_command: Sequence[str],
    code_hashes: Mapping[str, str],
) -> str:
    existence = {path: path.exists() for path in artifacts.all_paths()}
    if not any(existence.values()):
        return "missing"
    if not all(existence.values()):
        present = [str(path) for path, exists in existence.items() if exists]
        absent = [str(path) for path, exists in existence.items() if not exists]
        raise PartialCellError(
            f"partial cell {cell.cell_id}; stop and manually quarantine before identical rerun; "
            f"present={present}, absent={absent}"
        )
    if artifacts.checkpoint.stat().st_size <= 0:
        raise BatchIntegrityError(f"empty checkpoint for {cell.cell_id}")
    payload = validate_result_json(cell, artifacts)
    validate_predictions_npz(cell, artifacts, payload)
    validate_log(cell, artifacts, expected_command, code_hashes)
    return "complete"


def _write_event(log_file, event: Mapping[str, object]) -> None:
    log_file.write(BATCH_EVENT_PREFIX + json.dumps(dict(event), sort_keys=True) + "\n")
    log_file.flush()


def tee_command(
    cell: Cell,
    command: Sequence[str],
    log_path: Path,
    code_hashes: Mapping[str, str],
    *,
    cwd: Path,
) -> int:
    """Run one child sequentially and tee merged stdout/stderr to its unique log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", newline="") as log_file:
        _write_event(
            log_file,
            {
                "event": "cell_start",
                "protocol_version": PROTOCOL_VERSION,
                "cell": cell.metadata(),
                "command": list(command),
                "code_sha256": dict(code_hashes),
            },
        )
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        returncode = int(process.wait())
        _write_event(
            log_file,
            {"event": "cell_exit", "cell_id": cell.cell_id, "return_code": returncode},
        )
    return returncode


def stage1_approval_template(stage1: Sequence[tuple[Cell, Artifacts]]) -> dict[str, object]:
    cells: dict[str, object] = {}
    for cell, artifacts in stage1:
        cells[cell.cell_id] = {
            "json_sha256": file_sha256(artifacts.output),
            "checkpoint_sha256": file_sha256(artifacts.checkpoint),
            "predictions_sha256": file_sha256(artifacts.predictions),
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": "PASS",
        "partition_seed": PARTITION_SEED,
        "optimization_seed": 2024,
        "code_sha256": dict(FROZEN_CODE_SHA256),
        "cells": cells,
    }


def validate_stage1_approval(
    approval_path: Path,
    stage1: Sequence[tuple[Cell, Artifacts]],
    code_hashes: Mapping[str, str],
) -> None:
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"invalid stage-1 approval file {approval_path}: {exc}") from exc
    if not isinstance(approval, Mapping):
        raise BatchIntegrityError("stage-1 approval must be a JSON object")
    for field, expected in (
        ("protocol_version", PROTOCOL_VERSION),
        ("verdict", "PASS"),
        ("partition_seed", PARTITION_SEED),
        ("optimization_seed", 2024),
        ("code_sha256", dict(code_hashes)),
    ):
        if approval.get(field) != expected:
            raise BatchIntegrityError(f"stage-1 approval field {field!r} mismatch")
    approved_cells = approval.get("cells")
    if not isinstance(approved_cells, Mapping):
        raise BatchIntegrityError("stage-1 approval has no cell hash map")
    expected_ids = {cell.cell_id for cell, _ in stage1}
    if set(approved_cells) != expected_ids:
        raise BatchIntegrityError("stage-1 approval must bind exactly the two fold-1 cells")
    for cell, artifacts in stage1:
        expected_hashes = {
            "json_sha256": file_sha256(artifacts.output),
            "checkpoint_sha256": file_sha256(artifacts.checkpoint),
            "predictions_sha256": file_sha256(artifacts.predictions),
        }
        if approved_cells[cell.cell_id] != expected_hashes:
            raise BatchIntegrityError(f"stage-1 artifact hash mismatch for {cell.cell_id}")


def _assert_unique_matrix_paths(cells: Sequence[Cell], args: argparse.Namespace) -> None:
    paths = [
        path.resolve()
        for cell in cells
        for path in artifact_paths(cell, args.output_root, args.log_root).all_paths()
    ]
    if len(paths) != len(set(paths)):
        raise BatchIntegrityError("the frozen matrix does not have unique artifact paths")


def run_batch(args: argparse.Namespace) -> int:
    repo_root = repository_root()
    code_hashes = verify_frozen_code(repo_root)
    cells = frozen_schedule()
    _assert_unique_matrix_paths(cells, args)
    stage1_pairs = [
        (cell, artifact_paths(cell, args.output_root, args.log_root))
        for cell in cells
        if cell.stage == 1
    ]
    approval_checked = False

    print(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "mode": "dry-run" if args.dry_run else "execute",
                "cells": len(cells),
                "code_sha256": code_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for cell in cells:
        artifacts = artifact_paths(cell, args.output_root, args.log_root)
        command = build_command(cell, artifacts, args, repo_root)

        # Recompute before every execution or resume decision, not only once at
        # startup, so a code edit during a long matrix cannot be silently mixed.
        code_hashes = verify_frozen_code(repo_root)

        if not args.dry_run and cell.stage > 1 and not approval_checked:
            if args.stage1_approval is None:
                template = stage1_approval_template(stage1_pairs)
                print(
                    "STAGE1 COMPLETE: remaining 28 cells are withheld until an external exact "
                    "fold-1 regression audit writes --stage1-approval with this schema:\n"
                    + json.dumps(template, indent=2),
                    flush=True,
                )
                return 0
            for stage1_cell, stage1_artifacts in stage1_pairs:
                stage1_command = build_command(stage1_cell, stage1_artifacts, args, repo_root)
                if inspect_cell(
                    stage1_cell,
                    stage1_artifacts,
                    stage1_command,
                    verify_frozen_code(repo_root),
                ) != "complete":
                    raise BatchIntegrityError("stage-1 approval cannot precede complete fold-1 cells")
            validate_stage1_approval(args.stage1_approval, stage1_pairs, code_hashes)
            approval_checked = True
            print(f"STAGE1 APPROVED {args.stage1_approval}", flush=True)

        state = inspect_cell(cell, artifacts, command, code_hashes)
        if state == "complete":
            print(f"RESUME-SKIP {cell.cell_id}: verified complete", flush=True)
            continue

        if args.dry_run:
            gate = " stage1-gated" if cell.stage > 1 and args.stage1_approval is None else ""
            print(f"DRY-RUN{gate} {cell.cell_id}: {json.dumps(command)}", flush=True)
            continue

        artifacts.output.parent.mkdir(parents=True, exist_ok=True)
        returncode = tee_command(cell, command, artifacts.log, code_hashes, cwd=repo_root)
        if returncode != 0:
            raise ChildProcessFailed(cell, returncode)
        if inspect_cell(cell, artifacts, command, verify_frozen_code(repo_root)) != "complete":
            raise BatchIntegrityError(f"post-run validation did not complete for {cell.cell_id}")
        print(f"COMPLETED {cell.cell_id}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_batch(args)
    except ChildProcessFailed as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return exc.returncode
    except BatchIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
