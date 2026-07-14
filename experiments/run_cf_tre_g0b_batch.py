"""Fail-closed, resumable launcher for the frozen CF-TRE G0-B matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


@dataclass(frozen=True)
class Cell:
    family: str
    dataset: str
    session: int
    subject: int
    model: str | None
    output: Path
    predictions: Path
    checkpoint: Path | None
    log: Path

    @property
    def cell_id(self) -> str:
        prefix = self.family if self.model is None else f"{self.family}_{self.model}"
        return f"{prefix}_{self.dataset}_s{self.session:02d}_sub{self.subject:02d}"


def build_cells(family: str, root: Path, log_root: Path) -> tuple[Cell, ...]:
    if family not in {"classical", "deep"}:
        raise ValueError("family must be classical or deep")
    models: tuple[str | None, ...] = (None,) if family == "classical" else ("dgcnn", "gcbnet")
    cells: list[Cell] = []
    for model in models:
        for dataset in ("seed", "seediv"):
            for session in range(1, 4):
                for subject in range(1, 16):
                    directory = root / family
                    if model is not None:
                        directory = directory / model
                    directory = directory / dataset
                    stem = f"s{session:02d}_sub{subject:02d}"
                    cells.append(
                        Cell(
                            family=family,
                            dataset=dataset,
                            session=session,
                            subject=subject,
                            model=model,
                            output=directory / f"{stem}.json",
                            predictions=directory / f"{stem}.npz",
                            checkpoint=directory / f"{stem}.pt" if model is not None else None,
                            log=log_root / f"{family}_{model or 'seven'}_{dataset}_{stem}.log",
                        )
                    )
    return tuple(cells)


def _split_manifest(cell: Cell) -> Path:
    return Path(f"results/cf_tre/g0a/track_a_{cell.dataset}_split_v1.json")


def cell_command(cell: Cell) -> list[str]:
    common = [
        sys.executable,
        "-u",
        (
            "experiments/run_cf_tre_classical_subject_session.py"
            if cell.family == "classical"
            else "experiments/run_cf_tre_deep_subject_session.py"
        ),
        "--dataset",
        cell.dataset,
        "--session",
        str(cell.session),
        "--subject",
        str(cell.subject),
        "--split-manifest",
        str(_split_manifest(cell)),
        "--output",
        str(cell.output),
        "--predictions",
        str(cell.predictions),
        "--optimization-seed",
        "2024",
    ]
    if cell.family == "deep":
        assert cell.model is not None and cell.checkpoint is not None
        common.extend(
            [
                "--model",
                cell.model,
                "--checkpoint",
                str(cell.checkpoint),
                "--epochs",
                "150",
                "--device",
                "cuda",
            ]
        )
    return common


def expected_artifacts(cell: Cell) -> tuple[Path, ...]:
    result = [cell.output, cell.predictions]
    if cell.checkpoint is not None:
        result.append(cell.checkpoint)
    return tuple(result)


def audit_cell(cell: Cell) -> dict[str, Any]:
    missing = [path for path in expected_artifacts(cell) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{cell.cell_id} missing artifacts: {missing}")
    payload = json.loads(cell.output.read_text(encoding="utf-8"))
    expected = {
        "protocol": "cf_tre_seed_family_v1",
        "stage": "G0-B",
        "dataset": cell.dataset,
        "session": cell.session,
        "subject": cell.subject,
        "unit_id": f"{cell.dataset}:s{cell.session:02d}:sub{cell.subject:02d}",
        "optimization_seed": 2024,
        "validation_only": False,
        "test_contacted": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{cell.cell_id} {key} mismatch: {payload.get(key)!r} != {value!r}")
    if cell.family == "classical":
        components = payload.get("components", {})
        if len(components) != 7 or any("test_window" not in row for row in components.values()):
            raise ValueError(f"{cell.cell_id} incomplete seven-component test matrix")
    else:
        if payload.get("model") != cell.model or payload.get("epochs") != 150:
            raise ValueError(f"{cell.cell_id} deep model/epoch mismatch")
        if not isinstance(payload.get("test"), dict):
            raise ValueError(f"{cell.cell_id} missing test metrics")
    with __import__("numpy").load(cell.predictions, allow_pickle=False) as predictions:
        if "test_labels" not in predictions.files or len(predictions["test_labels"]) == 0:
            raise ValueError(f"{cell.cell_id} prediction file has no test rows")
    return payload


def _write_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("classical", "deep"), required=True)
    parser.add_argument("--root", type=Path, default=Path("results/cf_tre/g0b/formal"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/runs/cf_tre_g0b"))
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args(argv)
    cells = build_cells(args.family, args.root, args.log_root)
    if args.max_cells is not None:
        if args.max_cells <= 0:
            raise ValueError("max-cells must be positive")
        cells = cells[: args.max_cells]
    ledger_path = args.ledger or Path(f"results/cf_tre/g0b/g0b_{args.family}_ledger.json")
    if ledger_path.exists() and not args.resume:
        raise FileExistsError(f"ledger already exists; use --resume: {ledger_path}")
    ledger: dict[str, Any] = {
        "protocol": "cf_tre_seed_family_v1",
        "stage": "G0-B",
        "family": args.family,
        "status": "RUNNING",
        "expected_cells": len(cells),
        "completed_cells": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if previous.get("family") != args.family or previous.get("expected_cells") != len(cells):
            raise ValueError("existing ledger does not match this batch")
        ledger.update(previous)
        ledger["status"] = "RUNNING"
        ledger["resumed_utc"] = datetime.now(timezone.utc).isoformat()
    completed = set(ledger.get("completed_cells", []))

    for index, cell in enumerate(cells, start=1):
        present = [path.exists() for path in expected_artifacts(cell)]
        if any(present):
            if not all(present):
                raise RuntimeError(f"{cell.cell_id} has a partial artifact set")
            if not args.resume:
                raise FileExistsError(f"{cell.cell_id} artifacts already exist")
            audit_cell(cell)
            if cell.cell_id not in completed:
                completed.add(cell.cell_id)
                ledger["completed_cells"] = sorted(completed)
                _write_ledger(ledger_path, ledger)
            print(json.dumps({"event": "SKIP_AUDITED", "cell": cell.cell_id, "index": index}), flush=True)
            continue
        if cell.cell_id in completed:
            raise RuntimeError(f"ledger claims completion but artifacts are absent: {cell.cell_id}")
        cell.log.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "START", "cell": cell.cell_id, "index": index, "total": len(cells)}), flush=True)
        with cell.log.open("x", encoding="utf-8") as handle:
            process = subprocess.run(
                cell_command(cell),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            tail = cell.log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print("\n".join(tail), file=sys.stderr, flush=True)
            ledger["status"] = "FAILED"
            ledger["failed_cell"] = cell.cell_id
            ledger["returncode"] = process.returncode
            _write_ledger(ledger_path, ledger)
            raise RuntimeError(f"{cell.cell_id} exited {process.returncode}; see {cell.log}")
        payload = audit_cell(cell)
        completed.add(cell.cell_id)
        ledger["completed_cells"] = sorted(completed)
        ledger["last_completed"] = cell.cell_id
        ledger["last_elapsed_seconds"] = payload.get("elapsed_seconds")
        _write_ledger(ledger_path, ledger)
        print(
            json.dumps(
                {
                    "event": "DONE",
                    "cell": cell.cell_id,
                    "index": index,
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                }
            ),
            flush=True,
        )

    ledger["status"] = "PASS"
    ledger["completed_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["completed_count"] = len(completed)
    _write_ledger(ledger_path, ledger)
    print(json.dumps({"status": "PASS", "ledger": str(ledger_path), "cells": len(cells)}), flush=True)


if __name__ == "__main__":
    main()
