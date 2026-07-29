"""Fail-closed DGCNN compatibility C1 batch against pinned LibEER table cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


PROTOCOL = "libeer_39dc27e_clean_table_compat_v1"


def _settings(dataset: str) -> tuple[int, float]:
    return (80, 0.0015) if dataset == "seed" else (150, 0.001)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _paths(root: Path, dataset: str, session: int, subject: int) -> tuple[Path, Path, Path]:
    directory = root / "dgcnn" / dataset
    stem = f"s{session:02d}_sub{subject:02d}"
    return directory / f"{stem}.json", directory / f"{stem}.npz", directory / f"{stem}.pt"


def _audit(path: Path, dataset: str, session: int, subject: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    epochs, lr = _settings(dataset)
    expected = {
        "protocol": PROTOCOL,
        "stage": "C1",
        "dataset": dataset,
        "session": session,
        "subject": subject,
        "model": "dgcnn",
        "epochs": epochs,
        "batch_size": 32,
        "learning_rate": lr,
        "sampler_semantics": "upstream_global_rng",
        "test_contacted": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key} mismatch {payload.get(key)!r} != {value!r}")
    if not isinstance(payload.get("test"), dict):
        raise ValueError(f"{path}: missing test metrics")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/libeer_clean_compat/c1/formal"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/runs/libeer_clean_compat_c1"))
    parser.add_argument("--ledger", type=Path, default=Path("results/libeer_clean_compat/c1/c1_dgcnn_ledger.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args(argv)
    cells = [(dataset, session, subject) for dataset in ("seed", "seediv") for session in range(1, 4) for subject in range(1, 16)]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    if args.ledger.exists() and not args.resume:
        raise FileExistsError(f"ledger exists: {args.ledger}")
    ledger: dict[str, Any] = {"protocol": PROTOCOL, "stage": "C1", "status": "RUNNING", "expected_cells": len(cells), "completed_cells": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    if args.ledger.exists():
        previous = json.loads(args.ledger.read_text(encoding="utf-8"))
        if previous.get("expected_cells") != len(cells):
            raise ValueError("ledger matrix mismatch")
        ledger.update(previous)
        ledger["status"] = "RUNNING"
    completed = set(ledger.get("completed_cells", []))
    for index, (dataset, session, subject) in enumerate(cells, start=1):
        cell_id = f"dgcnn_{dataset}_s{session:02d}_sub{subject:02d}"
        output, predictions, checkpoint = _paths(args.root, dataset, session, subject)
        artifacts = (output, predictions, checkpoint)
        if any(path.exists() for path in artifacts):
            if not all(path.is_file() for path in artifacts) or not args.resume:
                raise RuntimeError(f"partial or unapproved existing cell: {cell_id}")
            _audit(output, dataset, session, subject)
            completed.add(cell_id)
            continue
        epochs, lr = _settings(dataset)
        split = Path(f"results/libeer_clean_compat/c1/splits/libeer_clean_{dataset}_split_v1.json")
        log = args.log_root / f"{cell_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "experiments/run_cf_tre_deep_subject_session.py", "--dataset", dataset, "--model", "dgcnn", "--session", str(session), "--subject", str(subject), "--split-manifest", str(split), "--output", str(output), "--predictions", str(predictions), "--checkpoint", str(checkpoint), "--optimization-seed", "2024", "--protocol", PROTOCOL, "--stage", "C1", "--epochs", str(epochs), "--batch-size", "32", "--eval-batch-size", "32", "--lr", str(lr), "--device", "cuda", "--upstream-sampler"]
        print(json.dumps({"event": "START", "cell": cell_id, "index": index, "total": len(cells)}), flush=True)
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        if process.returncode != 0:
            ledger.update({"status": "FAILED", "failed_cell": cell_id, "returncode": process.returncode})
            _write(args.ledger, ledger)
            raise RuntimeError(f"{cell_id} failed; see {log}")
        payload = _audit(output, dataset, session, subject)
        completed.add(cell_id)
        ledger["completed_cells"] = sorted(completed)
        ledger["last_completed"] = cell_id
        _write(args.ledger, ledger)
        print(json.dumps({"event": "DONE", "cell": cell_id, "elapsed_seconds": payload.get("elapsed_seconds")}), flush=True)
    ledger.update({"status": "PASS", "completed_cells": sorted(completed), "completed_count": len(completed), "completed_utc": datetime.now(timezone.utc).isoformat()})
    _write(args.ledger, ledger)
    print(json.dumps({"status": "PASS", "cells": len(cells), "ledger": str(args.ledger)}), flush=True)


if __name__ == "__main__":
    main()
