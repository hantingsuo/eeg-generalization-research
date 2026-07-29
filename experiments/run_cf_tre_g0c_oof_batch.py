"""Fail-closed 90-unit launcher for G0-C train-only OOF probabilities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


PROTOCOL = "cf_tre_g0c_development_feasibility_v1"
COMPONENTS = (
    "linear_de310", "linear_structural245", "linear_all555", "rbf_de310",
    "rbf_all555", "xgboost_all555", "lightgbm_all555",
)


def _paths(root: Path, dataset: str, session: int, subject: int) -> tuple[Path, Path]:
    stem = f"s{session:02d}_sub{subject:02d}"
    return root / dataset / f"{stem}.json", root / dataset / f"{stem}.npz"


def _audit(output: Path, predictions: Path, dataset: str, session: int, subject: int) -> dict[str, Any]:
    if not output.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"incomplete OOF cell {dataset}/{session}/{subject}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected = {"protocol": PROTOCOL, "stage": "G0-C-OOF", "dataset": dataset, "session": session, "subject": subject, "optimization_seed": 2024, "outer_partition_loaded": "train_only", "contains_validation_predictions": False, "contains_test_predictions": False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{output}: {key} mismatch")
    with np.load(predictions, allow_pickle=False) as arrays:
        expected_keys = {"train_labels", "train_trials", "inner_fold"} | {f"oof_probabilities__{component}" for component in COMPONENTS}
        if set(arrays.files) != expected_keys:
            raise ValueError(f"{predictions}: key mismatch")
        labels = arrays["train_labels"]
        folds = arrays["inner_fold"]
        if set(np.unique(folds).tolist()) != {1, 2, 3}:
            raise ValueError(f"{predictions}: fold coverage mismatch")
        for component in COMPONENTS:
            probabilities = arrays[f"oof_probabilities__{component}"]
            if probabilities.shape[0] != len(labels) or not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
                raise ValueError(f"{predictions}: malformed {component}")
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cf_tre/g0c/oof"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/runs/cf_tre_g0c/oof"))
    parser.add_argument("--ledger", type=Path, default=Path("results/cf_tre/g0c/g0c_oof_ledger.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    cells = [(dataset, session, subject) for dataset in ("seed", "seediv") for session in range(1, 4) for subject in range(1, 16)]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    if args.ledger.exists() and not args.resume:
        raise FileExistsError(args.ledger)
    ledger: dict[str, Any] = {"protocol": PROTOCOL, "stage": "G0-C-OOF", "status": "RUNNING", "expected_cells": len(cells), "completed_cells": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    if args.ledger.exists():
        old = json.loads(args.ledger.read_text(encoding="utf-8"))
        if old.get("expected_cells") != len(cells):
            raise ValueError("ledger cell-count mismatch")
        ledger.update(old)
        ledger["status"] = "RUNNING"
    completed = set(ledger.get("completed_cells", []))
    for index, (dataset, session, subject) in enumerate(cells, start=1):
        cell_id = f"{dataset}_s{session:02d}_sub{subject:02d}"
        output, predictions = _paths(args.root, dataset, session, subject)
        if output.exists() or predictions.exists():
            if not args.resume:
                raise FileExistsError(cell_id)
            _audit(output, predictions, dataset, session, subject)
            completed.add(cell_id)
            continue
        stem = f"s{session:02d}_sub{subject:02d}"
        command = [sys.executable, "-u", "experiments/run_cf_tre_g0c_oof_subject_session.py", "--dataset", dataset, "--session", str(session), "--subject", str(subject), "--split-manifest", f"results/cf_tre/g0a/track_a_{dataset}_split_v1.json", "--selection", f"results/cf_tre/g0c/inputs/{dataset}/{stem}.selection.json", "--output", str(output), "--predictions", str(predictions), "--optimization-seed", "2024"]
        log = args.log_root / f"{cell_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "START", "cell": cell_id, "index": index, "total": len(cells)}), flush=True)
        with log.open("x", encoding="utf-8") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        if process.returncode != 0:
            ledger.update({"status": "FAILED", "failed_cell": cell_id, "returncode": process.returncode})
            _write(args.ledger, ledger)
            raise RuntimeError(f"{cell_id} failed; see {log}")
        payload = _audit(output, predictions, dataset, session, subject)
        completed.add(cell_id)
        ledger.update({"completed_cells": sorted(completed), "last_completed": cell_id, "last_elapsed_seconds": payload.get("elapsed_seconds")})
        _write(args.ledger, ledger)
        print(json.dumps({"event": "DONE", "cell": cell_id, "elapsed_seconds": payload.get("elapsed_seconds")}), flush=True)
    ledger.update({"status": "PASS", "completed_count": len(completed), "completed_cells": sorted(completed), "completed_utc": datetime.now(timezone.utc).isoformat()})
    _write(args.ledger, ledger)
    print(json.dumps({"status": "PASS", "cells": len(cells), "ledger": str(args.ledger)}), flush=True)


if __name__ == "__main__":
    main()
