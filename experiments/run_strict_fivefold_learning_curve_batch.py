"""Fail-closed sequential runner for the 30 frozen learning-curve cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("seed", "seediv")
FOLDS = range(1, 6)
SEEDS = (2024, 2025, 2026)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--libeer-root", type=Path, required=True)
    result.add_argument("--seed-feature-root", type=Path, required=True)
    result.add_argument("--seediv-dataset-root", type=Path, required=True)
    result.add_argument("--seediv-cache-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--log-root", type=Path, required=True)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--dry-run", action="store_true")
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def schedule() -> list[dict[str, Any]]:
    return [
        {
            "cell_id": f"{dataset}_fold{fold}_opt{seed}",
            "dataset": dataset,
            "fold": fold,
            "optimization_seed": seed,
            "status": "PENDING",
        }
        for seed in SEEDS
        for fold in FOLDS
        for dataset in DATASETS
    ]


def command(args: argparse.Namespace, cell: dict[str, Any]) -> list[str]:
    output = args.output_root.resolve() / "cells" / f"{cell['cell_id']}.json"
    return [
        str(args.python.resolve()),
        str((ROOT / "experiments/run_strict_fivefold_learning_curve_cell.py").resolve()),
        "--dataset",
        str(cell["dataset"]),
        "--fold",
        str(cell["fold"]),
        "--optimization-seed",
        str(cell["optimization_seed"]),
        "--libeer-root",
        str(args.libeer_root.resolve()),
        "--seed-feature-root",
        str(args.seed_feature_root.resolve()),
        "--seediv-dataset-root",
        str(args.seediv_dataset_root.resolve()),
        "--seediv-cache-root",
        str(args.seediv_cache_root.resolve()),
        "--reference-root",
        str((ROOT / "results/strict_fivefold").resolve()),
        "--train-fit-result",
        str(
            (
                ROOT
                / "results/strict_fivefold/train_fit_diagnostic/train_fit_diagnostic.json"
            ).resolve()
        ),
        "--output",
        str(output),
        "--device",
        args.device,
    ]


def write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    cells = schedule()
    if args.dry_run:
        for cell in cells:
            print(json.dumps(command(args, cell)))
        return 0

    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    ledger_path = output_root / "learning_curve_ledger.json"
    collisions = [
        path
        for path in (
            ledger_path,
            *(output_root / "cells" / f"{cell['cell_id']}.json" for cell in cells),
            *(log_root / f"{cell['cell_id']}.log" for cell in cells),
        )
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite existing learning-curve artifacts: {collisions[:3]}")
    (output_root / "cells").mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    ledger = {
        "protocol": "strict_fivefold_dgcnn_learning_curve_v1",
        "status": "RUNNING",
        "started_at": now(),
        "completed_at": None,
        "current_cell": None,
        "cells": cells,
    }
    write_ledger(ledger_path, ledger)

    for index, cell in enumerate(cells):
        cell["status"] = "RUNNING"
        cell["started_at"] = now()
        ledger["current_cell"] = cell["cell_id"]
        write_ledger(ledger_path, ledger)
        log_path = log_root / f"{cell['cell_id']}.log"
        cmd = command(args, cell)
        with log_path.open("x", encoding="utf-8", newline="") as log:
            log.write("COMMAND\t" + json.dumps(cmd) + "\n")
            log.flush()
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
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
                log.write(line)
                log.flush()
            return_code = int(process.wait())
        cell["return_code"] = return_code
        cell["completed_at"] = now()
        if return_code != 0:
            cell["status"] = "FAILED"
            ledger["status"] = "FAILED"
            ledger["current_cell"] = None
            ledger["completed_at"] = now()
            write_ledger(ledger_path, ledger)
            return return_code
        cell["status"] = "PASS"
        ledger["completed_cells"] = index + 1
        write_ledger(ledger_path, ledger)

    ledger["status"] = "PASS"
    ledger["current_cell"] = None
    ledger["completed_at"] = now()
    write_ledger(ledger_path, ledger)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
