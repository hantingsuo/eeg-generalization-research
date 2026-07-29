"""Fail-closed sequential 90-unit runner for post-review LR/RF controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_postreview_lr_rf_subject_session import (  # noqa: E402
    METHODS,
    PROTOCOL,
    model_configurations,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--log-root", type=Path, required=True)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def schedule() -> list[dict[str, Any]]:
    return [
        {
            "cell_id": f"{dataset}_s{session:02d}_sub{subject:02d}",
            "dataset": dataset,
            "session": session,
            "subject": subject,
            "status": "PENDING",
        }
        for dataset in ("seed", "seediv")
        for session in range(1, 4)
        for subject in range(1, 16)
    ]


def command(args: argparse.Namespace, cell: dict[str, Any]) -> list[str]:
    dataset = str(cell["dataset"])
    stem = f"s{cell['session']:02d}_sub{cell['subject']:02d}"
    output = args.output_root.resolve() / "formal" / dataset / f"{stem}.json"
    split = ROOT / f"results/cf_tre/g0a/track_a_{dataset}_split_v1.json"
    return [
        str(args.python.resolve()),
        str((ROOT / "experiments/run_postreview_lr_rf_subject_session.py").resolve()),
        "--dataset",
        dataset,
        "--session",
        str(cell["session"]),
        "--subject",
        str(cell["subject"]),
        "--split-manifest",
        str(split.resolve()),
        "--output",
        str(output),
        "--predictions",
        str(output.with_suffix(".npz")),
        "--seed-root",
        str((ROOT / "data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s").resolve()),
        "--seediv-root",
        str((ROOT / "data/SEED/SEED_IV").resolve()),
        "--libeer-root",
        "C:\\Users\\ASUS\\AppData\\Local\\Temp\\codex-paper-eeg-libeer-2024",
        "--seediv-cache",
        str((ROOT / "results/libeer_gate_b/cache_seediv_1s_de_lds_39dc27e").resolve()),
    ]


def write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    payload = json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    last_error: PermissionError | None = None
    for attempt in range(20):
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.1 * (attempt + 1))
    assert last_error is not None
    raise last_error


def audit_complete_cell(
    output: Path,
    predictions: Path,
    cell: dict[str, Any],
) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "stage": "post_review_descriptive_control",
        "post_review": True,
        "confirmatory_gate_member": False,
        "dataset": cell["dataset"],
        "session": cell["session"],
        "subject": cell["subject"],
        "test_contacted": True,
        "test_evaluation_count": 1,
        "feature_space": "engineered_all555",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"{output}: {key} mismatch {payload.get(key)!r} != {value!r}"
            )
    if any(
        payload["methods"][method]["configuration"] != model_configurations()[method]
        for method in METHODS
    ):
        raise ValueError(f"{output}: frozen method configuration mismatch")
    with np.load(predictions, allow_pickle=False) as arrays:
        required = {
            "validation_labels",
            "validation_trials",
            "test_labels",
            "test_trials",
            *(
                f"{partition}_probabilities__{method}"
                for partition in ("validation", "test")
                for method in METHODS
            ),
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{predictions}: missing arrays {sorted(missing)}")
        class_count = 3 if cell["dataset"] == "seed" else 4
        for partition in ("validation", "test"):
            labels = arrays[f"{partition}_labels"]
            trials = arrays[f"{partition}_trials"]
            if labels.ndim != 1 or trials.shape != labels.shape:
                raise ValueError(f"{predictions}: invalid {partition} labels/trials")
            for method in METHODS:
                probabilities = arrays[f"{partition}_probabilities__{method}"]
                if probabilities.shape != (len(labels), class_count):
                    raise ValueError(
                        f"{predictions}: invalid {partition} {method} shape"
                    )
                if not np.all(np.isfinite(probabilities)):
                    raise ValueError(f"{predictions}: non-finite probabilities")
                if not np.allclose(
                    probabilities.sum(axis=1),
                    1.0,
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"{predictions}: probabilities do not sum to one"
                    )


def run(args: argparse.Namespace) -> int:
    cells = schedule()
    if args.dry_run:
        for cell in cells:
            print(json.dumps(command(args, cell)))
        return 0
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    ledger_path = output_root / "postreview_lr_rf_ledger.json"
    artifacts = [
        output_root
        / "formal"
        / cell["dataset"]
        / f"s{cell['session']:02d}_sub{cell['subject']:02d}{suffix}"
        for cell in cells
        for suffix in (".json", ".npz")
    ]
    logs = [log_root / f"{cell['cell_id']}.log" for cell in cells]
    collisions = [path for path in (ledger_path, *artifacts, *logs) if path.exists()]
    if collisions and not args.resume:
        raise FileExistsError(f"refusing to overwrite LR/RF artifacts: {collisions[:3]}")
    if args.resume and not ledger_path.is_file():
        raise FileNotFoundError("resume requested without an existing ledger")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    ledger = {
        "protocol": "cf_tre_postreview_lr_rf_v1",
        "status": "RUNNING",
        "started_at": now(),
        "completed_at": None,
        "current_cell": None,
        "cells": cells,
    }
    if args.resume:
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        if previous.get("protocol") != PROTOCOL:
            raise ValueError("resume ledger protocol mismatch")
        audited = 0
        for cell in cells:
            output = (
                output_root
                / "formal"
                / cell["dataset"]
                / f"s{cell['session']:02d}_sub{cell['subject']:02d}.json"
            )
            predictions = output.with_suffix(".npz")
            log = log_root / f"{cell['cell_id']}.log"
            present = (output.exists(), predictions.exists(), log.exists())
            if any(present) and not all(present):
                raise RuntimeError(
                    f"partial existing resume cell: {cell['cell_id']} {present}"
                )
            if all(present):
                audit_complete_cell(output, predictions, cell)
                cell["status"] = "PASS"
                cell["resume_audited"] = True
                audited += 1
        ledger["resume"] = True
        ledger["resume_audited_cells"] = audited
        ledger["completed_cells"] = audited
    write_ledger(ledger_path, ledger)
    for index, cell in enumerate(cells):
        if cell["status"] == "PASS":
            continue
        cell["status"] = "RUNNING"
        cell["started_at"] = now()
        ledger["current_cell"] = cell["cell_id"]
        write_ledger(ledger_path, ledger)
        cmd = command(args, cell)
        log_path = log_root / f"{cell['cell_id']}.log"
        with log_path.open("x", encoding="utf-8", newline="") as log:
            log.write("COMMAND\t" + json.dumps(cmd) + "\n")
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
        ledger["completed_cells"] = sum(
            item["status"] == "PASS" for item in cells
        )
        write_ledger(ledger_path, ledger)
    ledger["status"] = "PASS"
    ledger["current_cell"] = None
    ledger["completed_at"] = now()
    write_ledger(ledger_path, ledger)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
