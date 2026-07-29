"""Independent structural and numeric audit of learning-curve outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("."))
    result.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/strict_fivefold/learning_curves/learning_curve_audit.json"
        ),
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    aggregate_path = (
        root
        / "results/strict_fivefold/learning_curves/learning_curve_aggregate.json"
    )
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    cell_root = root / "results/strict_fivefold/learning_curves/cells"
    paths = sorted(cell_root.glob("*.json"))
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    identities = {
        (cell["dataset"], int(cell["fold"]), int(cell["optimization_seed"]))
        for cell in cells
    }
    expected = {
        (dataset, fold, seed)
        for dataset in ("seed", "seediv")
        for fold in range(1, 6)
        for seed in (2024, 2025, 2026)
    }
    checks = {
        "thirty_unique_cells": len(cells) == 30 and identities == expected,
        "all_reproduced": all(
            cell["status"] == "PASS_REPRODUCED"
            and cell["exact_reproduction"]["history_match"]
            and cell["exact_reproduction"]["best_checkpoint_metadata_match"]
            and cell["exact_reproduction"]["state_dict_match"]
            for cell in cells
        ),
        "no_test_access": all(
            cell["test_rows_loaded"] == 0 and cell["test_evaluation_count"] == 0
            for cell in cells
        ),
        "histories_complete": all(
            len(cell["history"]) == 150
            and [row["epoch"] for row in cell["history"]] == list(range(1, 151))
            for cell in cells
        ),
        "metrics_finite_and_bounded": all(
            np.isfinite(
                [
                    row["train_loss"],
                    row["train_online_window_accuracy"],
                    row["validation_window_accuracy"],
                    row["validation_window_macro_f1"],
                    row["validation_trial_accuracy"],
                ]
            ).all()
            and 0.0 <= row["train_online_window_accuracy"] <= 1.0
            and 0.0 <= row["validation_window_accuracy"] <= 1.0
            and 0.0 <= row["validation_window_macro_f1"] <= 1.0
            and 0.0 <= row["validation_trial_accuracy"] <= 1.0
            for cell in cells
            for row in cell["history"]
        ),
        "aggregate_complete": (
            aggregate["status"] == "PASS"
            and aggregate["cell_count"] == 30
            and all(
                len(aggregate["datasets"][dataset]["epochs"]) == 150
                for dataset in ("seed", "seediv")
            )
        ),
    }
    audit = {
        "protocol": "strict_fivefold_dgcnn_learning_curve_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "aggregate_sha256": sha256(aggregate_path),
        "cell_sha256": {path.name: sha256(path) for path in paths},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    audit = run(parser().parse_args(argv))
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
