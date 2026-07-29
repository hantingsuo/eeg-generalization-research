"""Independent integrity and metric audit for post-review LR/RF controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_cf_tre_g1_outer_test import _aggregate  # noqa: E402
from experiments.run_postreview_lr_rf_subject_session import (  # noqa: E402
    METHODS,
    model_configurations,
)


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
            "results/cf_tre/postreview_baselines/postreview_lr_rf_audit.json"
        ),
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    aggregate_path = (
        root
        / "results/cf_tre/postreview_baselines/postreview_lr_rf_aggregate.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    output = root / args.output
    if output.exists():
        raise FileExistsError(output)
    json_paths = sorted(
        (root / "results/cf_tre/postreview_baselines/formal").glob("*/*.json")
    )
    npz_paths = sorted(
        (root / "results/cf_tre/postreview_baselines/formal").glob("*/*.npz")
    )
    identities = set()
    configurations_ok = True
    access_ok = True
    units_by_dataset: dict[str, list[dict[str, Any]]] = {"seed": [], "seediv": []}
    expected_config = model_configurations()
    for path in json_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        identity = (row["dataset"], int(row["session"]), int(row["subject"]))
        identities.add(identity)
        configurations_ok &= all(
            row["methods"][method]["configuration"] == expected_config[method]
            for method in METHODS
        )
        access_ok &= (
            row["post_review"]
            and not row["confirmatory_gate_member"]
            and row["test_contacted"]
            and row["test_evaluation_count"] == 1
        )
        npz_path = path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as arrays:
            units_by_dataset[row["dataset"]].append(
                {
                    "session": row["session"],
                    "subject": row["subject"],
                    "labels": arrays["test_labels"].astype(np.int64),
                    "trials": arrays["test_trials"].astype(np.int64),
                    "probabilities": {
                        method: arrays[f"test_probabilities__{method}"].astype(
                            np.float64
                        )
                        for method in METHODS
                    },
                }
            )
    expected_identities = {
        (dataset, session, subject)
        for dataset in ("seed", "seediv")
        for session in range(1, 4)
        for subject in range(1, 16)
    }
    metric_match = True
    for dataset in ("seed", "seediv"):
        alpha = float(
            aggregate["datasets"][dataset]["methods"][METHODS[0]]["cvar_alpha"]
        )
        rebuilt = {
            method: _aggregate(units_by_dataset[dataset], method, alpha)
            for method in METHODS
        }
        for method in METHODS:
            for metric in (
                "subject_equal_accuracy",
                "subject_equal_macro_f1",
                "subject_equal_balanced_accuracy",
                "subject_equal_trial_accuracy",
                "mean_environment_log_loss",
                "environment_cvar_log_loss",
            ):
                metric_match &= np.isclose(
                    rebuilt[method][metric],
                    aggregate["datasets"][dataset]["methods"][method][metric],
                    rtol=0.0,
                    atol=1e-12,
                )
    checks = {
        "ninety_json_and_npz": bool(
            len(json_paths) == 90 and len(npz_paths) == 90
        ),
        "identity_matrix_complete": bool(identities == expected_identities),
        "frozen_configurations_match": bool(configurations_ok),
        "post_review_access_boundary_match": bool(access_ok),
        "aggregate_metrics_reconstructed": bool(metric_match),
        "confirmatory_gate_unchanged": bool(
            aggregate["confirmatory_gate_unchanged"]
        ),
    }
    audit = {
        "protocol": "cf_tre_postreview_lr_rf_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "aggregate_sha256": sha256(aggregate_path),
        "json_sha256": {str(path.relative_to(root)): sha256(path) for path in json_paths},
        "npz_sha256": {str(path.relative_to(root)): sha256(path) for path in npz_paths},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    audit = run(parser().parse_args(argv))
    print(json.dumps(audit["checks"], indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
