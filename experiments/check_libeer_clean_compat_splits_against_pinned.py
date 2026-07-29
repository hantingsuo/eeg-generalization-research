"""Directly compare frozen compatibility manifests with pinned LibEER code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


EXPECTED_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libeer-root",
        type=Path,
        default=Path(
            os.environ.get("LIBEER_CODE_ROOT", "third_party/libeer/LibEER")
        ),
    )
    parser.add_argument("--split-root", type=Path, default=Path("results/libeer_clean_compat/c1/splits"))
    parser.add_argument("--output", type=Path, default=Path("results/libeer_clean_compat/c1/pinned_split_direct_check.json"))
    args = parser.parse_args()
    root = args.libeer_root.resolve()
    split_root = args.split_root.resolve()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(root))
    errors: list[str] = []
    checks = 0
    try:
        os.chdir(root)
        from data_utils.split import get_split_index
        from utils.utils import setup_seed
        setting = SimpleNamespace(
            split_type="train-val-test",
            experiment_mode="subject-dependent",
            test_size=0.2,
            val_size=0.2,
            pr=None,
            sr=None,
        )
        for dataset in ("seed", "seediv"):
            path = split_root / f"libeer_clean_{dataset}_split_v1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for unit in payload["units"]:
                setup_seed(2024)
                labels = [np.array([int(label)]) for label in unit["label_sequence"]]
                upstream = get_split_index([None] * len(labels), labels, setting)
                expected = {
                    "train_trials": [int(value) + 1 for value in upstream["train"][0]],
                    "validation_trials": [int(value) + 1 for value in upstream["val"][0]],
                    "test_trials": [int(value) + 1 for value in upstream["test"][0]],
                }
                for key, value in expected.items():
                    checks += 1
                    if unit[key] != value:
                        errors.append(f"{unit['unit_id']} {key} mismatch")
    finally:
        os.chdir(old_cwd)
        sys.path.pop(0)
    result = {"pinned_commit": EXPECTED_COMMIT, "checks": checks, "errors": errors, "status": "PASS" if not errors else "FAIL"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
