"""Production-independent audit of LibEER clean compatibility split artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


PROTOCOL = "libeer_39dc27e_clean_table_compat_v1"


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected(labels: list[int]) -> dict[str, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels, start=1):
        groups.setdefault(int(label), []).append(index)
    rng = random.Random(2024)
    test: list[int] = []
    validation: list[int] = []
    train: list[int] = []
    others: list[int] = []
    for indexes in groups.values():
        rng.shuffle(indexes)
        n = len(indexes)
        nt, nv, nr = int(0.2 * n), int(0.2 * n), int(0.6 * n)
        test.extend(indexes[:nt])
        validation.extend(indexes[nt : nt + nv])
        train.extend(indexes[nt + nv : nt + nv + nr])
        others.extend(indexes[nt + nv + nr :])
    if others:
        rng.shuffle(others)
        nt = int(len(labels) * 0.2) - len(test)
        nv = int(len(labels) * 0.2) - len(validation)
        test.extend(others[:nt])
        validation.extend(others[nt : nt + nv])
        train.extend(others[nt + nv :])
    return {"train_trials": train, "validation_trials": validation, "test_trials": test}


def audit(root: Path) -> dict[str, Any]:
    checks = 0
    errors: list[str] = []
    for dataset in ("seed", "seediv"):
        path = root / f"libeer_clean_{dataset}_split_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL or payload.get("canonical_sha256") != _canonical(payload):
            errors.append(f"{dataset}: header/hash mismatch")
        units = payload.get("units", [])
        if len(units) != 45:
            errors.append(f"{dataset}: expected 45 units")
        for unit in units:
            expected = _expected([int(value) for value in unit["label_sequence"]])
            for key, value in expected.items():
                checks += 1
                if unit.get(key) != value:
                    errors.append(f"{unit.get('unit_id')} {key} mismatch")
            partitions = [set(unit[key]) for key in ("train_trials", "validation_trials", "test_trials")]
            checks += 2
            if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
                errors.append(f"{unit.get('unit_id')} overlap")
            if set().union(*partitions) != set(range(1, len(unit["label_sequence"]) + 1)):
                errors.append(f"{unit.get('unit_id')} coverage")
    return {"protocol": PROTOCOL, "audit": "independent pinned split reconstruction", "checks": checks, "errors": errors, "status": "PASS" if not errors else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/libeer_clean_compat/c1/splits"))
    parser.add_argument("--output", type=Path, default=Path("results/libeer_clean_compat/c1/split_independent_audit.json"))
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
