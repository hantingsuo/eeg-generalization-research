"""Production-independent coverage and metric audit for all G0-C OOF cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


COMPONENTS = (
    "linear_de310", "linear_structural245", "linear_all555", "rbf_de310",
    "rbf_all555", "xgboost_all555", "lightgbm_all555",
)


def _expected_folds(labels: np.ndarray, trials: np.ndarray) -> np.ndarray:
    trial_label: dict[int, int] = {}
    for trial in np.unique(trials):
        values = np.unique(labels[trials == trial])
        if len(values) != 1:
            raise ValueError("trial contains multiple labels")
        trial_label[int(trial)] = int(values[0])
    assigned: dict[int, int] = {}
    for label in sorted(set(trial_label.values())):
        values = [trial for trial, value in trial_label.items() if value == label]
        ordered = sorted(values, key=lambda trial: hashlib.sha256(f"cf_tre_seed_family_v1|calibration_trials|2024|{label}|{trial}".encode()).hexdigest())
        for index, trial in enumerate(ordered):
            assigned[trial] = index % 3 + 1
    return np.asarray([assigned[int(trial)] for trial in trials], dtype=np.int8)


def audit(root: Path, input_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks = 0
    for dataset in ("seed", "seediv"):
        for session in range(1, 4):
            for subject in range(1, 16):
                stem = f"s{session:02d}_sub{subject:02d}"
                json_path = root / dataset / f"{stem}.json"
                npz_path = root / dataset / f"{stem}.npz"
                selection_path = input_root / dataset / f"{stem}.selection.json"
                if not json_path.is_file() or not npz_path.is_file():
                    errors.append(f"missing {dataset}/{stem}")
                    continue
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                checks += 4
                if payload.get("outer_partition_loaded") != "train_only" or payload.get("contains_validation_predictions") is not False or payload.get("contains_test_predictions") is not False:
                    errors.append(f"access boundary {dataset}/{stem}")
                allowed_boundary_flags = {
                    "contains_validation_predictions",
                    "contains_test_predictions",
                }
                if any(
                    str(key).lower().startswith(("test", "validation"))
                    and key not in allowed_boundary_flags
                    for key in payload
                ):
                    errors.append(f"forbidden JSON key {dataset}/{stem}")
                with np.load(npz_path, allow_pickle=False) as arrays:
                    if any(key.lower().startswith(("test", "validation")) for key in arrays.files):
                        errors.append(f"forbidden NPZ key {dataset}/{stem}")
                    labels, trials, folds = arrays["train_labels"], arrays["train_trials"], arrays["inner_fold"]
                    checks += 3
                    if set(np.unique(trials).tolist()) != set(selection["trial_split_train"]):
                        errors.append(f"train trial coverage {dataset}/{stem}")
                    if not np.array_equal(folds, _expected_folds(labels, trials)):
                        errors.append(f"fold derivation {dataset}/{stem}")
                    for component in COMPONENTS:
                        probabilities = arrays[f"oof_probabilities__{component}"]
                        checks += 3
                        if probabilities.shape[0] != len(labels) or not np.all(np.isfinite(probabilities)) or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
                            errors.append(f"probability integrity {dataset}/{stem}/{component}")
                            continue
                        predicted = probabilities.argmax(axis=1)
                        expected = {"accuracy": float(accuracy_score(labels, predicted)), "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0))}
                        for key, value in expected.items():
                            if not np.isclose(float(payload["components"][component][key]), value, rtol=0, atol=1e-12):
                                errors.append(f"metric {dataset}/{stem}/{component}/{key}")
    return {"protocol": "cf_tre_g0c_development_feasibility_v1", "audit": "independent train-only OOF fold/probability/metric reconstruction", "unit_count": 90, "checks": checks, "errors": errors, "status": "PASS" if not errors else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cf_tre/g0c/oof"))
    parser.add_argument("--input-root", type=Path, default=Path("results/cf_tre/g0c/inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g0c/g0c_oof_independent_audit.json"))
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.input_root.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
