"""Independent access-boundary audit for sanitized G0-C inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = (
    "linear_de310", "linear_structural245", "linear_all555", "rbf_de310",
    "rbf_all555", "xgboost_all555", "lightgbm_all555",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(value: Any, context: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower().startswith("test"):
                errors.append(f"{context}: forbidden key {key}")
            _scan(nested, f"{context}.{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan(nested, f"{context}[{index}]", errors)


def audit(root: Path) -> dict[str, Any]:
    manifest_path = root / "g0c_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = 0
    listed = {Path(row["path"]).resolve(): row for row in manifest["files"]}
    for dataset in ("seed", "seediv"):
        for session in range(1, 4):
            for subject in range(1, 16):
                stem = f"s{session:02d}_sub{subject:02d}"
                selection_path = (root / dataset / f"{stem}.selection.json").resolve()
                validation_path = (root / dataset / f"{stem}.validation.npz").resolve()
                for path in (selection_path, validation_path):
                    checks += 2
                    if path not in listed or not path.is_file():
                        errors.append(f"missing/unlisted {path}")
                        continue
                    if _sha256(path) != listed[path]["sha256"]:
                        errors.append(f"hash mismatch {path}")
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                _scan(selection, str(selection_path), errors)
                checks += 1
                if set(selection.get("components", {})) != set(COMPONENTS):
                    errors.append(f"component mismatch {selection_path}")
                with np.load(validation_path, allow_pickle=False) as arrays:
                    expected = {"validation_labels", "validation_trials"} | {
                        f"validation_probabilities__{component}" for component in COMPONENTS
                    }
                    checks += 2
                    if set(arrays.files) != expected:
                        errors.append(f"array key mismatch {validation_path}")
                    if any(key.lower().startswith("test") for key in arrays.files):
                        errors.append(f"test array retained {validation_path}")
    checks += 2
    if manifest.get("file_count") != 180 or len(listed) != 180:
        errors.append("manifest file count mismatch")
    if manifest.get("contains_test_keys") is not False:
        errors.append("manifest access flag mismatch")
    return {"protocol": "cf_tre_g0c_development_feasibility_v1", "audit": "independent sanitized-input hash/key audit", "checks": checks, "errors": errors, "status": "PASS" if not errors else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/cf_tre/g0c/inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g0c/g0c_input_independent_audit.json"))
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
