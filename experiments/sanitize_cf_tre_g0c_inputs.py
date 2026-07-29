"""Create test-free G0-C selection and validation artifacts from G0-B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pcma.model.cf_tre_baselines import COMPONENT_SPECS


PROTOCOL = "cf_tre_g0c_development_feasibility_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_test_key(value: Any, context: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower().startswith("test"):
                raise ValueError(f"forbidden test key at {context}.{key}")
            _assert_no_test_key(nested, f"{context}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_test_key(nested, f"{context}[{index}]")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _assert_no_test_key(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sanitize(source_root: Path, output_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for dataset in ("seed", "seediv"):
        for session in range(1, 4):
            for subject in range(1, 16):
                stem = f"s{session:02d}_sub{subject:02d}"
                source_json = source_root / "classical" / dataset / f"{stem}.json"
                source_npz = source_root / "classical" / dataset / f"{stem}.npz"
                payload = json.loads(source_json.read_text(encoding="utf-8"))
                selected: dict[str, Any] = {}
                for component in COMPONENT_SPECS:
                    row = payload["components"][component]
                    selected[component] = {
                        "component_index": int(row["component_index"]),
                        "family": row["family"],
                        "views": row["views"],
                        "dimension": int(row["dimension"]),
                        "selected_candidate_index": int(row["selected_candidate_index"]),
                        "selected_parameters": row["selected_parameters"],
                    }
                selection = {
                    "protocol": PROTOCOL,
                    "stage": "G0-C-INPUT",
                    "source_protocol": payload["protocol"],
                    "dataset": dataset,
                    "session": session,
                    "subject": subject,
                    "unit_id": payload["unit_id"],
                    "optimization_seed": int(payload["optimization_seed"]),
                    "split_manifest": payload["split_manifest"],
                    "split_manifest_sha256": payload["split_manifest_sha256"],
                    "trial_split_train": payload["trial_split"]["train"],
                    "trial_split_validation": payload["trial_split"]["validation"],
                    "components": selected,
                    "source_selection_json_sha256": _sha256(source_json),
                }
                destination = output_root / dataset
                selection_path = destination / f"{stem}.selection.json"
                validation_path = destination / f"{stem}.validation.npz"
                _write_json(selection_path, selection)
                with np.load(source_npz, allow_pickle=False) as arrays:
                    validation = {
                        "validation_labels": arrays["validation_labels"],
                        "validation_trials": arrays["validation_trials"],
                    }
                    for component in COMPONENT_SPECS:
                        validation[f"validation_probabilities__{component}"] = arrays[
                            f"validation_probabilities__{component}"
                        ]
                if any(key.lower().startswith("test") for key in validation):
                    raise AssertionError("sanitizer retained a test array")
                with validation_path.open("xb") as handle:
                    np.savez_compressed(handle, **validation)
                files.extend(
                    [
                        {"path": str(selection_path), "sha256": _sha256(selection_path), "role": "selection_only"},
                        {"path": str(validation_path), "sha256": _sha256(validation_path), "role": "validation_only"},
                    ]
                )
    result = {
        "protocol": PROTOCOL,
        "stage": "G0-C-INPUT",
        "status": "PASS",
        "contains_test_keys": False,
        "unit_count": 90,
        "file_count": len(files),
        "files": files,
    }
    _write_json(output_root / "g0c_input_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("results/cf_tre/g0b/formal"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cf_tre/g0c/inputs"))
    args = parser.parse_args()
    result = sanitize(args.source_root.resolve(), args.output_root.resolve())
    print(json.dumps({key: result[key] for key in ("protocol", "status", "unit_count", "file_count", "contains_test_keys")}, indent=2))


if __name__ == "__main__":
    main()
