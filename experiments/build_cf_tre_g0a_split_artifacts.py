"""Build explicit Track A split artifacts for frozen CF-TRE protocol v1.

This command reads labels only.  It does not load feature matrices, fit a
model, calculate accuracy, or contact any test prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.io import loadmat

from pcma.data.cf_tre_splits import (
    PROTOCOL,
    build_track_a_split_manifest,
    canonical_json_sha256,
)
from pcma.data.libeer_seediv import SEEDIV_SESSION_LABELS


DEFAULT_SEED_LABEL_FILE = Path(
    "data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s/label.mat"
)
DEFAULT_OUTPUT_DIR = Path("results/cf_tre/g0a")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_split_artifacts(seed_label_file: Path, output_dir: Path) -> dict[str, Any]:
    seed_label_path = seed_label_file.expanduser().resolve()
    output_path = output_dir.expanduser().resolve()
    if not seed_label_path.is_file():
        raise FileNotFoundError(seed_label_path)

    payload = loadmat(seed_label_path, variable_names=["label"])
    if "label" not in payload:
        raise KeyError(f"missing label variable in {seed_label_path}")
    seed_labels = np.asarray(payload["label"], dtype=np.int64).reshape(-1) + 1
    if seed_labels.shape != (15,) or set(seed_labels.tolist()) != {0, 1, 2}:
        raise ValueError(f"unexpected mapped SEED labels: {seed_labels.tolist()}")

    seed_manifest = build_track_a_split_manifest("SEED", [seed_labels] * 3)
    seed_manifest["label_source"] = {
        "path": str(seed_label_path),
        "sha256": _sha256_file(seed_label_path),
        "mat_variable": "label",
        "mapping": "official {-1,0,1} plus one to {0,1,2}",
    }
    seed_manifest["canonical_sha256"] = canonical_json_sha256(seed_manifest)

    seediv_labels = [np.asarray(labels, dtype=np.int64) for labels in SEEDIV_SESSION_LABELS]
    seediv_manifest = build_track_a_split_manifest("SEED-IV", seediv_labels)
    labels_bytes = json.dumps(
        [labels.tolist() for labels in seediv_labels],
        separators=(",", ":"),
    ).encode("utf-8")
    seediv_manifest["label_source"] = {
        "module": "pcma.data.libeer_seediv.SEEDIV_SESSION_LABELS",
        "canonical_sequence_sha256": hashlib.sha256(labels_bytes).hexdigest(),
    }
    seediv_manifest["canonical_sha256"] = canonical_json_sha256(seediv_manifest)

    seed_file = output_path / "track_a_seed_split_v1.json"
    seediv_file = output_path / "track_a_seediv_split_v1.json"
    artifact_manifest_file = output_path / "track_a_split_artifact_manifest_v1.json"
    for target in (seed_file, seediv_file, artifact_manifest_file):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {target}")
    _write_json_exclusive(seed_file, seed_manifest)
    _write_json_exclusive(seediv_file, seediv_manifest)

    artifact_manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "artifact_role": "labels_and_split_assignments_only",
        "contains_feature_data": False,
        "contains_predictions": False,
        "contains_accuracy": False,
        "files": [],
    }
    for path in (seed_file, seediv_file):
        artifact_manifest["files"].append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    artifact_manifest["canonical_sha256"] = canonical_json_sha256(artifact_manifest)
    _write_json_exclusive(artifact_manifest_file, artifact_manifest)
    return {
        "seed": seed_file,
        "seediv": seediv_file,
        "artifact_manifest": artifact_manifest_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-label-file", type=Path, default=DEFAULT_SEED_LABEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    outputs = build_split_artifacts(args.seed_label_file, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

