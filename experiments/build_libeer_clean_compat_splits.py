"""Build labels-only split artifacts for frozen LibEER compatibility C1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from pcma.data.libeer_compat_splits import (
    PROTOCOL,
    build_compat_manifest,
    canonical_json_sha256,
)
from pcma.data.libeer_seediv import SEEDIV_SESSION_LABELS


DEFAULT_LABEL = Path("data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s/label.mat")
DEFAULT_OUTPUT = Path("results/libeer_clean_compat/c1/splits")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build(label_file: Path, output_dir: Path) -> dict[str, str]:
    labels_path = label_file.resolve()
    raw = np.asarray(loadmat(labels_path, variable_names=["label"])["label"]).reshape(-1)
    seed_labels = raw.astype(np.int64) + 1
    seed = build_compat_manifest("seed", [seed_labels] * 3)
    seed["label_source"] = {"path": str(labels_path), "sha256": _sha256(labels_path)}
    seed["canonical_sha256"] = canonical_json_sha256(seed)
    seediv = build_compat_manifest("seediv", SEEDIV_SESSION_LABELS)
    seediv["label_source"] = {"module": "pcma.data.libeer_seediv.SEEDIV_SESSION_LABELS"}
    seediv["canonical_sha256"] = canonical_json_sha256(seediv)
    output = output_dir.resolve()
    files = {
        "seed": output / "libeer_clean_seed_split_v1.json",
        "seediv": output / "libeer_clean_seediv_split_v1.json",
    }
    _write_new(files["seed"], seed)
    _write_new(files["seediv"], seediv)
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "contains_features": False,
        "contains_predictions": False,
        "files": [
            {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in files.values()
        ],
    }
    manifest["canonical_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output / "libeer_clean_split_artifact_manifest_v1.json"
    _write_new(manifest_path, manifest)
    return {**{key: str(value) for key, value in files.items()}, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-label-file", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.seed_label_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
