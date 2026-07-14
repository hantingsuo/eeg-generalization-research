"""Production-independent audit of frozen CF-TRE Track A split artifacts.

Independence rule: this file must not import ``pcma.data.cf_tre_splits`` or the
SEED-IV loader.  The protocol constants and label sequences are duplicated
here intentionally so the production builder cannot validate itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat


PROTOCOL = "cf_tre_seed_family_v1"
ROOT_SEED = 2024
SEEDIV_LABELS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
    (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
    (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("canonical_sha256", None)
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derive_seed(dataset: str, session: int, subject: int, label: int) -> tuple[str, int]:
    material = f"{PROTOCOL}|{dataset}|{session}|{subject}|{label}|{ROOT_SEED}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return material, int.from_bytes(digest[:8], "big", signed=False)


def _expected_unit(
    dataset: str,
    labels: Sequence[int],
    session: int,
    subject: int,
) -> dict[str, Any]:
    counts = (3, 1, 1) if dataset == "seed" else (4, 1, 1)
    classes = (0, 1, 2) if dataset == "seed" else (0, 1, 2, 3)
    label_array = np.asarray(labels, dtype=np.int64)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    materials: list[str] = []
    seeds: list[int] = []
    for label in classes:
        trials = (np.flatnonzero(label_array == label) + 1).tolist()
        if len(trials) != sum(counts):
            raise ValueError(f"wrong class count for {dataset} class {label}")
        material, derived_seed = _derive_seed(dataset, session, subject, label)
        random.Random(derived_seed).shuffle(trials)
        train.extend(trials[: counts[0]])
        validation.extend(trials[counts[0] : counts[0] + counts[1]])
        test.extend(trials[-counts[2] :])
        materials.append(material)
        seeds.append(derived_seed)
    return {
        "unit_id": f"{dataset}:s{session:02d}:sub{subject:02d}",
        "session": session,
        "subject": subject,
        "label_sequence": label_array.tolist(),
        "train_trials": sorted(train),
        "validation_trials": sorted(validation),
        "test_trials": sorted(test),
        "derivation_strings": materials,
        "derived_seeds": seeds,
    }


def _audit_dataset(
    path: Path,
    *,
    dataset: str,
    expected_labels: Sequence[Sequence[int]],
) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("canonical_sha256") != _canonical_hash(payload):
        errors.append(f"{dataset}: canonical hash mismatch")
    if payload.get("protocol") != PROTOCOL or payload.get("dataset") != dataset:
        errors.append(f"{dataset}: header mismatch")
    if payload.get("root_seed") != ROOT_SEED:
        errors.append(f"{dataset}: root seed mismatch")
    units = payload.get("units")
    if not isinstance(units, list) or len(units) != 45:
        errors.append(f"{dataset}: expected 45 units")
        return errors
    by_id = {unit.get("unit_id"): unit for unit in units}
    if len(by_id) != 45:
        errors.append(f"{dataset}: duplicate unit IDs")
    for session in range(1, 4):
        labels = expected_labels[session - 1]
        for subject in range(1, 16):
            expected = _expected_unit(dataset, labels, session, subject)
            actual = by_id.get(expected["unit_id"])
            if actual is None:
                errors.append(f"{dataset}: missing {expected['unit_id']}")
                continue
            for key, value in expected.items():
                if actual.get(key) != value:
                    errors.append(f"{dataset}: {expected['unit_id']} {key} mismatch")
    return errors


def audit_split_artifacts(
    artifact_dir: Path,
    seed_label_file: Path,
) -> dict[str, Any]:
    root = artifact_dir.expanduser().resolve()
    seed_path = root / "track_a_seed_split_v1.json"
    seediv_path = root / "track_a_seediv_split_v1.json"
    manifest_path = root / "track_a_split_artifact_manifest_v1.json"
    errors: list[str] = []
    for path in (seed_path, seediv_path, manifest_path):
        if not path.is_file():
            errors.append(f"missing artifact {path}")
    if errors:
        return {"status": "FAIL", "errors": errors, "checks_passed": 0, "checks_total": 4}

    seed_source = seed_label_file.expanduser().resolve()
    raw = np.asarray(loadmat(seed_source, variable_names=["label"])["label"], dtype=np.int64).reshape(-1)
    seed_labels = (raw + 1).tolist()
    errors.extend(_audit_dataset(seed_path, dataset="seed", expected_labels=[seed_labels] * 3))
    errors.extend(_audit_dataset(seediv_path, dataset="seediv", expected_labels=SEEDIV_LABELS))

    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("canonical_sha256") != _canonical_hash(artifact_manifest):
        errors.append("artifact manifest canonical hash mismatch")
    expected_paths = {str(seed_path), str(seediv_path)}
    listed_paths = {entry.get("path") for entry in artifact_manifest.get("files", [])}
    if listed_paths != expected_paths:
        errors.append("artifact manifest path set mismatch")
    for entry in artifact_manifest.get("files", []):
        path = Path(entry["path"])
        if entry.get("sha256") != _sha256_file(path):
            errors.append(f"file hash mismatch: {path}")
        if entry.get("size_bytes") != path.stat().st_size:
            errors.append(f"file size mismatch: {path}")
    if artifact_manifest.get("contains_predictions") is not False:
        errors.append("artifact manifest prediction boundary mismatch")
    if artifact_manifest.get("contains_accuracy") is not False:
        errors.append("artifact manifest accuracy boundary mismatch")

    total = 7
    return {
        "protocol": PROTOCOL,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks_passed": total if not errors else max(0, total - len(errors)),
        "checks_total": total,
        "independence": {
            "imports_production_split_module": False,
            "imports_seediv_loader": False,
            "reconstructs_assignments_from_hash_rule": True,
            "reloads_seed_label_source": True,
            "duplicates_official_seediv_labels": True,
        },
        "artifacts": {
            "seed": str(seed_path),
            "seediv": str(seediv_path),
            "manifest": str(manifest_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/cf_tre/g0a"))
    parser.add_argument(
        "--seed-label-file",
        type=Path,
        default=Path("data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s/label.mat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cf_tre/g0a/track_a_split_independent_audit_v1.json"),
    )
    args = parser.parse_args()
    report = audit_split_artifacts(args.artifact_dir, args.seed_label_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {args.output}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

