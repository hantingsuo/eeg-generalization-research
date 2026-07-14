"""Close the CF-TRE G0-A implementation gate without loading EEG features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any


PROTOCOL = "cf_tre_seed_family_v1"
EXPECTED_MANIFEST_SHA256 = "52d36f0849722dca27ba4145d70bd14c9261f2abcf41e50f0e9f7e77199ffc02"
SOURCE_FILES = (
    "pcma/data/cf_tre_splits.py",
    "pcma/data/cf_tre_features.py",
    "pcma/model/cf_tre.py",
    "experiments/build_cf_tre_g0a_split_artifacts.py",
    "experiments/audit_cf_tre_g0a_split_artifacts.py",
    "experiments/check_cf_tre_libeer_imports.py",
    "tests/test_cf_tre_splits.py",
    "tests/test_cf_tre_features.py",
    "tests/test_cf_tre.py",
    "tests/test_cf_tre_g0a_artifacts.py",
    "requirements.txt",
    "pyproject.toml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pytest_pass_count(log_path: Path) -> int:
    raw = log_path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:200]:
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")
    matches = re.findall(r"(\d+) passed", text)
    if not matches:
        raise ValueError(f"no pytest pass count in {log_path}")
    return int(matches[-1])


def audit_g0a(
    repo_root: Path,
    *,
    targeted_log: Path,
    full_log: Path,
) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, detail: Any) -> None:
        checks.append(
            {"id": identifier, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    frozen = root / "plans/2026-07-14-cf-tre-g0a-frozen-manifest.md"
    sidecar_path = root / "plans/2026-07-14-cf-tre-g0a-frozen-manifest.sha256.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_frozen_hash = _sha256(frozen)
    check(
        "frozen_manifest_hash",
        actual_frozen_hash == EXPECTED_MANIFEST_SHA256 == sidecar.get("sha256"),
        actual_frozen_hash,
    )

    split_audit_path = root / "results/cf_tre/g0a/track_a_split_independent_audit_v1.json"
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    check(
        "independent_split_audit",
        split_audit.get("status") == "PASS"
        and split_audit.get("checks_passed") == split_audit.get("checks_total") == 7,
        f"{split_audit.get('checks_passed')}/{split_audit.get('checks_total')}",
    )
    independence = split_audit.get("independence", {})
    check(
        "split_audit_independence",
        independence.get("imports_production_split_module") is False
        and independence.get("imports_seediv_loader") is False
        and independence.get("reconstructs_assignments_from_hash_rule") is True,
        independence,
    )

    deep_path = root / "results/cf_tre/g0a/libeer_deep_import_forward_v1.json"
    deep = json.loads(deep_path.read_text(encoding="utf-8"))
    expected_models = {"DGCNN", "GCBNet", "GCBNet_BLS"}
    actual_models = {model.get("name") for model in deep.get("models", [])}
    check(
        "pinned_deep_import_forward",
        deep.get("status") == "PASS"
        and deep.get("libeer_commit") == "39dc27e504e14138767b87ce8bce485380fd4f5a"
        and deep.get("role") == "import_and_forward_only_no_training"
        and actual_models == expected_models
        and all(model.get("finite_output") is True for model in deep.get("models", [])),
        sorted(actual_models),
    )

    targeted_count = _pytest_pass_count((root / targeted_log).resolve())
    full_count = _pytest_pass_count((root / full_log).resolve())
    check("targeted_tests", targeted_count == 20, targeted_count)
    check("full_test_suite", full_count == 212, full_count)

    result_dir = root / "results/cf_tre/g0a"
    forbidden = sorted(
        str(path.relative_to(root))
        for path in result_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".npz", ".pkl", ".csv"}
    )
    check("no_trained_or_prediction_artifacts", not forbidden, forbidden)

    split_manifest = json.loads(
        (result_dir / "track_a_split_artifact_manifest_v1.json").read_text(encoding="utf-8")
    )
    check(
        "metadata_only_split_artifacts",
        split_manifest.get("contains_feature_data") is False
        and split_manifest.get("contains_predictions") is False
        and split_manifest.get("contains_accuracy") is False,
        split_manifest.get("artifact_role"),
    )

    source_hashes: dict[str, str] = {}
    missing_sources: list[str] = []
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            missing_sources.append(relative)
        else:
            source_hashes[relative] = _sha256(path)
    check("implementation_sources_present", not missing_sources, missing_sources)

    audit_source = (root / "experiments/audit_cf_tre_g0a_split_artifacts.py").read_text(
        encoding="utf-8"
    )
    forbidden_imports = (
        "from pcma.data.cf_tre_splits",
        "import pcma.data.cf_tre_splits",
        "from pcma.data.libeer_seediv",
        "import pcma.data.libeer_seediv",
    )
    actual_forbidden_imports = [token for token in forbidden_imports if token in audit_source]
    check("auditor_source_import_boundary", not actual_forbidden_imports, actual_forbidden_imports)

    pyyaml_version = importlib.metadata.version("PyYAML")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    check(
        "libeer_yaml_dependency_declared",
        "PyYAML" in requirements and "PyYAML" in pyproject,
        pyyaml_version,
    )

    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "protocol": PROTOCOL,
        "status": status,
        "verification_status": "ANALYZED",
        "gate": "G0-A",
        "real_data_training_started": False,
        "real_data_accuracy_computed": False,
        "checks_passed": sum(item["status"] == "PASS" for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "source_sha256": source_hashes,
        "known_failed_checks_preserved": [
            "logs/runs/check_cf_tre_libeer_imports_20260714_1415.log",
            "logs/runs/check_cf_tre_libeer_imports_20260714_1417_after_pyyaml.log",
        ],
        "successful_deep_check_log": (
            "logs/runs/check_cf_tre_libeer_imports_20260714_1420_direct_files.log"
        ),
        "targeted_test_log": str(targeted_log),
        "full_test_log": str(full_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--targeted-log",
        type=Path,
        default=Path("logs/runs/pytest_cf_tre_g0a_targeted_20260714_1428.log"),
    )
    parser.add_argument(
        "--full-log",
        type=Path,
        default=Path("logs/runs/pytest_cf_tre_g0a_full_20260714_1425.log"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cf_tre/g0a/g0a_implementation_audit_v1.json"),
    )
    args = parser.parse_args()
    report = audit_g0a(
        args.repo_root,
        targeted_log=args.targeted_log,
        full_log=args.full_log,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G0-A audit: {args.output}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
