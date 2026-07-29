"""Fail-closed preflight audit before the one-shot CF-TRE G1 outer test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "cf_tre_g1_one_shot_outer_test_v1"
MANIFEST_SHA256 = "93c1f730044255d8eb7b9ff910f1bf73ff646a5b16d815701436228dc9b0fc04"
G0C_SHA256 = "2e1725f68f265e514d6a1aecd8db5b27a3baab1af24e6075dbd5a78020275958"
EXPECTED = {
    "seed": {
        "strongest": "linear_de310",
        "mean": (0.0, 0.8, 0.0, [0.2850001599957752, 0.06693946159728863, 0.04777817754926334, 0.20592650350298738, 0.04719576589915263, 0.11531345143021111, 0.23184648002532168]),
        "tail": (0.75, 0.67, 0.0, [0.1808485428888424, 0.1044570719746394, 0.036395379687067075, 0.2535503499609486, 0.18345534659920987, 0.11567365381206707, 0.12561965507722578]),
    },
    "seediv": {
        "strongest": "linear_de310",
        "mean": (0.0, 0.8, 0.0, [0.335001662065595, 0.11351029648416651, 0.05311598319057763, 0.174830585413816, 0.064549428098917, 0.14229395637427858, 0.11669808837264943]),
        "tail": (0.25, 0.67, 0.0, [0.3288458061791317, 0.1433732291006174, 0.0517417927009323, 0.1511130100759164, 0.06396133376992434, 0.13172350925537868, 0.1292413189180992]),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict[str, Any]:
    manifest = root / "plans/2026-07-15-cf-tre-g1-one-shot-outer-test-frozen-manifest.md"
    g0c_path = root / "results/cf_tre/g0c/g0c_feasibility_result.json"
    sidecar_path = root / "plans/2026-07-15-cf-tre-g1-one-shot-outer-test-frozen-manifest.sha256.json"
    errors: list[str] = []
    checks = 0
    if sha256(manifest) != MANIFEST_SHA256:
        errors.append("frozen manifest hash mismatch")
    if sha256(g0c_path) != G0C_SHA256:
        errors.append("G0-C result hash mismatch")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("frozen_manifest_sha256") != MANIFEST_SHA256 or sidecar.get("g0c_result_sha256") != G0C_SHA256:
        errors.append("hash sidecar mismatch")
    checks += 3

    g0c = json.loads(g0c_path.read_text(encoding="utf-8"))
    if g0c.get("status") != "PASS_REVIEW_BEFORE_OUTER_TEST":
        errors.append("G0-C did not authorize review before outer test")
    checks += 1
    for dataset, expected in EXPECTED.items():
        rows = g0c["datasets"][dataset]
        if rows["strongest_single"]["component"] != expected["strongest"]:
            errors.append(f"{dataset}: strongest component changed")
        checks += 1
        for name in ("mean", "tail"):
            selected = rows[f"selected_{name}"]
            expected_lambda, expected_alpha, expected_delta, expected_weights = expected[name]
            observed = [selected["lambda_tail"], selected["alpha"], selected["delta"]]
            if not np.allclose(observed, [expected_lambda, expected_alpha, expected_delta], rtol=0.0, atol=0.0):
                errors.append(f"{dataset}: selected {name} configuration changed")
            if not np.allclose(selected["weights"], expected_weights, rtol=0.0, atol=0.0):
                errors.append(f"{dataset}: selected {name} weights changed")
            checks += 2

    files: list[dict[str, Any]] = []
    for dataset in ("seed", "seediv"):
        for session in range(1, 4):
            for subject in range(1, 16):
                stem = f"s{session:02d}_sub{subject:02d}.npz"
                paths = (
                    root / "results/cf_tre/g0b/formal/classical" / dataset / stem,
                    root / "results/cf_tre/g0b/formal/deep/dgcnn" / dataset / stem,
                    root / "results/cf_tre/g0b/formal/deep/gcbnet" / dataset / stem,
                )
                for path in paths:
                    checks += 1
                    if not path.is_file():
                        errors.append(f"missing source file: {path}")
                        continue
                    files.append({
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    })
    if len(files) != 270 or len({row["path"] for row in files}) != 270:
        errors.append("source file coverage is not exactly 270 unique NPZ files")
    checks += 1
    outer_result = root / "results/cf_tre/g1/g1_outer_test_result.json"
    prediction_paths = [root / f"results/cf_tre/g1/{dataset}_outer_predictions.npz" for dataset in ("seed", "seediv")]
    if outer_result.exists() or any(path.exists() for path in prediction_paths):
        errors.append("proposed-method outer-test output existed before release")
    checks += 1
    return {
        "protocol": PROTOCOL,
        "stage": "G1-PREFLIGHT",
        "status": "PASS_RELEASE_ONE_SHOT" if not errors else "FAIL_BLOCK_OUTER_TEST",
        "checks": checks,
        "errors": errors,
        "manifest_sha256": MANIFEST_SHA256,
        "g0c_result_sha256": G0C_SHA256,
        "source_file_count": len(files),
        "source_files": files,
        "outer_test_outputs_absent_before_release": not outer_result.exists() and not any(path.exists() for path in prediction_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g1/g1_preflight_audit.json"))
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("protocol", "status", "checks", "errors", "source_file_count")}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS_RELEASE_ONE_SHOT" else 2)


if __name__ == "__main__":
    main()
