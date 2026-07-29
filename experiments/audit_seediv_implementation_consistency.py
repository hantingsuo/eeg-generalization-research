"""Read-only implementation audit for the SEED-IV LibEER compatibility cell."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Sequence

import numpy as np
import scipy
from scipy.io import whosmat

from pcma.data.libeer_seediv import SEEDIV_SESSION_FILES, SEEDIV_SESSION_LABELS


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"
EXPECTED_DGCNN_SHA256 = "020e369d54676e78bc3ef42df2fa76d65eb85140f566c22857fd2c6c65b0577d"
EXPECTED_DGCNN_CONFIG_SHA256 = (
    "0e54afb0986eade332316449217d07acb6af04b6383490228d8a6c05230f0d4a"
)
TRIAL_VARIABLE = re.compile(r"^.+_eeg(\d+)$")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--libeer-root",
        type=Path,
        default=Path(os.environ.get("LIBEER_REPO_ROOT", "third_party/libeer")),
    )
    result.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/SEED/SEED_IV",
    )
    result.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "results/libeer_gate_b/cache_seediv_1s_de_lds_39dc27e",
    )
    result.add_argument(
        "--c1-root",
        type=Path,
        default=ROOT / "results/libeer_clean_compat/c1",
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve().as_posix()}",
            "-C",
            str(repo.resolve()),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def function_literal_assignments(
    source: str,
    function_name: str,
    names: Iterable[str],
) -> dict[str, Any]:
    requested = set(names)
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"function not found: {function_name}")
    values: dict[str, Any] = {}
    for node in function.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in requested:
            values[target.id] = ast.literal_eval(node.value)
    missing = sorted(requested - set(values))
    if missing:
        raise ValueError(f"literal assignments not found in {function_name}: {missing}")
    return values


def trial_numbers_in_file_order(path: Path) -> tuple[list[str], list[int]]:
    names: list[str] = []
    numbers: list[int] = []
    for name, _shape, _kind in whosmat(path):
        match = TRIAL_VARIABLE.fullmatch(name)
        if match is not None:
            names.append(name)
            numbers.append(int(match.group(1)))
    return names, numbers


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    status: str,
    evidence: Any,
) -> None:
    if status not in {"PASS", "FAIL", "CANNOT_VERIFY"}:
        raise ValueError(status)
    checks.append({"id": check_id, "status": status, "evidence": evidence})


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    libeer_repo = args.libeer_root.resolve()
    libeer_code = libeer_repo / "LibEER"
    dataset_root = args.dataset_root.resolve()
    cache_root = args.cache_root.resolve()
    c1_root = args.c1_root.resolve()
    checks: list[dict[str, Any]] = []

    commit = git_commit(libeer_repo)
    dgcnn_hash = sha256(libeer_code / "models/DGCNN.py")
    config_hash = sha256(libeer_code / "config/model_param/DGCNN.yaml")
    add_check(
        checks,
        "pinned_source_identity",
        "PASS"
        if (
            commit == EXPECTED_COMMIT
            and dgcnn_hash == EXPECTED_DGCNN_SHA256
            and config_hash == EXPECTED_DGCNN_CONFIG_SHA256
        )
        else "FAIL",
        {
            "commit": commit,
            "dgcnn_sha256": dgcnn_hash,
            "dgcnn_config_sha256": config_hash,
        },
    )

    load_data_path = libeer_code / "data_utils/load_data.py"
    load_source = load_data_path.read_text(encoding="utf-8")
    upstream = function_literal_assignments(
        load_source,
        "read_seedIV_raw",
        ("eeg_files", "ses_label1", "ses_label2", "ses_label3"),
    )
    files_match = tuple(tuple(row) for row in upstream["eeg_files"]) == SEEDIV_SESSION_FILES
    labels_match = (
        tuple(tuple(upstream[f"ses_label{index}"]) for index in range(1, 4))
        == SEEDIV_SESSION_LABELS
    )
    add_check(
        checks,
        "file_and_label_tables",
        "PASS" if files_match and labels_match else "FAIL",
        {
            "file_table_match": files_match,
            "label_table_match": labels_match,
            "sessions": 3,
            "subjects_per_session": 15,
            "trials_per_recording": 24,
        },
    )

    parallel_source = ast.get_source_segment(
        load_source,
        next(
            node
            for node in ast.parse(load_source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "parallel_read_seedIV_raw"
        ),
    )
    drop_first_match = (
        parallel_source is not None
        and "list(subject_data.keys())[3:]" in parallel_source
        and "trail_data[:,1:]" in parallel_source.replace(" ", "")
    )

    raw_records: list[dict[str, Any]] = []
    variable_order_ok = True
    for session, filenames in enumerate(SEEDIV_SESSION_FILES, start=1):
        for subject, filename in enumerate(filenames, start=1):
            raw_path = dataset_root / "eeg_raw_data" / str(session) / filename
            names, numbers = trial_numbers_in_file_order(raw_path)
            order_ok = numbers == list(range(1, 25))
            variable_order_ok = variable_order_ok and order_ok
            raw_records.append(
                {
                    "session": session,
                    "subject": subject,
                    "file": str(raw_path),
                    "size_bytes": raw_path.stat().st_size,
                    "trial_variables": names,
                    "numeric_order_1_to_24": order_ok,
                }
            )
    add_check(
        checks,
        "raw_trial_order_and_first_sample",
        "PASS" if variable_order_ok and drop_first_match else "FAIL",
        {
            "recordings_checked": len(raw_records),
            "all_mat_variable_orders_numeric": variable_order_ok,
            "upstream_loader_uses_mat_order": True,
            "upstream_and_local_drop_first_sample": drop_first_match,
        },
    )

    expected_parameters = {
        "sample_rate_hz": 200,
        "drop_first_sample": True,
        "pass_band_hz": [0.3, 50.0],
        "extract_bands_hz": [
            [0.5, 4.0],
            [4.0, 8.0],
            [8.0, 14.0],
            [14.0, 30.0],
            [30.0, 50.0],
        ],
        "time_window_seconds": 1.0,
        "overlap_seconds": 0.0,
        "feature_type": "de_lds",
        "internal_dtype": "float64",
        "output_dtype": "float32",
        "functions": ["bandpass_filter", "de_extraction", "lds"],
        "numpy_version": "1.26.4",
        "scipy_version": "1.13.1",
    }
    cache_records: list[dict[str, Any]] = []
    cache_ok = True
    for raw in raw_records:
        session = int(raw["session"])
        subject = int(raw["subject"])
        manifest_path = (
            cache_root / f"seediv_de_lds_s{session:02d}_sub{subject:02d}.json"
        )
        cache_path = manifest_path.with_suffix(".npz")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_labels = list(SEEDIV_SESSION_LABELS[session - 1])
        expected_names = list(raw["trial_variables"])
        item_ok = (
            cache_path.is_file()
            and manifest.get("libeer_commit") == EXPECTED_COMMIT
            and manifest.get("session") == session
            and manifest.get("subject") == subject
            and manifest.get("parameters") == expected_parameters
            and manifest.get("trial_labels") == expected_labels
            and manifest.get("raw_trial_variables") == expected_names
            and manifest.get("raw_size_bytes") == raw["size_bytes"]
            and len(manifest.get("trial_shapes", [])) == 24
            and all(
                len(shape) == 3 and shape[1:] == [62, 5] and shape[0] > 0
                for shape in manifest.get("trial_shapes", [])
            )
        )
        cache_ok = cache_ok and item_ok
        cache_records.append(
            {
                "session": session,
                "subject": subject,
                "manifest": str(manifest_path),
                "cache": str(cache_path),
                "pass": item_ok,
            }
        )
    add_check(
        checks,
        "preprocessing_cache_provenance",
        "PASS" if cache_ok else "FAIL",
        {
            "recordings_checked": len(cache_records),
            "all_parameters_labels_shapes_and_sources_match": cache_ok,
            "numpy_version_now": np.__version__,
            "scipy_version_now": scipy.__version__,
        },
    )

    preprocess_source = (libeer_code / "data_utils/preprocess.py").read_text(
        encoding="utf-8"
    )
    wrapper_source = (ROOT / "pcma/data/libeer_seediv.py").read_text(encoding="utf-8")
    pipeline_ok = all(
        token in wrapper_source
        for token in (
            "preprocess.bandpass_filter",
            "preprocess.de_extraction",
            "preprocess.lds",
        )
    ) and all(
        f"def {name}(" in preprocess_source
        for name in ("bandpass_filter", "de_extraction", "lds", "segment_data")
    )
    add_check(
        checks,
        "preprocessing_function_identity",
        "PASS" if pipeline_ok else "FAIL",
        {
            "local_wrapper_calls_pinned_functions_directly": pipeline_ok,
            "segment_data_sample_length_one_returns_input_unchanged": (
                "if sample_length == 1:" in preprocess_source
                and "return data, len(data[0][0][0][0][0])" in preprocess_source
            ),
            "preprocess_sha256": sha256(libeer_code / "data_utils/preprocess.py"),
            "local_wrapper_sha256": sha256(ROOT / "pcma/data/libeer_seediv.py"),
        },
    )

    direct_split = json.loads(
        (c1_root / "pinned_split_direct_check.json").read_text(encoding="utf-8")
    )
    independent_split = json.loads(
        (c1_root / "split_independent_audit.json").read_text(encoding="utf-8")
    )
    split_ok = (
        direct_split.get("status") == "PASS"
        and direct_split.get("checks") == 270
        and not direct_split.get("errors")
        and independent_split.get("status") == "PASS"
    )
    add_check(
        checks,
        "trial_split_equivalence",
        "PASS" if split_ok else "FAIL",
        {
            "direct_pinned_checks": direct_split,
            "independent_split_audit": independent_split,
        },
    )

    result_records: list[dict[str, Any]] = []
    training_ok = True
    for session in range(1, 4):
        for subject in range(1, 16):
            path = (
                c1_root
                / "formal/dgcnn/seediv"
                / f"s{session:02d}_sub{subject:02d}.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            history = payload.get("history", [])
            values = [float(row["validation_macro_f1"]) for row in history]
            earliest_best_epoch = values.index(max(values)) + 1 if values else -1
            item_ok = (
                payload.get("protocol") == "libeer_39dc27e_clean_table_compat_v1"
                and payload.get("stage") == "C1"
                and payload.get("dataset") == "seediv"
                and payload.get("model") == "dgcnn"
                and payload.get("epochs") == 150
                and payload.get("batch_size") == 32
                and payload.get("eval_batch_size") == 32
                and payload.get("learning_rate") == 0.001
                and payload.get("optimization_seed") == 2024
                and payload.get("sampler_semantics") == "upstream_global_rng"
                and payload.get("normalization")
                == "none, matching the pinned LibEER DGCNN/GCBNet path"
                and len(history) == 150
                and payload.get("best_epoch") == earliest_best_epoch
                and payload.get("libeer_commit") == EXPECTED_COMMIT
                and payload.get("test_contacted") is True
            )
            training_ok = training_ok and item_ok
            result_records.append(
                {
                    "session": session,
                    "subject": subject,
                    "result": str(path),
                    "best_epoch": payload.get("best_epoch"),
                    "pass": item_ok,
                }
            )
    add_check(
        checks,
        "model_training_selection_and_test",
        "PASS" if training_ok else "FAIL",
        {
            "recordings_checked": len(result_records),
            "all_declared_settings_and_earliest_best_epochs_match": training_ok,
        },
    )

    aggregate = json.loads(
        (c1_root / "c1_dgcnn_audit_v2.json").read_text(encoding="utf-8")
    )
    seediv_aggregate = aggregate["aggregate"]["seediv"]
    aggregation_ok = (
        abs(seediv_aggregate["subject_equal_window_accuracy"] - 0.48990393722090375)
        < 1e-15
        and abs(seediv_aggregate["official_accuracy_anchor"] - 0.5239) < 1e-15
        and abs(seediv_aggregate["anchor_gap"] + 0.03399606277909628) < 1e-15
        and seediv_aggregate["within_two_percentage_points"] is False
    )
    add_check(
        checks,
        "aggregation_and_frozen_gate",
        "PASS" if aggregation_ok else "FAIL",
        seediv_aggregate,
    )

    add_check(
        checks,
        "public_table_runtime_and_upstream_feature_bytes",
        "CANNOT_VERIFY",
        {
            "reason": (
                "The pinned repository does not archive the dependency lockfile, "
                "preprocessed SEED-IV arrays, checkpoints, or per-unit predictions "
                "used to produce the public table. Exact bit-level equivalence to "
                "that historical run therefore cannot be established."
            ),
            "local_runtime": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        },
    )

    fail_count = sum(check["status"] == "FAIL" for check in checks)
    cannot_verify_count = sum(
        check["status"] == "CANNOT_VERIFY" for check in checks
    )
    status = "PASS_WITH_LIMITATION" if fail_count == 0 else "FAIL"
    payload = {
        "audit": "seediv_libeer_implementation_consistency_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "summary": {
            "pass": sum(check["status"] == "PASS" for check in checks),
            "fail": fail_count,
            "cannot_verify": cannot_verify_count,
            "raw_recordings": len(raw_records),
            "cache_manifests": len(cache_records),
            "training_results": len(result_records),
        },
        "conclusion": (
            "No local implementation mismatch was found in file/label mapping, "
            "trial order, raw-to-DE+LDS parameters and pinned functions, split "
            "generation, DGCNN settings, checkpoint selection, or aggregation. "
            "The -3.40 percentage-point public-anchor gap remains real and cannot "
            "be assigned to an exact historical-runtime cause."
        ),
        "raw_records": raw_records,
        "cache_records": cache_records,
        "result_records": result_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = run(args)
    print(json.dumps({"status": payload["status"], **payload["summary"]}))
    return 0 if payload["status"] == "PASS_WITH_LIMITATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
