"""Prebuild and audit the pinned LibEER SEED-IV DE+LDS cache for G0-B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from pcma.data.libeer_seediv import load_seediv_de_lds_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/SEED/SEED_IV"))
    parser.add_argument(
        "--libeer-root",
        type=Path,
        default=Path(
            os.environ.get("LIBEER_CODE_ROOT", "third_party/libeer/LibEER")
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("results/libeer_gate_b/cache_seediv_1s_de_lds_39dc27e"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cf_tre/g0b/seediv_cache_prebuild.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.time()
    rows: list[dict[str, object]] = []
    for session in range(1, 4):
        for subject in range(1, 16):
            unit_started = time.time()
            loaded = load_seediv_de_lds_subject(
                args.dataset_root,
                args.libeer_root,
                session,
                subject,
                cache_root=args.cache_root,
                expected_commit="39dc27e504e14138767b87ce8bce485380fd4f5a",
            )
            if loaded.cache_file is None or not loaded.cache_file.is_file():
                raise RuntimeError(f"cache not materialized for session={session}, subject={subject}")
            if loaded.cache_manifest_file is None or not loaded.cache_manifest_file.is_file():
                raise RuntimeError(f"cache manifest missing for session={session}, subject={subject}")
            row = {
                "unit_id": f"seediv:s{session:02d}:sub{subject:02d}",
                "loaded_from_cache": bool(loaded.loaded_from_cache),
                "cache_file": str(loaded.cache_file.resolve()),
                "cache_manifest": str(loaded.cache_manifest_file.resolve()),
                "trial_count": len(loaded.trials),
                "window_count": sum(len(trial) for trial in loaded.trials),
                "seconds": time.time() - unit_started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "protocol": "cf_tre_seed_family_v1",
        "stage": "G0-B",
        "status": "PASS",
        "dataset": "seediv",
        "pipeline": "raw EEG -> pinned LibEER bandpass -> 1s DE -> LDS",
        "expected_libeer_commit": "39dc27e504e14138767b87ce8bce485380fd4f5a",
        "unit_count": len(rows),
        "newly_built_count": sum(not bool(row["loaded_from_cache"]) for row in rows),
        "cache_hit_count": sum(bool(row["loaded_from_cache"]) for row in rows),
        "elapsed_seconds": time.time() - started,
        "units": rows,
    }
    if len(rows) != 45 or any(row["trial_count"] != 24 for row in rows):
        raise RuntimeError("incomplete SEED-IV cache matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
