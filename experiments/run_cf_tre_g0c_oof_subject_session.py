"""Generate seven-component outer-train OOF probabilities for one G0-C unit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from pcma.data.libeer_seed import load_seed_de_lds_subject
from pcma.data.libeer_seediv import load_seediv_de_lds_subject
from pcma.model.cf_tre_baselines import (
    COMPONENT_SPECS,
    build_estimator,
    calibration_cv_by_trial,
    component_matrix,
    feature_rows_for_recording,
    load_track_a_unit,
    normalize_probabilities,
    probability_metrics,
)


PROTOCOL = "cf_tre_g0c_development_feasibility_v1"


def _load_recording(args: argparse.Namespace) -> Any:
    if args.dataset == "seed":
        return load_seed_de_lds_subject(
            args.seed_root,
            session=args.session,
            subject=args.subject,
        )
    return load_seediv_de_lds_subject(
        args.seediv_root,
        args.libeer_root,
        args.session,
        args.subject,
        cache_root=args.seediv_cache,
        expected_commit="39dc27e504e14138767b87ce8bce485380fd4f5a",
    )


def _partition_features(
    recording: Any, trial_numbers: Sequence[int]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    numbers = [int(value) for value in trial_numbers]
    trials = [recording.trials[value - 1] for value in numbers]
    labels = np.asarray(
        [recording.trial_labels[value - 1] for value in numbers], dtype=np.int64
    )
    return feature_rows_for_recording(trials, labels, trial_numbers=numbers)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", choices=("seed", "seediv"), required=True)
    result.add_argument("--session", type=int, required=True)
    result.add_argument("--subject", type=int, required=True)
    result.add_argument("--split-manifest", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--optimization-seed", type=int, default=2024)
    result.add_argument("--seed-root", type=Path, default=Path("data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s"))
    result.add_argument("--seediv-root", type=Path, default=Path("data/SEED/SEED_IV"))
    result.add_argument(
        "--libeer-root",
        type=Path,
        default=Path(
            os.environ.get("LIBEER_CODE_ROOT", "third_party/libeer/LibEER")
        ),
    )
    result.add_argument("--seediv-cache", type=Path, default=Path("results/libeer_gate_b/cache_seediv_1s_de_lds_39dc27e"))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not 1 <= args.session <= 3 or not 1 <= args.subject <= 15:
        raise ValueError("session/subject out of range")
    for path in (args.output, args.predictions):
        if path.exists():
            raise FileExistsError(path)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("protocol") != PROTOCOL:
        raise ValueError("selection protocol mismatch")
    if any(str(key).lower().startswith("test") for key in selection):
        raise ValueError("selection contains a test key")
    expected = {"dataset": args.dataset, "session": args.session, "subject": args.subject, "optimization_seed": args.optimization_seed}
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"selection {key} mismatch")

    unit = load_track_a_unit(args.split_manifest, dataset=args.dataset, session=args.session, subject=args.subject)
    if selection["trial_split_train"] != unit["train_trials"] or selection["trial_split_validation"] != unit["validation_trials"]:
        raise ValueError("selection split mismatch")
    started = time.time()
    recording = _load_recording(args)
    train_views, labels, trials = _partition_features(recording, unit["train_trials"])
    folds = calibration_cv_by_trial(labels, trials, seed=args.optimization_seed, n_splits=3)
    inner_fold = np.zeros(len(labels), dtype=np.int8)
    arrays: dict[str, np.ndarray] = {"train_labels": labels, "train_trials": trials}
    ledger: list[dict[str, Any]] = []
    component_metrics: dict[str, Any] = {}
    for fold_index, (fit, score) in enumerate(folds, start=1):
        inner_fold[score] = fold_index
        ledger.append(
            {
                "inner_fold": fold_index,
                "fit_trials": sorted(np.unique(trials[fit]).astype(int).tolist()),
                "score_trials": sorted(np.unique(trials[score]).astype(int).tolist()),
            }
        )
    if not np.all(inner_fold > 0):
        raise AssertionError("OOF fold coverage incomplete")
    arrays["inner_fold"] = inner_fold

    for component in COMPONENT_SPECS:
        component_x = component_matrix(train_views, component)
        probabilities = np.full((len(labels), len(np.unique(labels))), np.nan, dtype=np.float64)
        parameters = selection["components"][component]["selected_parameters"]
        for fit, score in folds:
            estimator = build_estimator(
                component,
                parameters,
                train_labels=labels[fit],
                train_trials=trials[fit],
                seed=args.optimization_seed,
                calibration_splits=2,
            )
            estimator.fit(component_x[fit], labels[fit])
            probabilities[score] = normalize_probabilities(estimator.predict_proba(component_x[score]))
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"{component} incomplete OOF probabilities")
        arrays[f"oof_probabilities__{component}"] = probabilities
        component_metrics[component] = probability_metrics(labels, probabilities)
        print(json.dumps({"component": component, "oof": component_metrics[component]}), flush=True)

    result = {
        "protocol": PROTOCOL,
        "stage": "G0-C-OOF",
        "dataset": args.dataset,
        "session": args.session,
        "subject": args.subject,
        "unit_id": unit["unit_id"],
        "optimization_seed": args.optimization_seed,
        "outer_partition_loaded": "train_only",
        "contains_validation_predictions": False,
        "contains_test_predictions": False,
        "inner_folds": ledger,
        "components": component_metrics,
        "selection_path": str(args.selection.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "provenance": {
            "source_file": str(recording.source_file.resolve()),
            "representation": "official_de_LDS_1s" if args.dataset == "seed" else "pinned_libeer_raw_to_de_lds_1s",
        },
        "row_count": int(len(labels)),
        "elapsed_seconds": time.time() - started,
        "predictions": str(args.predictions.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
