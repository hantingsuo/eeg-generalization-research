"""Run the seven frozen CF-TRE classical components for one recording.

All hyperparameters are selected on outer validation.  Test features are not
transformed or predicted until every component has a fixed selected estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Sequence

import lightgbm
import numpy as np
import sklearn
import xgboost

from pcma.data.libeer_seed import load_seed_de_lds_subject
from pcma.data.libeer_seediv import load_seediv_de_lds_subject
from pcma.model.cf_tre_baselines import (
    COMPONENT_SPECS,
    build_estimator,
    component_matrix,
    feature_rows_for_recording,
    load_track_a_unit,
    normalize_probabilities,
    probability_metrics,
    select_validation_candidate,
    trial_mean_probability_metrics,
)


DEFAULT_SEED_ROOT = Path("data/SEED/SEED/SEED/SEED_EEG/ExtractedFeatures_1s")
DEFAULT_SEEDIV_ROOT = Path("data/SEED/SEED_IV")
DEFAULT_LIBEER_ROOT = Path(os.environ.get("LIBEER_ROOT", "external/LibEER"))
DEFAULT_SEEDIV_CACHE = Path("results/libeer_gate_b/cache_seediv_1s_de_lds_39dc27e")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", choices=("seed", "seediv"), required=True)
    result.add_argument("--session", type=int, required=True)
    result.add_argument("--subject", type=int, required=True)
    result.add_argument("--split-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--predictions", type=Path)
    result.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    result.add_argument("--seediv-root", type=Path, default=DEFAULT_SEEDIV_ROOT)
    result.add_argument("--libeer-root", type=Path, default=DEFAULT_LIBEER_ROOT)
    result.add_argument("--seediv-cache", type=Path, default=DEFAULT_SEEDIV_CACHE)
    result.add_argument("--optimization-seed", type=int, default=2024)
    result.add_argument("--validation-only", action="store_true")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_trials(recording: Any, trial_numbers: Sequence[int]) -> tuple[list[np.ndarray], np.ndarray]:
    numbers = [int(value) for value in trial_numbers]
    if any(value < 1 or value > len(recording.trials) for value in numbers):
        raise ValueError("split manifest contains an out-of-range trial")
    return (
        [recording.trials[value - 1] for value in numbers],
        np.asarray([recording.trial_labels[value - 1] for value in numbers], dtype=np.int64),
    )


def _load_recording(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.dataset == "seed":
        loaded = load_seed_de_lds_subject(
            args.seed_root,
            session=args.session,
            subject=args.subject,
        )
        provenance = {
            "source_file": str(loaded.source_file.resolve()),
            "source_sha256": _sha256(loaded.source_file),
            "label_file": str(loaded.label_file.resolve()),
            "label_sha256": _sha256(loaded.label_file),
            "representation": "official_de_LDS_1s",
        }
        return loaded, provenance
    loaded = load_seediv_de_lds_subject(
        args.seediv_root,
        args.libeer_root,
        args.session,
        args.subject,
        cache_root=args.seediv_cache,
        expected_commit="39dc27e504e14138767b87ce8bce485380fd4f5a",
    )
    provenance = dict(loaded.provenance)
    provenance.update(
        {
            "source_file": str(loaded.source_file.resolve()),
            "cache_file": str(loaded.cache_file.resolve()) if loaded.cache_file else None,
            "loaded_from_cache": bool(loaded.loaded_from_cache),
            "representation": "pinned_libeer_raw_to_de_lds_1s",
        }
    )
    return loaded, provenance


def _partition_features(recording: Any, trial_numbers: Sequence[int]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    trials, labels = _selected_trials(recording, trial_numbers)
    return feature_rows_for_recording(trials, labels, trial_numbers=trial_numbers)


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not 1 <= args.session <= 3 or not 1 <= args.subject <= 15:
        raise ValueError("session must be 1..3 and subject must be 1..15")
    output = args.output.resolve()
    predictions = (args.predictions or args.output.with_suffix(".npz")).resolve()
    if output == predictions:
        raise ValueError("JSON and prediction paths must differ")
    collisions = [path for path in (output, predictions) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite artifacts: {collisions}")

    started = time.time()
    split_path = args.split_manifest.resolve()
    unit = load_track_a_unit(
        split_path,
        dataset=args.dataset,
        session=args.session,
        subject=args.subject,
    )
    recording, provenance = _load_recording(args)

    train_views, train_y, train_trials = _partition_features(recording, unit["train_trials"])
    validation_views, validation_y, validation_trials = _partition_features(
        recording, unit["validation_trials"]
    )
    selected_models: dict[str, Any] = {}
    component_rows: dict[str, Any] = {}
    prediction_payload: dict[str, np.ndarray] = {
        "validation_labels": validation_y,
        "validation_trials": validation_trials,
    }

    for component_index, (component, spec) in enumerate(COMPONENT_SPECS.items()):
        train_x = component_matrix(train_views, component)
        validation_x = component_matrix(validation_views, component)
        candidates: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(spec.candidates):
            candidate_started = time.time()
            estimator = build_estimator(
                component,
                candidate,
                train_labels=train_y,
                train_trials=train_trials,
                seed=args.optimization_seed,
            )
            estimator.fit(train_x, train_y)
            probabilities = normalize_probabilities(estimator.predict_proba(validation_x))
            metrics = probability_metrics(validation_y, probabilities)
            row = {
                "candidate_index": candidate_index,
                "parameters": dict(candidate),
                **metrics,
                "seconds": time.time() - candidate_started,
            }
            candidates.append(row)
            print(json.dumps({"component": component, "stage": "candidate", **row}), flush=True)

        selected = select_validation_candidate(candidates)
        selected_index = int(selected["candidate_index"])
        final_estimator = build_estimator(
            component,
            spec.candidates[selected_index],
            train_labels=train_y,
            train_trials=train_trials,
            seed=args.optimization_seed,
        )
        final_estimator.fit(train_x, train_y)
        final_validation_probabilities = normalize_probabilities(final_estimator.predict_proba(validation_x))
        final_validation = probability_metrics(validation_y, final_validation_probabilities)
        selected_models[component] = final_estimator
        prediction_payload[f"validation_probabilities__{component}"] = final_validation_probabilities
        component_rows[component] = {
            "component_index": component_index,
            "family": spec.family,
            "views": list(spec.views),
            "dimension": spec.dimension,
            "candidate_table": candidates,
            "selected_candidate_index": selected_index,
            "selected_parameters": dict(spec.candidates[selected_index]),
            "validation_window": final_validation,
            "validation_trial_mean_probability": trial_mean_probability_metrics(
                validation_y, final_validation_probabilities, validation_trials
            ),
        }
        print(
            json.dumps(
                {
                    "component": component,
                    "stage": "selected",
                    "candidate_index": selected_index,
                    "validation": final_validation,
                }
            ),
            flush=True,
        )

    test_contacted = not args.validation_only
    if test_contacted:
        test_views, test_y, test_trials = _partition_features(recording, unit["test_trials"])
        prediction_payload["test_labels"] = test_y
        prediction_payload["test_trials"] = test_trials
        for component, estimator in selected_models.items():
            test_x = component_matrix(test_views, component)
            test_probabilities = normalize_probabilities(estimator.predict_proba(test_x))
            prediction_payload[f"test_probabilities__{component}"] = test_probabilities
            component_rows[component]["test_window"] = probability_metrics(test_y, test_probabilities)
            component_rows[component]["test_trial_mean_probability"] = trial_mean_probability_metrics(
                test_y, test_probabilities, test_trials
            )

    result = {
        "protocol": "cf_tre_seed_family_v1",
        "stage": "G0-B",
        "runner": "seven_classical_components_subject_session_v1",
        "dataset": args.dataset,
        "session": args.session,
        "subject": args.subject,
        "unit_id": unit["unit_id"],
        "optimization_seed": args.optimization_seed,
        "validation_only": bool(args.validation_only),
        "test_contacted": test_contacted,
        "split_manifest": str(split_path),
        "split_manifest_sha256": _sha256(split_path),
        "trial_split": {
            "train": unit["train_trials"],
            "validation": unit["validation_trials"],
            "test": unit["test_trials"],
        },
        "row_counts": {
            "train": int(len(train_y)),
            "validation": int(len(validation_y)),
            "test": int(len(prediction_payload.get("test_labels", []))),
        },
        "calibration": "CalibratedClassifierCV sigmoid with explicit class-stratified complete-trial folds for SVM; no SVC implicit probability calibration",
        "selection": "higher validation window accuracy, then macro-F1, then frozen candidate order",
        "fit_boundary": "selected models fit outer train only; validation is not refit; test transformed and predicted only after all selections are fixed",
        "components": component_rows,
        "provenance": provenance,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "predictions": str(predictions),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("xb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(output), "predictions": str(predictions)}), flush=True)


if __name__ == "__main__":
    main()
