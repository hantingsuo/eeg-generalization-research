"""Run frozen post-review LR and RF controls for one subject-session unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_cf_tre_classical_subject_session as classical  # noqa: E402
from pcma.model.cf_tre_baselines import (  # noqa: E402
    component_matrix,
    load_track_a_unit,
    normalize_probabilities,
    probability_metrics,
    trial_mean_probability_metrics,
)


PROTOCOL = "cf_tre_postreview_lr_rf_v1"
METHODS = ("logistic_regression_all555", "random_forest_all555")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("seed", "seediv"), required=True)
    result.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    result.add_argument("--subject", type=int, choices=range(1, 16), required=True)
    result.add_argument("--split-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--predictions", type=Path)
    result.add_argument("--seed-root", type=Path, default=classical.DEFAULT_SEED_ROOT)
    result.add_argument("--seediv-root", type=Path, default=classical.DEFAULT_SEEDIV_ROOT)
    result.add_argument("--libeer-root", type=Path, default=classical.DEFAULT_LIBEER_ROOT)
    result.add_argument("--seediv-cache", type=Path, default=classical.DEFAULT_SEEDIV_CACHE)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_models() -> dict[str, Any]:
    return {
        "logistic_regression_all555": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        tol=1e-4,
                        random_state=2024,
                    ),
                ),
            ]
        ),
        "random_forest_all555": RandomForestClassifier(
            n_estimators=300,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            n_jobs=1,
            random_state=2024,
        ),
    }


def model_configurations() -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression_all555": {
            "feature_dimension": 555,
            "scaler": "StandardScaler",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 1e-4,
            "random_state": 2024,
        },
        "random_forest_all555": {
            "feature_dimension": 555,
            "n_estimators": 300,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "bootstrap": True,
            "n_jobs": 1,
            "random_state": 2024,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    predictions = (args.predictions or output.with_suffix(".npz")).resolve()
    if output == predictions or output.exists() or predictions.exists():
        raise FileExistsError("refusing to overwrite or alias LR/RF artifacts")
    started = time.time()
    unit = load_track_a_unit(
        args.split_manifest.resolve(),
        dataset=args.dataset,
        session=args.session,
        subject=args.subject,
    )
    recording, provenance = classical._load_recording(args)
    train_views, train_y, train_trials = classical._partition_features(
        recording, unit["train_trials"]
    )
    validation_views, validation_y, validation_trials = classical._partition_features(
        recording, unit["validation_trials"]
    )
    train_x = component_matrix(train_views, "linear_all555")
    validation_x = component_matrix(validation_views, "linear_all555")
    models = build_models()
    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        "validation_labels": validation_y,
        "validation_trials": validation_trials,
    }
    for method, estimator in models.items():
        method_started = time.time()
        estimator.fit(train_x, train_y)
        classes = np.asarray(estimator.classes_, dtype=np.int64)
        expected_classes = np.arange(3 if args.dataset == "seed" else 4)
        if not np.array_equal(classes, expected_classes):
            raise ValueError(f"{method} learned unexpected class order {classes.tolist()}")
        probability = normalize_probabilities(estimator.predict_proba(validation_x))
        arrays[f"validation_probabilities__{method}"] = probability
        rows[method] = {
            "configuration": model_configurations()[method],
            "validation_window": probability_metrics(validation_y, probability),
            "validation_trial_mean_probability": trial_mean_probability_metrics(
                validation_y, probability, validation_trials
            ),
            "fit_seconds": time.time() - method_started,
        }

    # Outer test is contacted once, after both fixed estimators have been fit.
    test_views, test_y, test_trials = classical._partition_features(
        recording, unit["test_trials"]
    )
    test_x = component_matrix(test_views, "linear_all555")
    arrays["test_labels"] = test_y
    arrays["test_trials"] = test_trials
    for method, estimator in models.items():
        probability = normalize_probabilities(estimator.predict_proba(test_x))
        arrays[f"test_probabilities__{method}"] = probability
        rows[method]["test_window"] = probability_metrics(test_y, probability)
        rows[method]["test_trial_mean_probability"] = trial_mean_probability_metrics(
            test_y, probability, test_trials
        )

    result = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "stage": "post_review_descriptive_control",
        "post_review": True,
        "confirmatory_gate_member": False,
        "dataset": args.dataset,
        "session": args.session,
        "subject": args.subject,
        "unit_id": unit["unit_id"],
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256(args.split_manifest.resolve()),
        "trial_split": {
            "train": unit["train_trials"],
            "validation": unit["validation_trials"],
            "test": unit["test_trials"],
        },
        "fit_boundary": (
            "outer train only; validation is descriptive; fixed estimators contact outer test once"
        ),
        "test_contacted": True,
        "test_evaluation_count": 1,
        "feature_space": "engineered_all555",
        "row_counts": {
            "train": int(len(train_y)),
            "validation": int(len(validation_y)),
            "test": int(len(test_y)),
        },
        "methods": rows,
        "provenance": provenance,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "predictions": str(predictions),
        "elapsed_seconds": time.time() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(output)}), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(parser().parse_args(argv))


if __name__ == "__main__":
    main()
