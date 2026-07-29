"""Production-independent reconstruction of the one-shot CF-TRE G1 result."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROTOCOL = "cf_tre_g1_one_shot_outer_test_v1"
COMPONENTS = (
    "linear_de310", "linear_structural245", "linear_all555", "rbf_de310",
    "rbf_all555", "xgboost_all555", "lightgbm_all555",
)
METHODS = ("strongest_single", "uniform", "mean_risk", "tail_risk", "dgcnn", "gcbnet")
TOL = 1e-10


def _mix(base: np.ndarray, weights: list[float]) -> np.ndarray:
    mixed = np.einsum("nmc,m->nc", base, np.asarray(weights, dtype=np.float64))
    mixed = np.clip(mixed, 1e-6, 1.0)
    return mixed / mixed.sum(axis=1, keepdims=True)


def _cvar(values: list[float], alpha: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(min(eta + np.maximum(array - eta, 0.0).mean() / (1.0 - alpha) for eta in np.unique(array)))


def _bca(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    observed = float(array.mean())
    if np.all(array == array[0]):
        return {"observed": observed, "low": observed, "high": observed}
    rng = np.random.default_rng(240714)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    bootstrap = array[indices].mean(axis=1)
    proportion = (np.sum(bootstrap < observed) + 0.5 * np.sum(bootstrap == observed)) / 10_000
    proportion = float(np.clip(proportion, 0.5 / 10_000, 1.0 - 0.5 / 10_000))
    z0 = float(norm.ppf(proportion))
    jackknife = np.asarray([np.delete(array, index).mean() for index in range(len(array))])
    delta = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(delta**2) ** 1.5)
    acceleration = 0.0 if denominator == 0.0 else float(np.sum(delta**3) / denominator)
    adjusted = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        divisor = 1.0 - acceleration * (z0 + z_alpha)
        if divisor == 0.0:
            raise ValueError("BCa divisor is zero")
        quantile = float(norm.cdf(z0 + (z0 + z_alpha) / divisor))
        if not math.isfinite(quantile):
            raise ValueError("BCa quantile is non-finite")
        adjusted.append(quantile)
    low, high = np.quantile(bootstrap, adjusted)
    return {"observed": observed, "low": float(low), "high": float(high)}


def _unit_metrics(labels: np.ndarray, trials: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    trial_correct = []
    for trial in np.unique(trials):
        keep = trials == trial
        unique_labels = np.unique(labels[keep])
        if len(unique_labels) != 1:
            raise ValueError("trial label mismatch")
        trial_correct.append(float(probabilities[keep].mean(axis=0).argmax() == unique_labels[0]))
    true_probability = np.clip(probabilities[np.arange(len(labels)), labels], 1e-6, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "trial_accuracy": float(np.mean(trial_correct)),
        "log_loss": float(np.mean(-np.log(true_probability))),
    }


def _aggregate(units: list[dict[str, Any]], method: str, alpha: float) -> dict[str, Any]:
    by_subject: dict[int, list[dict[str, float]]] = defaultdict(list)
    environments: list[dict[str, Any]] = []
    for unit in units:
        metrics = _unit_metrics(unit["labels"], unit["trials"], unit[method])
        by_subject[unit["subject"]].append(metrics)
        environments.append({"session": unit["session"], "subject": unit["subject"], **metrics})
    per_subject = {
        str(subject): {metric: float(np.mean([row[metric] for row in by_subject[subject]])) for metric in ("accuracy", "macro_f1", "balanced_accuracy", "trial_accuracy", "log_loss")}
        for subject in range(1, 16)
    }
    def mean(metric: str) -> float:
        return float(np.mean([per_subject[str(subject)][metric] for subject in range(1, 16)]))
    losses = [row["log_loss"] for row in environments]
    return {
        "subject_equal_accuracy": mean("accuracy"),
        "subject_equal_macro_f1": mean("macro_f1"),
        "subject_equal_balanced_accuracy": mean("balanced_accuracy"),
        "subject_equal_trial_accuracy": mean("trial_accuracy"),
        "mean_environment_log_loss": float(np.mean(losses)),
        "environment_cvar_log_loss": _cvar(losses, alpha),
        "per_subject": per_subject,
        "per_environment": environments,
    }


def _close(context: str, observed: Any, expected: Any, errors: list[str]) -> None:
    if isinstance(expected, (list, np.ndarray)):
        if not np.allclose(np.asarray(observed), np.asarray(expected), rtol=0.0, atol=TOL):
            errors.append(f"{context} mismatch")
    elif not np.isclose(float(observed), float(expected), rtol=0.0, atol=TOL):
        errors.append(f"{context} mismatch")


def _source_units(root: Path, dataset: str, result: dict[str, Any], archive: dict[str, np.ndarray], errors: list[str]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    cursor = 0
    tail_weights = result["datasets"][dataset]["selected_tail_configuration"]["weights"]
    mean_weights = result["datasets"][dataset]["selected_mean_configuration"]["weights"]
    fixed_weights = {
        "strongest_single": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "uniform": [1.0 / 7.0] * 7,
        "mean_risk": mean_weights,
        "tail_risk": tail_weights,
    }
    for session in range(1, 4):
        for subject in range(1, 16):
            stem = f"s{session:02d}_sub{subject:02d}.npz"
            with np.load(root / "results/cf_tre/g0b/formal/classical" / dataset / stem, allow_pickle=False) as arrays:
                labels = arrays["test_labels"].astype(np.int64)
                trials = arrays["test_trials"].astype(np.int64)
                base = np.stack([arrays[f"test_probabilities__{component}"] for component in COMPONENTS], axis=1)
            stop = cursor + len(labels)
            for key, expected in (("labels", labels), ("trials", trials), ("subject", np.full(len(labels), subject)), ("session", np.full(len(labels), session))):
                if not np.array_equal(archive[key][cursor:stop], expected):
                    errors.append(f"{dataset}.{stem}: prediction metadata {key} mismatch")
            probabilities: dict[str, np.ndarray] = {}
            for method, weights in fixed_weights.items():
                probabilities[method] = _mix(base, weights)
                if not np.allclose(archive[f"probabilities__{method}"][cursor:stop], probabilities[method], rtol=0.0, atol=TOL):
                    errors.append(f"{dataset}.{stem}: stored {method} probabilities mismatch")
            for method in ("dgcnn", "gcbnet"):
                with np.load(root / "results/cf_tre/g0b/formal/deep" / method / dataset / stem, allow_pickle=False) as arrays:
                    probabilities[method] = arrays["test_probabilities"].astype(np.float64)
                if not np.allclose(archive[f"probabilities__{method}"][cursor:stop], probabilities[method], rtol=0.0, atol=TOL):
                    errors.append(f"{dataset}.{stem}: stored {method} probabilities mismatch")
            units.append({"session": session, "subject": subject, "labels": labels, "trials": trials, **probabilities})
            cursor = stop
    if cursor != len(archive["labels"]):
        errors.append(f"{dataset}: stored prediction row count mismatch")
    return units


def audit(root: Path) -> dict[str, Any]:
    result = json.loads((root / "results/cf_tre/g1/g1_outer_test_result.json").read_text(encoding="utf-8"))
    preflight = json.loads((root / "results/cf_tre/g1/g1_preflight_audit.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = 0
    if result.get("protocol") != PROTOCOL or preflight.get("status") != "PASS_RELEASE_ONE_SHOT":
        errors.append("protocol or preflight status mismatch")
    checks += 2
    recomputed: dict[str, Any] = {}
    for dataset in ("seed", "seediv"):
        with np.load(root / f"results/cf_tre/g1/{dataset}_outer_predictions.npz", allow_pickle=False) as arrays:
            expected_keys = {"labels", "trials", "subject", "session"} | {f"probabilities__{method}" for method in METHODS}
            if set(arrays.files) != expected_keys:
                errors.append(f"{dataset}: prediction archive key mismatch")
            archive = {key: arrays[key].copy() for key in arrays.files}
        units = _source_units(root, dataset, result, archive, errors)
        alpha = float(result["datasets"][dataset]["selected_tail_configuration"]["alpha"])
        methods = {method: _aggregate(units, method, alpha) for method in METHODS}
        for method in METHODS:
            observed = result["datasets"][dataset]["methods"][method]
            expected = methods[method]
            for key in ("subject_equal_accuracy", "subject_equal_macro_f1", "subject_equal_balanced_accuracy", "subject_equal_trial_accuracy", "mean_environment_log_loss", "environment_cvar_log_loss"):
                checks += 1
                _close(f"{dataset}.{method}.{key}", observed[key], expected[key], errors)
            for subject in range(1, 16):
                checks += 1
                _close(f"{dataset}.{method}.subject{subject}.accuracy", observed["per_subject"][str(subject)]["accuracy"], expected["per_subject"][str(subject)]["accuracy"], errors)
        paired = [methods["tail_risk"]["per_subject"][str(subject)]["accuracy"] - methods["dgcnn"]["per_subject"][str(subject)]["accuracy"] for subject in range(1, 16)]
        interval = _bca(paired)
        for key in ("observed", "low", "high"):
            checks += 1
            _close(f"{dataset}.bca.{key}", result["datasets"][dataset]["paired_tail_minus_dgcnn_accuracy_bca"][key], interval[key], errors)
        recomputed[dataset] = {"methods": methods, "paired_interval": interval}
    accuracy = {dataset: rows["methods"]["tail_risk"]["subject_equal_accuracy"] for dataset, rows in recomputed.items()}
    gain = {dataset: accuracy[dataset] - rows["methods"]["dgcnn"]["subject_equal_accuracy"] for dataset, rows in recomputed.items()}
    macro_gain = {dataset: rows["methods"]["tail_risk"]["subject_equal_macro_f1"] - rows["methods"]["dgcnn"]["subject_equal_macro_f1"] for dataset, rows in recomputed.items()}
    cvar_gain = {dataset: rows["methods"]["mean_risk"]["environment_cvar_log_loss"] - rows["methods"]["tail_risk"]["environment_cvar_log_loss"] for dataset, rows in recomputed.items()}
    ci_low = {dataset: rows["paired_interval"]["low"] for dataset, rows in recomputed.items()}
    gates = {
        "integrity": not errors,
        "absolute_accuracy": accuracy["seed"] >= 0.84 and accuracy["seediv"] >= 0.55,
        "point_gain_vs_dgcnn": max(gain.values()) >= 0.02 and min(gain.values()) >= 0.0,
        "paired_ci": max(ci_low.values()) > 0.0,
        "macro_f1": min(macro_gain.values()) >= -0.005,
        "cvar_mechanism": max(cvar_gain.values()) > 0.005 and min(cvar_gain.values()) >= -0.005,
    }
    for gate, expected in gates.items():
        checks += 1
        if bool(result["gates"][gate]) != expected:
            errors.append(f"gate mismatch: {gate}")
    expected_status = "PASS_METHOD_ROUTE_REVIEW" if all(gates.values()) else "FAIL_STOP_METHOD_ROUTE"
    checks += 1
    if result.get("status") != expected_status:
        errors.append("final status mismatch")
    return {"protocol": PROTOCOL, "audit": "independent source-to-prediction, metric, BCa, contrast, and gate reconstruction", "checks": checks, "errors": errors, "recomputed_gates": gates, "recomputed_status": expected_status, "status": "PASS" if not errors else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g1/g1_outer_test_independent_audit.json"))
    args = parser.parse_args()
    result = audit(args.root.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
