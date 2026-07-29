"""Run the single frozen CF-TRE G1 outer-test evaluation without refitting."""

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
BOOTSTRAP_SEED = 240714
BOOTSTRAP_RESAMPLES = 10_000


def _mixture(probabilities: np.ndarray, weights: list[float]) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (len(COMPONENTS),) or weight_array.min() < -1e-8 or not np.isclose(weight_array.sum(), 1.0, atol=1e-8):
        raise ValueError("invalid frozen weight vector")
    mixed = np.einsum("nmc,m->nc", probabilities, weight_array)
    mixed = np.clip(mixed, 1e-6, 1.0)
    return mixed / mixed.sum(axis=1, keepdims=True)


def bca_mean_interval(values: list[float], *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("paired subject differences must be finite")
    observed = float(array.mean())
    if np.all(array == array[0]):
        return {"method": "BCa paired subject-row bootstrap", "confidence_level": 0.95, "n_subjects": len(array), "n_resamples": n_resamples, "seed": seed, "status": "DEGENERATE_CONSTANT", "observed": observed, "low": observed, "high": observed, "z0": 0.0, "acceleration": 0.0, "adjusted_quantiles": [0.025, 0.975]}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_resamples, len(array)))
    bootstrap = array[indices].mean(axis=1)
    less = int(np.sum(bootstrap < observed))
    equal = int(np.sum(bootstrap == observed))
    proportion = float(np.clip((less + 0.5 * equal) / n_resamples, 0.5 / n_resamples, 1.0 - 0.5 / n_resamples))
    z0 = float(norm.ppf(proportion))
    jackknife = np.asarray([np.delete(array, index).mean() for index in range(len(array))])
    delta = jackknife.mean() - jackknife
    denominator = 6.0 * float(np.sum(delta**2) ** 1.5)
    acceleration = 0.0 if denominator == 0.0 else float(np.sum(delta**3) / denominator)
    adjusted: list[float] = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        divisor = 1.0 - acceleration * (z0 + z_alpha)
        if divisor == 0.0:
            raise ValueError("BCa adjusted quantile is undefined")
        quantile = float(norm.cdf(z0 + (z0 + z_alpha) / divisor))
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("BCa adjusted quantile outside [0,1]")
        adjusted.append(quantile)
    low, high = np.quantile(bootstrap, adjusted)
    return {"method": "BCa paired subject-row bootstrap", "confidence_level": 0.95, "n_subjects": len(array), "n_resamples": n_resamples, "seed": seed, "status": "OK", "observed": observed, "low": float(low), "high": float(high), "z0": z0, "acceleration": acceleration, "adjusted_quantiles": adjusted}


def _unit_metrics(labels: np.ndarray, trials: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    trial_correct: list[float] = []
    for trial in np.unique(trials):
        keep = trials == trial
        trial_labels = np.unique(labels[keep])
        if len(trial_labels) != 1:
            raise ValueError("one trial contains multiple labels")
        trial_correct.append(float(probabilities[keep].mean(axis=0).argmax() == trial_labels[0]))
    true_probability = np.clip(probabilities[np.arange(len(labels)), labels], 1e-6, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "trial_accuracy": float(np.mean(trial_correct)),
        "log_loss": float(np.mean(-np.log(true_probability))),
    }


def _cvar(values: list[float], alpha: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(min(eta + np.maximum(array - eta, 0.0).mean() / (1.0 - alpha) for eta in np.unique(array)))


def _aggregate(units: list[dict[str, Any]], method: str, alpha: float) -> dict[str, Any]:
    environment_rows: list[dict[str, Any]] = []
    by_subject: dict[int, list[dict[str, float]]] = defaultdict(list)
    for unit in units:
        metrics = _unit_metrics(unit["labels"], unit["trials"], unit["probabilities"][method])
        by_subject[unit["subject"]].append(metrics)
        environment_rows.append({"session": unit["session"], "subject": unit["subject"], **metrics})
    if set(by_subject) != set(range(1, 16)) or any(len(rows) != 3 for rows in by_subject.values()):
        raise ValueError("subject/session coverage mismatch")
    per_subject = {
        str(subject): {
            metric: float(np.mean([row[metric] for row in by_subject[subject]]))
            for metric in ("accuracy", "macro_f1", "balanced_accuracy", "trial_accuracy", "log_loss")
        }
        for subject in range(1, 16)
    }
    def subject_equal(metric: str) -> float:
        return float(np.mean([per_subject[str(subject)][metric] for subject in range(1, 16)]))
    losses = [float(row["log_loss"]) for row in environment_rows]
    return {
        "subject_equal_accuracy": subject_equal("accuracy"),
        "subject_equal_macro_f1": subject_equal("macro_f1"),
        "subject_equal_balanced_accuracy": subject_equal("balanced_accuracy"),
        "subject_equal_trial_accuracy": subject_equal("trial_accuracy"),
        "mean_environment_log_loss": float(np.mean(losses)),
        "environment_cvar_log_loss": _cvar(losses, alpha),
        "cvar_alpha": alpha,
        "per_subject": per_subject,
        "per_environment": environment_rows,
    }


def evaluate_gates(datasets: dict[str, Any], *, integrity_pass: bool) -> tuple[dict[str, bool], dict[str, Any]]:
    accuracy = {dataset: rows["methods"]["tail_risk"]["subject_equal_accuracy"] for dataset, rows in datasets.items()}
    gain = {dataset: accuracy[dataset] - rows["methods"]["dgcnn"]["subject_equal_accuracy"] for dataset, rows in datasets.items()}
    macro_gain = {dataset: rows["methods"]["tail_risk"]["subject_equal_macro_f1"] - rows["methods"]["dgcnn"]["subject_equal_macro_f1"] for dataset, rows in datasets.items()}
    cvar_improvement = {dataset: rows["methods"]["mean_risk"]["environment_cvar_log_loss"] - rows["methods"]["tail_risk"]["environment_cvar_log_loss"] for dataset, rows in datasets.items()}
    ci_low = {dataset: rows["paired_tail_minus_dgcnn_accuracy_bca"]["low"] for dataset, rows in datasets.items()}
    gates = {
        "integrity": bool(integrity_pass),
        "absolute_accuracy": accuracy["seed"] >= 0.84 and accuracy["seediv"] >= 0.55,
        "point_gain_vs_dgcnn": max(gain.values()) >= 0.02 and min(gain.values()) >= 0.0,
        "paired_ci": max(ci_low.values()) > 0.0,
        "macro_f1": min(macro_gain.values()) >= -0.005,
        "cvar_mechanism": max(cvar_improvement.values()) > 0.005 and min(cvar_improvement.values()) >= -0.005,
    }
    contrasts = {"tail_minus_dgcnn_accuracy": gain, "tail_minus_dgcnn_macro_f1": macro_gain, "mean_minus_tail_cvar_improvement": cvar_improvement, "paired_accuracy_bca_low": ci_low}
    return gates, contrasts


def _load_dataset(root: Path, dataset: str, g0c: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    selected = g0c["datasets"][dataset]
    weights = {
        "strongest_single": selected["strongest_single"]["weights"],
        "uniform": selected["uniform"]["weights"],
        "mean_risk": selected["selected_mean"]["weights"],
        "tail_risk": selected["selected_tail"]["weights"],
    }
    units: list[dict[str, Any]] = []
    prediction_chunks: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    metadata: dict[str, list[np.ndarray]] = {key: [] for key in ("labels", "trials", "subject", "session")}
    for session in range(1, 4):
        for subject in range(1, 16):
            stem = f"s{session:02d}_sub{subject:02d}.npz"
            classical_path = root / "results/cf_tre/g0b/formal/classical" / dataset / stem
            with np.load(classical_path, allow_pickle=False) as arrays:
                labels = arrays["test_labels"].astype(np.int64)
                trials = arrays["test_trials"].astype(np.int64)
                base = np.stack([arrays[f"test_probabilities__{component}"] for component in COMPONENTS], axis=1)
            probabilities = {method: _mixture(base, weights[method]) for method in weights}
            for deep in ("dgcnn", "gcbnet"):
                path = root / "results/cf_tre/g0b/formal/deep" / deep / dataset / stem
                with np.load(path, allow_pickle=False) as arrays:
                    if not np.array_equal(arrays["test_labels"], labels) or not np.array_equal(arrays["test_trials"], trials):
                        raise ValueError(f"deep/classical test alignment mismatch: {path}")
                    probabilities[deep] = arrays["test_probabilities"].astype(np.float64)
            units.append({"session": session, "subject": subject, "labels": labels, "trials": trials, "probabilities": probabilities})
            metadata["labels"].append(labels)
            metadata["trials"].append(trials)
            metadata["subject"].append(np.full(len(labels), subject, dtype=np.int16))
            metadata["session"].append(np.full(len(labels), session, dtype=np.int8))
            for method in METHODS:
                prediction_chunks[method].append(probabilities[method])
    archive = {key: np.concatenate(chunks) for key, chunks in metadata.items()}
    archive.update({f"probabilities__{method}": np.concatenate(chunks) for method, chunks in prediction_chunks.items()})
    return units, archive


def run(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    preflight_path = root / "results/cf_tre/g1/g1_preflight_audit.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS_RELEASE_ONE_SHOT" or not preflight.get("outer_test_outputs_absent_before_release"):
        raise RuntimeError("G1 preflight did not release the one-shot test")
    g0c = json.loads((root / "results/cf_tre/g0c/g0c_feasibility_result.json").read_text(encoding="utf-8"))
    datasets: dict[str, Any] = {}
    archives: dict[str, dict[str, np.ndarray]] = {}
    for dataset in ("seed", "seediv"):
        units, archive = _load_dataset(root, dataset, g0c)
        alpha = float(g0c["datasets"][dataset]["selected_tail"]["alpha"])
        methods = {method: _aggregate(units, method, alpha) for method in METHODS}
        paired = [methods["tail_risk"]["per_subject"][str(subject)]["accuracy"] - methods["dgcnn"]["per_subject"][str(subject)]["accuracy"] for subject in range(1, 16)]
        datasets[dataset] = {
            "selected_tail_configuration": {key: g0c["datasets"][dataset]["selected_tail"][key] for key in ("lambda_tail", "alpha", "delta", "weights")},
            "selected_mean_configuration": {key: g0c["datasets"][dataset]["selected_mean"][key] for key in ("lambda_tail", "alpha", "delta", "weights")},
            "binding_strongest_local_baseline": "dgcnn",
            "methods": methods,
            "paired_tail_minus_dgcnn_accuracy_by_subject": {str(subject): paired[subject - 1] for subject in range(1, 16)},
            "paired_tail_minus_dgcnn_accuracy_bca": bca_mean_interval(paired),
        }
        archives[dataset] = archive
    gates, contrasts = evaluate_gates(datasets, integrity_pass=True)
    status = "PASS_METHOD_ROUTE_REVIEW" if all(gates.values()) else "FAIL_STOP_METHOD_ROUTE"
    result = {"protocol": PROTOCOL, "stage": "G1", "status": status, "access_boundary": "one-shot scoring of frozen G0-C weights on pre-existing G0-B outer-test probabilities; no refit or reselection", "datasets": datasets, "contrasts": contrasts, "gates": gates}
    return result, archives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("results/cf_tre/g1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result, archives = run(root)
    for dataset, archive in archives.items():
        path = output_root / f"{dataset}_outer_predictions.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **archive)
    result_path = output_root / "g1_outer_test_result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "contrasts": result["contrasts"], "gates": result["gates"]}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS_METHOD_ROUTE_REVIEW" else 2)


if __name__ == "__main__":
    main()
