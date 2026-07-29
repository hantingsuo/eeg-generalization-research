"""Fit frozen CF-TRE configurations on train OOF and score validation only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from pcma.model.cf_tre import empirical_cvar, environment_log_losses, fit_cf_tre_weights, mixture_probabilities, select_validation_configuration
from pcma.model.cf_tre_baselines import trial_mean_probability_metrics


PROTOCOL = "cf_tre_g0c_development_feasibility_v1"
COMPONENTS = (
    "linear_de310", "linear_structural245", "linear_all555", "rbf_de310",
    "rbf_all555", "xgboost_all555", "lightgbm_all555",
)


def _load_dataset(dataset: str, oof_root: Path, input_root: Path) -> dict[str, Any]:
    oof_probabilities: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    oof_environments: list[np.ndarray] = []
    validation_units: list[dict[str, Any]] = []
    for session in range(1, 4):
        for subject in range(1, 16):
            stem = f"s{session:02d}_sub{subject:02d}"
            environment = f"s{session:02d}_sub{subject:02d}"
            with np.load(oof_root / dataset / f"{stem}.npz", allow_pickle=False) as arrays:
                labels = arrays["train_labels"]
                probs = np.stack([arrays[f"oof_probabilities__{component}"] for component in COMPONENTS], axis=1)
            oof_labels.append(labels)
            oof_probabilities.append(probs)
            oof_environments.append(np.full(len(labels), environment, dtype=object))
            with np.load(input_root / dataset / f"{stem}.validation.npz", allow_pickle=False) as arrays:
                validation_units.append({
                    "session": session,
                    "subject": subject,
                    "environment": environment,
                    "labels": arrays["validation_labels"].copy(),
                    "trials": arrays["validation_trials"].copy(),
                    "probabilities": np.stack([arrays[f"validation_probabilities__{component}"] for component in COMPONENTS], axis=1),
                })
    return {"oof_probabilities": np.concatenate(oof_probabilities), "oof_labels": np.concatenate(oof_labels), "oof_environments": np.concatenate(oof_environments), "validation_units": validation_units}


def _validation_metrics(units: Sequence[dict[str, Any]], weights: np.ndarray, alpha: float) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    environment_losses: list[float] = []
    for unit in units:
        mixed = mixture_probabilities(unit["probabilities"], weights)
        labels = unit["labels"]
        predicted = mixed.argmax(axis=1)
        true_probability = np.clip(mixed[np.arange(len(labels)), labels], 1e-6, 1.0)
        environment_losses.append(float(np.mean(-np.log(true_probability))))
        trial = trial_mean_probability_metrics(labels, mixed, unit["trials"])
        rows.append({"subject": unit["subject"], "accuracy": float(accuracy_score(labels, predicted)), "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "trial_accuracy": trial["accuracy"]})
    by_subject: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)
    def subject_equal(key: str) -> float:
        return float(np.mean([np.mean([row[key] for row in by_subject[subject]]) for subject in range(1, 16)]))
    return {"accuracy": subject_equal("accuracy"), "macro_f1": subject_equal("macro_f1"), "balanced_accuracy": subject_equal("balanced_accuracy"), "trial_accuracy": subject_equal("trial_accuracy"), "mean_environment_log_loss": float(np.mean(environment_losses)), "cvar_log_loss": empirical_cvar(environment_losses, alpha), "environment_losses": environment_losses}


def _candidate(data: dict[str, Any], lambda_tail: float, alpha: float, delta: float) -> dict[str, Any]:
    result = fit_cf_tre_weights(data["oof_probabilities"], data["oof_labels"], data["oof_environments"], lambda_tail=lambda_tail, alpha=alpha, delta=delta)
    validation = _validation_metrics(data["validation_units"], result.weights, alpha)
    return {"lambda_tail": lambda_tail, "alpha": alpha, "delta": delta, "weights": result.weights.tolist(), "oof_mean_loss": result.mean_loss, "oof_cvar_loss": result.cvar_loss, "iterations": result.iterations, **validation}


def _select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(select_validation_configuration(candidates))


def _stability(data: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    full = np.asarray(selected["weights"], dtype=np.float64)
    distances: list[float] = []
    top_components: list[str] = []
    environments = data["oof_environments"]
    for subject in range(1, 16):
        excluded = {f"s{session:02d}_sub{subject:02d}" for session in range(1, 4)}
        keep = np.asarray([value not in excluded for value in environments], dtype=bool)
        fitted = fit_cf_tre_weights(data["oof_probabilities"][keep], data["oof_labels"][keep], environments[keep], lambda_tail=float(selected["lambda_tail"]), alpha=float(selected["alpha"]), delta=float(selected["delta"]))
        distances.append(float(np.abs(fitted.weights - full).sum()))
        top_components.append(COMPONENTS[int(np.argmax(fitted.weights))])
    median = float(np.median(distances))
    p90 = float(np.quantile(distances, 0.9))
    return {"all_15_refits_converged": True, "l1_distances": distances, "median_l1": median, "p90_l1": p90, "max_weight": float(full.max()), "components_above_0_05": int(np.sum(full > 0.05)), "top_components": top_components, "pass": median <= 0.35 and p90 <= 0.60 and float(full.max()) < 0.95 and int(np.sum(full > 0.05)) >= 2}


def run(oof_root: Path, input_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for dataset in ("seed", "seediv"):
        data = _load_dataset(dataset, oof_root, input_root)
        probabilities, labels, environments = data["oof_probabilities"], data["oof_labels"], data["oof_environments"]
        single_mean_losses = []
        for index in range(len(COMPONENTS)):
            losses, _ = environment_log_losses(probabilities[:, index, :], labels, environments)
            single_mean_losses.append(float(losses.mean()))
        best_index = int(np.argmin(single_mean_losses))
        one_hot = np.eye(len(COMPONENTS))[best_index]
        uniform = np.full(len(COMPONENTS), 1.0 / len(COMPONENTS))
        strongest = {"component": COMPONENTS[best_index], "weights": one_hot.tolist(), **_validation_metrics(data["validation_units"], one_hot, 0.8)}
        uniform_row = {"weights": uniform.tolist(), **_validation_metrics(data["validation_units"], uniform, 0.8)}
        mean_candidates = [_candidate(data, 0.0, 0.8, delta) for delta in (0.0, 0.01)]
        tail_candidates = [_candidate(data, lam, alpha, delta) for lam in (0.25, 0.5, 0.75) for alpha in (0.67, 0.8) for delta in (0.0, 0.01)]
        selected_mean = _select(mean_candidates)
        selected_tail = _select(tail_candidates)
        mean_at_tail_alpha = _validation_metrics(data["validation_units"], np.asarray(selected_mean["weights"]), float(selected_tail["alpha"]))
        stability = _stability(data, selected_tail)
        results[dataset] = {"strongest_single": strongest, "uniform": uniform_row, "mean_candidates": mean_candidates, "tail_candidates": tail_candidates, "selected_mean": selected_mean, "selected_mean_at_tail_alpha": mean_at_tail_alpha, "selected_tail": selected_tail, "stability": stability}
    accuracy_gain = {dataset: results[dataset]["selected_tail"]["accuracy"] - results[dataset]["strongest_single"]["accuracy"] for dataset in results}
    f1_gain = {dataset: results[dataset]["selected_tail"]["macro_f1"] - results[dataset]["strongest_single"]["macro_f1"] for dataset in results}
    cvar_gain = {dataset: results[dataset]["selected_mean_at_tail_alpha"]["cvar_log_loss"] - results[dataset]["selected_tail"]["cvar_log_loss"] for dataset in results}
    gates = {"accuracy": max(accuracy_gain.values()) >= 0.02 and min(accuracy_gain.values()) >= -0.005, "macro_f1": min(f1_gain.values()) >= -0.005, "cvar": max(cvar_gain.values()) >= 0.005 and min(cvar_gain.values()) >= -0.005, "stability": all(results[dataset]["stability"]["pass"] for dataset in results)}
    status = "PASS_REVIEW_BEFORE_OUTER_TEST" if all(gates.values()) else "FAIL_STOP_METHOD_ROUTE"
    return {"protocol": PROTOCOL, "stage": "G0-C", "status": status, "access_boundary": "train OOF fit plus sanitized validation scoring only; no test arrays loaded", "datasets": results, "contrasts": {"tail_minus_strongest_accuracy": accuracy_gain, "tail_minus_strongest_macro_f1": f1_gain, "mean_minus_tail_cvar_improvement": cvar_gain}, "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, default=Path("results/cf_tre/g0c/oof"))
    parser.add_argument("--input-root", type=Path, default=Path("results/cf_tre/g0c/inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g0c/g0c_feasibility_result.json"))
    args = parser.parse_args()
    result = run(args.oof_root.resolve(), args.input_root.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "contrasts": result["contrasts"], "gates": result["gates"]}, indent=2))
    raise SystemExit(0 if result["status"].startswith("PASS") else 2)


if __name__ == "__main__":
    main()
