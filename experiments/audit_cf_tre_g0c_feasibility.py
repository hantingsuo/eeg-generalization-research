"""Independent audit of the frozen G0-C development-only result.

This auditor recomputes validation metrics and gates directly from train-OOF
and sanitized validation artifacts.  It does not import the production runner
or open any G0-B/test artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROTOCOL = "cf_tre_g0c_development_feasibility_v1"
COMPONENTS = (
    "linear_de310", "linear_structural245", "linear_all555", "rbf_de310",
    "rbf_all555", "xgboost_all555", "lightgbm_all555",
)
TOLERANCE = 1e-10


def _cvar(values: list[float], alpha: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(min(
        eta + np.maximum(array - eta, 0.0).mean() / (1.0 - alpha)
        for eta in np.unique(array)
    ))


def _mix(probabilities: np.ndarray, weights: list[float]) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (len(COMPONENTS),):
        raise ValueError("weight shape mismatch")
    if weight_array.min() < -1e-8 or not np.isclose(weight_array.sum(), 1.0, atol=1e-8):
        raise ValueError("invalid simplex weights")
    mixed = np.einsum("nmc,m->nc", probabilities, weight_array)
    mixed = np.clip(mixed, 1e-6, 1.0)
    return mixed / mixed.sum(axis=1, keepdims=True)


def _metrics(dataset: str, input_root: Path, weights: list[float], alpha: float) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    environment_losses: list[float] = []
    for session in range(1, 4):
        for subject in range(1, 16):
            path = input_root / dataset / f"s{session:02d}_sub{subject:02d}.validation.npz"
            with np.load(path, allow_pickle=False) as arrays:
                expected = {"validation_labels", "validation_trials"} | {
                    f"validation_probabilities__{component}" for component in COMPONENTS
                }
                if set(arrays.files) != expected or any(key.lower().startswith("test") for key in arrays.files):
                    raise ValueError(f"invalid sanitized validation keys: {path}")
                labels = arrays["validation_labels"].astype(np.int64)
                trials = arrays["validation_trials"]
                probabilities = np.stack([
                    arrays[f"validation_probabilities__{component}"] for component in COMPONENTS
                ], axis=1)
            mixed = _mix(probabilities, weights)
            predicted = mixed.argmax(axis=1)
            true_probability = np.clip(mixed[np.arange(len(labels)), labels], 1e-6, 1.0)
            environment_losses.append(float(np.mean(-np.log(true_probability))))
            trial_correct: list[float] = []
            for trial in np.unique(trials):
                keep = trials == trial
                trial_labels = np.unique(labels[keep])
                if len(trial_labels) != 1:
                    raise ValueError(f"trial has multiple labels: {path} trial={trial}")
                trial_correct.append(float(mixed[keep].mean(axis=0).argmax() == trial_labels[0]))
            rows.append({
                "subject": float(subject),
                "accuracy": float(accuracy_score(labels, predicted)),
                "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "trial_accuracy": float(np.mean(trial_correct)),
            })
    by_subject: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)

    def subject_equal(key: str) -> float:
        return float(np.mean([
            np.mean([row[key] for row in by_subject[subject]]) for subject in range(1, 16)
        ]))

    return {
        "accuracy": subject_equal("accuracy"),
        "macro_f1": subject_equal("macro_f1"),
        "balanced_accuracy": subject_equal("balanced_accuracy"),
        "trial_accuracy": subject_equal("trial_accuracy"),
        "mean_environment_log_loss": float(np.mean(environment_losses)),
        "cvar_log_loss": _cvar(environment_losses, alpha),
        "environment_losses": environment_losses,
    }


def _strongest_oof_component(dataset: str, oof_root: Path) -> str:
    losses: list[list[float]] = [[] for _ in COMPONENTS]
    for session in range(1, 4):
        for subject in range(1, 16):
            path = oof_root / dataset / f"s{session:02d}_sub{subject:02d}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                if any(key.lower().startswith(("test", "validation")) for key in arrays.files):
                    raise ValueError(f"non-train array in OOF input: {path}")
                labels = arrays["train_labels"].astype(np.int64)
                for index, component in enumerate(COMPONENTS):
                    probabilities = arrays[f"oof_probabilities__{component}"]
                    true_probability = np.clip(
                        probabilities[np.arange(len(labels)), labels], 1e-6, 1.0
                    )
                    losses[index].append(float(np.mean(-np.log(true_probability))))
    means = [float(np.mean(component_losses)) for component_losses in losses]
    return COMPONENTS[int(np.argmin(means))]


def _assert_close(name: str, observed: Any, expected: Any, errors: list[str]) -> None:
    if isinstance(expected, list):
        if not np.allclose(np.asarray(observed), np.asarray(expected), rtol=0.0, atol=TOLERANCE):
            errors.append(f"{name} mismatch")
    elif not np.isclose(float(observed), float(expected), rtol=0.0, atol=TOLERANCE):
        errors.append(f"{name} mismatch: {observed} != {expected}")


def _selected_independently(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(candidates, key=lambda row: (
        -float(row["accuracy"]), -float(row["macro_f1"]), float(row["cvar_log_loss"]),
        float(row["lambda_tail"]), float(row["alpha"]), float(row["delta"]),
    ))


def audit(result_path: Path, oof_root: Path, input_root: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks = 0
    if result.get("protocol") != PROTOCOL or result.get("stage") != "G0-C":
        errors.append("protocol/stage mismatch")
    if any(str(key).lower().startswith("test") for key in result):
        errors.append("forbidden top-level test key")
    checks += 2

    recomputed_contrasts = {
        "tail_minus_strongest_accuracy": {},
        "tail_minus_strongest_macro_f1": {},
        "mean_minus_tail_cvar_improvement": {},
    }
    stability_passes: list[bool] = []
    metric_keys = (
        "accuracy", "macro_f1", "balanced_accuracy", "trial_accuracy",
        "mean_environment_log_loss", "cvar_log_loss", "environment_losses",
    )
    for dataset in ("seed", "seediv"):
        rows = result["datasets"][dataset]
        strongest_component = _strongest_oof_component(dataset, oof_root)
        checks += 1
        if rows["strongest_single"]["component"] != strongest_component:
            errors.append(f"{dataset}: strongest OOF component mismatch")

        for candidate_group, selected_name in (
            ("mean_candidates", "selected_mean"), ("tail_candidates", "selected_tail")
        ):
            independent = _selected_independently(rows[candidate_group])
            selected = rows[selected_name]
            checks += 1
            for key in ("lambda_tail", "alpha", "delta", "weights"):
                _assert_close(f"{dataset}.{selected_name}.{key}", selected[key], independent[key], errors)

        selected_tail = rows["selected_tail"]
        selected_mean = rows["selected_mean"]
        strongest = rows["strongest_single"]
        uniform = rows["uniform"]
        recomputed = {
            "strongest_single": _metrics(dataset, input_root, strongest["weights"], 0.8),
            "uniform": _metrics(dataset, input_root, uniform["weights"], 0.8),
            "selected_mean": _metrics(dataset, input_root, selected_mean["weights"], float(selected_mean["alpha"])),
            "selected_tail": _metrics(dataset, input_root, selected_tail["weights"], float(selected_tail["alpha"])),
            "selected_mean_at_tail_alpha": _metrics(dataset, input_root, selected_mean["weights"], float(selected_tail["alpha"])),
        }
        for row_name, expected in recomputed.items():
            for key in metric_keys:
                checks += 1
                _assert_close(f"{dataset}.{row_name}.{key}", rows[row_name][key], expected[key], errors)

        stability = rows["stability"]
        distances = np.asarray(stability["l1_distances"], dtype=np.float64)
        full_weights = np.asarray(selected_tail["weights"], dtype=np.float64)
        stability_pass = bool(
            stability.get("all_15_refits_converged") is True
            and len(distances) == 15 and np.all(np.isfinite(distances))
            and float(np.median(distances)) <= 0.35
            and float(np.quantile(distances, 0.9)) <= 0.60
            and float(full_weights.max()) < 0.95
            and int(np.sum(full_weights > 0.05)) >= 2
        )
        stability_passes.append(stability_pass)
        for name, observed, expected in (
            ("median_l1", stability["median_l1"], np.median(distances)),
            ("p90_l1", stability["p90_l1"], np.quantile(distances, 0.9)),
            ("max_weight", stability["max_weight"], full_weights.max()),
            ("components_above_0_05", stability["components_above_0_05"], np.sum(full_weights > 0.05)),
        ):
            checks += 1
            _assert_close(f"{dataset}.stability.{name}", observed, expected, errors)
        checks += 1
        if bool(stability["pass"]) != stability_pass:
            errors.append(f"{dataset}: stability gate mismatch")

        recomputed_contrasts["tail_minus_strongest_accuracy"][dataset] = (
            recomputed["selected_tail"]["accuracy"] - recomputed["strongest_single"]["accuracy"]
        )
        recomputed_contrasts["tail_minus_strongest_macro_f1"][dataset] = (
            recomputed["selected_tail"]["macro_f1"] - recomputed["strongest_single"]["macro_f1"]
        )
        recomputed_contrasts["mean_minus_tail_cvar_improvement"][dataset] = (
            recomputed["selected_mean_at_tail_alpha"]["cvar_log_loss"]
            - recomputed["selected_tail"]["cvar_log_loss"]
        )

    for contrast, datasets in recomputed_contrasts.items():
        for dataset, expected in datasets.items():
            checks += 1
            _assert_close(f"contrasts.{contrast}.{dataset}", result["contrasts"][contrast][dataset], expected, errors)
    accuracy = recomputed_contrasts["tail_minus_strongest_accuracy"]
    macro_f1 = recomputed_contrasts["tail_minus_strongest_macro_f1"]
    cvar = recomputed_contrasts["mean_minus_tail_cvar_improvement"]
    gates = {
        "accuracy": max(accuracy.values()) >= 0.02 and min(accuracy.values()) >= -0.005,
        "macro_f1": min(macro_f1.values()) >= -0.005,
        "cvar": max(cvar.values()) >= 0.005 and min(cvar.values()) >= -0.005,
        "stability": all(stability_passes),
    }
    for gate, expected in gates.items():
        checks += 1
        if bool(result["gates"][gate]) != expected:
            errors.append(f"gate mismatch: {gate}")
    expected_status = "PASS_REVIEW_BEFORE_OUTER_TEST" if all(gates.values()) else "FAIL_STOP_METHOD_ROUTE"
    checks += 1
    if result.get("status") != expected_status:
        errors.append("status mismatch")
    return {
        "protocol": PROTOCOL,
        "audit": "independent validation metric, selection, stability-threshold, and final-gate reconstruction",
        "checks": checks,
        "errors": errors,
        "recomputed_contrasts": recomputed_contrasts,
        "recomputed_gates": gates,
        "recomputed_status": expected_status,
        "status": "PASS" if not errors else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("results/cf_tre/g0c/g0c_feasibility_result.json"))
    parser.add_argument("--oof-root", type=Path, default=Path("results/cf_tre/g0c/oof"))
    parser.add_argument("--input-root", type=Path, default=Path("results/cf_tre/g0c/inputs"))
    parser.add_argument("--output", type=Path, default=Path("results/cf_tre/g0c/g0c_feasibility_independent_audit.json"))
    args = parser.parse_args()
    result = audit(args.result.resolve(), args.oof_root.resolve(), args.input_root.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
