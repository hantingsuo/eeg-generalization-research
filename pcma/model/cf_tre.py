"""Cross-Fitted Tail-Risk Ensemble (CF-TRE) core contracts.

The optimizer consumes already cross-fitted probabilities.  It deliberately
contains no base-model training code, which keeps row exclusion auditable and
prevents accidental use of in-sample or outer-test predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


DEFAULT_EPSILON = 1e-6
DEFAULT_SOLVER_TOLERANCE = 1e-8


@dataclass(frozen=True)
class OOFPredictionRecord:
    """One base model's out-of-fold probability row and exclusion ledger."""

    row_id: str
    group_id: str
    environment_id: str
    outer_split: str
    inner_fold: int
    base_model: str
    fit_group_ids: tuple[str, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class CFTREResult:
    """Deterministic result of one frozen CF-TRE configuration."""

    weights: np.ndarray
    lambda_tail: float
    alpha: float
    delta: float
    mean_loss: float
    cvar_loss: float
    objective: float
    mean_loss_cap: float
    best_single_model_index: int
    solver_success: bool
    solver_status: int
    solver_message: str
    iterations: int


def validate_base_probabilities(
    base_probabilities: np.ndarray,
    *,
    atol: float = 1e-8,
) -> np.ndarray:
    """Return float64 probabilities or raise on a malformed simplex row."""

    probabilities = np.asarray(base_probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError(
            "base_probabilities must have shape (rows, models, classes), "
            f"got {probabilities.shape}"
        )
    if min(probabilities.shape) <= 0:
        raise ValueError("base_probabilities dimensions must be non-empty")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("base_probabilities contains non-finite values")
    if probabilities.min() < -atol or probabilities.max() > 1.0 + atol:
        raise ValueError("base_probabilities contains values outside [0, 1]")
    row_sums = probabilities.sum(axis=2)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=atol):
        raise ValueError("base probability rows must sum to one")
    return probabilities


def _validate_weights(weights: Sequence[float], num_models: int, atol: float = 1e-8) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (num_models,):
        raise ValueError(f"weights must have shape ({num_models},), got {array.shape}")
    if not np.all(np.isfinite(array)) or array.min() < -atol:
        raise ValueError("weights must be finite and nonnegative")
    if not np.isclose(array.sum(), 1.0, rtol=0.0, atol=atol):
        raise ValueError("weights must sum to one")
    return array


def mixture_probabilities(
    base_probabilities: np.ndarray,
    weights: Sequence[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Form a clipped and renormalized convex probability mixture."""

    probabilities = validate_base_probabilities(base_probabilities)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    weight_array = _validate_weights(weights, probabilities.shape[1])
    mixed = np.einsum("nmc,m->nc", probabilities, weight_array)
    mixed = np.clip(mixed, epsilon, 1.0)
    mixed /= mixed.sum(axis=1, keepdims=True)
    return mixed


def environment_log_losses(
    mixed_probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    environments: Sequence[str] | np.ndarray,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compute equal-environment mean multiclass log losses."""

    mixed = np.asarray(mixed_probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    environment_array = np.asarray([str(value) for value in environments], dtype=object)
    if mixed.ndim != 2 or len(mixed) != len(label_array) or len(mixed) != len(environment_array):
        raise ValueError("probabilities, labels, and environments must have equal rows")
    if not len(mixed):
        raise ValueError("at least one probability row is required")
    if label_array.min() < 0 or label_array.max() >= mixed.shape[1]:
        raise ValueError("labels outside probability class range")
    if not np.all(np.isfinite(mixed)):
        raise ValueError("mixed probabilities contain non-finite values")
    if not np.allclose(mixed.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("mixed probability rows must sum to one")

    true_probability = np.clip(mixed[np.arange(len(label_array)), label_array], epsilon, 1.0)
    row_loss = -np.log(true_probability)
    environment_ids = tuple(sorted(set(environment_array.tolist())))
    losses = np.asarray(
        [row_loss[environment_array == environment].mean() for environment in environment_ids],
        dtype=np.float64,
    )
    return losses, environment_ids


def empirical_cvar(losses: Sequence[float], alpha: float) -> float:
    """Evaluate empirical upper-tail CVaR via the exact finite epigraph minimum."""

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("losses must be a non-empty finite vector")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    candidates = np.unique(values)
    objectives = [
        eta + np.maximum(values - eta, 0.0).mean() / (1.0 - alpha)
        for eta in candidates
    ]
    return float(min(objectives))


def _losses_for_weights(
    probabilities: np.ndarray,
    labels: np.ndarray,
    environments: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    mixed = np.einsum("nmc,m->nc", probabilities, weights)
    mixed = np.clip(mixed, epsilon, 1.0)
    mixed /= mixed.sum(axis=1, keepdims=True)
    losses, _ = environment_log_losses(
        mixed,
        labels,
        environments,
        epsilon=epsilon,
    )
    return losses


def fit_cf_tre_weights(
    base_probabilities: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    environments: Sequence[str] | np.ndarray,
    *,
    lambda_tail: float,
    alpha: float = 0.8,
    delta: float = 0.0,
    epsilon: float = DEFAULT_EPSILON,
    maxiter: int = 2000,
    ftol: float = 1e-10,
    constraint_tolerance: float = DEFAULT_SOLVER_TOLERANCE,
    mean_loss_cap: float | None = None,
) -> CFTREResult:
    """Fit simplex weights under the frozen mean-risk guard.

    ``mean_loss_cap`` exists only for fail-closed solver testing. Production
    calls leave it as ``None``, which uses best-single mean loss plus ``delta``.
    """

    probabilities = validate_base_probabilities(base_probabilities)
    label_array = np.asarray(labels, dtype=np.int64)
    environment_array = np.asarray([str(value) for value in environments], dtype=object)
    if len(label_array) != len(probabilities) or len(environment_array) != len(probabilities):
        raise ValueError("labels and environments must match probability rows")
    if not 0.0 <= lambda_tail <= 1.0:
        raise ValueError("lambda_tail must lie in [0, 1]")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    if delta < 0.0:
        raise ValueError("delta must be nonnegative")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    if maxiter <= 0 or ftol <= 0 or constraint_tolerance <= 0:
        raise ValueError("solver controls must be positive")

    num_models = probabilities.shape[1]
    single_losses = np.stack(
        [
            _losses_for_weights(
                probabilities,
                label_array,
                environment_array,
                np.eye(num_models, dtype=np.float64)[model],
                epsilon,
            )
            for model in range(num_models)
        ]
    )
    single_means = single_losses.mean(axis=1)
    best_single = int(np.argmin(single_means))
    default_cap = float(single_means[best_single] + delta)
    cap = default_cap if mean_loss_cap is None else float(mean_loss_cap)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError("mean_loss_cap must be finite and nonnegative")

    uniform = np.full(num_models, 1.0 / num_models, dtype=np.float64)
    uniform_mean = float(
        _losses_for_weights(
            probabilities, label_array, environment_array, uniform, epsilon
        ).mean()
    )
    initial_weights = uniform if uniform_mean <= cap else np.eye(num_models)[best_single]

    def environment_losses_for_vector(vector: np.ndarray) -> np.ndarray:
        return _losses_for_weights(
            probabilities,
            label_array,
            environment_array,
            vector[:num_models],
            epsilon,
        )

    def mean_constraint(vector: np.ndarray) -> float:
        return cap - float(environment_losses_for_vector(vector).mean())

    constraints = (
        {"type": "eq", "fun": lambda vector: float(np.sum(vector[:num_models]) - 1.0)},
        {"type": "ineq", "fun": mean_constraint},
    )

    if lambda_tail == 0.0:
        x0 = initial_weights

        def objective(vector: np.ndarray) -> float:
            return float(environment_losses_for_vector(vector).mean())

        bounds = [(0.0, 1.0)] * num_models
    else:
        initial_losses = environment_losses_for_vector(initial_weights)
        eta0 = float(np.quantile(initial_losses, alpha, method="higher"))
        x0 = np.concatenate([initial_weights, [eta0]])

        def objective(vector: np.ndarray) -> float:
            losses = environment_losses_for_vector(vector)
            mean_loss = float(losses.mean())
            eta = float(vector[-1])
            cvar_epigraph = eta + float(np.maximum(losses - eta, 0.0).mean()) / (1.0 - alpha)
            return float((1.0 - lambda_tail) * mean_loss + lambda_tail * cvar_epigraph)

        bounds = [(0.0, 1.0)] * num_models + [(0.0, float(-np.log(epsilon)))]

    optimized = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": False},
    )
    if not optimized.success:
        raise RuntimeError(
            f"CF-TRE optimization failed ({optimized.status}): {optimized.message}"
        )

    weights = np.asarray(optimized.x[:num_models], dtype=np.float64)
    weights[np.abs(weights) < constraint_tolerance] = 0.0
    if weights.sum() <= 0:
        raise RuntimeError("CF-TRE optimizer returned zero total weight")
    weights /= weights.sum()
    final_losses = _losses_for_weights(
        probabilities, label_array, environment_array, weights, epsilon
    )
    mean_loss = float(final_losses.mean())
    if weights.min() < -constraint_tolerance or not np.isclose(
        weights.sum(), 1.0, rtol=0.0, atol=constraint_tolerance
    ):
        raise RuntimeError("CF-TRE solution violates simplex constraints")
    if mean_loss > cap + constraint_tolerance:
        raise RuntimeError(
            f"CF-TRE solution violates mean-risk cap: {mean_loss} > {cap}"
        )
    cvar_loss = empirical_cvar(final_losses, alpha)
    final_objective = float((1.0 - lambda_tail) * mean_loss + lambda_tail * cvar_loss)
    return CFTREResult(
        weights=weights,
        lambda_tail=float(lambda_tail),
        alpha=float(alpha),
        delta=float(delta),
        mean_loss=mean_loss,
        cvar_loss=cvar_loss,
        objective=final_objective,
        mean_loss_cap=cap,
        best_single_model_index=best_single,
        solver_success=bool(optimized.success),
        solver_status=int(optimized.status),
        solver_message=str(optimized.message),
        iterations=int(getattr(optimized, "nit", -1)),
    )


def audit_oof_prediction_ledger(
    records: Iterable[OOFPredictionRecord],
    *,
    expected_row_ids: Sequence[str],
    expected_base_models: Sequence[str],
    require_environment_exclusion: bool = False,
    atol: float = 1e-8,
) -> None:
    """Raise if an OOF ledger is incomplete, duplicated, or group-contaminated."""

    rows = tuple(str(row) for row in expected_row_ids)
    models = tuple(str(model) for model in expected_base_models)
    if not rows or not models:
        raise ValueError("expected rows and base models must be non-empty")
    if len(set(rows)) != len(rows) or len(set(models)) != len(models):
        raise ValueError("expected rows and models must be unique")

    expected_pairs = {(row, model) for row in rows for model in models}
    seen_pairs: set[tuple[str, str]] = set()
    for record in records:
        pair = (str(record.row_id), str(record.base_model))
        if pair not in expected_pairs:
            raise ValueError(f"unexpected ledger pair {pair}")
        if pair in seen_pairs:
            raise ValueError(f"duplicate ledger pair {pair}")
        seen_pairs.add(pair)
        if record.outer_split != "train":
            raise ValueError(f"OOF ledger contains non-train row {record.row_id}")
        if record.inner_fold <= 0:
            raise ValueError("inner_fold must be positive")
        fit_groups = {str(group) for group in record.fit_group_ids}
        if not fit_groups:
            raise ValueError("fit_group_ids cannot be empty")
        if str(record.group_id) in fit_groups:
            raise ValueError(f"scored group leaked into fit for {pair}")
        if require_environment_exclusion and str(record.environment_id) in fit_groups:
            raise ValueError(f"scored environment leaked into fit for {pair}")
        probability = np.asarray(record.probabilities, dtype=np.float64)
        if probability.ndim != 1 or len(probability) < 2:
            raise ValueError(f"invalid probability vector for {pair}")
        if not np.all(np.isfinite(probability)) or probability.min() < -atol:
            raise ValueError(f"invalid probability values for {pair}")
        if not np.isclose(probability.sum(), 1.0, rtol=0.0, atol=atol):
            raise ValueError(f"probability vector does not sum to one for {pair}")

    missing = expected_pairs - seen_pairs
    if missing:
        raise ValueError(f"OOF ledger is incomplete; missing {sorted(missing)[:5]}")


def select_validation_configuration(
    candidates: Iterable[Mapping[str, float]],
) -> Mapping[str, float]:
    """Apply the frozen validation-only lexicographic selection rule."""

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("at least one validation candidate is required")
    required = {"accuracy", "macro_f1", "cvar_log_loss", "lambda_tail", "alpha", "delta"}
    for candidate in candidate_list:
        missing = required - set(candidate)
        if missing:
            raise ValueError(f"candidate missing fields: {sorted(missing)}")
        if not all(np.isfinite(float(candidate[key])) for key in required):
            raise ValueError("candidate selection fields must be finite")

    def key(candidate: Mapping[str, float]) -> tuple[float, ...]:
        return (
            -float(candidate["accuracy"]),
            -float(candidate["macro_f1"]),
            float(candidate["cvar_log_loss"]),
            float(candidate["lambda_tail"]),
            float(candidate["alpha"]),
            float(candidate["delta"]),
        )

    return min(candidate_list, key=key)

