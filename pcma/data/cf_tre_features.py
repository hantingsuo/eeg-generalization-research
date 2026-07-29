"""Label-free multi-view features frozen for CF-TRE G0-A."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np


SEED_CHANNELS: tuple[str, ...] = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4",
    "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1",
    "O1", "OZ", "O2", "CB2",
)

ASYMMETRY_PAIRS: tuple[tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F5", "F6"),
    ("FC5", "FC6"),
    ("C5", "C6"),
    ("CP5", "CP6"),
    ("P5", "P6"),
    ("PO5", "PO6"),
    ("O1", "O2"),
)


def _region_for_channel(channel: str) -> str:
    if channel.startswith(("PO", "O", "CB")):
        return "occipital"
    if channel.startswith(("CP", "P")):
        return "parietal"
    if channel.startswith(("TP", "T")):
        return "temporal"
    if channel.startswith("C"):
        return "central"
    if channel.startswith(("FP", "AF", "F")):
        return "frontal"
    raise ValueError(f"unassigned channel {channel!r}")


REGION_CHANNELS: Mapping[str, tuple[str, ...]] = OrderedDict(
    (
        region,
        tuple(channel for channel in SEED_CHANNELS if _region_for_channel(channel) == region),
    )
    for region in ("frontal", "central", "temporal", "parietal", "occipital")
)

VIEW_DIMENSIONS: Mapping[str, int] = OrderedDict(
    (
        ("de310", 310),
        ("asymmetry45", 45),
        ("regional50", 50),
        ("trial_context150", 150),
    )
)


def audit_feature_schema() -> None:
    """Raise if the frozen channel/region schema is internally inconsistent."""

    if len(SEED_CHANNELS) != 62 or len(set(SEED_CHANNELS)) != 62:
        raise ValueError("SEED channel schema must contain 62 unique channels")
    flattened = [channel for channels in REGION_CHANNELS.values() for channel in channels]
    if len(flattened) != 62 or set(flattened) != set(SEED_CHANNELS):
        raise ValueError("regions must form a disjoint cover of all 62 channels")
    for left, right in ASYMMETRY_PAIRS:
        if left not in SEED_CHANNELS or right not in SEED_CHANNELS:
            raise ValueError(f"unknown asymmetry pair {(left, right)}")


def _as_trial_tensor(trial_windows: np.ndarray) -> np.ndarray:
    array = np.asarray(trial_windows, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (62, 5):
        raise ValueError(
            "trial_windows must have shape (windows, 62, 5), "
            f"got {array.shape}"
        )
    if len(array) == 0:
        raise ValueError("trial_windows cannot be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("trial_windows contains non-finite values")
    return array


def _upper_correlations(matrix: np.ndarray) -> np.ndarray:
    """Return upper off-diagonal correlations, with constants mapped to zero."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("correlation input must be two-dimensional")
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    denom = np.outer(norms, norms)
    numer = centered.T @ centered
    correlation = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=np.float64),
        where=denom > 0,
    )
    indices = np.triu_indices(values.shape[1], k=1)
    return correlation[indices]


def extract_cf_tre_feature_views(trial_windows: np.ndarray) -> dict[str, np.ndarray]:
    """Extract all four frozen feature views from one complete trial.

    The trial context is computed solely from the supplied trial and broadcast
    back to its windows.  Callers must invoke this function separately for each
    trial to preserve the frozen leakage boundary.
    """

    audit_feature_schema()
    trial = _as_trial_tensor(trial_windows)
    num_windows = len(trial)
    channel_index = {channel: index for index, channel in enumerate(SEED_CHANNELS)}

    de310 = trial.reshape(num_windows, -1)
    asymmetry = np.stack(
        [
            trial[:, channel_index[left], :] - trial[:, channel_index[right], :]
            for left, right in ASYMMETRY_PAIRS
        ],
        axis=1,
    ).reshape(num_windows, -1)

    regional_means: list[np.ndarray] = []
    regional_stds: list[np.ndarray] = []
    for channels in REGION_CHANNELS.values():
        indices = [channel_index[channel] for channel in channels]
        regional_means.append(trial[:, indices, :].mean(axis=1))
        regional_stds.append(trial[:, indices, :].std(axis=1, ddof=0))
    region_mean_tensor = np.stack(regional_means, axis=1)  # windows x regions x bands
    region_std_tensor = np.stack(regional_stds, axis=1)
    regional50 = np.concatenate(
        [
            region_mean_tensor.reshape(num_windows, -1),
            region_std_tensor.reshape(num_windows, -1),
        ],
        axis=1,
    )

    context_distribution = np.concatenate(
        [
            region_mean_tensor.mean(axis=0).reshape(-1),
            region_mean_tensor.std(axis=0, ddof=0).reshape(-1),
        ]
    )
    band_correlations = np.concatenate(
        [_upper_correlations(region_mean_tensor[:, region, :]) for region in range(5)]
    )
    region_correlations = np.concatenate(
        [_upper_correlations(region_mean_tensor[:, :, band]) for band in range(5)]
    )
    context = np.concatenate(
        [context_distribution, band_correlations, region_correlations]
    )
    if context.shape != (150,):
        raise AssertionError(f"internal trial-context shape error: {context.shape}")
    trial_context150 = np.broadcast_to(context, (num_windows, 150)).copy()

    views = {
        "de310": np.ascontiguousarray(de310, dtype=np.float32),
        "asymmetry45": np.ascontiguousarray(asymmetry, dtype=np.float32),
        "regional50": np.ascontiguousarray(regional50, dtype=np.float32),
        "trial_context150": np.ascontiguousarray(trial_context150, dtype=np.float32),
    }
    for name, expected_dim in VIEW_DIMENSIONS.items():
        if views[name].shape != (num_windows, expected_dim):
            raise AssertionError(f"internal {name} shape error: {views[name].shape}")
        if not np.all(np.isfinite(views[name])):
            raise AssertionError(f"internal {name} non-finite value")
    return views


def concatenate_feature_views(
    views: Mapping[str, np.ndarray],
    names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Concatenate named views after fail-closed row/dimension checks."""

    if not names:
        raise ValueError("at least one feature view is required")
    arrays: list[np.ndarray] = []
    rows: int | None = None
    for name in names:
        if name not in VIEW_DIMENSIONS:
            raise KeyError(f"unknown feature view {name!r}")
        if name not in views:
            raise KeyError(f"missing feature view {name!r}")
        array = np.asarray(views[name])
        if array.ndim != 2 or array.shape[1] != VIEW_DIMENSIONS[name]:
            raise ValueError(f"invalid {name} shape: {array.shape}")
        if rows is None:
            rows = len(array)
        elif len(array) != rows:
            raise ValueError("feature views have inconsistent row counts")
        arrays.append(array)
    return np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float32)
