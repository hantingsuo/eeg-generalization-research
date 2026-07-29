"""Cue-locked ERP features for MODMA.

This module implements the fixed v2 MODMA gate: baseline-corrected cue epochs,
central-parietal HydroCel channels, and P300/LPP mean-amplitude features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pcma.data.modma import (
    DEFAULT_EVENT_CHANNELS,
    DEFAULT_MODMA_ROOT,
    ModmaFeatureData,
    egi_event_samples,
    load_modma_metadata,
)


ERP_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("p300", 0.300, 0.500),
    ("lpp", 0.400, 1.000),
)

ERP_MONTAGE_TARGETS: tuple[str, ...] = ("Cz", "CPz", "Pz", "POz")


@dataclass(frozen=True)
class ModmaErpConfig:
    event_channels: tuple[str, ...] = DEFAULT_EVENT_CHANNELS
    tmin: float = -0.200
    tmax: float = 1.000
    baseline: tuple[float, float] = (-0.200, 0.000)
    windows: tuple[tuple[str, float, float], ...] = ERP_WINDOWS
    montage_targets: tuple[str, ...] = ERP_MONTAGE_TARGETS
    nearest_per_target: int = 4
    reject_mad_z: float = 6.0
    max_peak_to_peak_uv: float = 8000.0


def _hydrocel_to_raw_name(name: str) -> str:
    # MNE's HydroCel-129 montage names the vertex channel "Cz"; EGI raw files
    # in this dataset name the same 129th EEG channel "E129".
    return "E129" if name == "Cz" else name


def egi_centroparietal_cluster(
    targets: tuple[str, ...] = ERP_MONTAGE_TARGETS,
    nearest_per_target: int = 4,
) -> list[str]:
    """Select a fixed central-parietal HydroCel cluster from 10-20 landmarks."""
    import mne

    hydrocel = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ten_twenty = mne.channels.make_standard_montage("standard_1020")
    hpos = hydrocel.get_positions()["ch_pos"]
    tpos = ten_twenty.get_positions()["ch_pos"]
    selected: list[str] = []
    for target in targets:
        if target not in tpos:
            raise ValueError(f"target {target!r} is not in standard_1020 montage")
        distances = sorted(
            (float(np.linalg.norm(np.asarray(pos) - np.asarray(tpos[target]))), name)
            for name, pos in hpos.items()
        )
        for _, name in distances[:nearest_per_target]:
            raw_name = _hydrocel_to_raw_name(name)
            if raw_name not in selected:
                selected.append(raw_name)
    return selected


def erp_feature_names(ch_names: list[str], windows: tuple[tuple[str, float, float], ...] = ERP_WINDOWS) -> list[str]:
    names: list[str] = []
    for win_name, _, _ in windows:
        names.extend(f"{win_name}_mean_uv_{ch}" for ch in ch_names)
        names.append(f"{win_name}_mean_uv_cluster")
    return names


def baseline_average_reference(epoch: np.ndarray, sfreq: float, tmin: float, baseline: tuple[float, float]) -> np.ndarray:
    """Baseline-correct one epoch and apply average reference."""
    x = np.asarray(epoch, dtype=float).copy()
    times = tmin + np.arange(x.shape[1]) / float(sfreq)
    mask = (times >= baseline[0]) & (times < baseline[1])
    if not np.any(mask):
        raise ValueError("baseline window has no samples")
    x -= x[:, mask].mean(axis=1, keepdims=True)
    x -= x.mean(axis=0, keepdims=True)
    return x


def erp_window_features(
    epoch: np.ndarray,
    sfreq: float,
    tmin: float,
    channel_indices: list[int],
    windows: tuple[tuple[str, float, float], ...] = ERP_WINDOWS,
) -> np.ndarray:
    """Extract P300/LPP mean amplitudes in microvolts for selected channels."""
    x = np.asarray(epoch, dtype=float)
    times = tmin + np.arange(x.shape[1]) / float(sfreq)
    feats: list[np.ndarray] = []
    for _, start, stop in windows:
        mask = (times >= start) & (times < stop)
        if not np.any(mask):
            raise ValueError(f"ERP window {start}-{stop}s has no samples")
        values = x[np.asarray(channel_indices), :][:, mask].mean(axis=1) * 1e6
        feats.append(values.astype(float))
        feats.append(np.asarray([values.mean()], dtype=float))
    return np.concatenate(feats).astype(np.float32)


def _robust_epoch_keep_mask(scores: np.ndarray, reject_mad_z: float, max_peak_to_peak_uv: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        return np.zeros_like(scores, dtype=bool)
    med = float(np.median(scores[finite]))
    mad = float(np.median(np.abs(scores[finite] - med)))
    robust_sd = 1.4826 * mad
    robust_limit = med + reject_mad_z * robust_sd if robust_sd > 1e-12 else max_peak_to_peak_uv
    limit = min(float(max_peak_to_peak_uv), float(robust_limit))
    return finite & (scores <= limit)


def load_modma_erp_features(
    root: str | Path = DEFAULT_MODMA_ROOT,
    config: ModmaErpConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> ModmaFeatureData:
    """Build baseline-corrected cue-locked ERP features from MODMA raw files."""
    import mne

    config = config or ModmaErpConfig()
    records = load_modma_metadata(root)
    cluster_names = egi_centroparietal_cluster(config.montage_targets, config.nearest_per_target)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    populations: list[str] = []
    events: list[str] = []
    scales_by_subject = {r.subject_id: dict(r.scales) for r in records}
    phq9_by_subject = {r.subject_id: r.scales["PHQ-9"] for r in records if "PHQ-9" in r.scales}
    feature_names = erp_feature_names(cluster_names, config.windows)
    reject_summary: dict[str, dict[str, object]] = {}

    for i, record in enumerate(records, start=1):
        if record.raw_path is None:
            raise FileNotFoundError(f"{record.subject_id}: no joined MODMA raw file")
        if progress is not None:
            progress(f"[{i}/{len(records)}] ERP {record.subject_id} {record.population}")
        raw = mne.io.read_raw_egi(record.raw_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude=[])
        eeg_names = [raw.ch_names[p] for p in eeg_picks]
        missing = [ch for ch in cluster_names if ch not in eeg_names]
        if missing:
            raise ValueError(f"{record.subject_id}: missing ERP cluster channels {missing}")
        cluster_indices = [eeg_names.index(ch) for ch in cluster_names]
        eeg = raw.get_data(picks=eeg_picks)
        event_samples = egi_event_samples(raw, config.event_channels)
        start_offset = int(round(config.tmin * sfreq))
        stop_offset = int(round(config.tmax * sfreq))
        if stop_offset <= start_offset:
            raise ValueError("ERP epoch must have positive duration")

        subject_epochs: list[np.ndarray] = []
        subject_labels: list[str] = []
        subject_scores: list[float] = []
        for event_name in config.event_channels:
            for sample in event_samples[event_name]:
                start = int(sample) + start_offset
                stop = int(sample) + stop_offset
                if start < 0 or stop > eeg.shape[1]:
                    continue
                epoch = baseline_average_reference(eeg[:, start:stop], sfreq, config.tmin, config.baseline)
                score = float(np.max(np.ptp(epoch, axis=1)) * 1e6)
                subject_epochs.append(epoch)
                subject_labels.append(event_name)
                subject_scores.append(score)

        keep = _robust_epoch_keep_mask(np.asarray(subject_scores), config.reject_mad_z, config.max_peak_to_peak_uv)
        kept_counts = {label: 0 for label in config.event_channels}
        total_counts = {label: 0 for label in config.event_channels}
        for epoch, label, keep_epoch in zip(subject_epochs, subject_labels, keep):
            total_counts[label] += 1
            if not keep_epoch:
                continue
            kept_counts[label] += 1
            rows.append(erp_window_features(epoch, sfreq, config.tmin, cluster_indices, config.windows))
            labels.append(label)
            subjects.append(record.subject_id)
            populations.append(record.population)
            events.append(label)
        reject_summary[record.subject_id] = {
            "total": int(len(subject_labels)),
            "kept": int(keep.sum()),
            "total_by_label": total_counts,
            "kept_by_label": kept_counts,
            "cluster_channels": cluster_names,
        }

    if not rows:
        raise ValueError("no MODMA ERP feature rows were extracted")
    return ModmaFeatureData(
        F=np.vstack(rows).astype(np.float32),
        y_emotion=np.asarray(labels),
        subject=np.asarray(subjects),
        population=np.asarray(populations),
        phq9_by_subject=phq9_by_subject,
        event=np.asarray(events),
        scales_by_subject=scales_by_subject,
        feature_names=feature_names,
        source=str(Path(root)),
        config={
            "erp_v2": {
                "event_channels": list(config.event_channels),
                "tmin": config.tmin,
                "tmax": config.tmax,
                "baseline": list(config.baseline),
                "windows": [[name, start, stop] for name, start, stop in config.windows],
                "montage_targets": list(config.montage_targets),
                "nearest_per_target": config.nearest_per_target,
                "cluster_channels": cluster_names,
                "reject_mad_z": config.reject_mad_z,
                "max_peak_to_peak_uv": config.max_peak_to_peak_uv,
                "reject_summary": reject_summary,
            }
        },
    )
