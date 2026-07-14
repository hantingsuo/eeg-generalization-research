"""Shared helpers for novelty-method competition evaluations."""
from __future__ import annotations

import numpy as np


def align_proba(proba, classes, target_classes=(0, 1)) -> np.ndarray:
    """Return probability columns in target_classes order, filling absent classes with 0."""
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    out = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for j, cls in enumerate(target_classes):
        where = np.where(classes == cls)[0]
        if len(where):
            out[:, j] = proba[:, where[0]]
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        out[row_sum.ravel() <= 0] = 1.0 / len(target_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def pred_from_proba(proba, classes=(0, 1)) -> np.ndarray:
    classes = np.asarray(classes)
    return classes[np.asarray(proba).argmax(axis=1)]


def aggregate_proba_by_group(y_true, proba, classes, group_id, subject=None, population=None):
    """Average probabilities within each verified group, e.g. 5 segments per video."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    group_id = np.asarray(group_id).astype(str)
    subject = np.asarray(subject) if subject is not None else None
    population = np.asarray(population) if population is not None else None

    y_out, p_out, s_out, pop_out, group_out = [], [], [], [], []
    for group in dict.fromkeys(group_id.tolist()):
        mask = group_id == group
        labels = np.unique(y_true[mask])
        if len(labels) != 1:
            raise ValueError(f"{group}: y_true is not constant")
        y_out.append(labels[0])
        p_out.append(proba[mask].mean(axis=0))
        group_out.append(group)
        if subject is not None:
            subjects = np.unique(subject[mask])
            if len(subjects) != 1:
                raise ValueError(f"{group}: subject is not constant")
            s_out.append(subjects[0])
        if population is not None:
            pops = np.unique(population[mask])
            if len(pops) != 1:
                raise ValueError(f"{group}: population is not constant")
            pop_out.append(pops[0])
    return {
        "y": np.asarray(y_out),
        "proba": np.asarray(p_out),
        "pred": pred_from_proba(np.asarray(p_out), classes),
        "classes": classes,
        "subject": np.asarray(s_out) if subject is not None else None,
        "population": np.asarray(pop_out) if population is not None else None,
        "group": np.asarray(group_out),
    }


def source_subject_validation_mask(subject, val_fraction: float = 0.25, seed: int = 0):
    """Deterministic source-only validation split by subject."""
    subject = np.asarray(subject)
    subjects = np.unique(subject)
    if len(subjects) < 2:
        train = np.ones(len(subject), dtype=bool)
        return train, ~train
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)
    val_subjects = set(shuffled[:n_val].tolist())
    val = np.asarray([s in val_subjects for s in subject], dtype=bool)
    train = ~val
    return train, val


def keep_from_gate(p1_video_boot: dict, p3_video: dict | None = None) -> tuple[bool, str]:
    """Apply the fixed keep/drop gate from the roadmap."""
    if p1_video_boot["ci_low"] > 0:
        return True, "keep: P1 video paired-bootstrap CI excludes 0"
    if p3_video is not None:
        p = p3_video.get("wilcoxon_method_vs_baseline_p", 1.0)
        method = p3_video.get("method_video_mean", -np.inf)
        baseline = p3_video.get("baseline_video_mean", np.inf)
        if method > baseline and p < 0.05:
            return True, "keep: P3 video Wilcoxon significant"
    return False, "drop: no significant improvement over rich-SVM gate"
