"""Reconstruct reporting choices and audit declared information boundaries.

This utility checks supplied records. It cannot detect omitted data contacts.
No data, network access, or model training is required for the synthetic demo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_trial_partitions(partitions, trial_labels):
    labels = np.asarray(trial_labels)
    expected = {"train", "validation", "selection", "audit"}
    if set(partitions) != expected:
        raise ValueError("four declared partitions required")
    seen = set()
    for name, ids in partitions.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"empty or duplicate trial IDs: {name}")
        current = set(ids)
        if any(not isinstance(i, int) or i < 1 or i > len(labels) for i in ids):
            raise ValueError("trial ID outside data")
        if seen & current:
            raise ValueError("trial overlap between partitions")
        if set(labels[np.asarray(ids) - 1]) != set(labels):
            raise ValueError(f"missing class in {name}")
        seen |= current
    if seen != set(range(1, len(labels) + 1)):
        raise ValueError("unassigned trials")


def earliest_best(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("nonempty finite score vector required")
    return int(np.argmax(values)) + 1


def reporting_summary(logits, labels, subjects, trials):
    logits, labels, subjects, trials = map(np.asarray, (logits, labels, subjects, trials))
    if logits.ndim != 2 or logits.shape[0] == 0 or not np.isfinite(logits).all():
        raise ValueError("finite nonempty logits required")
    if any(a.ndim != 1 or len(a) != len(logits) for a in (labels, subjects, trials)):
        raise ValueError("row metadata must match logits")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("invalid class labels")
    correct = logits.argmax(1) == labels
    per_subject = {}
    for subject in np.unique(subjects):
        mask = subjects == subject
        trial_correct = []
        for trial in np.unique(trials[mask]):
            group = mask & (trials == trial)
            if len(np.unique(labels[group])) != 1:
                raise ValueError("trial contains conflicting labels")
            trial_correct.append(int(logits[group].mean(0).argmax() == labels[group][0]))
        per_subject[str(subject)] = {
            "window_accuracy": float(correct[mask].mean()),
            "trial_accuracy": float(np.mean(trial_correct)),
            "windows": int(mask.sum()),
            "trials": len(trial_correct),
        }
    return {
        "pooled_window_accuracy": float(correct.mean()),
        "subject_equal_window_accuracy": float(np.mean([x["window_accuracy"] for x in per_subject.values()])),
        "subject_equal_trial_accuracy": float(np.mean([x["trial_accuracy"] for x in per_subject.values()])),
        "per_subject": per_subject,
    }


def paired_subject_interval(differences, *, replicates=20000, seed=20260910):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent-unit values required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "n_subjects": len(values), "unit": "subject", "method": "paired percentile bootstrap",
            "conditioning": "fixed split and fitted models; sessions and optimization seeds averaged within subject"}


def audit_contact_log(events, expected_cells):
    frozen, audit_loaded = set(), set()
    errors = []
    for event in events:
        kind, cell = event.get("event"), event.get("cell")
        if kind == "selection_frozen":
            if cell in frozen:
                errors.append("duplicate selection freeze")
            frozen.add(cell)
        elif kind == "audit_loaded":
            if len(frozen) != expected_cells:
                errors.append("audit accessed before all selections frozen")
            if cell not in frozen:
                errors.append("audit accessed without own frozen selection")
            if cell in audit_loaded:
                errors.append("duplicate audit access")
            audit_loaded.add(cell)
    if len(frozen) != expected_cells or len(audit_loaded) != expected_cells:
        errors.append("incomplete cell coverage")
    return {"valid": not errors, "errors": sorted(set(errors)), "frozen_cells": len(frozen),
            "audited_cells": len(audit_loaded), "scope": "checks supplied log only; cannot detect unlogged access"}


def demo():
    logits = np.array([[2., 0.], [2., 0.], [2., 0.], [2., 0.]])
    summary = reporting_summary(logits, np.array([0, 0, 0, 1]), np.array([1, 1, 1, 2]), np.array([1, 1, 1, 1]))
    good = [{"event": "selection_frozen", "cell": "one"}, {"event": "audit_loaded", "cell": "one"}]
    return {"synthetic_reporting": summary, "valid_log": audit_contact_log(good, 1),
            "invalid_log": audit_contact_log(good[::-1], 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--events", type=Path)
    p.add_argument("--expected-cells", type=int)
    args = p.parse_args()
    if args.demo:
        result = demo()
    elif args.events is not None and args.expected_cells is not None:
        result = audit_contact_log([json.loads(x) for x in args.events.read_text(encoding="utf-8").splitlines()], args.expected_cells)
    else:
        p.error("use --demo or --events with --expected-cells")
    print(json.dumps(result, indent=2))
    if result.get("valid") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
