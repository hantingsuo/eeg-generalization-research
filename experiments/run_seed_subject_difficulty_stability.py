"""Frozen Gate-C prerequisite: subject-difficulty stability on SEED/SEED-IV.

Importing this module is side-effect free. The CLI is the only real-data entry
point; tests use pure helpers and synthetic inputs only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import norm, rankdata, spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.data.seed import SeedData, load_seed_family


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "results" / "subject_difficulty_stability.json"
DEFAULT_MD = ROOT / "results" / "subject_difficulty_stability.md"
EXPECTED = {"SEED": ({-1, 0, 1}, 15), "SEED-IV": ({0, 1, 2, 3}, 24)}
OFFICIAL_TRIAL_LABELS = {
    "SEED": {
        session: (1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1)
        for session in (1, 2, 3)
    },
    "SEED-IV": {
        1: (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
        2: (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
        3: (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
    },
}


def _subject_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def make_fold_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")),
    ])


def strict_loso_oof(X, y, subject, estimator_factory=make_fold_pipeline):
    """Fit every preprocessing/model component without held-out-subject data."""
    X, y, subject = np.asarray(X), np.asarray(y), np.asarray(subject)
    if not (len(X) == len(y) == len(subject)):
        raise ValueError("X, y, and subject lengths differ")
    subjects = sorted(np.unique(subject).tolist(), key=_subject_key)
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two subjects")
    pred = np.empty(y.shape, dtype=y.dtype)
    folds = []
    for held_out in subjects:
        test = subject == held_out
        train = ~test
        model = estimator_factory()
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        folds.append({"held_out_subject": str(held_out),
                      "n_train_subjects": int(len(np.unique(subject[train]))),
                      "n_train_trials": int(train.sum()), "n_test_trials": int(test.sum())})
    return pred, folds


def balanced_blocks(y, trial):
    """Alternate trial-id-sorted observations within each true-label stratum."""
    y, trial = np.asarray(y), np.asarray(trial)
    if len(y) != len(trial) or len(np.unique(trial)) != len(trial):
        raise ValueError("labels/trials mismatch or duplicate trial ids")
    block = np.empty(len(y), dtype="<U1")
    for label in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == label)
        idx = idx[np.argsort(trial[idx], kind="stable")]
        block[idx[::2]], block[idx[1::2]] = "A", "B"
    return block


def check_integrity(data: SeedData):
    """Verify the frozen n/session/label/common trial-block hierarchy."""
    expected_labels, expected_trials = EXPECTED.get(data.dataset, (set(), -1))
    lengths = [len(data.X), len(data.y), len(data.subject), len(data.session), len(data.trial)]
    arrays_aligned = len(set(lengths)) == 1 and lengths[0] > 0
    subjects = sorted(np.unique(data.subject).tolist(), key=_subject_key)
    sessions = sorted(np.unique(data.session).tolist())
    checks = {
        "known_dataset": data.dataset in EXPECTED,
        "arrays_have_equal_nonzero_length": arrays_aligned,
        "features_are_2d": data.X.ndim == 2,
        "n_subjects_is_15": len(subjects) == 15,
        "sessions_are_1_2_3": sessions == [1, 2, 3],
        "features_are_finite": bool(np.isfinite(data.X).all()),
        "labels_match_dataset": set(np.unique(data.y).tolist()) == expected_labels,
        "trial_ids_are_official_sequence": True,
        "official_trial_label_schedule": True,
        "every_session_has_same_15_subjects": True,
        "every_subject_has_expected_trials_and_labels": True,
        "fixed_trial_label_block_mapping": True,
        "all_classes_and_blocks_nonempty": True,
    }
    maps = {}
    serial_maps = {}
    if not arrays_aligned or data.X.ndim != 2:
        checks["every_session_has_same_15_subjects"] = False
        checks["every_subject_has_expected_trials_and_labels"] = False
        checks["fixed_trial_label_block_mapping"] = False
        checks["all_classes_and_blocks_nonempty"] = False
        checks["trial_ids_are_official_sequence"] = False
        checks["official_trial_label_schedule"] = False
    elif not subjects or sessions != [1, 2, 3]:
        checks["every_session_has_same_15_subjects"] = False
        checks["every_subject_has_expected_trials_and_labels"] = False
        checks["fixed_trial_label_block_mapping"] = False
        checks["all_classes_and_blocks_nonempty"] = False
        checks["trial_ids_are_official_sequence"] = False
        checks["official_trial_label_schedule"] = False
    else:
        expected_subjects = {str(s) for s in subjects}
        for session in sessions:
            sm = data.session == session
            session_subjects = sorted(np.unique(data.subject[sm]).tolist(), key=_subject_key)
            if {str(s) for s in session_subjects} != expected_subjects:
                checks["every_session_has_same_15_subjects"] = False
                continue
            ref = session_subjects[0]
            rm = sm & (data.subject == ref)
            ref_trials, ref_y = data.trial[rm], data.y[rm]
            try:
                ref_blocks = balanced_blocks(ref_y, ref_trials)
            except ValueError:
                checks["fixed_trial_label_block_mapping"] = False
                continue
            ref_map = {int(t): (int(label), str(block)) for t, label, block in zip(ref_trials, ref_y, ref_blocks)}
            maps[int(session)] = ref_map
            serial_maps[str(session)] = [
                {"trial": trial, "label": values[0], "block": values[1]}
                for trial, values in sorted(ref_map.items())
            ]
            if len(ref_map) != expected_trials or {v[0] for v in ref_map.values()} != expected_labels:
                checks["every_subject_has_expected_trials_and_labels"] = False
            official_labels = OFFICIAL_TRIAL_LABELS.get(data.dataset, {}).get(int(session), ())
            official_map = {trial: int(label) for trial, label in enumerate(official_labels, start=1)}
            observed_ref_labels = {trial: values[0] for trial, values in ref_map.items()}
            if set(ref_map) != set(range(1, expected_trials + 1)):
                checks["trial_ids_are_official_sequence"] = False
            if observed_ref_labels != official_map:
                checks["official_trial_label_schedule"] = False
            for label in expected_labels:
                label_blocks = {block for value, block in ref_map.values() if value == label}
                if label_blocks != {"A", "B"}:
                    checks["all_classes_and_blocks_nonempty"] = False
            for subject in session_subjects:
                mask = sm & (data.subject == subject)
                trials, labels = data.trial[mask], data.y[mask]
                if len(trials) != expected_trials or len(np.unique(trials)) != expected_trials or set(labels.tolist()) != expected_labels:
                    checks["every_subject_has_expected_trials_and_labels"] = False
                    continue
                observed = {int(t): int(label) for t, label in zip(trials, labels)}
                if set(observed) != set(range(1, expected_trials + 1)):
                    checks["trial_ids_are_official_sequence"] = False
                if observed != {trial: values[0] for trial, values in ref_map.items()}:
                    checks["fixed_trial_label_block_mapping"] = False
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "session_trial_label_block_maps": serial_maps}
    return report, maps


def _rho(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x, y).statistic)


def median_spearman(columns, pairs):
    pairwise = [{"pair": name, "rho": _rho(columns[left], columns[right])}
                for name, left, right in pairs]
    valid = all(np.isfinite(row["rho"]) for row in pairwise)
    return {"status": "OK" if valid else "NOT_IDENTIFIABLE",
            "pairwise": [{"pair": row["pair"], "rho": float(row["rho"]) if np.isfinite(row["rho"]) else None}
                         for row in pairwise],
            "median_rho": float(np.median([row["rho"] for row in pairwise])) if valid else None}


def _row_rho(x, y):
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
    return np.divide((rx * ry).sum(axis=1), den, out=np.full(len(x), np.nan), where=den > 0)


def _row_statistics(columns, pairs, index_maps):
    rhos = np.column_stack([
        _row_rho(np.asarray(columns[left])[index_maps[left]], np.asarray(columns[right])[index_maps[right]])
        for _, left, right in pairs
    ])
    valid = np.all(np.isfinite(rhos), axis=1)
    stats = np.full(len(rhos), np.nan)
    stats[valid] = np.median(rhos[valid], axis=1)
    return stats, valid


def bca_interval(columns, pairs, n_resamples=20_000, seed=20240713):
    """Aligned subject-row bootstrap plus leave-one-subject-out BCa interval."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    observed = median_spearman(columns, pairs)
    n = len(next(iter(columns.values())))
    base = {key: None for key in columns}
    result = {"method": "BCa subject-cluster bootstrap", "confidence_level": 0.95,
              "n_resamples": int(n_resamples), "seed": int(seed), "valid_resamples": 0,
              "invalid_resamples": int(n_resamples), "jackknife_valid": 0,
              "jackknife_invalid": int(n), "z0": None, "acceleration": None,
              "adjusted_quantiles": None, "low": None, "high": None,
              "status": "NOT_IDENTIFIABLE"}
    if observed["status"] != "OK":
        return result
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_maps = {key: idx for key in base}
    boot, valid = _row_statistics(columns, pairs, boot_maps)
    valid_boot = boot[valid]
    result["valid_resamples"] = int(valid.sum())
    result["invalid_resamples"] = int((~valid).sum())
    jack = []
    for omitted in range(n):
        keep = np.delete(np.arange(n), omitted).reshape(1, -1)
        value, ok = _row_statistics(columns, pairs, {key: keep for key in columns})
        jack.append(float(value[0]) if ok[0] else np.nan)
    jack = np.asarray(jack)
    result["jackknife_valid"] = int(np.isfinite(jack).sum())
    result["jackknife_invalid"] = int((~np.isfinite(jack)).sum())
    if len(valid_boot) == 0 or not np.isfinite(jack).all():
        return result
    observed_value = observed["median_rho"]
    proportion = float(np.mean(valid_boot < observed_value))
    if proportion <= 0.0 or proportion >= 1.0:
        result["failure_reason"] = "BCa bias-correction proportion is on the 0/1 boundary"
        return result
    z0 = float(norm.ppf(proportion))
    result["z0"] = z0
    jack_mean = float(jack.mean())
    delta = jack_mean - jack
    if np.allclose(delta, 0.0, rtol=0.0, atol=1e-12):
        result["failure_reason"] = "BCa jackknife distribution is numerically degenerate"
        return result
    denominator = 6.0 * float(np.sum(delta ** 2) ** 1.5)
    if denominator <= 0:
        result["failure_reason"] = "BCa jackknife acceleration denominator is zero"
        return result
    acceleration = float(np.sum(delta ** 3) / denominator)
    result["acceleration"] = acceleration
    adjusted = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        denom = 1.0 - acceleration * (z0 + z_alpha)
        adjusted.append(float(norm.cdf(z0 + (z0 + z_alpha) / denom)) if denom != 0 else np.nan)
    result["adjusted_quantiles"] = [float(v) if np.isfinite(v) else None for v in adjusted]
    if (not np.isfinite(adjusted).all() or any(value < 0.0 or value > 1.0 for value in adjusted)):
        result["failure_reason"] = "BCa adjusted quantile is nonfinite or outside [0,1]"
        return result
    low, high = np.quantile(valid_boot, adjusted)
    result.update({"status": "OK", "z0": z0, "acceleration": acceleration,
                   "adjusted_quantiles": [float(v) for v in adjusted],
                   "low": float(low), "high": float(high)})
    return result


def permutation_index_maps(keys, n_subjects, n_permutations, shared_groups, seed):
    """Create row maps; keys in one group receive exactly the same permutation."""
    base = np.broadcast_to(np.arange(n_subjects), (n_permutations, n_subjects))
    maps = {key: base for key in keys}
    assigned = set()
    rng = np.random.default_rng(seed)
    for group in shared_groups:
        if assigned.intersection(group):
            raise ValueError("permutation key assigned more than once")
        permutation = np.argsort(rng.random((n_permutations, n_subjects)), axis=1)
        for key in group:
            if key not in maps:
                raise ValueError(f"unknown permutation key {key}")
            maps[key] = permutation
            assigned.add(key)
    return maps


def pairing_permutation_test(columns, pairs, shared_groups, n_permutations=20_000, seed=20240713):
    """One-sided joint pairing test; all scheduled permutations must be valid."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    observed = median_spearman(columns, pairs)
    n = len(next(iter(columns.values())))
    maps = permutation_index_maps(columns.keys(), n, n_permutations, shared_groups, seed)
    null, valid = _row_statistics(columns, pairs, maps)
    all_valid = bool(valid.all())
    result = {"method": "joint subject-pairing permutation", "alternative": "greater",
              "n_permutations": int(n_permutations), "seed": int(seed),
              "valid_permutations": int(valid.sum()), "invalid_permutations": int((~valid).sum()),
              "observed_median_rho": observed["median_rho"], "p_value": None,
              "status": "NOT_IDENTIFIABLE"}
    if observed["status"] == "OK" and all_valid:
        result["p_value"] = float((np.sum(null >= observed["median_rho"]) + 1) / (n_permutations + 1))
        result["status"] = "OK"
    return result


def _unavailable_stat(reason):
    return {"status": "NOT_IDENTIFIABLE", "reason": reason, "pairwise": [], "median_rho": None,
            "bca_95_ci": {"status": "NOT_IDENTIFIABLE"},
            "permutation_test": {"status": "NOT_IDENTIFIABLE", "p_value": None}}


def stability_statistics(subject_rows, n_boot, n_permutations, seed):
    sessions = sorted(subject_rows)
    ids = [r["subject"] for r in subject_rows.get(1, [])]
    if sessions != [1, 2, 3] or any([r["subject"] for r in subject_rows[s]] != ids for s in sessions):
        reason = "unaligned subject rows across sessions"
        return {"cross_session_subject_accuracy": _unavailable_stat(reason),
                "within_session_a_vs_b": _unavailable_stat(reason)}
    cross_columns = {f"S{s}": np.asarray([r["accuracy"] for r in subject_rows[s]]) for s in sessions}
    cross_pairs = [(f"S{a}-S{b}", f"S{a}", f"S{b}") for a, b in itertools.combinations(sessions, 2)]
    split_columns = {f"{part}{s}": np.asarray([r[f"block_{part.lower()}_accuracy"] for r in subject_rows[s]])
                     for s in sessions for part in ("A", "B")}
    split_pairs = [(f"A{s}-B{s}", f"A{s}", f"B{s}") for s in sessions]

    def summarize(columns, pairs, permutation_groups):
        observed = median_spearman(columns, pairs)
        return {**observed,
                "bca_95_ci": bca_interval(columns, pairs, n_boot, seed),
                "permutation_test": pairing_permutation_test(
                    columns, pairs, permutation_groups, n_permutations, seed)}

    return {
        "cross_session_subject_accuracy": summarize(
            cross_columns, cross_pairs, [("S2",), ("S3",)]),  # S1 fixed; S2/S3 independent.
        "within_session_a_vs_b": summarize(
            split_columns, split_pairs, [("B1", "B2", "B3")]),  # One shared B permutation.
    }


def analyze_dataset(data: SeedData, aggregate, n_boot=20_000, n_permutations=20_000,
                    seed=20240713, progress=print):
    integrity, block_maps = check_integrity(data)
    base = {"dataset": data.dataset, "aggregate": aggregate, "n_subjects": int(data.n_subjects),
            "integrity": integrity, "sessions": {}}
    if integrity["status"] != "PASS":
        base["stability"] = {"cross_session_subject_accuracy": _unavailable_stat("integrity failure"),
                             "within_session_a_vs_b": _unavailable_stat("integrity failure")}
        return base
    subject_rows = {}
    for session in (1, 2, 3):
        if progress:
            progress(f"[{data.dataset}/{aggregate}] session {session}: LOSO start")
        mask = data.session == session
        X, y, subjects, trials = data.X[mask], data.y[mask], data.subject[mask], data.trial[mask]
        pred, folds = strict_loso_oof(X, y, subjects)
        records, rows = [], []
        for subject in sorted(np.unique(subjects).tolist(), key=_subject_key):
            sm = subjects == subject
            sy, sp, st = y[sm], pred[sm], trials[sm]
            block = np.asarray([block_maps[session][int(t)][1] for t in st])
            correct = sy == sp
            rows.append({"subject": str(subject), "n_trials": int(sm.sum()),
                         "accuracy": float(correct.mean()),
                         "block_a_accuracy": float(correct[block == "A"].mean()),
                         "block_b_accuracy": float(correct[block == "B"].mean())})
            for i in np.argsort(st, kind="stable"):
                records.append({"subject": str(subject), "session": session, "trial": int(st[i]),
                                "y_true": int(sy[i]), "y_pred": int(sp[i]),
                                "correct": bool(correct[i]), "block": str(block[i])})
        subject_rows[session] = rows
        base["sessions"][str(session)] = {"n_trials": int(len(y)), "folds": folds,
                                          "subject_metrics": rows, "oof_predictions": records}
        if progress:
            progress(f"[{data.dataset}/{aggregate}] session {session}: LOSO complete")
    if progress:
        progress(f"[{data.dataset}/{aggregate}] resampling start")
    base["stability"] = stability_statistics(subject_rows, n_boot, n_permutations, seed)
    if progress:
        progress(f"[{data.dataset}/{aggregate}] resampling complete")
    return base


def _positive_count(stat):
    return sum(row["rho"] is not None and row["rho"] > 0 for row in stat["pairwise"])


def decide_gate(configurations):
    """Apply the manifest's hierarchical confirmatory/replication gate exactly."""
    primary = configurations.get("mean_std", {})
    sensitivity = configurations.get("mean", {})
    seediv, seed = primary.get("SEED-IV"), primary.get("SEED")
    decision = {"decision": "NOT_IDENTIFIABLE", "may_proceed": False,
                "confirmatory_dataset": "SEED-IV", "replication_dataset": "SEED",
                "sensitivity_cannot_rescue_primary": True, "checks": {},
                "representation_direction_reversal": [],
                "representation_dependent_warning": False}

    def mark_direction_reversal(dataset):
        primary_stat = primary.get(dataset, {}).get("stability", {}).get("cross_session_subject_accuracy", {})
        sensitivity_stat = sensitivity.get(dataset, {}).get("stability", {}).get("cross_session_subject_accuracy", {})
        if (primary_stat.get("median_rho") is not None and primary_stat["median_rho"] > 0 and
                sensitivity_stat.get("median_rho") is not None and sensitivity_stat["median_rho"] < 0):
            decision["representation_direction_reversal"].append(dataset)
            decision["representation_dependent_warning"] = True
    if not seediv or seediv.get("integrity", {}).get("status") != "PASS":
        decision["reason"] = "SEED-IV integrity failed"
        return decision
    split = seediv["stability"]["within_session_a_vs_b"]
    cross = seediv["stability"]["cross_session_subject_accuracy"]
    # Undefined required quantities dominate threshold failures: they may never
    # be silently converted into FAIL_RELIABILITY/FAIL_CROSS_SESSION.
    if (split["status"] != "OK" or cross["status"] != "OK" or
            cross["permutation_test"].get("status") != "OK"):
        decision["reason"] = "required SEED-IV correlation/permutation undefined"
        return decision
    split_ok = split["median_rho"] >= 0.30 and _positive_count(split) >= 2
    decision["checks"]["seediv_split_median_ge_0p30"] = split["median_rho"] >= 0.30
    decision["checks"]["seediv_split_two_positive"] = _positive_count(split) >= 2
    if not split_ok:
        decision.update(decision="FAIL_RELIABILITY", reason="SEED-IV split-half gate failed")
        return decision
    cross_values = [row["rho"] for row in cross["pairwise"]]
    cross_checks = {
        "seediv_cross_median_ge_0p40": cross["median_rho"] >= 0.40,
        "seediv_cross_permutation_p_le_0p05": cross["permutation_test"]["p_value"] <= 0.05,
        "seediv_cross_two_positive": sum(v > 0 for v in cross_values) >= 2,
        "seediv_no_cross_rho_le_minus_0p20": all(v > -0.20 for v in cross_values),
    }
    decision["checks"].update(cross_checks)
    if not all(cross_checks.values()):
        decision.update(decision="FAIL_CROSS_SESSION", reason="SEED-IV cross-session gate failed")
        return decision
    # The confirmatory representation has passed; a sensitivity reversal must
    # now be disclosed even if replication later becomes non-identifiable.
    mark_direction_reversal("SEED-IV")
    if not seed or seed.get("integrity", {}).get("status") != "PASS":
        decision["reason"] = "SEED replication integrity failed"
        return decision
    seed_cross = seed["stability"]["cross_session_subject_accuracy"]
    if seed_cross["status"] != "OK":
        decision["reason"] = "required SEED replication correlation undefined"
        return decision
    seed_values = [row["rho"] for row in seed_cross["pairwise"]]
    replication_checks = {
        "seed_cross_median_ge_0p30": seed_cross["median_rho"] >= 0.30,
        "seed_cross_two_positive": sum(v > 0 for v in seed_values) >= 2,
        "seed_no_cross_rho_le_minus_0p20": all(v > -0.20 for v in seed_values),
    }
    decision["checks"].update(replication_checks)
    decision["decision"] = "PASS" if all(replication_checks.values()) else "DATASET_SPECIFIC"
    decision["may_proceed"] = decision["decision"] == "PASS"
    decision["reason"] = "both frozen gates passed" if decision["may_proceed"] else "SEED directional replication failed"
    if all(replication_checks.values()):
        mark_direction_reversal("SEED")
    return decision


def build_document(configurations, n_boot, n_permutations, seed):
    return {"schema_version": "2.0", "analysis": "Gate C subject-difficulty stability prerequisite",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {"confirmatory_dataset": "SEED-IV", "replication_dataset": "SEED",
                         "primary_aggregate": "mean_std", "sensitivity_aggregate": "mean",
                         "independent_unit": "subject", "expected_n_subjects": 15,
                         "evaluation": "strict session-wise LOSO",
                         "model": "StandardScaler + RBF SVC(C=1,gamma=scale,class_weight=balanced)",
                         "target_statistics": False, "per_subject_zscore": False,
                         "bootstrap": "aligned subject-row BCa with leave-one-subject jackknife",
                         "bootstrap_resamples": int(n_boot), "permutations": int(n_permutations),
                         "seed": int(seed)},
            "configurations": configurations, "gate": decide_gate(configurations)}


def validate_schema(document):
    if not {"schema_version", "analysis", "generated_at_utc", "protocol", "configurations", "gate"} <= set(document):
        raise ValueError("incomplete top-level schema")
    for aggregate in ("mean_std", "mean"):
        for dataset in ("SEED", "SEED-IV"):
            item = document.get("configurations", {}).get(aggregate, {}).get(dataset)
            if not item or "integrity" not in item or "stability" not in item:
                raise ValueError(f"incomplete result for {aggregate}/{dataset}")
            if item["integrity"]["status"] == "PASS" and set(item.get("sessions", {})) != {"1", "2", "3"}:
                raise ValueError(f"incomplete sessions for {aggregate}/{dataset}")
            required_stats = {"cross_session_subject_accuracy", "within_session_a_vs_b"}
            if not required_stats <= set(item.get("stability", {})):
                raise ValueError(f"incomplete stability statistics for {aggregate}/{dataset}")
            for name in required_stats:
                stat = item["stability"][name]
                if not {"status", "pairwise", "median_rho", "bca_95_ci", "permutation_test"} <= set(stat):
                    raise ValueError(f"incomplete {name} schema for {aggregate}/{dataset}")
                if stat["status"] == "OK" and (len(stat["pairwise"]) != 3 or stat["median_rho"] is None):
                    raise ValueError(f"invalid identified {name} for {aggregate}/{dataset}")
    if document["gate"].get("decision") not in {
        "PASS", "DATASET_SPECIFIC", "FAIL_RELIABILITY", "FAIL_CROSS_SESSION", "NOT_IDENTIFIABLE"
    }:
        raise ValueError("unknown gate decision")
    return True


def _fmt(value):
    return "NA" if value is None else f"{value:.3f}"


def render_markdown(document):
    gate = document["gate"]
    lines = ["# Gate C Subject-Difficulty Stability Diagnostic", "",
             f"**Frozen decision: {gate['decision']}**", "", gate.get("reason", ""), "",
             "All scaling and classification were fitted only on source subjects within each session.", ""]
    for aggregate, by_dataset in document["configurations"].items():
        lines += [f"## {aggregate} ({'primary' if aggregate == 'mean_std' else 'sensitivity'})", "",
                  "| Dataset | Integrity | Statistic | Median rho | BCa 95% CI | Permutation p | Status |",
                  "|---|---|---|---:|---:|---:|---|"]
        for dataset in ("SEED-IV", "SEED"):
            result = by_dataset[dataset]
            for key, label in (("within_session_a_vs_b", "A-vs-B"),
                               ("cross_session_subject_accuracy", "Cross-session")):
                stat = result["stability"][key]
                ci, perm = stat.get("bca_95_ci", {}), stat.get("permutation_test", {})
                lines.append(f"| {dataset} | {result['integrity']['status']} | {label} | "
                             f"{_fmt(stat.get('median_rho'))} | [{_fmt(ci.get('low'))}, {_fmt(ci.get('high'))}] | "
                             f"{_fmt(perm.get('p_value'))} | {stat.get('status')} |")
        lines.append("")
    if gate.get("representation_dependent_warning"):
        lines += ["## Sensitivity warning", "",
                  "The mean representation reverses the cross-session direction for: " +
                  ", ".join(gate["representation_direction_reversal"]) + ".", ""]
    return "\n".join(lines) + "\n"


def guard_output_paths(json_path, md_path):
    json_path, md_path = Path(json_path), Path(md_path)
    if json_path.resolve() == md_path.resolve():
        raise ValueError("JSON and Markdown outputs must be distinct")
    existing = [str(path) for path in (json_path, md_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output(s): " + ", ".join(existing))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", default=None)
    parser.add_argument("--boot", "--bootstrap", dest="bootstrap", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20240713)
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    json_path, md_path = Path(args.output_json), Path(args.output_md)
    guard_output_paths(json_path, md_path)  # Must precede any data loading/computation.
    print(f"[plan] seed={args.seed} bootstrap={args.bootstrap} permutations={args.permutations}", flush=True)
    configurations = {}
    for aggregate in ("mean_std", "mean"):
        configurations[aggregate] = {}
        for dataset in ("SEED-IV", "SEED"):  # Confirmatory dataset is evaluated first.
            print(f"[{dataset}/{aggregate}] loading", flush=True)
            kwargs = {"aggregate": aggregate}
            if args.seed_root is not None:
                kwargs["root"] = args.seed_root
            data = load_seed_family(dataset, **kwargs)
            print(f"[{dataset}/{aggregate}] loaded X={data.X.shape}", flush=True)
            configurations[aggregate][dataset] = analyze_dataset(
                data, aggregate, args.bootstrap, args.permutations, args.seed)
    document = build_document(configurations, args.bootstrap, args.permutations, args.seed)
    validate_schema(document)
    guard_output_paths(json_path, md_path)  # Close the compute/write race without overwriting.
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(document), encoding="utf-8")
    print(f"[done] wrote {json_path} and {md_path}; decision={document['gate']['decision']}", flush=True)


if __name__ == "__main__":
    main()
