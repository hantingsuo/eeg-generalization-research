from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.run_seed_subject_difficulty_stability import (
    DEFAULT_JSON,
    DEFAULT_MD,
    OFFICIAL_TRIAL_LABELS,
    balanced_blocks,
    bca_interval,
    build_document,
    build_parser,
    check_integrity,
    decide_gate,
    guard_output_paths,
    make_fold_pipeline,
    median_spearman,
    pairing_permutation_test,
    permutation_index_maps,
    strict_loso_oof,
    validate_schema,
)
from pcma.data.seed import SeedData


def test_pipeline_is_frozen_and_fold_fit_never_sees_target_subject():
    pipeline = make_fold_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    svc = pipeline.named_steps["svc"]
    assert isinstance(svc, SVC)
    assert (svc.C, svc.kernel, svc.gamma, svc.class_weight) == (1.0, "rbf", "scale", "balanced")
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(set(X[:, 0].astype(int)))
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    subjects = np.repeat(np.arange(3), 2)
    X = np.column_stack([subjects, np.arange(6)])
    strict_loso_oof(X, np.tile([0, 1], 3), subjects, estimator_factory=Spy)
    assert seen == [{1, 2}, {0, 2}, {0, 1}]


def test_balanced_blocks_are_label_stratified_deterministic():
    y = np.array([1, 0, 1, 0, 1, 0])
    trial = np.array([6, 1, 2, 5, 4, 3])
    expected = np.array(["A", "A", "A", "A", "B", "B"])
    assert np.array_equal(balanced_blocks(y, trial), expected)
    assert np.array_equal(balanced_blocks(y, trial), expected)


def test_bca_has_required_metadata_and_is_reproducible():
    columns = {"x": np.arange(15.0),
               "y": np.array([0, 2, 1, 4, 3, 5, 7, 6, 9, 8, 10, 12, 11, 14, 13], dtype=float)}
    pairs = [("x-y", "x", "y")]
    first = bca_interval(columns, pairs, n_resamples=200, seed=9)
    second = bca_interval(columns, pairs, n_resamples=200, seed=9)
    assert first == second
    assert first["status"] == "OK"
    assert {"z0", "acceleration", "valid_resamples", "invalid_resamples",
            "jackknife_valid", "jackknife_invalid", "adjusted_quantiles"} <= set(first)
    assert first["valid_resamples"] + first["invalid_resamples"] == 200


def test_midrank_spearman_and_degenerate_bca_are_handled_explicitly():
    tied = median_spearman(
        {"x": np.array([1, 1, 2, 3]), "y": np.array([1, 2, 2, 3])},
        [("x-y", "x", "y")],
    )
    assert tied["median_rho"] == pytest.approx(5 / 6)
    degenerate = bca_interval(
        {"x": np.arange(15.0), "y": np.arange(15.0)},
        [("x-y", "x", "y")],
        n_resamples=200,
        seed=9,
    )
    assert degenerate["status"] == "NOT_IDENTIFIABLE"
    assert degenerate["failure_reason"]


def test_integrity_detects_corrupted_subject_trial_label_mapping():
    subjects, sessions, trials, labels = [], [], [], []
    for session in (1, 2, 3):
        schedule = OFFICIAL_TRIAL_LABELS["SEED"][session]
        for subject in range(1, 16):
            subjects.extend([str(subject)] * 15)
            sessions.extend([session] * 15)
            trials.extend(range(1, 16))
            labels.extend(schedule)
    data = SeedData(X=np.zeros((len(labels), 2)), y=np.asarray(labels),
                    subject=np.asarray(subjects), session=np.asarray(sessions),
                    trial=np.asarray(trials), dataset="SEED")
    assert check_integrity(data)[0]["status"] == "PASS"
    corrupt_y = data.y.copy()
    corrupt_y[15] = 0
    corrupt = SeedData(X=data.X, y=corrupt_y, subject=data.subject,
                       session=data.session, trial=data.trial, dataset=data.dataset)
    report, _ = check_integrity(corrupt)
    assert report["status"] == "FAIL"
    assert report["checks"]["fixed_trial_label_block_mapping"] is False


def test_integrity_rejects_a_common_but_nonofficial_label_schedule():
    subjects, sessions, trials, labels = [], [], [], []
    for session in (1, 2, 3):
        corrupted = list(OFFICIAL_TRIAL_LABELS["SEED"][session])
        corrupted[0] = 0
        for subject in range(1, 16):
            subjects.extend([str(subject)] * 15)
            sessions.extend([session] * 15)
            trials.extend(range(1, 16))
            labels.extend(corrupted)
    data = SeedData(
        X=np.zeros((len(labels), 2)), y=np.asarray(labels), subject=np.asarray(subjects),
        session=np.asarray(sessions), trial=np.asarray(trials), dataset="SEED",
    )
    report, _ = check_integrity(data)
    assert report["status"] == "FAIL"
    assert report["checks"]["fixed_trial_label_block_mapping"] is True
    assert report["checks"]["official_trial_label_schedule"] is False


def test_constant_required_pair_is_not_identifiable_and_permutation_has_no_p_value():
    columns = {"x": np.ones(15), "y": np.arange(15.0)}
    pairs = [("x-y", "x", "y")]
    assert median_spearman(columns, pairs)["status"] == "NOT_IDENTIFIABLE"
    assert bca_interval(columns, pairs, 20, 3)["status"] == "NOT_IDENTIFIABLE"
    perm = pairing_permutation_test(columns, pairs, [("y",)], 20, 3)
    assert perm["status"] == "NOT_IDENTIFIABLE"
    assert perm["p_value"] is None


def test_undefined_cross_correlation_dominates_split_threshold_failure():
    bad_cross = _result(cross=0.8, split=0.1)
    bad_cross["stability"]["cross_session_subject_accuracy"]["status"] = "NOT_IDENTIFIABLE"
    bad_cross["stability"]["cross_session_subject_accuracy"]["median_rho"] = None
    bad_cross["stability"]["cross_session_subject_accuracy"]["pairwise"][0]["rho"] = None
    configurations = {
        "mean_std": {"SEED-IV": bad_cross, "SEED": _result()},
        "mean": {"SEED-IV": _result(), "SEED": _result()},
    }
    assert decide_gate(configurations)["decision"] == "NOT_IDENTIFIABLE"


def test_frozen_permutation_semantics_fix_s1_and_share_all_b_sides():
    cross = permutation_index_maps(["S1", "S2", "S3"], 8, 20, [("S2",), ("S3",)], 4)
    base = np.broadcast_to(np.arange(8), (20, 8))
    assert np.array_equal(cross["S1"], base)
    assert not np.array_equal(cross["S2"], cross["S3"])
    split = permutation_index_maps(["A1", "B1", "A2", "B2", "A3", "B3"], 8, 20,
                                   [("B1", "B2", "B3")], 4)
    assert np.array_equal(split["B1"], split["B2"])
    assert np.array_equal(split["B2"], split["B3"])
    assert np.array_equal(split["A1"], base)


def _result(cross=0.8, split=0.7, p=0.01):
    def stat(value, permutation_p):
        return {"status": "OK", "median_rho": value,
                "pairwise": [{"pair": str(i), "rho": value} for i in range(3)],
                "bca_95_ci": {"status": "OK", "low": value, "high": value,
                               "z0": 0.0, "acceleration": 0.1,
                               "valid_resamples": 20_000, "invalid_resamples": 0,
                               "jackknife_valid": 15, "jackknife_invalid": 0},
                "permutation_test": {"status": "OK", "p_value": permutation_p}}
    return {"integrity": {"status": "PASS"}, "sessions": {str(i): {} for i in (1, 2, 3)},
            "stability": {"cross_session_subject_accuracy": stat(cross, p),
                          "within_session_a_vs_b": stat(split, p)}}


def test_perfect_stable_primary_results_pass_exact_hierarchical_gate():
    configurations = {
        "mean_std": {"SEED-IV": _result(), "SEED": _result()},
        "mean": {"SEED-IV": _result(), "SEED": _result()},
    }
    gate = decide_gate(configurations)
    assert gate["decision"] == "PASS"
    assert gate["may_proceed"] is True


def test_output_guard_cli_defaults_and_schema(tmp_path):
    args = build_parser().parse_args([])
    assert Path(args.output_json) == DEFAULT_JSON
    assert Path(args.output_md) == DEFAULT_MD
    json_path, md_path = tmp_path / "out.json", tmp_path / "out.md"
    guard_output_paths(json_path, md_path)
    json_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        guard_output_paths(json_path, md_path)
    configurations = {
        "mean_std": {"SEED-IV": _result(), "SEED": _result()},
        "mean": {"SEED-IV": _result(), "SEED": _result()},
    }
    document = build_document(configurations, 20_000, 20_000, 20240713)
    assert validate_schema(document) is True
    broken = build_document(configurations, 20_000, 20_000, 20240713)
    broken["configurations"]["mean"]["SEED"]["stability"] = {}
    with pytest.raises(ValueError, match="incomplete stability"):
        validate_schema(broken)
