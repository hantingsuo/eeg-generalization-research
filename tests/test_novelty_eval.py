import numpy as np

from pcma.model.novelty_eval import aggregate_proba_by_group, align_proba, keep_from_gate, pred_from_proba


def test_align_proba_and_pred_from_proba():
    proba = np.array([[0.2, 0.8], [0.7, 0.3]])
    aligned = align_proba(proba, classes=np.array([1, 0]), target_classes=(0, 1))
    assert np.allclose(aligned, np.array([[0.8, 0.2], [0.3, 0.7]]))
    assert pred_from_proba(aligned, classes=(0, 1)).tolist() == [0, 1]


def test_aggregate_proba_by_group_checks_constants():
    out = aggregate_proba_by_group(
        y_true=np.array([1, 1, 0, 0]),
        proba=np.array([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]]),
        classes=np.array([0, 1]),
        group_id=np.array(["a", "a", "b", "b"]),
        subject=np.array(["S1", "S1", "S2", "S2"]),
        population=np.array(["DEP", "DEP", "HC", "HC"]),
    )
    assert out["y"].tolist() == [1, 0]
    assert out["pred"].tolist() == [1, 0]
    assert out["subject"].tolist() == ["S1", "S2"]


def test_keep_from_gate_requires_positive_ci_or_p3_sig():
    keep, _ = keep_from_gate({"ci_low": 0.01, "ci_high": 0.1})
    assert keep
    drop, _ = keep_from_gate(
        {"ci_low": -0.01, "ci_high": 0.1},
        {"wilcoxon_method_vs_baseline_p": 0.2, "method_video_mean": 0.8, "baseline_video_mean": 0.7},
    )
    assert not drop
