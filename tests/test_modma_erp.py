import numpy as np

from pcma.data.modma_erp import (
    baseline_average_reference,
    egi_centroparietal_cluster,
    erp_feature_names,
    erp_window_features,
)
from pcma.model.erp_gate import gate1_passed, loso_erp_decode


def test_egi_centroparietal_cluster_is_fixed_and_contains_expected_landmarks():
    cluster = egi_centroparietal_cluster()
    assert "E129" in cluster  # raw-file name for HydroCel Cz
    assert "E55" in cluster  # nearest to CPz
    assert "E62" in cluster  # nearest to Pz
    assert len(cluster) >= 8
    assert len(cluster) == len(set(cluster))


def test_baseline_average_reference_and_erp_window_features():
    sfreq = 100.0
    tmin = -0.2
    times = tmin + np.arange(120) / sfreq
    epoch = np.ones((3, 120)) * 5e-6
    epoch[0, (times >= 0.3) & (times < 0.5)] += 2e-6
    epoch[1, (times >= 0.4) & (times < 1.0)] += 4e-6
    corrected = baseline_average_reference(epoch, sfreq, tmin, (-0.2, 0.0))
    assert np.allclose(corrected[:, times < 0].mean(axis=1), 0.0, atol=1e-12)
    feats = erp_window_features(corrected, sfreq, tmin, [0, 1], (("p300", 0.3, 0.5), ("lpp", 0.4, 1.0)))
    names = erp_feature_names(["E1", "E2"], (("p300", 0.3, 0.5), ("lpp", 0.4, 1.0)))
    assert feats.shape == (len(names),)
    assert names == [
        "p300_mean_uv_E1",
        "p300_mean_uv_E2",
        "p300_mean_uv_cluster",
        "lpp_mean_uv_E1",
        "lpp_mean_uv_E2",
        "lpp_mean_uv_cluster",
    ]
    assert np.isfinite(feats).all()


def test_loso_erp_decode_gate_passes_on_subject_general_signal():
    rng = np.random.default_rng(0)
    F, y, subject = [], [], []
    for sid in range(6):
        offset = rng.normal(0, 0.1, size=4)
        for label, mu in zip(("fcue", "hcue", "scue"), (-1.0, 0.0, 1.0)):
            block = rng.normal(0, 0.15, size=(10, 4)) + offset
            block[:, 0] += mu
            F.append(block)
            y.extend([label] * len(block))
            subject.extend([f"S{sid}"] * len(block))
    result = loso_erp_decode(np.vstack(F), np.asarray(y), np.asarray(subject))
    assert result["mean_accuracy"] > 0.8
    assert gate1_passed(result)


def test_gate1_requires_ci_excluding_zero():
    result = {
        "mean_accuracy_minus_majority": 0.01,
        "accuracy_minus_majority_bootstrap_ci": {"ci_low": -0.01, "ci_high": 0.03},
        "wilcoxon_accuracy_gt_majority": {"p_value": 0.01},
    }
    assert not gate1_passed(result)
