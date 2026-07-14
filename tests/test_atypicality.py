import numpy as np

from pcma.model.atypicality import group_contrast, severity_correlations, subject_atypicality


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    F, y, subject, population = [], [], [], []
    for s in range(4):
        for label in (0, 1):
            block = rng.normal(0, 0.2, size=(8, 12))
            block[:, label * 3 : label * 3 + 3] += 1.0
            F.append(block)
            y.extend([label] * len(block))
            subject.extend([f"HC{s}"] * len(block))
            population.extend(["HC"] * len(block))
    for s in range(3):
        for label in (0, 1):
            block = rng.normal(0, 0.2, size=(8, 12))
            block[:, label * 3 : label * 3 + 3] += 0.25
            block[:, 6:] += 1.5
            F.append(block)
            y.extend([label] * len(block))
            subject.extend([f"DEP{s}"] * len(block))
            population.extend(["DEP"] * len(block))
    return np.vstack(F), np.asarray(y), np.asarray(subject), np.asarray(population)


def test_subject_atypicality_scores_subject_rows_and_dep_higher():
    F, y, subject, population = _toy()
    result = subject_atypicality(F, y, subject, population, max_components=5)
    rows = result["subjects"]
    assert len(rows) == 7
    assert all("atypicality" in r for r in rows)
    contrast = group_contrast(rows, n_perm=200, seed=1)
    assert contrast["target_mean"] > contrast["reference_mean"]
    assert contrast["cliffs_delta"] > 0


def test_subject_atypicality_hc_scaling_centers_hc_components():
    F, y, subject, population = _toy(2)
    rows = subject_atypicality(F, y, subject, population, max_components=4)["subjects"]
    hc = [r for r in rows if r["population"] == "HC"]
    for key in ("manifold_distance_z_hc", "classifier_error_z_hc", "low_confidence_z_hc"):
        assert abs(np.mean([r[key] for r in hc])) < 1e-7


def test_severity_correlations_positive_signal():
    rows = [
        {"subject": "a", "atypicality": 0.0},
        {"subject": "b", "atypicality": 1.0},
        {"subject": "c", "atypicality": 2.0},
        {"subject": "d", "atypicality": 3.0},
    ]
    sev = {"a": 1, "b": 2, "c": 3, "d": 4}
    out = severity_correlations(rows, sev, n_perm=100, seed=3)
    assert out["spearman_r"] > 0.99
    assert out["pearson_r"] > 0.99
    assert out["spearman_bootstrap_ci"]["ci_low"] > 0.99


def test_subject_atypicality_can_use_logreg_classifier():
    F, y, subject, population = _toy(4)
    result = subject_atypicality(F, y, subject, population, max_components=5, classifier="logreg")
    assert result["classifier"] == "logreg"
    assert {"manifold_distance", "classifier_error", "low_confidence"} == set(result["components"])


def test_subject_atypicality_allows_missing_emotion_labels():
    F, _, subject, population = _toy(5)
    result = subject_atypicality(F, None, subject, population, max_components=5)
    rows = result["subjects"]
    assert result["components"] == ["manifold_distance"]
    assert "classifier_error" not in rows[0]
    assert all("atypicality" in row for row in rows)
