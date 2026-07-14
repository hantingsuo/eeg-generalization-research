import numpy as np

from pcma.model.seed_da import (
    aggregate_scores_by_group,
    fit_predict_seed_coral_svm,
    fit_predict_seed_svm_proba,
    select_evenly_per_group,
    trial_group_keys,
)
from pcma.model.seed_spd import euclidean_align_bands, fit_predict_ea_logsvm_bands, logeuclid_band_features


def _make_shifted(seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr, str_ = [], [], []
    Xte, yte, ste = [], [], []
    for subject in range(3):
        for label in (0, 1, 2):
            block = rng.normal(0, 0.25, size=(20, 12))
            block[:, label * 4 : (label + 1) * 4] += 1.5
            block += subject * 0.05
            Xtr.append(block)
            ytr.extend([label] * len(block))
            str_.extend([f"S{subject}"] * len(block))
    for label in (0, 1, 2):
        block = rng.normal(0, 0.25, size=(20, 12))
        block[:, label * 4 : (label + 1) * 4] += 1.5
        block = block @ np.diag([1.5, 0.7, 1.1, 1.0, 1.3, 0.8, 1.2, 0.9, 1.4, 0.75, 1.15, 0.85])
        Xte.append(block)
        yte.extend([label] * len(block))
        ste.extend(["T0"] * len(block))
    return np.vstack(Xtr), np.array(ytr), np.array(str_), np.vstack(Xte), np.array(yte), np.array(ste)


def test_fit_predict_seed_coral_svm_valid_multiclass():
    Xtr, ytr, str_, Xte, yte, ste = _make_shifted()
    pred = fit_predict_seed_coral_svm(Xtr, ytr, str_, Xte, ste, C=1.0)
    assert pred.shape == yte.shape
    assert set(np.unique(pred)) <= {0, 1, 2}
    assert (pred == yte).mean() > 0.8


def test_fit_predict_seed_svm_proba_rows_sum_to_one():
    Xtr, ytr, str_, Xte, yte, ste = _make_shifted(1)
    pred, proba, classes = fit_predict_seed_svm_proba(Xtr, ytr, str_, Xte, ste, C=1.0)
    assert pred.shape == yte.shape
    assert proba.shape == (len(yte), len(classes))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_trial_group_sampling_and_score_aggregation():
    groups = trial_group_keys(["S1"] * 6 + ["S2"] * 3, [1] * 9, [1] * 6 + [2] * 3)
    picked = select_evenly_per_group(groups, max_per_group=3)
    assert picked.tolist() == [0, 2, 5, 6, 7, 8]

    scores = np.array([
        [2.0, 1.0],
        [3.0, 0.0],
        [1.0, 4.0],
        [0.0, 5.0],
    ])
    y, pred, out_groups = aggregate_scores_by_group(
        scores,
        classes=np.array([0, 1]),
        groups=np.array(["a", "a", "b", "b"]),
        y_true=np.array([0, 0, 1, 1]),
    )
    assert y.tolist() == [0, 1]
    assert pred.tolist() == [0, 1]
    assert out_groups.tolist() == ["a", "b"]


def test_euclidean_align_and_logsvm_bands_valid():
    rng = np.random.default_rng(4)
    C, y, subject = [], [], []
    for s in range(3):
        for label in (0, 1):
            for _ in range(10):
                bands = []
                for b in range(2):
                    A = rng.normal(size=(5, 5)) + (0.8 if label and b == 1 else 0.0) + 0.2 * s
                    bands.append(A @ A.T + 5 * np.eye(5))
                C.append(np.stack(bands))
                y.append(label)
                subject.append(str(s))
    C = np.asarray(C)
    y = np.asarray(y)
    subject = np.asarray(subject)
    aligned = euclidean_align_bands(C, subject)
    assert aligned.shape == C.shape
    feats = logeuclid_band_features(aligned)
    assert feats.shape == (len(C), 2 * 15)
    train = subject != "0"
    pred = fit_predict_ea_logsvm_bands(C[train], y[train], subject[train], C[~train], subject[~train])
    assert pred.shape == y[~train].shape
