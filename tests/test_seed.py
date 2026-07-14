import pickle

import numpy as np
from scipy.io import savemat

from pcma.data.seed import (
    aggregate_de_feature,
    de_windows,
    load_seed,
    load_seed_preprocessed,
    load_seed_iv,
    load_seed_v,
    parse_seed_iv_labels,
    raw_band_covariances,
)


def test_aggregate_de_feature_mean_and_mean_std():
    x = np.arange(62 * 4 * 5, dtype=float).reshape(62, 4, 5)
    mean = aggregate_de_feature(x, method="mean")
    mean_std = aggregate_de_feature(x, method="mean_std")

    assert mean.shape == (310,)
    assert mean_std.shape == (620,)
    assert np.allclose(mean, x.mean(axis=1).reshape(-1))
    assert np.allclose(mean_std[:310], mean)
    assert np.allclose(mean_std[310:], x.std(axis=1).reshape(-1))


def test_de_windows_flattens_each_window():
    x = np.arange(62 * 4 * 5, dtype=float).reshape(62, 4, 5)
    win = de_windows(x)
    assert win.shape == (4, 310)
    assert np.allclose(win[0], x[:, 0, :].reshape(-1))
    assert np.allclose(win[3], x[:, 3, :].reshape(-1))


def test_parse_seed_iv_labels_from_official_readme_text(tmp_path):
    readme = tmp_path / "ReadMe.txt"
    readme.write_text(
        """
        session1_label = [1,2,3,0];
        session2_label = [2,1,3,0];
        session3_label = [1,2,2,1];
        """,
        encoding="utf-8",
    )

    labels = parse_seed_iv_labels(readme)
    assert labels[1] == [1, 2, 3, 0]
    assert labels[2] == [2, 1, 3, 0]
    assert labels[3] == [1, 2, 2, 1]


def test_load_seed_synthetic_mat_tree(tmp_path):
    root = tmp_path / "SEED"
    feature_dir = root / "SEED" / "SEED" / "SEED_EEG" / "ExtractedFeatures_1s"
    feature_dir.mkdir(parents=True)
    labels = np.array([[1, 0, -1]])
    savemat(feature_dir / "label.mat", {"label": labels})
    for subject in (1, 2):
        payload = {}
        for trial in (1, 2, 3):
            payload[f"de_LDS{trial}"] = np.full((62, trial + 1, 5), subject + trial, dtype=float)
        savemat(feature_dir / f"{subject}_2020010{subject}.mat", payload)

    data = load_seed(root=root, aggregate="mean")
    assert data.X.shape == (6, 310)
    assert data.y.tolist() == [1, 0, -1, 1, 0, -1]
    assert data.subject.tolist() == ["1", "1", "1", "2", "2", "2"]
    assert data.session.tolist() == [1, 1, 1, 1, 1, 1]
    assert data.trial.tolist() == [1, 2, 3, 1, 2, 3]

    win = load_seed(root=root, unit="window")
    assert win.X.shape == (18, 310)
    assert win.y.tolist() == [1] * 2 + [0] * 3 + [-1] * 4 + [1] * 2 + [0] * 3 + [-1] * 4


def test_load_seed_preprocessed_and_raw_covariances(tmp_path):
    root = tmp_path / "SEED"
    raw_dir = root / "SEED" / "SEED" / "SEED_EEG" / "Preprocessed_EEG"
    raw_dir.mkdir(parents=True)
    savemat(raw_dir / "label.mat", {"label": np.array([[1, 0]])})
    t = np.linspace(0, 2, 400, endpoint=False)
    for subject in (1, 2):
        payload = {
            "abc_eeg1": np.vstack([np.sin(2 * np.pi * (i + 1) * t) for i in range(62)]),
            "abc_eeg2": np.vstack([np.cos(2 * np.pi * (i + 1) * t) for i in range(62)]),
        }
        savemat(raw_dir / f"{subject}_2020010{subject}.mat", payload)

    raw = load_seed_preprocessed(root=root, sfreq=200.0)
    assert len(raw.signals) == 4
    assert raw.y.tolist() == [1, 0, 1, 0]
    assert raw.subject.tolist() == ["1", "1", "2", "2"]
    C = raw_band_covariances(raw, bands=((1.0, 4.0), (4.0, 8.0)), decim=2)
    assert C.shape == (4, 2, 62, 62)
    for cov in C.reshape(-1, 62, 62):
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > 0


def test_load_seed_iv_synthetic_mat_tree(tmp_path):
    root = tmp_path / "SEED"
    seed_iv = root / "SEED_IV"
    (seed_iv / "eeg_feature_smooth" / "1").mkdir(parents=True)
    (seed_iv / "eeg_feature_smooth" / "2").mkdir(parents=True)
    (seed_iv / "eeg_feature_smooth" / "3").mkdir(parents=True)
    (seed_iv / "ReadMe.txt").write_text(
        """
        session1_label = [1,2];
        session2_label = [2,1];
        session3_label = [0,3];
        """,
        encoding="utf-8",
    )
    for session in (1, 2, 3):
        payload = {
            "de_LDS1": np.ones((62, 2, 5), dtype=float) * session,
            "de_LDS2": np.ones((62, 3, 5), dtype=float) * (session + 10),
        }
        savemat(seed_iv / "eeg_feature_smooth" / str(session) / "1_20200101.mat", payload)

    data = load_seed_iv(root=root, aggregate="mean")
    assert data.X.shape == (6, 310)
    assert data.y.tolist() == [1, 2, 2, 1, 0, 3]
    assert data.subject.tolist() == ["1"] * 6
    assert data.session.tolist() == [1, 1, 2, 2, 3, 3]


def test_load_seed_v_synthetic_npz_tree(tmp_path):
    root = tmp_path / "SEED"
    feature_dir = root / "SEED-V" / "EEG_DE_features"
    feature_dir.mkdir(parents=True)
    data = {
        0: np.ones((2, 310), dtype=float),
        15: np.ones((3, 310), dtype=float) * 2,
    }
    label = {
        0: np.array([4, 4], dtype=float),
        15: np.array([2, 2, 2], dtype=float),
    }
    np.savez(feature_dir / "1_123.npz", data=pickle.dumps(data), label=pickle.dumps(label))

    loaded = load_seed_v(root=root, aggregate="mean")
    assert loaded.X.shape == (2, 310)
    assert loaded.y.tolist() == [4, 2]
    assert loaded.subject.tolist() == ["1", "1"]
    assert loaded.session.tolist() == [1, 2]
    assert loaded.trial.tolist() == [1, 1]

    win = load_seed_v(root=root, unit="window")
    assert win.X.shape == (5, 310)
    assert win.y.tolist() == [4, 4, 2, 2, 2]
    assert win.session.tolist() == [1, 1, 2, 2, 2]
