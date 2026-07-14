import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

import pcma.data.libeer_seediv as seediv


FULL_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"


class _FakePreprocess:
    def __init__(self):
        self.calls = {"bandpass_filter": 0, "de_extraction": 0, "lds": 0}

    def bandpass_filter(self, wrapped, sample_rate, pass_band):
        self.calls["bandpass_filter"] += 1
        raw = wrapped[0][0][0]
        assert raw.dtype == np.float64
        assert sample_rate == 200
        assert pass_band == [0.3, 50.0]
        return wrapped

    def de_extraction(self, raw, sample_rate, bands, time_window, overlap):
        self.calls["de_extraction"] += 1
        assert raw.dtype == np.float64
        assert sample_rate == 200
        assert bands == [list(band) for band in seediv.EXTRACT_BANDS]
        assert time_window == 1.0
        assert overlap == 0.0
        # The synthetic raw first column is a sentinel; raw[:, 1:] must make
        # the requested numeric trial identifier the first retained sample.
        return np.full((2, 62, 5), raw[0, 0], dtype=np.float64)

    def lds(self, features):
        self.calls["lds"] += 1
        assert features.dtype == np.float64
        return features + 0.25


def _make_synthetic_recording(tmp_path: Path, *, session: int = 2, subject: int = 1) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    source = (
        dataset_root
        / "eeg_raw_data"
        / str(session)
        / seediv.SEEDIV_SESSION_FILES[session - 1][subject - 1]
    )
    source.parent.mkdir(parents=True)

    order = [10, 2, 24, 1] + [trial for trial in range(1, 25) if trial not in {1, 2, 10, 24}]
    payload = {"unrelated": np.array([[123]])}
    for trial in order:
        raw = np.full((62, 202), trial, dtype=np.float32)
        raw[:, 0] = -999.0
        payload[f"synthetic_eeg{trial}"] = raw
    savemat(source, payload)

    libeer_root = tmp_path / "libeer"
    libeer_root.mkdir()
    return dataset_root, libeer_root


def test_official_seediv_tables_and_all_three_label_sequences():
    assert tuple(map(len, seediv.SEEDIV_SESSION_FILES)) == (15, 15, 15)
    assert seediv.SEEDIV_SESSION_FILES[0][0] == "1_20160518.mat"
    assert seediv.SEEDIV_SESSION_FILES[2][-1] == "15_20150527.mat"
    assert seediv.SEEDIV_SESSION_LABELS == (
        (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
        (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
        (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
    )


def test_streaming_loader_sorts_trials_and_writes_provenance_cache(tmp_path, monkeypatch):
    dataset_root, libeer_root = _make_synthetic_recording(tmp_path)
    fake = _FakePreprocess()
    monkeypatch.setattr(seediv, "_git_commit", lambda _root: FULL_COMMIT)
    monkeypatch.setattr(seediv, "_load_libeer_preprocess", lambda _root: fake)

    loaded = seediv.load_seediv_de_lds_subject(
        dataset_root,
        libeer_root,
        session=2,
        subject=1,
        cache_root=tmp_path / "cache",
    )

    assert not loaded.loaded_from_cache
    assert len(loaded.trials) == 24
    assert all(trial.shape == (2, 62, 5) for trial in loaded.trials)
    assert all(trial.dtype == np.float32 for trial in loaded.trials)
    assert [float(trial[0, 0, 0]) for trial in loaded.trials] == [
        trial + 0.25 for trial in range(1, 25)
    ]
    np.testing.assert_array_equal(loaded.labels, seediv.SEEDIV_SESSION_LABELS[1])
    assert fake.calls == {"bandpass_filter": 24, "de_extraction": 24, "lds": 24}

    assert loaded.cache_file is not None and loaded.cache_file.is_file()
    assert loaded.cache_manifest_file is not None and loaded.cache_manifest_file.is_file()
    manifest = json.loads(loaded.cache_manifest_file.read_text(encoding="utf-8"))
    assert len(manifest["raw_sha256"]) == 64
    assert manifest["raw_size_bytes"] == loaded.source_file.stat().st_size
    assert manifest["raw_mtime_ns"] == loaded.source_file.stat().st_mtime_ns
    assert manifest["libeer_commit"] == FULL_COMMIT
    assert manifest["parameters"]["feature_type"] == "de_lds"
    assert manifest["parameters"]["internal_dtype"] == "float64"
    assert manifest["parameters"]["output_dtype"] == "float32"
    assert manifest["trial_shapes"] == [[2, 62, 5]] * 24
    assert manifest["raw_trial_variables"][0] == "synthetic_eeg1"
    assert manifest["raw_trial_variables"][-1] == "synthetic_eeg24"

    cached = seediv.load_seediv_de_lds_subject(
        dataset_root,
        libeer_root,
        session=2,
        subject=1,
        cache_root=tmp_path / "cache",
    )
    assert cached.loaded_from_cache
    assert fake.calls == {"bandpass_filter": 24, "de_extraction": 24, "lds": 24}
    np.testing.assert_array_equal(cached.trials[9], loaded.trials[9])


def test_flatten_seediv_subject_windows_retains_subject_session_and_trial(tmp_path, monkeypatch):
    dataset_root, libeer_root = _make_synthetic_recording(tmp_path, session=3, subject=1)
    fake = _FakePreprocess()
    monkeypatch.setattr(seediv, "_git_commit", lambda _root: FULL_COMMIT)
    monkeypatch.setattr(seediv, "_load_libeer_preprocess", lambda _root: fake)
    loaded = seediv.load_seediv_de_lds_subject(
        dataset_root,
        libeer_root,
        session=3,
        subject=1,
    )

    rows = seediv.flatten_seediv_subject_windows(loaded)
    assert rows.features.shape == (48, 62, 5)
    assert rows.features.dtype == np.float32
    assert set(rows.subject) == {1}
    assert set(rows.session) == {3}
    assert rows.trial[:2].tolist() == [1, 1]
    assert rows.trial[-2:].tolist() == [24, 24]
    assert rows.labels[:4].tolist() == [1, 1, 2, 2]
