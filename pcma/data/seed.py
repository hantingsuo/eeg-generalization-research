# pcma/data/seed.py
"""Load SEED-family DE features for cross-subject emotion classification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


DEFAULT_SEED_ROOT = Path(__file__).resolve().parents[2] / "data" / "SEED"


@dataclass
class SeedData:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray
    dataset: str

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject))

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))


@dataclass
class SeedRawData:
    signals: list[np.ndarray]
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray
    dataset: str
    sfreq: float = 200.0

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject))


def aggregate_de_feature(de: np.ndarray, method: str = "mean") -> np.ndarray:
    """Aggregate a trial matrix shaped (channels, windows, bands) to flat features."""
    arr = np.asarray(de, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D DE feature, got shape {arr.shape}")
    if arr.shape[0] != 62 or arr.shape[2] != 5:
        raise ValueError(f"Expected shape (62, windows, 5), got {arr.shape}")
    mean = arr.mean(axis=1).reshape(-1)
    if method == "mean":
        return mean
    if method == "mean_std":
        return np.concatenate([mean, arr.std(axis=1).reshape(-1)])
    raise ValueError(f"Unknown aggregate method: {method}")


def de_windows(de: np.ndarray) -> np.ndarray:
    """Return one flattened 62x5 DE feature row per time window."""
    arr = np.asarray(de, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D DE feature, got shape {arr.shape}")
    if arr.shape[0] != 62 or arr.shape[2] != 5:
        raise ValueError(f"Expected shape (62, windows, 5), got {arr.shape}")
    return np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1)


def _natural_subject_key(path: Path) -> tuple[int, str]:
    m = re.match(r"(\d+)_", path.stem)
    return (int(m.group(1)) if m else 10**9, path.name)


def _trial_keys(mat: dict, prefix: str = "de_LDS") -> list[str]:
    keys = [k for k in mat if re.fullmatch(prefix + r"\d+", k)]
    return sorted(keys, key=lambda k: int(k.replace(prefix, "")))


def _eeg_trial_keys(mat: dict) -> list[str]:
    keys = [k for k in mat if re.fullmatch(r".*eeg\d+", k)]
    return sorted(keys, key=lambda k: int(re.search(r"(\d+)$", k).group(1)))


def _subject_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def _stack_records(records: list[tuple[np.ndarray, int, str, int, int]], dataset: str) -> SeedData:
    if not records:
        raise ValueError(f"No records loaded for {dataset}")
    X, y, subject, session, trial = zip(*records)
    return SeedData(
        X=np.vstack([x.reshape(1, -1) for x in X]).astype(np.float32),
        y=np.asarray(y, dtype=int),
        subject=np.asarray(subject),
        session=np.asarray(session, dtype=int),
        trial=np.asarray(trial, dtype=int),
        dataset=dataset,
    )


def load_seed(
    root: str | Path = DEFAULT_SEED_ROOT,
    feature_dir: str = "ExtractedFeatures_1s",
    aggregate: str = "mean",
    unit: str = "trial",
) -> SeedData:
    """Load SEED de_LDS trials. Labels are read from label.mat."""
    root = Path(root)
    feature_root = root / "SEED" / "SEED" / "SEED_EEG" / feature_dir
    label_path = feature_root / "label.mat"
    labels = loadmat(label_path)["label"].ravel().astype(int).tolist()
    files = sorted(
        [p for p in feature_root.glob("*.mat") if p.name.lower() != "label.mat"],
        key=_natural_subject_key,
    )
    sessions_seen: dict[str, int] = {}
    records: list[tuple[np.ndarray, int, str, int, int]] = []
    for path in files:
        subject = _subject_from_stem(path.stem)
        sessions_seen[subject] = sessions_seen.get(subject, 0) + 1
        session = sessions_seen[subject]
        mat = loadmat(path)
        keys = _trial_keys(mat)
        if len(keys) != len(labels):
            raise ValueError(f"{path.name}: {len(keys)} de_LDS keys but {len(labels)} labels")
        for i, key in enumerate(keys, start=1):
            if unit == "trial":
                records.append((aggregate_de_feature(mat[key], aggregate), labels[i - 1], subject, session, i))
            elif unit == "window":
                for row in de_windows(mat[key]):
                    records.append((row, labels[i - 1], subject, session, i))
            else:
                raise ValueError(f"Unknown unit: {unit}")
    return _stack_records(records, "SEED")


def load_seed_preprocessed(root: str | Path = DEFAULT_SEED_ROOT, sfreq: float = 200.0) -> SeedRawData:
    """Load SEED preprocessed EEG trials as variable-length raw trial matrices."""
    root = Path(root)
    feature_root = root / "SEED" / "SEED" / "SEED_EEG" / "Preprocessed_EEG"
    labels = loadmat(feature_root / "label.mat")["label"].ravel().astype(int).tolist()
    files = sorted(
        [p for p in feature_root.glob("*.mat") if p.name.lower() != "label.mat"],
        key=_natural_subject_key,
    )
    sessions_seen: dict[str, int] = {}
    signals: list[np.ndarray] = []
    y: list[int] = []
    subject: list[str] = []
    session: list[int] = []
    trial: list[int] = []
    for path in files:
        sid = _subject_from_stem(path.stem)
        sessions_seen[sid] = sessions_seen.get(sid, 0) + 1
        sess = sessions_seen[sid]
        mat = loadmat(path)
        keys = _eeg_trial_keys(mat)
        if len(keys) != len(labels):
            raise ValueError(f"{path.name}: {len(keys)} EEG keys but {len(labels)} labels")
        for i, key in enumerate(keys, start=1):
            x = np.asarray(mat[key], dtype=np.float32)
            if x.ndim != 2 or x.shape[0] != 62:
                raise ValueError(f"{path.name} {key}: expected (62, times), got {x.shape}")
            signals.append(x)
            y.append(labels[i - 1])
            subject.append(sid)
            session.append(sess)
            trial.append(i)
    return SeedRawData(
        signals=signals,
        y=np.asarray(y, dtype=int),
        subject=np.asarray(subject),
        session=np.asarray(session, dtype=int),
        trial=np.asarray(trial, dtype=int),
        dataset="SEED",
        sfreq=sfreq,
    )


def raw_band_covariances(
    raw: SeedRawData,
    bands: tuple[tuple[float, float], ...] = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
    shrink: float = 0.01,
    decim: int = 2,
    per_trial_zscore: bool = False,
) -> np.ndarray:
    """Compute empirical SPD covariances per raw trial and band."""
    covs = []
    eye = np.eye(62)
    for x in raw.signals:
        trial = np.asarray(x, dtype=float)
        trial = trial - trial.mean(axis=1, keepdims=True)
        if per_trial_zscore:
            trial = trial / (trial.std(axis=1, keepdims=True) + 1e-8)
        trial_covs = []
        for lo, hi in bands:
            sos = butter(4, [lo / (raw.sfreq / 2), hi / (raw.sfreq / 2)], btype="band", output="sos")
            xf = sosfiltfilt(sos, trial, axis=-1)
            if decim > 1:
                xf = xf[:, ::decim]
            cov = np.cov(xf, bias=False)
            cov = (cov + cov.T) / 2
            tr = float(np.trace(cov)) / cov.shape[0]
            cov = (1.0 - shrink) * cov + shrink * tr * eye
            cov = (cov + cov.T) / 2 + 1e-8 * eye
            trial_covs.append(cov.astype(np.float64))
        covs.append(np.stack(trial_covs, axis=0))
    return np.stack(covs, axis=0)


def parse_seed_iv_labels(readme_path: str | Path) -> dict[int, list[int]]:
    """Parse official SEED-IV ReadMe session labels."""
    text = Path(readme_path).read_text(encoding="utf-8")
    labels: dict[int, list[int]] = {}
    for m in re.finditer(r"session(\d+)_label\s*=\s*\[([^\]]+)\]", text):
        session = int(m.group(1))
        labels[session] = [int(x) for x in re.findall(r"\d+", m.group(2))]
    if set(labels) != {1, 2, 3}:
        raise ValueError(f"Expected labels for sessions 1,2,3 in {readme_path}")
    return labels


def load_seed_iv(root: str | Path = DEFAULT_SEED_ROOT, aggregate: str = "mean", unit: str = "trial") -> SeedData:
    """Load SEED-IV de_LDS trials. Trial labels are parsed from official ReadMe.txt."""
    root = Path(root)
    seed_iv = root / "SEED_IV"
    labels_by_session = parse_seed_iv_labels(seed_iv / "ReadMe.txt")
    records: list[tuple[np.ndarray, int, str, int, int]] = []
    for session in (1, 2, 3):
        feature_root = seed_iv / "eeg_feature_smooth" / str(session)
        labels = labels_by_session[session]
        for path in sorted(feature_root.glob("*.mat"), key=_natural_subject_key):
            subject = _subject_from_stem(path.stem)
            mat = loadmat(path)
            keys = _trial_keys(mat)
            if len(keys) != len(labels):
                raise ValueError(f"{path.name}: {len(keys)} de_LDS keys but {len(labels)} labels")
            for i, key in enumerate(keys, start=1):
                if unit == "trial":
                    records.append((aggregate_de_feature(mat[key], aggregate), labels[i - 1], subject, session, i))
                elif unit == "window":
                    for row in de_windows(mat[key]):
                        records.append((row, labels[i - 1], subject, session, i))
                else:
                    raise ValueError(f"Unknown unit: {unit}")
    return _stack_records(records, "SEED-IV")


def _pickle_loads_npz_scalar(value: np.ndarray):
    payload = value.item() if getattr(value, "shape", None) == () else value
    return pickle.loads(payload)


def load_seed_v(root: str | Path = DEFAULT_SEED_ROOT, aggregate: str = "mean", unit: str = "trial") -> SeedData:
    """Load SEED-V DE features stored as pickled dicts inside npz files."""
    root = Path(root)
    feature_root = root / "SEED-V" / "EEG_DE_features"
    records: list[tuple[np.ndarray, int, str, int, int]] = []
    for path in sorted(feature_root.glob("*.npz"), key=_natural_subject_key):
        subject = _subject_from_stem(path.stem)
        z = np.load(path, allow_pickle=True)
        data = _pickle_loads_npz_scalar(z["data"])
        labels = _pickle_loads_npz_scalar(z["label"])
        for key in sorted(data):
            x = np.asarray(data[key], dtype=float)
            lab = np.asarray(labels[key]).ravel()
            if len(lab) != x.shape[0]:
                raise ValueError(f"{path.name} trial {key}: {len(lab)} labels for {x.shape[0]} windows")
            if len(np.unique(lab.astype(int))) != 1:
                raise ValueError(f"{path.name} trial {key}: label changes within a trial")
            # SEED-V notebook stores each trial as (windows, 310) = flattened 62x5 DE.
            if x.ndim != 2 or x.shape[1] != 310:
                raise ValueError(f"{path.name} trial {key}: expected (windows, 310), got {x.shape}")
            if unit == "window":
                features = x
            elif unit == "trial":
                mean = x.mean(axis=0)
                if aggregate == "mean":
                    features = mean.reshape(1, -1)
                elif aggregate == "mean_std":
                    features = np.concatenate([mean, x.std(axis=0)]).reshape(1, -1)
                else:
                    raise ValueError(f"Unknown aggregate method: {aggregate}")
            else:
                raise ValueError(f"Unknown unit: {unit}")
            session = int(key) // 15 + 1
            trial = int(key) % 15 + 1
            for row in features:
                records.append((row, int(lab[0]), subject, session, trial))
    return _stack_records(records, "SEED-V")


def load_seed_family(
    name: str,
    root: str | Path = DEFAULT_SEED_ROOT,
    aggregate: str = "mean",
    unit: str = "trial",
) -> SeedData:
    key = name.lower().replace("_", "-")
    if key == "seed":
        return load_seed(root=root, aggregate=aggregate, unit=unit)
    if key in {"seed-iv", "seediv"}:
        return load_seed_iv(root=root, aggregate=aggregate, unit=unit)
    if key in {"seed-v", "seedv"}:
        return load_seed_v(root=root, aggregate=aggregate, unit=unit)
    raise ValueError(f"Unknown SEED-family dataset: {name}")
