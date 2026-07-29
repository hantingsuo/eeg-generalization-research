"""Small, auditable SEED adapters for protocol-faithful LibEER checks.

The upstream LibEER loader eagerly reads every SEED file before applying its
``pr`` filter.  That is unnecessary for a one-subject smoke test and can exceed
the memory available on the development machine.  These helpers load only the
official LDS-DE variables needed by a requested subject/session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SEED_SESSION_FILES: tuple[tuple[str, ...], ...] = (
    (
        "1_20131027.mat", "2_20140404.mat", "3_20140603.mat",
        "4_20140621.mat", "5_20140411.mat", "6_20130712.mat",
        "7_20131027.mat", "8_20140511.mat", "9_20140620.mat",
        "10_20131130.mat", "11_20140618.mat", "12_20131127.mat",
        "13_20140527.mat", "14_20140601.mat", "15_20130709.mat",
    ),
    (
        "1_20131030.mat", "2_20140413.mat", "3_20140611.mat",
        "4_20140702.mat", "5_20140418.mat", "6_20131016.mat",
        "7_20131030.mat", "8_20140514.mat", "9_20140627.mat",
        "10_20131204.mat", "11_20140625.mat", "12_20131201.mat",
        "13_20140603.mat", "14_20140615.mat", "15_20131016.mat",
    ),
    (
        "1_20131107.mat", "2_20140419.mat", "3_20140629.mat",
        "4_20140705.mat", "5_20140506.mat", "6_20131113.mat",
        "7_20131106.mat", "8_20140521.mat", "9_20140704.mat",
        "10_20131211.mat", "11_20140630.mat", "12_20131207.mat",
        "13_20140610.mat", "14_20140627.mat", "15_20131105.mat",
    ),
)


@dataclass(frozen=True)
class SeedSubjectWindows:
    """One SEED subject/session represented as trial-level LDS-DE windows."""

    trials: tuple[np.ndarray, ...]
    trial_labels: np.ndarray
    source_file: Path
    label_file: Path


@dataclass(frozen=True)
class SeedWindowRows:
    """Flattened windows with explicit subject/session/trial provenance."""

    features: np.ndarray
    labels: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


def seed_de_lds_variable_names(num_trials: int = 15) -> tuple[str, ...]:
    """Return the official MATLAB variable names without relying on key order."""

    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    return tuple(f"de_LDS{trial}" for trial in range(1, num_trials + 1))


def resolve_seed_subject_file(feature_root: Path, session: int, subject: int) -> Path:
    """Resolve a 1-based SEED session/subject to its official feature file."""

    if not 1 <= session <= len(SEED_SESSION_FILES):
        raise ValueError("session must be in 1..3")
    if not 1 <= subject <= len(SEED_SESSION_FILES[session - 1]):
        raise ValueError("subject must be in 1..15")
    return feature_root / SEED_SESSION_FILES[session - 1][subject - 1]


def load_seed_de_lds_subject(
    feature_root: str | Path,
    *,
    session: int,
    subject: int,
) -> SeedSubjectWindows:
    """Load only the 15 LDS-DE variables for one official SEED recording.

    Each returned trial has shape ``(windows, 62, 5)``.  Labels are mapped from
    the official ``{-1, 0, 1}`` coding to ``{0, 1, 2}``, matching LibEER.
    """

    root = Path(feature_root).resolve()
    source_file = resolve_seed_subject_file(root, session, subject)
    label_file = root / "label.mat"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    if not label_file.is_file():
        raise FileNotFoundError(label_file)

    names = seed_de_lds_variable_names()
    payload = loadmat(source_file, variable_names=list(names))
    missing = [name for name in names if name not in payload]
    if missing:
        raise KeyError(f"missing LDS-DE variables: {missing}")

    trials: list[np.ndarray] = []
    for name in names:
        raw = np.asarray(payload[name])
        if raw.ndim != 3 or raw.shape[0] != 62 or raw.shape[2] != 5:
            raise ValueError(f"unexpected {name} shape: {raw.shape}")
        trials.append(np.ascontiguousarray(raw.transpose(1, 0, 2), dtype=np.float32))

    labels_raw = np.asarray(loadmat(label_file, variable_names=["label"])["label"]).reshape(-1)
    if labels_raw.shape != (15,):
        raise ValueError(f"unexpected label shape: {labels_raw.shape}")
    labels = labels_raw.astype(np.int64) + 1
    if set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError(f"unexpected mapped labels: {np.unique(labels).tolist()}")

    return SeedSubjectWindows(
        trials=tuple(trials),
        trial_labels=labels,
        source_file=source_file,
        label_file=label_file,
    )


def make_clean_front_back_split(
    subject_data: SeedSubjectWindows,
    *,
    validation_trial: int = 9,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create a leakage-free smoke split from the conventional 9/6 protocol.

    Trials 1--9 form the development side and trials 10--15 are untouched test
    trials.  One predeclared development trial is used for checkpoint selection;
    the other eight train the model.  This is deliberately *not* claimed as an
    exact reproduction of the upstream score.
    """

    if not 1 <= validation_trial <= 9:
        raise ValueError("validation_trial must be one of the first nine trials")

    train_ids = [i for i in range(9) if i != validation_trial - 1]
    val_ids = [validation_trial - 1]
    test_ids = list(range(9, 15))

    def flatten(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([subject_data.trials[i] for i in ids], axis=0)
        y = np.concatenate(
            [np.full(len(subject_data.trials[i]), subject_data.trial_labels[i], dtype=np.int64) for i in ids]
        )
        return x, y

    return {
        "train": flatten(train_ids),
        "validation": flatten(val_ids),
        "test": flatten(test_ids),
    }


def make_fixed_front_back_split(
    subject_data: SeedSubjectWindows,
    *,
    front: int = 9,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create a fixed-epoch front/back split with no validation/test reuse.

    The returned third array is a one-based trial identifier for aggregation.
    The conventional SEED setting uses trials 1--9 for training and 10--15 for
    a single final test evaluation.
    """

    if not 1 <= front < len(subject_data.trials):
        raise ValueError("front must leave at least one train and one test trial")

    def flatten(ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.concatenate([subject_data.trials[i] for i in ids], axis=0)
        labels = np.concatenate(
            [np.full(len(subject_data.trials[i]), subject_data.trial_labels[i], dtype=np.int64) for i in ids]
        )
        trials = np.concatenate(
            [np.full(len(subject_data.trials[i]), i + 1, dtype=np.int16) for i in ids]
        )
        return features, labels, trials

    return {
        "train": flatten(list(range(front))),
        "test": flatten(list(range(front, len(subject_data.trials)))),
    }


def one_hot(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Convert integer labels to LibEER-compatible floating one-hot targets."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels outside requested class range")
    return np.eye(num_classes, dtype=np.float32)[labels]


def flatten_seed_subject_windows(
    subject_data: SeedSubjectWindows,
    *,
    subject: int,
    session: int,
) -> SeedWindowRows:
    """Flatten one recording while retaining the clustering identifiers."""

    if subject <= 0 or session <= 0:
        raise ValueError("subject and session identifiers must be positive")
    features = np.concatenate(subject_data.trials, axis=0)
    labels = np.concatenate(
        [
            np.full(len(trial), subject_data.trial_labels[index], dtype=np.int64)
            for index, trial in enumerate(subject_data.trials)
        ]
    )
    trials = np.concatenate(
        [np.full(len(trial), index + 1, dtype=np.int16) for index, trial in enumerate(subject_data.trials)]
    )
    return SeedWindowRows(
        features=features,
        labels=labels,
        subject=np.full(len(labels), subject, dtype=np.int16),
        session=np.full(len(labels), session, dtype=np.int8),
        trial=trials,
    )
