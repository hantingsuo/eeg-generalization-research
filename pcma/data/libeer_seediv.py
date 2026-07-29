"""Auditable, streaming SEED-IV raw-to-LDS-DE loading for LibEER.

LibEER's stock SEED-IV loader reads all 45 recordings before preprocessing and
uses MATLAB key insertion order to identify trials.  This adapter resolves one
official subject/session file, discovers the 24 ``*_eegN`` variables by name,
and preprocesses one trial at a time with the functions from a pinned LibEER
checkout.  An optional cache retains both the features and their provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import scipy
from scipy.io import loadmat, whosmat

from pcma.data.libeer_seed import SeedWindowRows


SEEDIV_SESSION_FILES: tuple[tuple[str, ...], ...] = (
    (
        "1_20160518.mat", "2_20150915.mat", "3_20150919.mat",
        "4_20151111.mat", "5_20160406.mat", "6_20150507.mat",
        "7_20150715.mat", "8_20151103.mat", "9_20151028.mat",
        "10_20151014.mat", "11_20150916.mat", "12_20150725.mat",
        "13_20151115.mat", "14_20151205.mat", "15_20150508.mat",
    ),
    (
        "1_20161125.mat", "2_20150920.mat", "3_20151018.mat",
        "4_20151118.mat", "5_20160413.mat", "6_20150511.mat",
        "7_20150717.mat", "8_20151110.mat", "9_20151119.mat",
        "10_20151021.mat", "11_20150921.mat", "12_20150804.mat",
        "13_20151125.mat", "14_20151208.mat", "15_20150514.mat",
    ),
    (
        "1_20161126.mat", "2_20151012.mat", "3_20151101.mat",
        "4_20151123.mat", "5_20160420.mat", "6_20150512.mat",
        "7_20150721.mat", "8_20151117.mat", "9_20151209.mat",
        "10_20151023.mat", "11_20151011.mat", "12_20150807.mat",
        "13_20161130.mat", "14_20151215.mat", "15_20150527.mat",
    ),
)

SEEDIV_SESSION_LABELS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
    (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
    (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0),
)

SAMPLE_RATE = 200
PASS_BAND = (0.3, 50.0)
EXTRACT_BANDS: tuple[tuple[float, float], ...] = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 14.0),
    (14.0, 30.0),
    (30.0, 50.0),
)
TIME_WINDOW_SECONDS = 1.0
OVERLAP_SECONDS = 0.0
_CACHE_FORMAT_VERSION = 1
_TRIAL_VARIABLE = re.compile(r"^.+_eeg(\d+)$")


@dataclass(frozen=True)
class SeedIVSubjectWindows:
    """One SEED-IV subject/session represented by 1 s LDS-DE windows."""

    trials: tuple[np.ndarray, ...]
    trial_labels: np.ndarray
    source_file: Path
    session: int
    subject: int
    loaded_from_cache: bool
    cache_file: Path | None
    cache_manifest_file: Path | None
    manifest: dict[str, Any]

    @property
    def labels(self) -> np.ndarray:
        """Alias retained for consumers that call trial labels ``labels``."""

        return self.trial_labels

    @property
    def provenance(self) -> dict[str, Any]:
        """Cache/raw provenance used to create these windows."""

        return self.manifest


def resolve_seediv_subject_file(dataset_root: str | Path, session: int, subject: int) -> Path:
    """Resolve a 1-based session/subject using the official SEED-IV file table."""

    if not 1 <= session <= len(SEEDIV_SESSION_FILES):
        raise ValueError("session must be in 1..3")
    if not 1 <= subject <= len(SEEDIV_SESSION_FILES[session - 1]):
        raise ValueError("subject must be in 1..15")

    root = Path(dataset_root).expanduser().resolve()
    relative = Path(str(session)) / SEEDIV_SESSION_FILES[session - 1][subject - 1]
    candidates = (root / "eeg_raw_data" / relative, root / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[0])


def _discover_seediv_trial_variables(source_file: Path) -> tuple[str, ...]:
    """Return exactly trials 1..24 in numeric order, independent of MAT order."""

    by_trial: dict[int, str] = {}
    for name, _shape, _matlab_type in whosmat(source_file):
        match = _TRIAL_VARIABLE.fullmatch(name)
        if match is None:
            continue
        trial = int(match.group(1))
        if not 1 <= trial <= 24:
            continue
        if trial in by_trial:
            raise ValueError(
                f"duplicate SEED-IV trial {trial}: {by_trial[trial]!r} and {name!r}"
            )
        by_trial[trial] = name

    missing = sorted(set(range(1, 25)) - set(by_trial))
    if missing:
        raise KeyError(f"missing SEED-IV raw trial variables: {missing}")
    return tuple(by_trial[trial] for trial in range(1, 25))


def _git_commit(libeer_root: Path) -> str:
    safe_directory = libeer_root.resolve().as_posix()
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={safe_directory}",
            "-C",
            str(libeer_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"unexpected LibEER commit identifier: {commit!r}")
    return commit


def _load_libeer_preprocess(libeer_root: Path) -> ModuleType:
    root = libeer_root.expanduser().resolve()
    candidates = (
        root / "LibEER" / "data_utils" / "preprocess.py",
        root / "data_utils" / "preprocess.py",
    )
    module_file = next((path for path in candidates if path.is_file()), None)
    if module_file is None:
        raise FileNotFoundError(candidates[0])

    module_name = "_paper_eeg_libeer_preprocess_" + hashlib.sha256(
        str(module_file).encode("utf-8")
    ).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import LibEER preprocessing from {module_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for function_name in ("bandpass_filter", "de_extraction", "lds"):
        if not callable(getattr(module, function_name, None)):
            raise AttributeError(f"LibEER preprocessing lacks {function_name}")
    return module


def _sha256_stable(path: Path) -> tuple[str, int, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source file changed while hashing: {path}")
    return digest.hexdigest(), after.st_size, after.st_mtime_ns


def _base_manifest(
    source_file: Path,
    *,
    session: int,
    subject: int,
    libeer_commit: str,
    raw_sha256: str,
    raw_size: int,
    raw_mtime_ns: int,
    trial_variables: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "dataset": "SEED-IV",
        "session": session,
        "subject": subject,
        "source_file": str(source_file),
        "raw_sha256": raw_sha256,
        "raw_size_bytes": raw_size,
        "raw_mtime_ns": raw_mtime_ns,
        "raw_trial_variables": list(trial_variables),
        "libeer_commit": libeer_commit,
        "parameters": {
            "sample_rate_hz": SAMPLE_RATE,
            "drop_first_sample": True,
            "pass_band_hz": list(PASS_BAND),
            "extract_bands_hz": [list(band) for band in EXTRACT_BANDS],
            "time_window_seconds": TIME_WINDOW_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS,
            "feature_type": "de_lds",
            "internal_dtype": "float64",
            "output_dtype": "float32",
            "functions": ["bandpass_filter", "de_extraction", "lds"],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
    }


def _manifest_matches(cached: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(cached.get(key) == value for key, value in expected.items())


def _load_cache(
    cache_file: Path,
    manifest_file: Path,
    expected_manifest: Mapping[str, Any],
) -> tuple[tuple[np.ndarray, ...], np.ndarray, dict[str, Any]] | None:
    if not cache_file.is_file() or not manifest_file.is_file():
        return None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not _manifest_matches(manifest, expected_manifest):
            return None
        with np.load(cache_file, allow_pickle=False) as cached:
            labels = np.asarray(cached["trial_labels"], dtype=np.int64)
            trials = tuple(
                np.ascontiguousarray(cached[f"trial_{trial:02d}"], dtype=np.float32)
                for trial in range(1, 25)
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None

    if labels.shape != (24,):
        return None
    if any(array.ndim != 3 or array.shape[1:] != (62, 5) for array in trials):
        return None
    expected_shapes = manifest.get("trial_shapes")
    if expected_shapes != [list(array.shape) for array in trials]:
        return None
    return trials, labels, manifest


def _write_cache(
    cache_file: Path,
    manifest_file: Path,
    trials: tuple[np.ndarray, ...],
    labels: np.ndarray,
    manifest: Mapping[str, Any],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid4().hex
    cache_temp = cache_file.with_name(f".{cache_file.name}.{nonce}.tmp")
    manifest_temp = manifest_file.with_name(f".{manifest_file.name}.{nonce}.tmp")
    payload = {f"trial_{index:02d}": trial for index, trial in enumerate(trials, start=1)}
    payload["trial_labels"] = labels
    try:
        with cache_temp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        manifest_temp.write_text(
            json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cache_temp.replace(cache_file)
        manifest_temp.replace(manifest_file)
    finally:
        cache_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)


def load_seediv_de_lds_subject(
    dataset_root: str | Path,
    libeer_root: str | Path,
    session: int,
    subject: int,
    cache_root: str | Path | None = None,
    expected_commit: str = "39dc27e",
) -> SeedIVSubjectWindows:
    """Stream one official SEED-IV recording through LibEER's 1 s DE+LDS.

    Raw MATLAB variables are loaded and processed one trial at a time.  All
    numerical work enters LibEER as ``float64`` and the retained arrays are
    contiguous ``float32`` with shape ``(windows, 62, 5)``.
    """

    source_file = resolve_seediv_subject_file(dataset_root, session, subject)
    trial_variables = _discover_seediv_trial_variables(source_file)
    libeer_path = Path(libeer_root).expanduser().resolve()
    libeer_commit = _git_commit(libeer_path)
    normalized_expected = expected_commit.strip().lower()
    if normalized_expected and not libeer_commit.startswith(normalized_expected):
        raise RuntimeError(
            f"LibEER commit mismatch: expected {expected_commit}, found {libeer_commit}"
        )

    raw_sha256, raw_size, raw_mtime_ns = _sha256_stable(source_file)
    base_manifest = _base_manifest(
        source_file,
        session=session,
        subject=subject,
        libeer_commit=libeer_commit,
        raw_sha256=raw_sha256,
        raw_size=raw_size,
        raw_mtime_ns=raw_mtime_ns,
        trial_variables=trial_variables,
    )

    cache_file: Path | None = None
    manifest_file: Path | None = None
    if cache_root is not None:
        cache_dir = Path(cache_root).expanduser().resolve()
        cache_file = cache_dir / f"seediv_de_lds_s{session:02d}_sub{subject:02d}.npz"
        manifest_file = cache_file.with_suffix(".json")
        cached = _load_cache(cache_file, manifest_file, base_manifest)
        if cached is not None:
            trials, labels, manifest = cached
            return SeedIVSubjectWindows(
                trials=trials,
                trial_labels=labels,
                source_file=source_file,
                session=session,
                subject=subject,
                loaded_from_cache=True,
                cache_file=cache_file,
                cache_manifest_file=manifest_file,
                manifest=manifest,
            )

    preprocess = _load_libeer_preprocess(libeer_path)
    trials_list: list[np.ndarray] = []
    extract_bands = [list(band) for band in EXTRACT_BANDS]
    for trial_number, variable_name in enumerate(trial_variables, start=1):
        payload = loadmat(source_file, variable_names=[variable_name])
        if variable_name not in payload:
            raise KeyError(f"missing {variable_name!r} while loading trial {trial_number}")
        raw = np.asarray(payload[variable_name])
        if raw.ndim != 2 or raw.shape[0] != 62:
            raise ValueError(f"unexpected {variable_name} shape: {raw.shape}")
        if raw.shape[1] <= 1:
            raise ValueError(f"{variable_name} has no samples after raw[:, 1:]")

        trial_raw = np.ascontiguousarray(raw[:, 1:], dtype=np.float64)
        wrapped = [[[trial_raw]]]
        filtered = preprocess.bandpass_filter(wrapped, SAMPLE_RATE, list(PASS_BAND))[0][0][0]
        de = preprocess.de_extraction(
            np.asarray(filtered, dtype=np.float64),
            SAMPLE_RATE,
            extract_bands,
            TIME_WINDOW_SECONDS,
            OVERLAP_SECONDS,
        )
        smoothed = preprocess.lds(np.asarray(de, dtype=np.float64))
        trial = np.ascontiguousarray(smoothed, dtype=np.float32)
        if trial.ndim != 3 or trial.shape[1:] != (62, 5):
            raise ValueError(f"unexpected processed trial {trial_number} shape: {trial.shape}")
        if trial.shape[0] == 0:
            raise ValueError(f"processed trial {trial_number} has no complete 1 s windows")
        trials_list.append(trial)

    trials = tuple(trials_list)
    labels = np.asarray(SEEDIV_SESSION_LABELS[session - 1], dtype=np.int64)
    manifest = dict(base_manifest)
    manifest["trial_shapes"] = [list(trial.shape) for trial in trials]
    manifest["trial_labels"] = labels.tolist()
    if cache_file is not None and manifest_file is not None:
        _write_cache(cache_file, manifest_file, trials, labels, manifest)

    return SeedIVSubjectWindows(
        trials=trials,
        trial_labels=labels,
        source_file=source_file,
        session=session,
        subject=subject,
        loaded_from_cache=False,
        cache_file=cache_file,
        cache_manifest_file=manifest_file,
        manifest=manifest,
    )


def flatten_seediv_subject_windows(
    subject_data: SeedIVSubjectWindows,
    *,
    subject: int | None = None,
    session: int | None = None,
) -> SeedWindowRows:
    """Flatten one SEED-IV recording while retaining clustering provenance."""

    subject_id = subject_data.subject if subject is None else subject
    session_id = subject_data.session if session is None else session
    if subject_id <= 0 or session_id <= 0:
        raise ValueError("subject and session identifiers must be positive")
    if len(subject_data.trials) != 24 or subject_data.trial_labels.shape != (24,):
        raise ValueError("SEED-IV subject data must contain exactly 24 labeled trials")

    features = np.concatenate(subject_data.trials, axis=0)
    labels = np.concatenate(
        [
            np.full(len(trial), subject_data.trial_labels[index], dtype=np.int64)
            for index, trial in enumerate(subject_data.trials)
        ]
    )
    trials = np.concatenate(
        [
            np.full(len(trial), index + 1, dtype=np.int16)
            for index, trial in enumerate(subject_data.trials)
        ]
    )
    return SeedWindowRows(
        features=features,
        labels=labels,
        subject=np.full(len(labels), subject_id, dtype=np.int16),
        session=np.full(len(labels), session_id, dtype=np.int8),
        trial=trials,
    )
