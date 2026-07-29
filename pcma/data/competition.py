# pcma/data/competition.py
from dataclasses import dataclass
import os
from typing import Optional
import numpy as np
from pyriemann.estimation import Covariances

DEFAULT_PATH = os.environ.get(
    "COMPETITION_TRAIN_DATA",
    "data/competition/train_data.npz",
)


@dataclass
class CompetitionData:
    X: np.ndarray            # (n_trials, n_channels, n_times)
    y: np.ndarray            # (n_trials,) emotion label in {0,1}
    subject: np.ndarray      # (n_trials,) subject id as string, e.g. "HC1003"
    population: np.ndarray   # (n_trials,) "HC"/"DEP"
    video_id: Optional[np.ndarray] = None  # (n_trials,) 5 consecutive 10s segments per training video

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject))


def _as_population(groups: np.ndarray) -> np.ndarray:
    g = np.asarray(groups)
    if g.dtype.kind in ("U", "S", "O"):
        s = np.array([str(x).upper() for x in g])
        return np.where(np.char.find(s, "DEP") >= 0, "DEP", "HC")
    # numeric encoding: 1 = DEP, 0 = HC (verified against real data)
    return np.where(g.astype(int) == 1, "DEP", "HC")


def infer_video_ids(subject: np.ndarray, y: np.ndarray, segments_per_video: int = 5) -> np.ndarray:
    """Infer train-video ids from verified cache order: subject -> label -> 5 segments."""
    subject = np.asarray(subject).astype(str)
    y = np.asarray(y).astype(int)
    video_id = np.empty(len(y), dtype=object)
    for s in np.unique(subject):
        for label in sorted(np.unique(y[subject == s]).tolist()):
            idx = np.where((subject == s) & (y == label))[0]
            if len(idx) % segments_per_video != 0:
                raise ValueError(f"{s} label {label}: {len(idx)} segments is not divisible by {segments_per_video}")
            if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
                raise ValueError(f"{s} label {label}: segments are not consecutive in cache order")
            local_video = np.arange(len(idx)) // segments_per_video
            for row, vid in zip(idx, local_video):
                video_id[row] = f"{s}_y{label}_v{int(vid)}"
    return video_id.astype(str)


def verify_video_grouping(subject: np.ndarray, y: np.ndarray, video_id: Optional[np.ndarray] = None) -> dict[str, object]:
    subject = np.asarray(subject).astype(str)
    y = np.asarray(y).astype(int)
    video_id = infer_video_ids(subject, y) if video_id is None else np.asarray(video_id).astype(str)
    per_video_counts = [int(np.sum(video_id == vid)) for vid in np.unique(video_id)]
    per_video_labels_constant = all(len(np.unique(y[video_id == vid])) == 1 for vid in np.unique(video_id))
    per_video_subject_constant = all(len(np.unique(subject[video_id == vid])) == 1 for vid in np.unique(video_id))
    per_subject_counts = {
        str(s): {int(label): int(np.sum((subject == s) & (y == label))) for label in sorted(np.unique(y[subject == s]))}
        for s in np.unique(subject)
    }
    return {
        "n_segments": int(len(y)),
        "n_videos": int(len(np.unique(video_id))),
        "per_video_counts_unique": sorted(set(per_video_counts)),
        "per_video_labels_constant": bool(per_video_labels_constant),
        "per_video_subject_constant": bool(per_video_subject_constant),
        "per_subject_label_counts_unique": sorted({tuple(sorted(v.items())) for v in per_subject_counts.values()}),
    }


def aggregate_by_video(
    values: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    population: np.ndarray,
    video_id: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average row-wise values within each verified video group."""
    values = np.asarray(values)
    subject = np.asarray(subject).astype(str)
    population = np.asarray(population).astype(str)
    y = np.asarray(y).astype(int)
    video_id = infer_video_ids(subject, y) if video_id is None else np.asarray(video_id).astype(str)

    rows, yv, sv, popv, vidv = [], [], [], [], []
    for vid in dict.fromkeys(video_id.tolist()):
        mask = video_id == vid
        labels = np.unique(y[mask])
        subjects = np.unique(subject[mask])
        pops = np.unique(population[mask])
        if len(labels) != 1:
            raise ValueError(f"{vid}: labels are not constant within video")
        if len(subjects) != 1:
            raise ValueError(f"{vid}: subjects are not constant within video")
        if len(pops) != 1:
            raise ValueError(f"{vid}: populations are not constant within video")
        rows.append(values[mask].mean(axis=0))
        yv.append(labels[0])
        sv.append(subjects[0])
        popv.append(pops[0])
        vidv.append(vid)
    return (
        np.asarray(rows),
        np.asarray(yv, dtype=int),
        np.asarray(sv).astype(str),
        np.asarray(popv).astype(str),
        np.asarray(vidv).astype(str),
    )


def load_competition(path: str = DEFAULT_PATH) -> CompetitionData:
    d = np.load(path, allow_pickle=True)
    sid_key = "subject_ids" if "subject_ids" in d else "subject"
    subject = np.asarray(d[sid_key])
    y = d["y"].astype(int)
    return CompetitionData(
        X=d["X"],
        y=y,
        subject=subject,   # keep as-is (strings), do NOT cast to int
        population=_as_population(d["groups"]),
        video_id=infer_video_ids(subject, y),
    )


def to_covariances(X: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Broadband SPD covariance per trial -> (n_trials, n_ch, n_ch)."""
    return Covariances(estimator=estimator).fit_transform(X)
