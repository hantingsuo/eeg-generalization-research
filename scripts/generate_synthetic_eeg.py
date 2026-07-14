"""Create a tiny, clearly synthetic EEG-like fixture for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_fixture(seed: int = 2026) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    subjects, trials, channels, samples = 4, 6, 8, 128
    time = np.linspace(0.0, 1.0, samples, endpoint=False)
    data = rng.normal(0.0, 0.15, size=(subjects, trials, channels, samples))
    labels = np.tile(np.arange(trials) % 3, (subjects, 1))

    for subject in range(subjects):
        for trial in range(trials):
            frequency = 6.0 + 2.0 * labels[subject, trial]
            signal = np.sin(2.0 * np.pi * frequency * time)
            data[subject, trial] += (0.25 + 0.02 * subject) * signal

    return {
        "eeg": data.astype(np.float32),
        "labels": labels.astype(np.int64),
        "subject_ids": np.arange(subjects, dtype=np.int64),
        "sampling_rate_hz": np.array(128, dtype=np.int64),
        "synthetic": np.array(True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **build_fixture(args.seed))
    print(f"Wrote synthetic fixture to {args.output}")


if __name__ == "__main__":
    main()
