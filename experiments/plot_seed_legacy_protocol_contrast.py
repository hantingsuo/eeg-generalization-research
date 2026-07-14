"""Create the paired fixed-epoch versus test-selected SEED audit figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in (args.png, args.pdf):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(
        payload["recordings"],
        key=lambda row: (
            row["selected"]["metrics"]["window"]["accuracy"],
            row["session"],
            row["subject"],
        ),
    )

    labels = [f"S{row['session']}-{row['subject']:02d}" for row in rows]
    final = [row["final_epoch"]["metrics"]["window"]["accuracy"] for row in rows]
    selected = [row["selected"]["metrics"]["window"]["accuracy"] for row in rows]
    epochs = [row["selected"]["epoch"] for row in rows]
    uplift = [row["selection_uplift_window_accuracy"] for row in rows]
    sessions = [row["session"] for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, (ax_pair, ax_epoch) = plt.subplots(
        1,
        2,
        figsize=(12.0, 8.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )

    y = list(range(len(rows)))
    for index, (left, right) in enumerate(zip(final, selected)):
        ax_pair.plot([left, right], [index, index], color="#a7a9ac", linewidth=1.0, zorder=1)
    ax_pair.scatter(final, y, color="#2b6cb0", s=28, label="Epoch 80", zorder=3)
    ax_pair.scatter(selected, y, color="#c53030", s=28, label="Test-selected maximum", zorder=4)
    ax_pair.axvline(0.8948, color="#222222", linestyle="--", linewidth=1.0, label="Public mean anchor 0.8948")
    ax_pair.set_yticks(y, labels)
    ax_pair.set_xlim(0.43, 1.015)
    ax_pair.set_xlabel("Window accuracy")
    ax_pair.set_ylabel("Subject-session unit")
    ax_pair.set_title("A  Paired accuracy on identical training trajectories", loc="left", fontweight="bold")
    ax_pair.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_pair.legend(loc="lower right", frameon=False)

    colors = ["#2f855a" if session == 1 else "#805ad5" for session in sessions]
    ax_epoch.scatter(epochs, uplift, c=colors, s=44, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax_epoch.axhline(0.0, color="#666666", linewidth=0.8)
    ax_epoch.set_xlim(0, 81)
    ax_epoch.set_ylim(-0.01, 0.35)
    ax_epoch.set_xlabel("Selected epoch (earliest strict maximum)")
    ax_epoch.set_ylabel("Selected minus epoch-80 accuracy")
    ax_epoch.set_title("B  Selection epoch and within-run uplift", loc="left", fontweight="bold")
    ax_epoch.grid(color="#e5e7eb", linewidth=0.8)
    ax_epoch.scatter([], [], color="#2f855a", label="Session 1")
    ax_epoch.scatter([], [], color="#805ad5", label="Session 2")
    ax_epoch.legend(frameon=False, loc="upper right")

    summary = payload["summary"]
    fig.suptitle(
        "SEED archival protocol audit: labelled-test checkpoint selection",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        (
            "30 subject-session units; 80 labelled-test selection evaluations + one reload retest per unit. "
            f"Selected mean={summary['selected_window_accuracy_mean_subject_session']:.4f}; "
            f"epoch-80 mean={summary['final_epoch_window_accuracy_mean_subject_session']:.4f}; "
            f"mean uplift={summary['within_run_selection_uplift_mean']:.4f}. "
            "The selected score is a protocol-compatibility quantity, not a leakage-free estimate."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )

    for output in (args.png, args.pdf):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
