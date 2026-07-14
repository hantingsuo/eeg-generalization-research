# experiments/inspect_competition_video_grouping.py
"""Inspect competition train-video grouping for 5x10s -> 50s aggregation."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pcma.data.competition import load_competition, verify_video_grouping


OUT = os.path.join(os.path.dirname(__file__), "..", "results")
MOSHISHIBIE_AUDIT = Path(
    os.environ.get(
        "PCMA_SOURCE_VIDEO_AUDIT", "results/source_video_grouping_audit.json"
    )
)


def main():
    data = load_competition()
    audit = {
        "local_cache": verify_video_grouping(data.subject, data.y, data.video_id),
        "source_audit": None,
    }
    if MOSHISHIBIE_AUDIT.exists():
        source = json.loads(MOSHISHIBIE_AUDIT.read_text(encoding="utf-8"))
        audit["source_audit"] = {
            "path": str(MOSHISHIBIE_AUDIT),
            "conclusion": source.get("conclusion", {}),
            "train_cache_vs_raw": {
                key: source.get("train_cache_vs_raw", {}).get(key)
                for key in (
                    "raw_to_cache_exact_match",
                    "max_abs_diff_raw_vs_cache",
                    "per_subject_20_neutral_20_positive",
                    "per_video_5_consecutive_10s_segments",
                    "subject_order_matches_raw_sorted_files",
                )
            },
        }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "competition_video_grouping_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    lines = [
        "# Competition Video Grouping Audit",
        "",
        f"- segments: {audit['local_cache']['n_segments']}",
        f"- videos: {audit['local_cache']['n_videos']}",
        f"- per-video segment counts: {audit['local_cache']['per_video_counts_unique']}",
        f"- labels constant per video: {audit['local_cache']['per_video_labels_constant']}",
        f"- subject constant per video: {audit['local_cache']['per_video_subject_constant']}",
    ]
    if audit["source_audit"] is not None:
        src = audit["source_audit"]
        lines += [
            "",
            f"- source audit: {src['path']}",
            f"- raw_to_cache_exact_match: {src['train_cache_vs_raw']['raw_to_cache_exact_match']}",
            f"- max_abs_diff_raw_vs_cache: {src['train_cache_vs_raw']['max_abs_diff_raw_vs_cache']}",
            f"- per_video_5_consecutive_10s_segments: {src['train_cache_vs_raw']['per_video_5_consecutive_10s_segments']}",
            f"- conclusion: {src['conclusion']}",
        ]
    with open(os.path.join(OUT, "competition_video_grouping_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
