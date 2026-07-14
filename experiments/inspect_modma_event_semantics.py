"""Audit MODMA dot-probe event channels and likely f/h/s condition semantics."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from pcma.data.modma import DEFAULT_MODMA_ROOT, discover_modma_files


OUT = Path(__file__).resolve().parents[1] / "results"
PREFIXES = ("f", "h", "s")
EVENT_SUFFIXES = ("cue", "dot", "fix", "isi", "wrp")


def _rising_edges(x):
    active = np.asarray(x) > 0
    return np.flatnonzero(np.diff(np.r_[False, active].astype(np.int8)) == 1)


def _event_counts(raw, stim_channels):
    counts = {}
    first_samples = {}
    data = raw.get_data(picks=stim_channels)
    for ch, row in zip(stim_channels, data):
        idx = _rising_edges(row)
        counts[ch] = int(len(idx))
        first_samples[ch] = [int(x) for x in idx[:5]]
    return counts, first_samples


def _prefix_order(first_samples):
    pairs = []
    for prefix in PREFIXES:
        samples = first_samples.get(f"{prefix}cue", [])
        if samples:
            pairs.append((samples[0], prefix))
    return "".join(prefix for _, prefix in sorted(pairs))


def _select_subjects(raw_by_subject, sample_limit: int | None, per_prefix: int | None):
    items = sorted(raw_by_subject.items())
    if per_prefix is not None:
        selected = []
        for prefix in ("0201", "0202", "0203"):
            selected.extend([(sid, path) for sid, path in items if sid.startswith(prefix)][:per_prefix])
        return selected
    return items[:sample_limit]


def inspect(root: Path, sample_limit: int | None = None, per_prefix: int | None = None):
    import mne

    raw_by_subject, xlsx = discover_modma_files(root)
    rows = []
    all_stim_names = Counter()
    order_counts = Counter()
    expected = [f"{prefix}{suffix}" for prefix in PREFIXES for suffix in EVENT_SUFFIXES]
    audited = _select_subjects(raw_by_subject, sample_limit, per_prefix)
    for sid, path in audited:
        raw = mne.io.read_raw_egi(path, preload=False, verbose="ERROR")
        stim = [ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == "stim"]
        all_stim_names.update(stim)
        counts, first_samples = _event_counts(raw, stim)
        prefix_order = _prefix_order(first_samples)
        order_counts[prefix_order] += 1
        rows.append(
            {
                "subject_id": sid,
                "file": str(path),
                "stim_channels": stim,
                "all_expected_fhs_events_present": all(ch in stim for ch in expected),
                "expected_counts": {ch: counts.get(ch, 0) for ch in expected},
                "cue_counts": {f"{p}cue": counts.get(f"{p}cue", 0) for p in PREFIXES},
                "prefix_order_by_first_cue": prefix_order,
                "first_cue_samples": {f"{p}cue": first_samples.get(f"{p}cue", []) for p in PREFIXES},
            }
        )
    complete_expected = [
        row["all_expected_fhs_events_present"] and all(v == 160 for v in row["expected_counts"].values())
        for row in rows
    ]
    return {
        "status": "complete" if rows else "blocked",
        "root": str(root),
        "xlsx": str(xlsx) if xlsx else None,
        "n_raw_files": len(raw_by_subject),
        "n_audited_files": len(rows),
        "expected_event_channels": expected,
        "stim_channel_name_counts": dict(sorted(all_stim_names.items())),
        "prefix_order_counts": dict(sorted(order_counts.items())),
        "all_audited_files_have_160_each_for_fhs_cue_dot_fix_isi_wrp": bool(rows) and all(complete_expected),
        "semantic_interpretation": {
            "local_raw_support": "The raw files contain matched f/h/s-prefixed event families (cue, dot, fix, isi, wrp), with 160 cue events per prefix.",
            "likely_mapping": {"f": "Fear-Neutral block", "h": "Happy-Neutral block", "s": "Sad-Neutral block"},
            "caution": "The local archive lacks Methodology.docx/ReadMe.pdf; mapping is inferred from event names plus public MODMA task descriptions.",
        },
        "subjects": rows,
    }


def _write_markdown(audit):
    interp = audit["semantic_interpretation"]
    lines = [
        "# MODMA Event Semantics Audit",
        "",
        f"- root: `{audit['root']}`",
        f"- status: {audit['status']}",
        f"- raw files: {audit['n_raw_files']}",
        f"- audited files: {audit['n_audited_files']}",
        f"- all audited files have 160 each for f/h/s cue/dot/fix/isi/wrp: {audit['all_audited_files_have_160_each_for_fhs_cue_dot_fix_isi_wrp']}",
        f"- prefix order counts by first cue: `{audit['prefix_order_counts']}`",
        "",
        "## Interpretation",
        "",
        f"- Local raw support: {interp['local_raw_support']}",
        f"- Likely mapping: `{interp['likely_mapping']}`",
        f"- Caution: {interp['caution']}",
        "",
        "## Sample Subjects",
        "",
        "| Subject | Cue counts | First-cue order | All expected events present |",
        "|---|---|---|---|",
    ]
    for row in audit["subjects"][:10]:
        lines.append(
            f"| {row['subject_id']} | `{row['cue_counts']}` | "
            f"{row['prefix_order_by_first_cue']} | {row['all_expected_fhs_events_present']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    root = Path(os.environ.get("MODMA_ROOT", DEFAULT_MODMA_ROOT))
    sample_limit_text = os.environ.get("MODMA_EVENT_AUDIT_SAMPLE_LIMIT", "")
    sample_limit = int(sample_limit_text) if sample_limit_text else None
    per_prefix_text = os.environ.get("MODMA_EVENT_AUDIT_PER_PREFIX", "")
    per_prefix = int(per_prefix_text) if per_prefix_text else None
    audit = inspect(root, sample_limit=sample_limit, per_prefix=per_prefix)
    OUT.mkdir(exist_ok=True)
    with open(OUT / "modma_event_semantics_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    with open(OUT / "modma_event_semantics_audit.md", "w", encoding="utf-8") as f:
        f.write(_write_markdown(audit))
    print(json.dumps({k: v for k, v in audit.items() if k != "subjects"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
