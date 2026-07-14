"""Inspect local MODMA EGI raw/xlsx schema before severity analysis."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from pcma.data.modma import DEFAULT_EVENT_CHANNELS, DEFAULT_MODMA_ROOT, discover_modma_files, egi_event_samples, load_modma_metadata


OUT = Path(__file__).resolve().parents[1] / "results"


def _range(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return {"n": len(values), "min": min(values), "max": max(values)} if values else {"n": 0, "min": None, "max": None}


def _metadata_summary(root: Path):
    records = load_modma_metadata(root)
    raw_by_subject, xlsx = discover_modma_files(root)
    ids = {r.subject_id for r in records}
    raw_ids = set(raw_by_subject)
    by_type = Counter(r.population for r in records)
    prefix_counts = Counter(r.subject_id[:4] for r in records)
    scales = {}
    for scale in ("PHQ-9", "GAD-7", "PSQI"):
        all_values = [r.scales.get(scale) for r in records]
        scales[scale] = {
            "all": _range(all_values),
            "by_type": {
                label: _range([r.scales.get(scale) for r in records if r.population == label])
                for label in sorted(by_type)
            },
        }
    return {
        "xlsx": str(xlsx) if xlsx else None,
        "n_rows": len(records),
        "type_counts": dict(sorted(by_type.items())),
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "n_raw_files": len(raw_by_subject),
        "join_count": len(ids & raw_ids),
        "raw_not_in_metadata": sorted(raw_ids - ids),
        "metadata_not_in_raw": sorted(ids - raw_ids),
        "scales": scales,
    }


def _stim_counts(raw, event_channels):
    out = {}
    samples = egi_event_samples(raw, event_channels)
    for name, idx in samples.items():
        out[name] = {
            "n_events": int(len(idx)),
            "first_samples": [int(x) for x in idx[:10]],
        }
    return out


def _raw_summary(path: Path):
    import mne

    raw = mne.io.read_raw_egi(path, preload=False, verbose="ERROR")
    ch_types = Counter(raw.get_channel_types())
    event_id = {str(k): int(v) for k, v in (getattr(raw, "event_id", {}) or {}).items()}
    try:
        events = mne.find_events(raw, stim_channel=[ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == "stim"], shortest_event=1, verbose="ERROR")
        event_codes = Counter(int(v) for v in events[:, 2]) if len(events) else Counter()
    except Exception as exc:  # pragma: no cover - only for local schema surprises
        events = np.empty((0, 3), dtype=int)
        event_codes = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "file": str(path),
        "subject_id": path.name[:8],
        "n_channels": len(raw.ch_names),
        "channel_type_counts": dict(sorted(ch_types.items())),
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "first_channels": raw.ch_names[:8],
        "last_channels": raw.ch_names[-8:],
        "n_annotations": int(len(raw.annotations)),
        "annotation_descriptions": raw.annotations.description[:20].tolist() if len(raw.annotations) else [],
        "event_id": event_id,
        "find_events_shape": [int(x) for x in events.shape],
        "find_events_code_counts": dict(event_codes),
        "cue_event_counts": _stim_counts(raw, DEFAULT_EVENT_CHANNELS),
    }


def inspect_tree(root: Path, max_examples: int = 20):
    if not root.exists():
        return {"status": "blocked", "exists": False, "root": str(root), "message": "MODMA root not found"}
    files = [p for p in root.rglob("*") if p.is_file()]
    suffix_counts = Counter(p.suffix.lower() or "<none>" for p in files)
    examples = {}
    for p in files:
        suffix = p.suffix.lower() or "<none>"
        examples.setdefault(suffix, [])
        if len(examples[suffix]) < max_examples:
            examples[suffix].append(str(p.relative_to(root)))
    metadata = _metadata_summary(root)
    raw_by_subject, _ = discover_modma_files(root)
    sample_paths = []
    for prefix in ("0201", "0202", "0203"):
        candidates = [p for sid, p in sorted(raw_by_subject.items()) if sid.startswith(prefix)]
        if candidates:
            sample_paths.append(candidates[0])
    raw_samples = [_raw_summary(p) for p in sample_paths]
    return {
        "status": "complete",
        "exists": True,
        "root": str(root),
        "n_files": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "examples_by_suffix": examples,
        "metadata": metadata,
        "raw_samples": raw_samples,
        "verified": {
            "mne_read_raw_egi": bool(raw_samples),
            "sfreq_250hz": all(abs(s["sfreq"] - 250.0) < 1e-6 for s in raw_samples),
            "egi_total_channels_150": all(s["n_channels"] == 150 for s in raw_samples),
            "eeg_channels_129": all(s["channel_type_counts"].get("eeg") == 129 for s in raw_samples),
            "stim_channels_21": all(s["channel_type_counts"].get("stim") == 21 for s in raw_samples),
            "cue_channels_have_160_events_each": all(
                all(c["n_events"] == 160 for c in s["cue_event_counts"].values()) for s in raw_samples
            ),
            "xlsx_raw_join_complete": metadata["join_count"] == metadata["n_raw_files"] == metadata["n_rows"],
        },
    }


def _write_markdown(audit):
    lines = [
        "# MODMA Schema Audit",
        "",
        f"- root: `{audit['root']}`",
        f"- status: {audit['status']}",
        f"- exists: {audit['exists']}",
    ]
    if not audit["exists"]:
        lines.append(f"- message: {audit['message']}")
        return "\n".join(lines) + "\n"
    md = audit["metadata"]
    lines += [
        f"- files: {audit['n_files']}",
        f"- suffix counts: `{audit['suffix_counts']}`",
        f"- metadata rows: {md['n_rows']} ({md['type_counts']})",
        f"- raw files: {md['n_raw_files']}; xlsx/raw join: {md['join_count']}",
        f"- prefix counts: {md['prefix_counts']}",
        "",
        "## Raw Samples",
        "",
        "| Subject | sfreq | channels | types | duration(s) | fcue/hcue/scue events |",
        "|---|---:|---:|---|---:|---|",
    ]
    for sample in audit["raw_samples"]:
        counts = "/".join(str(sample["cue_event_counts"][ch]["n_events"]) for ch in DEFAULT_EVENT_CHANNELS)
        lines.append(
            f"| {sample['subject_id']} | {sample['sfreq']:.1f} | {sample['n_channels']} | "
            f"{sample['channel_type_counts']} | {sample['duration_sec']:.1f} | {counts} |"
        )
    lines += [
        "",
        "## Scale Ranges",
        "",
        "| Scale | Group | n | min | max |",
        "|---|---|---:|---:|---:|",
    ]
    for scale, info in md["scales"].items():
        for group, vals in info["by_type"].items():
            lines.append(f"| {scale} | {group} | {vals['n']} | {vals['min']} | {vals['max']} |")
    lines += ["", f"Verified flags: `{audit['verified']}`"]
    return "\n".join(lines) + "\n"


def main():
    root = Path(os.environ.get("MODMA_ROOT", DEFAULT_MODMA_ROOT))
    OUT.mkdir(exist_ok=True)
    audit = inspect_tree(root)
    with open(OUT / "modma_schema_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    with open(OUT / "modma_schema_audit.md", "w", encoding="utf-8") as f:
        f.write(_write_markdown(audit))
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
