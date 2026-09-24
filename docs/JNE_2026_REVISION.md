# Code for the 2026 revision (Journal of Neural Engineering submission)

This directory set accompanies the manuscript *Choosing checkpoints on validation
participants yields no cross-participant gain in EEG emotion decoding*. It extends
the earlier release in this repository with the analyses added in September 2026.

## What was added

| Path | What it is |
|---|---|
| `plans/2026-09-17-jne-e1-seed-cross-subject.json` | Frozen plan, SEED participant hold-out (DGCNN, MLP) |
| `plans/2026-09-17-jne-e2-eegnet-seed.json` | Frozen plan, SEED participant hold-out (EEGNet) |
| `plans/2026-09-17-jne-e4-matched-session.json` | Frozen plan, matched-session control |
| `plans/2026-09-18-jne-e1-faced.json` | Frozen plan, FACED, source-statistics standardisation |
| `plans/2026-09-20-jne-e1-faced-v2.json` | Frozen plan, FACED, per-participant standardisation, with the four written predictions |
| `experiments/jne_*.py` | Runners and analyses for the experiments above, the common-time-range grid check and the accounting simulation |
| `experiments/run_throttled.py` | GPU telemetry logger used during the runs |
| `reports/jne_revision_2026-09/*.json` | Aggregate result summaries (means, intervals, p values) behind every number in the manuscript |

Each plan records the participant split, candidate grid, calibration rotations,
primary endpoints, SHA-256 hashes of every input file and the hash of the runner.
A runner refuses to start if its own hash, its plan or any input has changed.

## What is not included, and why

- **Raw or processed EEG** from SEED, SEED-IV, SEED-V or FACED. Their licences do not
  permit redistribution; obtain each dataset from its provider.
- **Per-epoch, per-clip model outputs** (the `trace.npz` file of each training run).
  These are participant-level derived records. Consistent with this repository's data
  policy they are not published; they are available from the corresponding author on
  reasonable request, subject to the data providers' terms.
- **Participant-level lists** inside the summaries. Keys named `participant_*` were
  removed before release; every aggregate value is unchanged.
- **The pinned LibEER DGCNN source.** The plans name the exact commit; scripts that
  need it fail closed when it is absent.

## Reproducing

```bash
python -m pip install -e .

# Runs without any restricted data and reproduces its JSON byte for byte
python experiments/jne_selection_simulation.py
```

The training runners (`jne_e1_seed_cross_subject.py`, `jne_e2_eegnet_seed.py`,
`jne_e4_matched_session.py`, `jne_e1_faced.py`) require the licensed datasets at the
paths recorded in their plans and verify the input hashes before training. The
analyses (`jne_e1_analysis.py`, `jne_e4_analysis.py`, `jne_faced_analysis.py`,
`jne_common_range.py`, `jne_selection_informativeness.py`) read the per-epoch records
those runners write. Training used deterministic GPU algorithms; the optional
`--duty` flag paces the GPU without changing the computation, which was checked by
re-fitting a completed run bit-identically before it was used.
