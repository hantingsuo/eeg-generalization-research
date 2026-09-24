# Evaluation Protocols and Cross-Subject Generalization in EEG Emotion Recognition

Research code accompanying the manuscript *Evaluation Protocols and Cross-Subject Generalization in EEG Emotion Recognition*.

The repository studies how subject partitioning, checkpoint selection, admissible target information, and aggregation change the interpretation of EEG emotion-recognition results. It supports an evaluation-methodology contribution and does not claim a new classifier, state-of-the-art accuracy, or a clinical biomarker.

## What is included

- `pcma/`: data interfaces, feature pipelines, split logic, models, metrics, and statistical utilities.
- `experiments/`: compatibility checks, strict subject-disjoint evaluation, learning-curve diagnostics, participant-ranking analyses, CF-TRE experiments, controls, and independent audit entry points.
- `tests/`: unit, protocol, and artifact-audit tests.
- `supplementary/ESM_2.json`: the final machine-readable CF-TRE parameters and mixture weights submitted as Online Resource 2.
- `docs/`: the public evidence boundary, data policy, and reproduction guidance.

**2026 revision.** The analyses added for the Journal of Neural Engineering submission (participant hold-outs with and without target calibration labels, the matched-session control, EEGNet, the FACED external dataset with a stimulus hold-out, the common-time-range grid check and the accounting simulation) are described in [docs/JNE_2026_REVISION.md](docs/JNE_2026_REVISION.md), with their frozen plans in `plans/` and aggregate result summaries in `reports/jne_revision_2026-09/`.

The frozen numerical results reported in the manuscript were produced before this public release. Scripts that consume restricted datasets or frozen result artifacts fail closed when those inputs are absent.

## Data are not redistributed

SEED, SEED-IV, and the competition EEG release are governed by their providers' access or licence terms. This repository contains no raw EEG, processed participant arrays, private labels, individual predictions, submissions, credentials, or trained checkpoints.

Obtain each dataset directly from its provider and keep it outside the repository. See [the data policy](docs/DATA_POLICY.md) before configuring local paths.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m pip install pytest
```

## Tests

```bash
pytest -q
```

Some real-data and frozen-artifact checks require files that cannot be redistributed. Their expected inputs and evidence boundaries are described in [the reproduction guide](docs/REPRODUCIBILITY.md).

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) when citing the software. Please cite the manuscript separately once its bibliographic record is available.

## License

Original code and documentation are released under the [BSD 3-Clause License](LICENSE). Dataset licences and third-party code remain separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
