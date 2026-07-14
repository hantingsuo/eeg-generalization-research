# Protocol-Aware EEG Generalization Research

Research code for studying EEG affective-computing pipelines under explicit subject, session, target-access, and aggregation protocols. The project emphasizes auditable evaluation and honest negative results rather than presenting protocol-sensitive scores as model innovation.

> Status: active research. The EEG manuscript is not yet written, and this repository does not claim a new state-of-the-art method or a clinical biomarker.

## Current evidence boundary

- Rich handcrafted EEG features with an RBF SVM and video-level aggregation are the strongest practical competition pipeline tested in this project.
- Several alignment/adaptation variants did not pass frozen retention gates.
- A high archival DGCNN score was reproduced only under a labelled-test-selected compatibility protocol; it is not clean generalization evidence.
- Strict subject-balanced five-fold evaluation is complete for SEED and SEED-IV, with materially lower estimates than the archival compatibility protocol.
- A MODMA ERP cue-decoding gate failed, so downstream PHQ-9 inference was not run.

See [docs/EVIDENCE_STATUS.md](docs/EVIDENCE_STATUS.md) for the concise public ledger.

## Repository map

- `pcma/` — data interfaces, feature pipelines, model implementations, splits, metrics, and statistical utilities.
- `experiments/` — runnable experiment, aggregation, and audit entry points.
- `tests/` — protocol and implementation tests.
- `scripts/generate_synthetic_eeg.py` — creates a small synthetic fixture for smoke testing without redistributing real EEG.
- `docs/` — data policy, evidence status, and reproduction guidance.

## Data and weights are not included

SEED/SEED-IV and competition EEG data are governed by their providers' access terms. This repository therefore contains no raw EEG, processed participant arrays, download credentials, private labels, submissions, or model checkpoints trained on restricted data.

Follow [docs/DATA_POLICY.md](docs/DATA_POLICY.md) and obtain each dataset directly from its provider. Do not open an issue asking maintainers to share restricted files or access credentials.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m pip install pytest
```

## Safe smoke test

```bash
python scripts/generate_synthetic_eeg.py --output sample_data/synthetic_eeg.npz
pytest -q
```

Some tests and experiment scripts require provider-obtained datasets and are skipped or fail closed when those inputs are absent. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Project role

Hanting Suo serves as project lead, coordinating protocol design, implementation, experiment auditing, and evidence tracking. No author order is claimed before the manuscript is completed.

## License

Original code and documentation are released under the [BSD 3-Clause License](LICENSE). Dataset licenses and third-party code are separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
