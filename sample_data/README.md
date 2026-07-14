# Synthetic fixtures only

This directory may contain generated synthetic fixtures for smoke tests. Never place real participant EEG, processed dataset arrays, hidden labels, or trained checkpoints here.

Generate a fixture with:

```bash
python scripts/generate_synthetic_eeg.py --output sample_data/synthetic_eeg.npz
```
