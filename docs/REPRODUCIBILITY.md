# Reproducibility guide

1. Create an isolated Python environment and install the project.
2. Obtain each dataset directly from its provider and keep it outside the Git repository.
3. Configure local paths with environment variables or an ignored local config file.
4. Start with unit tests and the synthetic fixture generator.
5. Run a single fold or smoke command before launching a full batch.
6. Record the split definition, random seed, target access, checkpoint selection rule, aggregation unit, and subject weighting for every result.
7. Keep compatibility protocols and clean generalization protocols in separate result tables.

Real-data commands are intentionally not automated around provider authentication. The project will not download or redistribute restricted EEG on a user's behalf.
