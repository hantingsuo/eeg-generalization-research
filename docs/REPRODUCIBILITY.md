# Reproducibility guide

1. Create an isolated Python environment and install the project.
2. Obtain SEED, SEED-IV, or competition data directly from the relevant provider and keep them outside the Git repository.
3. Configure local paths through command-line arguments, environment variables, or an ignored local configuration file.
4. Run `pytest -q` before any real-data job.
5. Start with a single fold or smoke command before launching a batch.
6. Record the participant/session partition, random seed, admissible target information, checkpoint rule, primitive prediction unit, aggregation rule, and participant weighting.
7. Keep compatibility checks, strict subject-disjoint estimates, and supporting sensitivity analyses in separate result tables.

## Manuscript experiment map

- `run_seed_dgcnn_legacy_test_selected.py` and `run_seed_dgcnn_fixed_epoch_subject_dependent.py`: matched checkpoint-selection contrast.
- `run_libeer_clean_compat_c1_batch.py` and `audit_libeer_clean_compat_c1.py`: public-implementation compatibility checks.
- `run_seed_family_strict_fivefold_batch.py`: strict subject-disjoint evaluation.
- `run_strict_fivefold_learning_curve_*.py`: train-fit and learning-curve diagnostics.
- `run_seed_subject_difficulty_stability.py`: participant-ranking stability analysis.
- `run_cf_tre_g0c_*.py` and `run_cf_tre_g1_outer_test.py`: development-only selection and the frozen separate final evaluation.
- `run_postreview_lr_rf_*.py`: logistic-regression and random-forest controls.
- `audit_*.py`: independent checks of split, provenance, aggregation, and result artifacts.

Real-data commands intentionally do not automate provider authentication. Restricted datasets, checkpoints, unit predictions, and frozen result arrays are not included. `supplementary/ESM_2.json` preserves the selected CF-TRE configuration independently of those files.
