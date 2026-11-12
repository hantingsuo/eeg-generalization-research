# Additions of November 2026

These files accompany the manuscript *How much of a checkpoint-selection gain survives
on new users? An apparent-versus-retained audit of cross-subject EEG emotion decoding*
and extend the September 2026 release described in
[JNE_2026_REVISION.md](JNE_2026_REVISION.md).

## All-session SEED repeat

| Path | What it is |
|---|---|
| `plans/2026-11-11-mbec-seed-allsessions.json` | Frozen plan, with the four predictions written before the run (`predictions_written_before_running`) |
| `experiments/mbec_seed_allsessions.py` | Runner (`freeze`, `run`, `status`) |
| `tests/test_mbec_seed_allsessions.py` | Unit tests of the runner |
| `reports/jne_revision_2026-09/seedas_summary.json` | Aggregate result summary, written by `experiments/jne_e1_analysis.py --plan plans/2026-11-11-mbec-seed-allsessions.json --tag seedas` |

The repeat uses the folds, target and validation participants, scored pools, models
(DGCNN, MLP), seeds, optimisation recipe, selection rules and analysis of the SEED
participant hold-out. It differs in two respects only: training uses all three
sessions of the nine source participants, and every recording is standardised per
feature with its own mean and standard deviation, computed without labels. The plan
and its predictions were fixed before the run and after all other results of the
manuscript were known. As in the other summaries, keys named `participant_*` were
removed before release; every aggregate value is unchanged.

## Reporting survey

`reports/mbec_2026-11/` holds the records of the survey of how articles on
cross-subject EEG emotion recognition report checkpoint selection.

| File | What it is |
|---|---|
| `e6_raw_ranked_dois.txt`, `e6_frame.csv`, `e6_frame_log.md` | Search frame, random ranking, screening decisions and log |
| `e6_codes_coderA.csv`, `e6_codes_coderB.csv` and their `_log.md` files | Independent codes of the two coders for the sampled articles |
| `e6_adjudicated.csv` | Adjudicated codes behind the tallies in the supplementary material |
| `e6_codes_coderA_mbec.csv`, `e6_codes_coderB_mbec.csv`, their logs, `e6_adjudicated_mbec.csv` | The five journal articles coded separately; not part of the tallies |

Quotations in these files are limited to short excerpts that support a code. The
full texts of the articles are not redistributed.
