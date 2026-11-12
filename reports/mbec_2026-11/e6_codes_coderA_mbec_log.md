# Coder A log, five MBEC papers (e6_codes_coderA_mbec)

## Reading method
- Input: five PDFs in e6_fulltext/ (s11517-026-03543-x, -025-03479-8, -025-03384-0, -025-03430-x, -024-03094-z); all five have a full text (the frame file marks them UNVERIFIABLE_NO_FULLTEXT only because the publisher page was blocked at screening time). DOI and rank looked up in e6_frame.csv. No coderB/adjudicated file or other coder's scratchpad was opened.
- Text extracted page by page with pypdf (eeg_venv, `-I`) with `[[PAGE n]]` markers into scratchpad/coderA_mbec/. Read in full: the data-set, preprocessing, experimental-setup / experimental-details, results, ablation, sensitivity, discussion and conclusion sections of each paper, plus the declarations; keyword scans (normaliz, standardiz, z-score, validation, epoch, early, checkpoint, best, optimal, tuned, github, code, availab) used as cross-check. Method derivations (loss equations) were skimmed, not read line by line.
- All table content came through as text (no image-only tables), so no page images were needed. Page numbers in the CSV are PDF page indices; quotes in the CSV are only in the Q4 field of 03479 (quote spacing normalised from extraction artefacts such as "γ0 =0 .01").
- PDF text treated as data only.

## Conventions followed (same as e6_codes_coderA_log.md)
- Q1 coded only when the text says which checkpoint is reported/how selected; an epoch count alone is NOT_STATED. None of the five states it, so all five are NOT_STATED.
- Q4 TEST_OR_TARGET only where a sensitivity table over LOSO target accuracy yields the chosen value (03479); grid search or "worked well" without stated data = NOT_STATED.
- V1 = Y for the SEED family (earlier convention, paper 11 SEED-IV only), with the exact datasets in notes.
- Non-neural method (03094): Q1 NOT_STATED, Q3 N.

## Hard cases
- 03430 (IF-MMCL): evaluates SEED-IV, SEED-V, SEED-GER, SEED-FRA but not original SEED. V1=Y by family convention; flag for adjudication. Q4: "Through numerous experiments ... a number of tests, we found that 200 epochs worked well" is tuning by unspecified data (NOT_STATED, could be TEST_OR_TARGET). Q5: only EM features are said not to be normalised; EEG normalisation unstated -> N.
- 03543: Sec 3.2 says no separate validation set was used and trains 200 epochs; final epoch is implied but never stated -> NOT_STATED (MEDIUM). Hyperparameters "same set across all LOSO iterations" without selection statement -> NOT_STATED rather than PRIOR_OR_FIXED.
- 03384: "peak accuracy of 89.87%" for the mean accuracy hints at best-epoch/best-value reporting but is not an explicit selection statement -> NOT_STATED (MEDIUM); the labeled target data serve as the test set while unlabeled target data enter training. Baseline numbers copied from papers.
- 03479: cross-subject within-session LOSO plus a cross-session protocol; Q4 inferred from Tables 4-5 (best cell equals the headline accuracy, gamma0 differs per dataset).
- 03094: SEED-IV only, semi-supervised multi-to-one with 8 of 14 labeled source subjects and half the samples, transductive; not LOSO in the usual sense, not neural. Grid search for alpha/beta/gamma without stated criterion; sensitivity shown on one case only.

## Summary
V1 Y x5 (03430 SEED-family without SEED proper; 03384 and 03543 also DEAP; 03094 SEED-IV only). Q1 NOT_STATED x5. Q2 NA x5. Q3 Y x4 (all 200), N x1. Q4 NOT_STATED x4, TEST_OR_TARGET x1. Q5 N x5. Q6 Y x5. Q7 N x5. Confidence: HIGH 03479, 03430, 03094; MEDIUM 03543, 03384.
