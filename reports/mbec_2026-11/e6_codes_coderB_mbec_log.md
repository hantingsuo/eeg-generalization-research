# E6 coding log (Coder B), MBEC supplement, 5 articles

- Input: five MBEC PDFs in `e6_fulltext/` (rank 25, rank 157, and three forced rows with screen_rank F). No coderA or adjudicated file was opened; no scratchpad of another coder was read. Only two output files were written (`e6_codes_coderB_mbec.csv`, this log); existing files were not touched.
- Text extraction: `pypdf` page by page with `[[PAGE n]]` markers, into `...\scratchpad\coderB_mbec\*.pdf.txt`. All five PDFs are full texts (11-23 pages), all readable. Two-column layout is partly interleaved in the extraction (e.g. rank 157 Sec 3.5 sentence is split across pdf pp. 5-6); I resolved this by reading the surrounding lines.
- Coding rules, vocabularies and conventions as in `e6_codes_coderB_log.md`. Quotes were machine-checked against the extracted text (whitespace-normalised, <=25 words). One quote (03430) differs from the raw extraction only by a line-end hyphenation ("num-ber"), restored to "number".

## Reading per paper

| paper | read in full | notes |
|---|---|---|
| rank 25, AGCDAN (s11517-026-03543-x) | Sec 2.1 data preprocessing, 3.1 datasets, 3.2 experimental setup, 3.3 results, ablation, discussion, conclusion; Table 1, 5 | Complete. No hyperparameter study. No code/data statement; Declarations only has funding/COI. |
| rank 157, MSSGDA (s11517-025-03479-8) | Sec 3.1-3.5 (data, preprocessing, training), 4.1-4.4, 5.x, limitations | Complete. Training sentence split over page columns. |
| F, ACAN (s11517-025-03384-0) | Sec 3 (architecture, training), 4.1-4.6, 5, 6 | Complete. Header says "REVIEW ARTICLE", but content is a method paper. |
| F, IF-MMCL (s11517-025-03430-x) | Sec 2.2-2.3 data and preprocessing, 2.4 network, Table 5, 3.1-3.2 setup, 4.x results, conclusion | Complete. Results/ablation tables are mostly numeric (read the text, not every cell). |
| F, SBGASS (s11517-024-03094-z) | Sec 3.1-3.4 (data, setup, learning details, results), parameter sensitivity, conclusion | Complete. Method sections 2.x (math) skimmed for training procedure only. |

Nothing was unavailable. No table was an image that I could not read in extracted text, so I did not need to Read PDF pages as images.

## Dataset and protocol per paper (V1)

- Rank 25: SEED + SEED-IV + DEAP; LOSO, first session only; multi-source UDA (target features used unlabeled).
- Rank 157: SEED + SEED-IV; LOSO within each session (all three sessions); plus a separate cross-session protocol.
- ACAN: SEED + SEED-IV + DEAP (15 chosen subjects); LOSO; plus cross-session.
- IF-MMCL: SEED-IV, SEED-V, SEED-GER, SEED-FRA only (no original SEED); LOSO, 3 sessions.
- SBGASS: SEED-IV only; multi-to-one, 8 of 14 source subjects, half of their samples, one unlabeled target; not full LOSO; transductive.
All five coded V1=Y on the "cross-subject protocol on a SEED-family dataset" criterion used in the main coding; the SEED-IV-only / non-SEED cases are stated in the notes.

## Hard cases (for adjudication)

1. SBGASS Q1 = BEST_ON_TEST (LOW). Not a neural method; there is no checkpoint. I followed convention 3 of the main log (configuration chosen by target accuracy counted as BEST_ON_TEST, as for ranks 64, 117, 121): the sample-rejection rate with the best average accuracy is reported as "ultimate accuracy". An alternative reading is NOT_STATED / NA because no epoch or checkpoint exists.
2. ACAN Q1 = NOT_STATED. "peak accuracy of 89.87%" and "highest cross-subject accuracy" are comparative statements, not a stated checkpoint rule; a reader could take "peak" as best epoch.
3. Rank 25 Q4 = NOT_STATED. The authors say no validation set was used and the same hyperparameters were used in every LOSO fold, but they never say where the values came from; PRIOR_OR_FIXED is the nearest alternative.
4. IF-MMCL Q5 = N. Sec 3.1 states the experiments focus on "EEG data normalization for domain adaptation" but no EEG normalization step is described; only eye-movement data is explicitly not normalized. Could be Y/OTHER_OR_UNCLEAR.
5. IF-MMCL and ACAN Q4 = NOT_STATED: epoch count / batch size chosen after "tests"/"in the hope of improving generalization" without stating on which data.
6. Rank 157 Q4 = TEST_OR_TARGET: beta and gamma0 sensitivity tables are the target-subject accuracies and the chosen setting reproduces the headline numbers; no validation split is mentioned.
7. SBGASS Q3 = N (no epochs); Q5 = N (graph row-normalization is a model constraint, not data normalization).

## Neural method with training epochs?

- Rank 25: yes (200 epochs). Rank 157: yes (200). ACAN: yes (200). IF-MMCL: yes (200; also 800/500 iterations per epoch in Table 5). SBGASS: no (non-neural graph learning, alternating optimization, no epochs).
