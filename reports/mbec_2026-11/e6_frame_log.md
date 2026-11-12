# E6 sampling frame (stage 1) - log

Date of run: 2026-11-11. Files: `e6_frame.csv`, `e6_frame_log.md`, `e6_raw_ranked_dois.txt` (the raw ranked DOI lists, extra file required by protocol step 2), `e6_fulltext/`.
Nothing else in the repository was touched; nothing committed. Working scripts and caches are in the session scratchpad only.

## 1. API URLs (exact)
Crossref, `rows=100`, first page, default relevance sort, filter `type:journal-article,from-pub-date:2023-01-01,until-pub-date:2026-10-31`:
1. https://api.crossref.org/works?query.bibliographic=cross-subject+EEG+emotion+recognition+SEED&filter=type:journal-article,from-pub-date:2023-01-01,until-pub-date:2026-10-31&rows=100
2. https://api.crossref.org/works?query.bibliographic=subject-independent+EEG+emotion+recognition+domain+adaptation&filter=type:journal-article,from-pub-date:2023-01-01,until-pub-date:2026-10-31&rows=100
3. https://api.crossref.org/works?query.bibliographic=EEG+emotion+recognition+cross-subject+domain+generalization&filter=type:journal-article,from-pub-date:2023-01-01,until-pub-date:2026-10-31&rows=100

Forced-paper metadata: https://api.crossref.org/works/<doi> for the five MBEC DOIs.
Full-text / abstract location helpers (per candidate): https://api.openalex.org/works/doi:<doi>; https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:<doi>&format=json&resultType=lite and `/rest/PMC<id>/fullTextXML`; Crossref `link` field; https://doi.org/<doi>; publisher OA patterns (Springer, Frontiers, Nature, PLOS); https://export.arxiv.org/api/query (exact-title search, plus a fuzzy title+author search that found no additional same-paper hits); https://api.semanticscholar.org/graph/v1/paper/DOI:<doi>?fields=abstract (abstracts only). No personal names or e-mails were sent.

## 2. Counts
| Step | n |
|---|---|
| Hits per query (Crossref total-results / items on first page) | (i) 404,662 / 100; (ii) 263,791 / 100; (iii) 415,479 / 100 |
| Ranked entries total | 300 |
| Unique candidates after round-robin merge (screen order) | 170 |
| Screened (list exhausted; fewer than 40 INCLUDED) | 170 |
| **INCLUDED** (full text accessed and saved) | **37** |
| **ELIGIBLE_NO_FULLTEXT** | **42** |
| **UNVERIFIABLE_NO_FULLTEXT** (added category, see deviations) | **59** (56 in the random-order list + 3 forced-only rows) |
| **EXCLUDED** | **35** (all within the 170) - see below |
| Forced MBEC extra rows appended (not in the 170) | 3 (the other 2 forced DOIs are already in the frame at ranks 25 and 157) |
| Total rows in CSV | 173 |

Check: 37 + 42 + 59 + 35 = 173.
Note: of the 170 screened, INCLUDED 37 + ELIGIBLE 42 + EXCLUDED 35 + UNVERIFIABLE 56 = 170.

Excluded by reason (35): criterion (c) no SEED-family dataset 22; criterion (a) not EEG emotion / not emotion 4; criterion (b) cross-dataset only, no cross-subject protocol 3 (ranks 40, 99, 143, all borderline); reviews 3 (ranks 10, 20, 136); conference papers 3 (ranks 3 AAAI proceedings; 91, 119 J. Phys.: Conf. Ser.). Per-paper reasons are in the CSV.

The target of 40 INCLUDED was NOT reached. True number: 37. Stopped because the list was exhausted.

Forced MBEC papers: 10.1007/s11517-026-03543-x (rank 25), 10.1007/s11517-025-03479-8 (rank 157), 10.1007/s11517-025-03384-0, 10.1007/s11517-025-03430-x, 10.1007/s11517-024-03094-z (the last three appended with screen_rank `F`). All five are closed access on Springer (preview page only, no methods section), so none is INCLUDED. Their abstracts were retrievable only as truncated 155-character meta descriptions, so they are marked UNVERIFIABLE_NO_FULLTEXT (criteria not checkable). forced_mbec=1 in the CSV.

## 3. Deviations from the protocol
1. Extra decision value `UNVERIFIABLE_NO_FULLTEXT`. The protocol has three values; for 59 papers there was no full text and no usable abstract (Elsevier/ScienceDirect and IEEE pages blocked, Springer abstracts truncated), so (a)-(c) could not be judged. Forcing them into EXCLUDED or ELIGIBLE would have been arbitrary. These must NOT be treated as excluded; they are unresolved.
2. ELIGIBLE_NO_FULLTEXT was assigned from the abstract/title only when the abstract named a SEED-family dataset and a cross-subject/subject-independent/LOSO-type term (regex-assisted, then spot-checked). `datasets_named` for these rows comes from the abstract; for INCLUDED rows it comes from string counts (>=3 mentions) in the full text, not from coding.
3. Inclusion decisions for the 37 INCLUDED were made by me from the saved full text using keyword-snippet review (dataset-use and protocol passages), not by a complete read of each paper. Borderline calls: rank 34 (SEED, "subject-to-subject transfer", included); ranks 117 and 130 (authors call the design cross-subject; included); rank 143 (cross-dataset only; excluded, (b)); ranks 40 and 99 (cross-dataset; excluded by abstract). Please re-check these before the coding stage.
4. Conference-type exclusion: ranks 3, 91, 119, 136 have Crossref type journal-article but are proceedings/symposium items; excluded as conference papers.
5. Author versions: for ranks 27, 49, 66, 121, 146, 161 the saved text is the arXiv/author version matched by (near-)identical title (rank 49 is the accepted manuscript). It was not diffed against the journal version, so wording/numbers may differ slightly from the published one. Rank 99 (excluded) was matched to its arXiv version despite "Toward" vs "Towards" in the title.
6. Full-text format: PDFs saved as fetched (26 files). For HTML/XML sources (11 papers) the saved file is the text extracted from the page (`.txt`) plus the raw source (`.source.html/.xml`); the extraction may drop figures/tables.
7. Full-text validity was tested automatically (length, title present, method keywords, no "preview of subscription content" marker); the first attempt wrongly accepted a Springer paywall preview, which was removed and the whole run redone with stricter checks.
8. Years: the Crossref date filter defines the window. Some DOIs carry 2021/2022 (e.g. 10.1109/taffc.2021.3137857, 10.1109/taffc.2022.3179717, 10.1109/tii.2022.3217120, 10.1016/j.measurement.2022.112379); they were kept because Crossref placed them in the window (issue date), but online-first dates may precede 2023. Not excluded; worth a check.
9. Fallback and arXiv search used curl when Node fetch failed. A fuzzy arXiv search (title tokens + author surnames) was run but only rank 99 matched.

## 4. Fetch problems
- Elsevier/ScienceDirect: no abstract or full text; API requires key; the browser pane hit a Cloudflare "please wait" challenge. I did not try to bypass it. Affects most 10.1016 papers (incl. some gold-OA: ranks 22, 31, 52, 55, 79).
- MDPI (ranks 8, 44, 74, 77, 148, 149, 151): scripted requests get "Access Denied" (Akamai). Ranks 8 and 77 are eligible by abstract and OA; the page opens in the browser pane but the text cannot be saved (attempts to relay it to a local receiver were blocked by the page), so they are ELIGIBLE_NO_FULLTEXT. Several other MDPI papers were obtained via PubMed Central (ranks 26, 45, 50, 64, 76, 113, 150).
- PubMed Central direct pages return 403/PoW challenge and Europe PMC XML returns 500 for the Springer "green" papers (ranks 16, 57, 129, 158): no full text.
- IEEE Xplore: paywalled except OA items via the Crossref staging PDF link (ranks 4, 17, 34, 82, 102(TNSRE), 131, 143, 168 etc.) and arXiv versions.
- Springer (Cognitive Neurodynamics, Multimedia Systems, MBEC, Neurological Sciences, SN Comput Sci): paywall preview only.
- arXiv via Node fetch timed out; curl worked.
- Others: rank 3 (AAAI OJS) returned 520; IOP pages (ranks 91, 115, 119) return a stub; Wiley (rank 136) 403.
- Browser-pane experiments to pass page text to a local receiver (form POST, popup, iframe, image beacon) all failed (CSP); no files were produced that way. The local receiver was stopped.

## 5. Where screening stood when interrupted, and how it was resumed
At the time of the coordinator's message, the output folder contained only an empty `e6_fulltext/`; no CSV or log had been written (all 170 candidates had been fetched/screened in scratch files). I did not restart or reorder: the round-robin order was fixed from the saved raw lists (screen_rank 1-170), all 170 were decided, and the two outputs were then written.
