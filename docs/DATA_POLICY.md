# Data and model-release policy

## Never commit

- SEED, SEED-IV, MODMA, competition, or other participant-derived EEG files;
- provider download links, passwords, application forms, or signed agreements;
- processed arrays that reproduce participant data, even if filenames are anonymized;
- hidden competition labels or submission workbooks;
- model checkpoints trained on restricted data unless written redistribution permission is confirmed;
- third-party source without a clear redistribution license.

## Safe to publish

- original algorithms, loaders, split logic, metrics, and audit code;
- documentation explaining how an authorized user should arrange local data;
- schemas, tiny synthetic fixtures, and generators that do not derive from real participants;
- aggregate results that comply with the source dataset's publication terms;
- citations and links to the official provider pages.

The `.gitignore` is a guardrail, not a legal or privacy review. Before every public release, inspect the full Git history and scan for large files, credentials, personal data, and dataset fragments.
