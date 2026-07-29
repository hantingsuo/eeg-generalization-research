# Public evidence status

Last updated: 2026-07-29

This ledger records the evidence boundary used in the accompanying manuscript.

## Supported

- A DGCNN compatibility run on SEED was within 1.47 percentage points of the selected public reference.
- The corresponding SEED-IV compatibility result differed by 3.40 percentage points. Eight implementation-audit groups did not explain the difference, so it is reported as unresolved and used only as secondary sensitivity evidence.
- Strict subject-balanced five-fold evaluation produced accuracies of 0.5348 on SEED and 0.3954 on SEED-IV.
- Training-participant trial accuracies reached 0.9990 on SEED and 0.9920 on SEED-IV under the same strict folds. This rules out simple failure to fit the training data but does not identify the cause of the generalization gap.
- Repeated selection using labelled test performance increased the matched SEED DGCNN score by 0.1036 relative to the fixed-epoch comparison.
- Participant rankings were not stable across sessions in the primary SEED-IV analysis.
- The CF-TRE tail objective changed the intended tail-loss criterion, but it did not produce a consistent recognition-accuracy advantage in the separate final evaluation.

## Not claimed

- exact reproduction of the historical SEED-IV public run;
- a new neural architecture or state-of-the-art classifier;
- a representation-independent advantage for the tail-risk objective;
- persistent "difficult participant" identities;
- a clinical biomarker or causal explanation for participant-level differences.

Compatibility results, strict subject-disjoint estimates, and supporting stress tests must remain labelled separately because they answer different questions.
