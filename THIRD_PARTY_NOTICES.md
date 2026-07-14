# Third-party notices

This repository contains original research code and adapters. It does not relicense datasets or upstream projects.

| Resource | Relationship | Public-release decision |
|---|---|---|
| SEED and SEED-IV | Provider-controlled EEG datasets | Not redistributed. Users must apply through the official provider and follow its agreement. |
| Competition EEG data | Organizer-provided data | Not redistributed. |
| MODMA | External clinical EEG dataset | Not redistributed. Follow the provider's access and citation conditions. |
| [LibEER](https://github.com/ButterSen/LibEER) | Baseline/protocol compatibility reference | MIT-licensed upstream; this repository keeps adapters and links upstream rather than copying the full project. |
| DSAINet research code | Method comparison reference | No clear license was found in the reviewed local copy; source is not redistributed. |
| PyTorch, scikit-learn, MNE, pyRiemann, POT | Runtime dependencies | Installed separately under their respective licenses. |

Model checkpoints are also excluded by default because their redistribution rights may depend on training data and upstream pretrained components.
