# experiments/inspect_competition.py
"""One-off: verify the competition npz shapes/keys before building loaders."""
import os
import numpy as np

PATH = os.environ.get(
    "PCMA_COMPETITION_DATA", "data/competition/train_data.npz"
)

def main():
    d = np.load(PATH, allow_pickle=True)
    print("keys:", list(d.keys()))
    for k in d.keys():
        a = d[k]
        print(f"{k}: shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}")
    y = d["y"]; sid = d["subject_ids"]; grp = d["groups"]
    print("unique y:", np.unique(y, return_counts=True))
    print("n subjects:", len(np.unique(sid)))
    print("unique groups:", np.unique(grp, return_counts=True))

if __name__ == "__main__":
    main()
