# experiments/inspect_competition.py
"""One-off: verify the competition npz shapes/keys before building loaders."""
import argparse
from pathlib import Path

import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    d = np.load(args.path, allow_pickle=True)
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
