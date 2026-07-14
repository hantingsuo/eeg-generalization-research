# experiments/inspect_seed.py
"""Verify DE-feature schema of SEED / SEED-IV / SEED-V before building loaders (the 'check')."""
import os, glob
import numpy as np

ROOT = os.environ.get("PCMA_SEED_ROOT", os.path.join("data", "SEED"))

def load_mat(path):
    try:
        from scipy.io import loadmat
        return loadmat(path), "scipy"
    except Exception as e:
        try:
            import h5py
            f = h5py.File(path, "r")
            return {k: f[k] for k in f.keys()}, "h5py"
        except Exception as e2:
            return {"__error__": f"{e} | {e2}"}, "none"

def show_keys(d, reader, n=14):
    keys = [k for k in d.keys() if not k.startswith("__")]
    print("  reader:", reader, "| n_keys:", len(keys))
    for k in keys[:n]:
        v = d[k]
        print(f"    {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',type(v).__name__)}")

print("==== SEED (ExtractedFeatures_1s) ====")
ef = os.path.join(ROOT, "SEED", "SEED", "SEED_EEG", "ExtractedFeatures_1s")
files = sorted(glob.glob(os.path.join(ef, "*.mat")))
print("n files:", len(files), "| sample:", [os.path.basename(f) for f in files[:5]])
sample = [f for f in files if os.path.basename(f).lower() != "label.mat"]
if sample:
    d, r = load_mat(sample[0]); print("file:", os.path.basename(sample[0])); show_keys(d, r)
lab = os.path.join(ef, "label.mat")
if os.path.exists(lab):
    d, r = load_mat(lab)
    print("label.mat:", {k: np.array(d[k]).ravel().tolist() for k in d if not k.startswith('__')})

print("\n==== SEED-IV (eeg_feature_smooth/1) ====")
s4 = os.path.join(ROOT, "SEED_IV", "eeg_feature_smooth", "1")
files = sorted(glob.glob(os.path.join(s4, "*.mat")))
print("n files:", len(files), "| sample:", [os.path.basename(f) for f in files[:3]])
if files:
    d, r = load_mat(files[0]); print("file:", os.path.basename(files[0])); show_keys(d, r)

print("\n==== SEED-V (EEG_DE_features) ====")
s5 = os.path.join(ROOT, "SEED-V", "EEG_DE_features")
files = sorted(glob.glob(os.path.join(s5, "*.npz")))
print("n files:", len(files), "| sample:", [os.path.basename(f) for f in files[:3]])
if files:
    z = np.load(files[0], allow_pickle=True)
    print("npz keys:", list(z.keys()))
    for k in list(z.keys())[:6]:
        v = z[k]
        print(f"  {k}: shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',None)}")
        if getattr(v, "dtype", None) == object:
            try:
                inner = v.item()
                print("   (object) ->", type(inner).__name__,
                      ("keys "+str(list(inner.keys())[:5]) if hasattr(inner, "keys") else "len "+str(len(inner))))
            except Exception as e:
                print("   object inspect err:", e)
