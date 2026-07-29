# pcma/data/features.py
"""Rich per-trial EEG features for the 30-channel competition data:
per-band DE + relative power + band ratios + Hjorth + left-right asymmetry."""
import numpy as np
from scipy.signal import butter, filtfilt

BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]  # d th a b g
# left-right mirror channel index pairs for the 30-ch competition montage (0-based).
# order: FP1 FP2 F7 F3 FZ F4 F8 FT7 FC3 FCZ FC4 FT8 T3 C3 CZ C4 T4 TP7 CP3 CPZ CP4 TP8 T5 P3 PZ P4 T6 O1 OZ O2
LR_PAIRS = [(0, 1), (2, 6), (3, 5), (7, 11), (8, 10), (12, 16), (13, 15),
            (17, 21), (18, 20), (22, 26), (23, 25), (27, 29)]  # 12 pairs (midline chans unpaired)

N_CH = 30
N_FEATURES = (N_CH * 5) + (N_CH * 5) + (N_CH * 5) + N_CH + N_CH + (len(LR_PAIRS) * 5)  # DE+relpow+ratios+mob+comp+asym = 570


def _bandpass(x, lo, hi, sf, order=4):
    b, a = butter(order, [lo / (sf / 2), hi / (sf / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def _de(x):
    """Differential entropy of ~Gaussian signal: 0.5 log(2 pi e var)."""
    var = x.var(axis=-1) + 1e-12
    return 0.5 * np.log(2 * np.pi * np.e * var)


def _hjorth(x):
    d1 = np.diff(x, axis=-1); d2 = np.diff(d1, axis=-1)
    v0 = x.var(axis=-1) + 1e-12; v1 = d1.var(axis=-1) + 1e-12; v2 = d2.var(axis=-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = (np.sqrt(v2 / v1)) / mob
    return mob, comp


def extract_rich_features(X, sfreq=250.0):
    """X: (n_trials, n_ch, n_times) -> (n_trials, N_FEATURES)."""
    X = np.asarray(X, float)
    n, ch, T = X.shape
    de, pw = [], []
    for lo, hi in BANDS:
        xf = _bandpass(X, lo, hi, sfreq)
        de.append(_de(xf))            # (n, ch)
        pw.append(xf.var(axis=-1))    # (n, ch)
    de = np.stack(de, axis=-1)        # (n, ch, 5)
    pw = np.stack(pw, axis=-1)        # (n, ch, 5)
    rel = pw / (pw.sum(axis=-1, keepdims=True) + 1e-12)
    d, th, a, b, g = [pw[..., i] for i in range(5)]
    ratios = np.stack([a / (b + 1e-12), th / (b + 1e-12), a / (th + 1e-12),
                       b / (g + 1e-12), d / (th + 1e-12)], axis=-1)  # (n, ch, 5)
    mob, comp = _hjorth(X)            # (n, ch), (n, ch)
    asym = np.concatenate([de[:, l, :] - de[:, r, :] for (l, r) in LR_PAIRS], axis=-1)  # (n, 5*pairs)
    feats = np.concatenate([de.reshape(n, -1), rel.reshape(n, -1), ratios.reshape(n, -1),
                            mob, comp, asym], axis=-1)
    return feats
