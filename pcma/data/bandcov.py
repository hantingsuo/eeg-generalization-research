# pcma/data/bandcov.py
import numpy as np
from scipy.signal import butter, filtfilt
from pyriemann.estimation import Covariances

# (low, high) Hz for delta, theta, alpha, beta, gamma
BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]


def _bandpass(X, lo, hi, sfreq, order=4):
    nyq = 0.5 * sfreq
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


def to_band_covariances(X, sfreq=250.0, bands=BANDS, order=4, estimator="oas"):
    """Filter-bank SPD covariances -> (n_trials, n_bands, n_ch, n_ch)."""
    cov = Covariances(estimator=estimator)
    out = []
    for lo, hi in bands:
        Xf = _bandpass(X, lo, hi, sfreq, order)
        out.append(cov.fit_transform(Xf))       # (n_trials, n_ch, n_ch)
    return np.stack(out, axis=1)                 # (n_trials, n_bands, n_ch, n_ch)
