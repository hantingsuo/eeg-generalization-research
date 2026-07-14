import numpy as np
from pcma.data.bandcov import to_band_covariances, BANDS


def test_band_cov_shape_and_spd():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 30, 1250))  # 6 trials, 30 ch, 5s @250Hz
    C = to_band_covariances(X, sfreq=250.0)
    assert C.shape == (6, len(BANDS), 30, 30)
    for trial in C:
        for Cb in trial:
            assert np.allclose(Cb, Cb.T, atol=1e-8)
            assert np.linalg.eigvalsh(Cb).min() > 0


def test_bands_are_five():
    assert len(BANDS) == 5
