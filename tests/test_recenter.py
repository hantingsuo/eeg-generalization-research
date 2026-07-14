# tests/test_recenter.py
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pcma.manifold.recenter import recenter_per_subject

def test_recentered_mean_is_identity(spd_batch):
    subject = np.array([0]*10 + [1]*10)
    Cr = recenter_per_subject(spd_batch, subject)
    for s in (0, 1):
        M = mean_riemann(Cr[subject == s])
        assert distance_riemann(M, np.eye(M.shape[0])) < 1e-4

def test_recentering_is_per_subject_unsupervised(spd_batch):
    # recentering must not use labels; only subject id + covariances
    subject = np.array([0]*10 + [1]*10)
    Cr = recenter_per_subject(spd_batch, subject)
    assert Cr.shape == spd_batch.shape
    # each recentered matrix stays SPD
    for Ci in Cr:
        assert np.linalg.eigvalsh(Ci).min() > 0
