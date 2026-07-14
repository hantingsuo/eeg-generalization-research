# pcma/manifold/recenter.py
import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm

def recenter_per_subject(C, subject, ref_means=None):
    """Whiten each subject's covariances by its Riemannian mean (Zanini RA).

    Unsupervised: uses only covariances + subject id. If ref_means is given
    (dict subject->mean), use it instead of recomputing (for held-out subjects
    whose mean is estimated from their own test covariances)."""
    C = np.asarray(C, dtype=float)
    out = np.empty_like(C)
    for s in np.unique(subject):
        m = subject == s
        M = ref_means[s] if ref_means is not None and s in ref_means else mean_riemann(C[m])
        W = invsqrtm(M)
        out[m] = W @ C[m] @ W
    return out
