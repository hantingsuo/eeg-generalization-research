# tests/test_pipelines.py
import numpy as np
from pcma.model.pipelines import fit_predict_noadapt, fit_predict_ra

def _make(rng, n_per, offset):
    # two classes separated on the manifold; SPD via A A^T
    C, y, sid = [], [], []
    for s in range(2):
        for lab in (0, 1):
            for _ in range(n_per):
                A = rng.standard_normal((6, 6)) + (offset if lab else 0) + 0.3 * s
                S = A @ A.T + 6 * np.eye(6)
                C.append((S + S.T) / 2); y.append(lab); sid.append(s)
    return np.array(C), np.array(y), np.array(sid)

def test_noadapt_beats_chance():
    rng = np.random.default_rng(0)
    Ctr, ytr, str_ = _make(rng, 30, 0.8)
    Cte, yte, ste = _make(rng, 30, 0.8)
    pred = fit_predict_noadapt(Ctr, ytr, Cte)
    assert (pred == yte).mean() > 0.6

def test_ra_returns_valid_labels():
    rng = np.random.default_rng(1)
    Ctr, ytr, str_ = _make(rng, 30, 0.8)
    Cte, yte, ste = _make(rng, 30, 0.8)
    pred = fit_predict_ra(Ctr, ytr, str_, Cte, ste)
    assert pred.shape == yte.shape
    assert set(np.unique(pred)) <= {0, 1}

from pcma.model.pipelines import fit_predict_noadapt_bands, fit_predict_ra_bands

def _make_bands(rng, n_per, n_bands=5, sep_band=3, offset=0.9):
    C, y, sid = [], [], []
    for s in range(2):
        for lab in (0, 1):
            for _ in range(n_per):
                bands = []
                for b in range(n_bands):
                    off = offset if (lab and b == sep_band) else 0.0
                    A = rng.standard_normal((6, 6)) + off + 0.2 * s
                    S = A @ A.T + 6 * np.eye(6)
                    bands.append((S + S.T) / 2)
                C.append(np.stack(bands)); y.append(lab); sid.append(s)
    return np.array(C), np.array(y), np.array(sid)

def test_noadapt_bands_beats_chance():
    rng = np.random.default_rng(0)
    Ctr, ytr, _ = _make_bands(rng, 30); Cte, yte, _ = _make_bands(rng, 30)
    pred = fit_predict_noadapt_bands(Ctr, ytr, Cte)
    assert pred.shape == yte.shape
    assert (pred == yte).mean() > 0.6

def test_ra_bands_valid_labels():
    rng = np.random.default_rng(1)
    Ctr, ytr, str_ = _make_bands(rng, 30); Cte, yte, ste = _make_bands(rng, 30)
    pred = fit_predict_ra_bands(Ctr, ytr, str_, Cte, ste)
    assert pred.shape == yte.shape
    assert set(np.unique(pred)) <= {0, 1}

def test_ra_bands_ref_means_equivalence():
    from pcma.model.pipelines import subject_band_means
    rng = np.random.default_rng(2)
    def mk(subjects):
        C, y, sid = [], [], []
        for s in subjects:
            for lab in (0, 1):
                for _ in range(15):
                    bands = []
                    for b in range(5):
                        off = 0.8 if (lab and b == 2) else 0.0
                        A = rng.standard_normal((6, 6)) + off + 0.2 * s
                        S = A @ A.T + 6 * np.eye(6); bands.append((S + S.T) / 2)
                    C.append(np.stack(bands)); y.append(lab); sid.append(s)
        return np.array(C), np.array(y), np.array(sid)
    Ctr, ytr, str_ = mk([0, 1])
    Cte, yte, ste = mk([2, 3])            # disjoint subjects from train
    Call = np.concatenate([Ctr, Cte]); sall = np.concatenate([str_, ste])
    sbm = subject_band_means(Call, sall)
    p_no  = fit_predict_ra_bands(Ctr, ytr, str_, Cte, ste)
    p_ref = fit_predict_ra_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm)
    assert np.array_equal(p_no, p_ref)

def test_band_tangent_features_shapes():
    from pcma.model.pipelines import band_tangent_features
    rng = np.random.default_rng(4)
    Ctr, ytr, str_ = _make_bands(rng, 20)
    Cte, yte, ste = _make_bands(rng, 20)
    Xtr, Xte = band_tangent_features(Ctr, Cte, recenter=False)
    assert Xtr.shape == (Ctr.shape[0], 5 * 21)   # 5 bands * (6*7/2)
    assert Xte.shape == (Cte.shape[0], 5 * 21)
    Xtr2, Xte2 = band_tangent_features(Ctr, Cte, str_, ste, recenter=True)
    assert Xtr2.shape == Xtr.shape and Xte2.shape == Xte.shape

def test_global_mean_align_matches_overall_mean():
    from pcma.model.pipelines import global_mean_align
    rng = np.random.default_rng(7)
    Xs = rng.normal(0, 1, (80, 4)); Xt = rng.normal(3, 1, (60, 4))
    Xt_a = global_mean_align(Xs, Xt)
    assert np.allclose(Xt_a.mean(axis=0), Xs.mean(axis=0), atol=1e-8)

def test_conditional_align_uses_confident_only_and_gates_small_classes():
    from pcma.model.pipelines import conditional_mean_align
    rng = np.random.default_rng(8)
    Xs = np.vstack([rng.normal(0, 1, (50, 3)), rng.normal(5, 1, (50, 3))])
    ys = np.array([0]*50 + [1]*50)
    Xt = np.vstack([rng.normal(2, 1, (40, 3)), rng.normal(2, 1, (40, 3))])
    yt = np.array([0]*40 + [1]*40)
    # class 1 has only 2 confident samples -> below min_count -> must stay unaligned
    conf = np.concatenate([np.ones(40), np.array([1.0, 1.0] + [0.0]*38)])
    Xt_a = conditional_mean_align(Xs, ys, Xt, yt, conf=conf, conf_thresh=0.5, min_count=5)
    assert np.allclose(Xt_a[yt == 0].mean(axis=0), Xs[ys == 0].mean(axis=0), atol=1e-8)
    assert np.allclose(Xt_a[yt == 1], Xt[yt == 1])   # class 1 gated out -> unchanged

def test_pcma_bands_valid_labels_all_modes():
    from pcma.model.pipelines import fit_predict_pcma_bands
    rng = np.random.default_rng(9)
    Ctr, ytr, str_ = _make_bands(rng, 25)
    Cte, yte, ste = _make_bands(rng, 25)
    for mode in ("global", "conditional", "iterative"):
        pred = fit_predict_pcma_bands(Ctr, ytr, str_, Cte, ste, mode=mode)
        assert pred.shape == yte.shape
        assert set(np.unique(pred)) <= {0, 1}

def test_pcma_bands_no_target_label_in_signature():
    import inspect
    from pcma.model.pipelines import fit_predict_pcma_bands
    params = list(inspect.signature(fit_predict_pcma_bands).parameters)
    assert "yte" not in params and "y_target" not in params and "y_te" not in params

def test_coral_align_reduces_covariance_gap():
    from pcma.model.pipelines import coral_align
    rng = np.random.default_rng(3)
    Xs = rng.normal(0, 1, (400, 5)) @ np.diag([3., 1, 1, 1, 1])
    Xt = rng.normal(0, 1, (400, 5)) @ np.diag([1., 1, 1, 1, 3])
    Xt_a = coral_align(Xs, Xt, shrink=0.0)
    Cs = np.cov(Xs, rowvar=False)
    d_before = np.linalg.norm(np.cov(Xt, rowvar=False) - Cs)
    d_after = np.linalg.norm(np.cov(Xt_a, rowvar=False) - Cs)
    assert d_after < 0.25 * d_before          # covariance strongly aligned to source

def test_band_tangent_blocks_shapes():
    from pcma.model.pipelines import band_tangent_blocks
    rng = np.random.default_rng(12)
    Ctr, ytr, str_ = _make_bands(rng, 15)
    Cte, yte, ste = _make_bands(rng, 15)
    ftr, fte = band_tangent_blocks(Ctr, Cte, recenter=False)
    assert len(ftr) == 5 and len(fte) == 5
    assert ftr[0].shape == (Ctr.shape[0], 21) and fte[0].shape == (Cte.shape[0], 21)

def test_fit_predict_coral_bands_valid_and_no_target_label():
    import inspect
    from pcma.model.pipelines import fit_predict_coral_bands
    rng = np.random.default_rng(13)
    Ctr, ytr, str_ = _make_bands(rng, 25)
    Cte, yte, ste = _make_bands(rng, 25)
    pred = fit_predict_coral_bands(Ctr, ytr, str_, Cte, ste)
    assert pred.shape == yte.shape and set(np.unique(pred)) <= {0, 1}
    params = list(inspect.signature(fit_predict_coral_bands).parameters)
    assert "yte" not in params and "y_target" not in params
