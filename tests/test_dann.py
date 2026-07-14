import inspect

import numpy as np

from pcma.model.dann import DANNConfig, fit_predict_dann_proba, grad_reverse


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    Ftr, ytr, str_, group = [], [], [], []
    Fte, ste = [], []
    for s in range(4):
        for y in (0, 1):
            x = rng.normal(size=(10, 8)) + y * 1.5 + s * 0.1
            Ftr.append(x)
            ytr.extend([y] * len(x))
            str_.extend([f"S{s}"] * len(x))
            group.extend(["A" if s < 2 else "B"] * len(x))
    for y in (0, 1):
        x = rng.normal(size=(8, 8)) + y * 1.5 + 0.3
        Fte.append(x)
        ste.extend(["T"] * len(x))
    return np.vstack(Ftr), np.array(ytr), np.array(str_), np.array(group), np.vstack(Fte), np.array(ste)


def test_dann_signature_has_no_target_labels():
    params = list(inspect.signature(fit_predict_dann_proba).parameters)
    assert "yte" not in params and "y_target" not in params


def test_dann_returns_probabilities():
    Ftr, ytr, str_, group, Fte, ste = _toy()
    pred, proba, classes = fit_predict_dann_proba(
        Ftr,
        ytr,
        str_,
        Fte,
        ste,
        group_tr=group,
        config=DANNConfig(epochs=3, patience=2, batch_size=16, seed=2),
    )
    assert pred.shape == (len(Fte),)
    assert proba.shape == (len(Fte), 2)
    assert set(classes.tolist()) == {0, 1}
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_grad_reverse_forward_identity():
    import torch

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = grad_reverse(x, 1.0)
    assert torch.allclose(x, y)
