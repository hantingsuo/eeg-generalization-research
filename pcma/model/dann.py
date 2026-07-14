"""DANN + group-DRO style classifier for cross-population EEG transfer."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from pcma.model.novelty_eval import source_subject_validation_mask
from pcma.model.rich import per_subject_zscore


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float):
    return _GradientReverse.apply(x, lambd)


class _DANNNet(torch.nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int, embedding: int, dropout: float):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, embedding),
            torch.nn.ReLU(),
        )
        self.label_head = torch.nn.Linear(embedding, n_classes)
        self.domain_head = torch.nn.Sequential(
            torch.nn.Linear(embedding, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 2),
        )

    def forward(self, x, lambd: float = 0.0):
        z = self.encoder(x)
        y = self.label_head(z)
        d = self.domain_head(grad_reverse(z, lambd))
        return y, d


@dataclass(frozen=True)
class DANNConfig:
    hidden: int = 128
    embedding: int = 64
    dropout: float = 0.25
    epochs: int = 60
    patience: int = 12
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_max: float = 0.5
    val_fraction: float = 0.25
    seed: int = 0


def _prepare(Ftr, subject_tr, Fte, subject_te):
    Ftr = per_subject_zscore(Ftr, subject_tr)
    Fte = per_subject_zscore(Fte, subject_te)
    scaler = StandardScaler().fit(Ftr)
    return scaler.transform(Ftr).astype(np.float32), scaler.transform(Fte).astype(np.float32)


def _sample_indices(rng, n: int, size: int):
    if n <= size:
        return rng.choice(n, size=size, replace=True)
    return rng.choice(n, size=size, replace=False)


def _group_dro_loss(loss_each, groups):
    unique = torch.unique(groups)
    if len(unique) <= 1:
        return loss_each.mean()
    losses = torch.stack([loss_each[groups == g].mean() for g in unique])
    return losses.max()


def fit_predict_dann_proba(
    Ftr,
    ytr,
    subject_tr,
    Fte,
    subject_te,
    group_tr=None,
    config: DANNConfig | None = None,
):
    """Train DANN on labeled source + unlabeled target X; target labels are never accepted."""
    cfg = config or DANNConfig()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    device = torch.device("cpu")

    Xs, Xt = _prepare(Ftr, subject_tr, Fte, subject_te)
    ytr = np.asarray(ytr)
    classes = np.unique(ytr)
    y_index = np.searchsorted(classes, ytr).astype(np.int64)
    group_tr = np.asarray(group_tr if group_tr is not None else subject_tr)
    _, group_index = np.unique(group_tr, return_inverse=True)

    train_mask, val_mask = source_subject_validation_mask(subject_tr, cfg.val_fraction, seed=cfg.seed)
    if val_mask.sum() == 0 or len(np.unique(y_index[val_mask])) < min(len(classes), 2):
        train_mask = np.ones(len(y_index), dtype=bool)
        val_mask = np.zeros(len(y_index), dtype=bool)

    Xs_t = torch.from_numpy(Xs).to(device)
    Xt_t = torch.from_numpy(Xt).to(device)
    y_t = torch.from_numpy(y_index).long().to(device)
    g_t = torch.from_numpy(group_index).long().to(device)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    net = _DANNNet(Xs.shape[1], len(classes), cfg.hidden, cfg.embedding, cfg.dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    domain_ce = torch.nn.CrossEntropyLoss()

    best_state = None
    best_score = -np.inf
    stale = 0
    n_batches = max(1, int(np.ceil(max(len(train_idx), len(Xt)) / cfg.batch_size)))

    for epoch in range(cfg.epochs):
        net.train()
        lambd = cfg.lambda_max * min(1.0, (epoch + 1) / max(1, cfg.epochs // 2))
        for _ in range(n_batches):
            src_local = _sample_indices(np_rng, len(train_idx), cfg.batch_size)
            src_idx = train_idx[src_local]
            tgt_idx = _sample_indices(np_rng, len(Xt), cfg.batch_size)
            xs = Xs_t[src_idx]
            xt = Xt_t[tgt_idx]
            logits_s, dom_s = net(xs, lambd)
            _, dom_t = net(xt, lambd)
            cls_loss_each = ce(logits_s, y_t[src_idx])
            cls_loss = _group_dro_loss(cls_loss_each, g_t[src_idx])
            dom_labels_s = torch.zeros(len(src_idx), dtype=torch.long, device=device)
            dom_labels_t = torch.ones(len(tgt_idx), dtype=torch.long, device=device)
            dom_loss = 0.5 * (domain_ce(dom_s, dom_labels_s) + domain_ce(dom_t, dom_labels_t))
            loss = cls_loss + lambd * dom_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            if len(val_idx):
                val_logits, _ = net(Xs_t[val_idx], 0.0)
                pred = val_logits.argmax(dim=1).cpu().numpy()
                score = balanced_accuracy_score(y_index[val_idx], pred)
            else:
                tr_logits, _ = net(Xs_t[train_idx], 0.0)
                pred = tr_logits.argmax(dim=1).cpu().numpy()
                score = balanced_accuracy_score(y_index[train_idx], pred)
        if score > best_score + 1e-5:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        logits, _ = net(Xt_t, 0.0)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    pred = classes[proba.argmax(axis=1)]
    return pred, proba, classes
