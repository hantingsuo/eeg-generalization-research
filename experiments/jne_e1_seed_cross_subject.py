"""E1 SEED anchor: checkpoint-search retention on unseen participants.

Third point of the retention gradient (trials within a session -> sessions ->
participants). Participants are split into train / validation / target by the
existing strict five-fold partition (partition seed 2024), so every participant is
a target exactly once. Training settings follow the v4 cross-session study so the
three generalization axes share one optimisation recipe.

Each epoch stores, for every validation and target trial, the number of correctly
classified windows and the mean window logits. Nothing else is needed to recompute
any selection rule afterwards, so the zero-label policy (select on validation
participants) and the calibration policy (select on a few labelled target trials)
come from the same fitted trajectories. Target labels never touch gradients,
normalisation, stopping or settings.

Modes:
  freeze   write the frozen plan (refuses to overwrite)
  run      fit every cell not yet completed; resumable after interruption
  status   print progress
"""
from pathlib import Path
import sys, json, time, argparse, contextlib, io, os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_selection_sensitivity_v1 as base  # noqa: E402  (sets CUBLAS env)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

PLAN = ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json"
OUT = ROOT / "results/jne_revision_2026-09/e1_seed_cross_subject"
CACHE = ROOT / "results/revision_2026-09-11/seed_family_extension_v4/cache"
V4_PLAN = ROOT / "plans/2026-09-11-seed-family-extension-v4.json"
STRICT = ROOT / "results/strict_fivefold/seed"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def freeze():
    if PLAN.exists():
        raise SystemExit(f"plan already frozen: {PLAN}")
    v4 = json.loads(V4_PLAN.read_text(encoding="utf-8"))
    folds = []
    for f in range(1, 6):
        d = json.loads((STRICT / f"seed_fold{f}_opt2024.json").read_text(encoding="utf-8"))
        folds.append({k: [int(s) for s in d["split_one_based"][k]] for k in ("train", "validation", "test")})
    targets = sorted(s for fold in folds for s in fold["test"])
    assert targets == list(range(1, 16)), "every participant must be a target exactly once"
    for fold in folds:
        assert not (set(fold["train"]) & set(fold["validation"]) or set(fold["train"]) & set(fold["test"])
                    or set(fold["validation"]) & set(fold["test"]))

    # Calibration rotations: per participant and class, a fixed permutation of that
    # class's session-1 trials. Rotation r takes the r-th trial of each class as C.
    labels = None
    inputs = {}
    for s in range(1, 16):
        p = CACHE / f"seed_s1_p{s:02d}.npz"
        inputs[p.relative_to(ROOT).as_posix()] = base.sha(p)
        with np.load(p, allow_pickle=False) as z:
            per_trial = {int(t): int(z["y"][z["trial"] == t][0]) for t in np.unique(z["trial"])}
        lab = [per_trial[t] for t in sorted(per_trial)]
        assert labels is None or lab == labels, "SEED session-1 label schedule must be shared"
        labels = lab
    labels = np.array(labels)
    rng = np.random.default_rng(20260917)
    perms = {int(c): [int(t) for t in rng.permutation(np.flatnonzero(labels == c) + 1)]
             for c in sorted(set(labels.tolist()))}
    assert all(len(v) == 5 for v in perms.values())

    cfg = {
        "protocol_id": "jne_e1_seed_cross_subject_v1",
        "design_status": (
            "Frozen 2026-09-17 before any E1 model outcome. SEED session 1 and the strict five-fold "
            "participant partition were used in earlier work (historical exposure); this is a "
            "protocol extension, not independent confirmation."
        ),
        "dataset": "SEED official 1-second DE/LDS, session 1, 15 participants, 15 trials each",
        "inputs_sha256": inputs,
        "trial_labels": labels.tolist(),
        "folds": folds,
        "fold_source": "results/strict_fivefold/seed/seed_fold{1..5}_opt2024.json split_one_based (partition seed 2024)",
        "models": ["dgcnn", "mlp"],
        "optimization_seeds": [2024, 2025, 2026],
        "training": {k: v4[k] for k in ("epochs", "batch_size", "evaluation_batch_size", "learning_rate",
                                         "optimizer", "weight_decay", "adam_epsilon", "dgcnn_regularizer", "mlp")},
        "training_source": "identical to v4 cross-session study (plans/2026-09-11-seed-family-extension-v4.json)",
        "vendor_root": v4["vendor_root"],
        "dgcnn_source_sha256": v4["dgcnn_source_sha256"],
        "dgcnn_config_sha256": v4["dgcnn_config_sha256"],
        "normalization": "per-feature mean/std from training participants' windows only; std floor 1e-6; both models",
        "budgets": [1, 5, 10, 20, 40, 80],
        "candidate_rule": "nested grid 80/K, 2(80/K), ..., 80 (same as rotation_v2 and v4)",
        "selection_metric": "window accuracy",
        "tie_rule": "earliest maximum",
        "calibration": {
            "seed": 20260917,
            "class_trial_permutations": perms,
            "primary_k": 1,
            "rotations": 5,
            "rule": "rotation r: C = perms[c][r] for each class c; B = remaining target trials",
            "secondary_k": 2,
            "secondary_rule": "rotation r: C = perms[c][r], perms[c][(r+1)%5]; B = remaining (B differs from k=1; reported separately)",
        },
        "policies": {
            "U_K": "select on pooled validation-participant windows; score on target B (zero target labels)",
            "A_K": "select on target C windows; score on target B (calibration labels, no weight update)",
            "O_K": "select on target B; score on B. Diagnostic only, computed after U/A are exported",
            "F": "epoch 80; score on B",
        },
        "primary_endpoints": [
            "zero-label retained change U_80 - U_5 on B",
            "calibration retained change A_80 - A_5 on B",
            "A_80 - U_80 on B",
        ],
        "retention_definitions": {
            "zero_label": "apparent = a_V(h_V(80)) - a_V(h_V(5)); retained = a_B(h_V(80)) - a_B(h_V(5))",
            "calibration": "apparent = a_C(h_C(80)) - a_C(h_C(5)); retained = a_B(h_C(80)) - a_B(h_C(5))",
            "summary": "aggregate ratio mean(retained)/mean(apparent) and OLS slope over participants, as in experiments/jne_retention_curve.py",
        },
        "aggregation": "average over 3 seeds and 5 calibration rotations within participant; n=15 participants; participant bootstrap 10000, seed 20260917",
        "secondary": ["balanced window accuracy", "trial-level accuracy", "k=2 calibration", "MLP", "seed dispersion"],
        "pilot": "fold1/dgcnn/seed2024 first; gate only finite loss, shapes, label validity; included unchanged; low accuracy is not an exclusion or tuning trigger",
        "stopping": "all 30 cells retained; no significance-dependent expansion or tuning",
        "boundary": (
            "Released features with upstream LDS smoothing; every participant watched the same 15 films, so "
            "participant hold-out is not stimulus hold-out. Session 1 only. No temporal-input model in this anchor."
        ),
    }
    dump(PLAN, cfg)
    print("frozen", PLAN)


def cell_list(cfg):
    pilot = (1, "dgcnn", 2024)
    cells = [(f, m, s) for f in range(1, 6) for m in cfg["models"] for s in cfg["optimization_seeds"]]
    return [pilot] + [c for c in cells if c != pilot]


def cell_key(cell):
    f, m, s = cell
    return f"fold{f}_{m}_seed{s}"


def load_subjects(subjects):
    xs, ys, ss, ts = [], [], [], []
    for s in subjects:
        with np.load(CACHE / f"seed_s1_p{s:02d}.npz", allow_pickle=False) as z:
            xs.append(z["x"]); ys.append(z["y"]); ts.append(z["trial"])
            ss.append(np.full(len(z["y"]), s, dtype=np.int16))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


def make_model(name, cfg, dgcnn):
    if name == "dgcnn":
        with contextlib.redirect_stdout(io.StringIO()):
            return dgcnn.DGCNN(num_electrodes=62, in_channels=5, num_classes=3)
    a, b = cfg["training"]["mlp"]["hidden"]
    p = cfg["training"]["mlp"]["dropout"]
    return nn.Sequential(nn.Flatten(), nn.Linear(310, a), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(a, b), nn.ReLU(), nn.Dropout(p), nn.Linear(b, 3))


def unit_index(subject, trial):
    """Stable (participant, trial) unit ordering for per-trial arrays."""
    pairs = sorted(set(zip(subject.tolist(), trial.tolist())))
    lookup = {p: i for i, p in enumerate(pairs)}
    idx = np.array([lookup[p] for p in zip(subject.tolist(), trial.tolist())])
    return np.array(pairs, dtype=np.int16), idx


def per_trial(logits, y, idx, n_units):
    correct = np.bincount(idx, weights=(logits.argmax(1) == y).astype(np.float64), minlength=n_units)
    sums = np.zeros((n_units, logits.shape[1]))
    np.add.at(sums, idx, logits.astype(np.float64))
    return correct.astype(np.int32), sums


def fit(cell, cfg, dgcnn, device):
    f, name, seed = cell
    key = cell_key(cell)
    folder = OUT / "cells" / key
    if (folder / "result.json").exists():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    fold = cfg["folds"][f - 1]
    tr = cfg["training"]
    x, y, _, _ = load_subjects(fold["train"])
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    x = (x - mean) / scale
    parts = {}
    for part, subs in (("validation", fold["validation"]), ("target", fold["test"])):
        px, py, ps, pt = load_subjects(subs)
        units, idx = unit_index(ps, pt)
        count = np.bincount(idx, minlength=len(units)).astype(np.int32)
        ulab = np.array([py[idx == i][0] for i in range(len(units))], dtype=np.int8)
        assert all((py[idx == i] == ulab[i]).all() for i in range(len(units)))
        parts[part] = dict(x=(px - mean) / scale, y=py, idx=idx, units=units, count=count, label=ulab)

    base.seed_all(seed)
    model = make_model(name, cfg, dgcnn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"],
                            eps=tr["adam_epsilon"])
    reg = dgcnn.NewSparseL2Regularization(tr["dgcnn_regularizer"]).to(device) if name == "dgcnn" else None
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=tr["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    epochs = tr["epochs"]
    rec = {p: dict(correct=np.zeros((epochs, len(v["units"])), np.int32),
                   logits=np.zeros((epochs, len(v["units"]), 3), np.float32)) for p, v in parts.items()}
    losses = []
    t0 = time.monotonic()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(bx.to(device)), by.to(device))
            if reg is not None:
                loss = loss + reg(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{key} epoch {epoch + 1}")
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(by)
        losses.append(total / len(y))
        with torch.no_grad():
            for p, v in parts.items():
                logits = base.predict(model, v["x"], tr["evaluation_batch_size"], device)
                c, sums = per_trial(logits, v["y"], v["idx"], len(v["units"]))
                rec[p]["correct"][epoch] = c
                rec[p]["logits"][epoch] = (sums / v["count"][:, None]).astype(np.float32)
    with torch.no_grad():
        train_acc = float(np.mean(base.predict(model, x, tr["evaluation_batch_size"], device).argmax(1) == y))
    arrays = {"loss": np.array(losses)}
    for p, v in parts.items():
        arrays[f"{p}_units"] = v["units"]
        arrays[f"{p}_count"] = v["count"]
        arrays[f"{p}_label"] = v["label"]
        arrays[f"{p}_correct"] = rec[p]["correct"]
        arrays[f"{p}_trial_logits"] = rec[p]["logits"]
    tmp = folder / "trace.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, folder / "trace.npz")
    tv = arrays["target_correct"][-1].sum() / arrays["target_count"].sum()
    row = {"cell": key, "fold": f, "model": name, "seed": seed, "epochs": epochs,
           "train_window_accuracy_final": train_acc,
           "target_window_accuracy_epoch80": float(tv),
           "loss_first_last": [losses[0], losses[-1]],
           "seconds": round(time.monotonic() - t0, 1),
           "trace_sha256": base.sha(folder / "trace.npz")}
    dump(folder / "result.json", row)
    return row


def run(limit=None):
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    for rel, h in cfg["inputs_sha256"].items():
        if base.sha(ROOT / rel) != h:
            raise RuntimeError(f"input changed since freeze: {rel}")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_copy = OUT / "frozen_manifest.json"
    if not frozen_copy.exists():
        frozen_copy.write_bytes(PLAN.read_bytes())
    elif frozen_copy.read_bytes() != PLAN.read_bytes():
        raise RuntimeError("plan differs from the copy used by earlier cells")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dgcnn = base.load_model_module(cfg)
    device = torch.device("cuda")
    runtime = OUT / "runtime.json"
    if not runtime.exists():
        dump(runtime, {"runner_sha256": base.sha(Path(__file__)), "base_sha256": base.sha(Path(base.__file__)),
                       "torch": torch.__version__, "numpy": np.__version__,
                       "gpu": torch.cuda.get_device_name(0)})
    elif json.loads(runtime.read_text())["runner_sha256"] != base.sha(Path(__file__)):
        raise RuntimeError("runner changed after cells were produced")
    done = 0
    for cell in cell_list(cfg):
        if limit is not None and done >= limit:
            break
        row = fit(cell, cfg, dgcnn, device)
        if row is None:
            continue
        done += 1
        n_done = len(list((OUT / "cells").glob("*/result.json")))
        msg = {"completed": n_done, "total": 30, "cell": row["cell"], "seconds": row["seconds"],
               "train_acc": round(row["train_window_accuracy_final"], 4),
               "loss_first_last": [round(v, 4) for v in row["loss_first_last"]]}
        dump(OUT / "progress.json", msg)
        print(json.dumps(msg), flush=True)
    n_done = len(list((OUT / "cells").glob("*/result.json")))
    if n_done == 30:
        dump(OUT / "completion.json", {"status": "complete", "cells": 30})
        print("COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "run", "status"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.mode == "freeze":
        freeze()
    elif a.mode == "run":
        run(a.limit)
    else:
        print((OUT / "progress.json").read_text() if (OUT / "progress.json").exists() else "not started")
