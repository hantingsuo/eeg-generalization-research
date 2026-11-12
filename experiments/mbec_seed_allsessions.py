"""SEED-AS: SEED new-participant hold-out with ALL-SESSIONS training.

A copy of the frozen E1 SEED cross-subject experiment (experiments/jne_e1_seed_cross_subject.py,
plan 2026-09-17) with exactly two changes of training regime:

  (2) training data: the nine source participants of a fold contribute all three SEED
      sessions (91,638 windows instead of 30,546). Validation and target participants are
      scored on SESSION 1 ONLY, with the same 45 (participant, trial) units as E1; their
      session-2/3 recordings are never loaded.
  (4) input standardisation: per recording and label-free. Every loaded recording
      (participant x session) is standardised per feature (62 x 5) with the mean/std of its
      own windows (float64 statistics, ddof=0, std floor 1e-6). For validation and target
      participants that recording is their session-1 recording. No statistics are shared
      across recordings.

Folds, models (DGCNN, MLP), optimiser recipe, 80 epochs, optimisation seeds, candidate
grid, calibration permutations and label schedule are read from / identical to the E1
plan. trace.npz has the same arrays and shapes as E1, so experiments/jne_e1_analysis.py runs
unmodified:  --plan <this plan> --out <this out dir> --tag seedas

Modes:
  freeze   write the frozen plan (refuses to overwrite; refuses while PREDICTIONS is empty)
  run      fit every cell not yet completed; resumable after interruption
  status   print progress
Options:  --limit N   fit at most N new cells in this invocation
          --duty D    GPU busy fraction in (0, 1] (pacing only, see _pace)

Smoke mode (SEEDAS_SMOKE=1): synthetic random arrays with the real shapes, a separate output
directory and plan, SEEDAS_SMOKE_EPOCHS epochs (default 2), optional SEEDAS_SMOKE_STRIDE (keep every
n-th synthetic window, default 1 = real size), PREDICTIONS check bypassed.
It never touches the SEED caches.
"""
from pathlib import Path
import sys, json, time, argparse, contextlib, io, os, hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_selection_sensitivity_v1 as base  # noqa: E402  (sets CUBLAS env)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

SMOKE = os.environ.get("SEEDAS_SMOKE") == "1"
SMOKE_EPOCHS = int(os.environ.get("SEEDAS_SMOKE_EPOCHS", "2"))
SMOKE_STRIDE = int(os.environ.get("SEEDAS_SMOKE_STRIDE", "1"))  # smoke only: keep every n-th synthetic window

# Filled in by the project lead before freeze: the written predictions that go into the plan.
PREDICTIONS = [
    "P1: target accuracy at epoch 80 (window accuracy on the scored clips, participant mean) is higher than "
    "in the 2026-09-17 participant hold-out for both models (reference values 0.486 for DGCNN, 0.528 for MLP).",
    "P2: for both models the retained gain of the zero-label policy, U80-U5, has a 95% participant-bootstrap "
    "interval that includes zero.",
    "P3: for DGCNN the retained gain of the calibrated policy (one labelled clip per class), A80-A5, is "
    "positive with a 95% participant-bootstrap interval that excludes zero.",
    "P4: for both models selection on the scored clips themselves exceeds the zero-label choice at K=80 "
    "(O80-U80 > 0) with a 95% participant-bootstrap interval that excludes zero.",
    "Written 2026-11-12 by the project lead with the 2026-09-17 participant hold-out results known and "
    "before any cell of this experiment was fitted on SEED data. Whichever way they come out, all are "
    "reported and no cell, model or policy is dropped.",
]

OUT = ROOT / ("results/mbec_2026-11/seed_allsessions_smoke" if SMOKE else "results/mbec_2026-11/seed_allsessions")
PLAN = (OUT / "smoke_plan.json") if SMOKE else ROOT / "plans/2026-11-11-mbec-seed-allsessions.json"
E1_PLAN = ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json"
CACHE = ROOT / "results/revision_2026-09-11/seed_family_extension_v4/cache"
AUDIT = ROOT / "results/revision_2026-09-11/seed_family_extension_v4/input_audit.json"

SESSIONS = (1, 2, 3)
N_PART = 15
N_CELLS = 30
# Fraction of wall time the GPU is kept busy (pacing only; computations are unchanged).
DUTY = 1.0

# Real SEED layout (identical in all 45 recordings): windows per trial, trials 1..15.
SYN_TRIAL_WINDOWS = (235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206)


def _pace(t_start):
    """Per-batch pacing, as in jne_e1_faced.py: wait for the GPU, then sleep (1-DUTY)/DUTY times
    the batch time. The sequence of computations is unchanged, so results are identical."""
    if DUTY < 1.0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep((time.perf_counter() - t_start) * (1.0 - DUTY) / DUTY)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def cache_name(session, subject):
    return f"seed_s{session}_p{subject:02d}.npz"


def all_cache_files():
    return [CACHE / cache_name(s, p) for s in SESSIONS for p in range(1, N_PART + 1)]


# ---------------------------------------------------------------- data
def load_recording(session, subject):
    """Raw (x, y, trial) of one recording. Smoke mode returns synthetic arrays of the real shape."""
    if SMOKE:
        e1 = json.loads(E1_PLAN.read_text(encoding="utf-8"))["trial_labels"]
        rng = np.random.default_rng([20261111, session, subject])
        trial = np.repeat(np.arange(1, 16, dtype=np.int16), SYN_TRIAL_WINDOWS)
        y = np.repeat(np.array(e1, dtype=np.int64), SYN_TRIAL_WINDOWS)
        x = (rng.normal(size=(len(y), 62, 5)) * 2.0 + 5.0 + 0.3 * session).astype(np.float32)
        for c in range(3):
            x[y == c, :, c] += 0.5
        return x[::SMOKE_STRIDE], y[::SMOKE_STRIDE], trial[::SMOKE_STRIDE]
    with np.load(CACHE / cache_name(session, subject), allow_pickle=False) as z:
        return z["x"], z["y"], z["trial"]


def standardise_recording(x):
    """Label-free per-recording standardisation (same arithmetic as jne_e1_faced._subject_standardise).

    Per feature (62 x 5): mean and population std (ddof=0) of this recording's own windows in
    float64, std floored at 1e-6. Takes no labels and no other recording's statistics.
    """
    mu = x.mean(0, dtype=np.float64)
    sd = np.maximum(x.std(0, dtype=np.float64), 1e-6)
    return (x - mu.astype(np.float32)) / sd.astype(np.float32)


def load_standardised(session, subject):
    x, y, t = load_recording(session, subject)
    return standardise_recording(x), y, t


def train_recordings(fold):
    """(session, participant) recordings used for training: every session of every train participant."""
    return [(s, p) for p in fold["train"] for s in SESSIONS]


def scored_recordings(subjects):
    """Validation / target participants are scored on session 1 only."""
    return [(1, p) for p in subjects]


def load_recordings(recordings, trial_labels=None):
    xs, ys, ss, ts = [], [], [], []
    for session, p in recordings:
        x, y, t = load_standardised(session, p)
        if trial_labels is not None:  # label schedule must be the one frozen in the plan
            per = [int(y[t == u][0]) for u in np.unique(t)]
            assert per == list(trial_labels), f"label schedule mismatch in s{session} p{p}"
        xs.append(x); ys.append(y); ts.append(t)
        ss.append(np.full(len(y), p, dtype=np.int16))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


# ---------------------------------------------------------------- freeze
def freeze():
    if not SMOKE and not PREDICTIONS:
        raise SystemExit("PREDICTIONS is empty: the project lead must write the predictions into "
                         "experiments/mbec_seed_allsessions.py before freeze")
    if PLAN.exists():
        raise SystemExit(f"plan already frozen: {PLAN}")
    e1 = json.loads(E1_PLAN.read_text(encoding="utf-8"))
    folds = e1["folds"]
    targets = sorted(s for fold in folds for s in fold["test"])
    assert targets == list(range(1, 16)), "every participant must be a target exactly once"
    for fold in folds:
        assert [len(fold[k]) for k in ("train", "validation", "test")] == [9, 3, 3]
        assert not (set(fold["train"]) & set(fold["validation"]) or set(fold["train"]) & set(fold["test"])
                    or set(fold["validation"]) & set(fold["test"]))
    labels = e1["trial_labels"]

    inputs, windows = {}, {}
    if SMOKE:
        inputs = {"SYNTHETIC": "SEEDAS_SMOKE synthetic arrays; no SEED cache is read"}
        windows = {cache_name(s, p): int(sum(SYN_TRIAL_WINDOWS)) for s in SESSIONS for p in range(1, N_PART + 1)}
    else:
        audit = {r["cache_file"]: r["cache_sha256"] for r in
                 json.loads(AUDIT.read_text(encoding="utf-8"))["recordings"] if r["dataset"] == "seed"}
        for p in all_cache_files():
            h = base.sha(p)
            assert audit[p.name] == h, f"cache differs from v4 input audit: {p.name}"
            if p.name.startswith("seed_s1_"):
                assert e1["inputs_sha256"][p.relative_to(ROOT).as_posix()] == h, f"differs from E1 plan: {p.name}"
            with np.load(p, allow_pickle=False) as z:
                y, t = z["y"], z["trial"]
            assert sorted(np.unique(t).tolist()) == list(range(1, 16)), f"trial ids {p.name}"
            assert [int(y[t == u][0]) for u in range(1, 16)] == labels, f"label schedule {p.name}"
            inputs[p.relative_to(ROOT).as_posix()] = h
            windows[p.name] = int(len(y))
        assert len(inputs) == 45

    train_windows = [sum(windows[cache_name(s, p)] for s, p in train_recordings(f)) for f in folds]
    training = dict(e1["training"])
    if SMOKE:
        training["epochs"] = SMOKE_EPOCHS

    cfg = {
        "protocol_id": "mbec_seed_allsessions_v1" + ("_SMOKE_SYNTHETIC" if SMOKE else ""),
        "name": "SEED-AS: SEED new-participant hold-out, all-sessions training",
        "design_status": (
            "Frozen before any SEED-AS model outcome exists. Copy of the 2026-09-17 E1 protocol "
            "(plans/2026-09-17-jne-e1-seed-cross-subject.json); see differs_from_e1 and exposure."
        ),
        "differs_from_e1": (
            "differs from the 2026-09-17 E1 plan only in items (2) and (4): "
            "(2) training data: source participants contribute all three sessions instead of session 1 only; "
            "(4) input standardisation: per recording, label-free, instead of training-participant statistics. "
            "Everything else (folds, models, optimiser recipe, 80 epochs, seeds, candidate grid, calibration "
            "permutations/rotations, label schedule, scored pools, endpoints, analysis code) is identical; "
            "only the protocol id, paths and descriptive text names differ."
        ),
        "exposure": (
            "E1 (session-1 training, source-statistics standardisation) results on the same folds and "
            "participants were known when this plan was written; this experiment extends that protocol and "
            "does not confirm it on untouched participants."
        ),
        "predictions_written_before_running": list(PREDICTIONS),
        "dataset": (
            "SEED official 1-second DE/LDS, 15 participants x 3 sessions x 15 trials. Training: the 9 source "
            "participants of a fold, sessions 1-3 (27 recordings, 91,638 windows). Validation and target "
            "participants: session 1 only (their session-2/3 recordings are not loaded), 15 trials each."
        ),
        "scored_pools": "validation and target participants, SESSION 1 only; units keyed (participant, trial), 45 per pool, as in E1",
        "inputs_sha256": inputs,
        "window_counts_per_recording": windows,
        "training_windows_per_fold": train_windows,
        "trial_labels": labels,
        "folds": folds,
        "fold_source": "identical to E1 plan folds (strict five-fold partition, seed 2024)",
        "e1_plan_sha256": base.sha(E1_PLAN),
        "models": ["dgcnn", "mlp"],
        "optimization_seeds": list(e1["optimization_seeds"]),
        "training": training,
        "training_source": "identical to the E1 plan (v4 cross-session recipe); epoch count counts passes over the larger training set, so gradient steps per epoch triple",
        "vendor_root": e1["vendor_root"],
        "dgcnn_source_sha256": e1["dgcnn_source_sha256"],
        "dgcnn_config_sha256": e1["dgcnn_config_sha256"],
        "normalization": (
            "per recording, label-free: every loaded recording (participant x session) is standardised per "
            "feature (62x5) with the mean and std of its own windows (ddof=0, float64 statistics, std floor 1e-6); "
            "for validation/target participants that recording is their session-1 recording (all 15 trials, "
            "unlabelled, as in jne_e1_faced._subject_standardise). No statistics are shared across recordings; "
            "both models."
        ),
        "budgets": e1["budgets"],
        "candidate_rule": e1["candidate_rule"],
        "selection_metric": e1["selection_metric"],
        "tie_rule": e1["tie_rule"],
        "calibration": e1["calibration"],
        "policies": e1["policies"],
        "primary_endpoints": [
            "zero-label retained change U_80 - U_5 on B",
            "calibration retained change A_80 - A_5 on B",
            "A_80 - U_80 on B",
        ],
        "primary_inference": (
            "per model (DGCNN, MLP): the three endpoints on the frozen nested grid; exact sign-flip permutation "
            "p over 15 participants with Holm correction across the three endpoints within model; "
            "participant bootstrap 10000 (seed 20260917); analysis = experiments/jne_e1_analysis.py unmodified"
        ),
        "retention_definitions": e1["retention_definitions"],
        "aggregation": e1["aggregation"],
        "secondary": e1["secondary"],
        "pilot": e1["pilot"],
        "stopping": e1["stopping"],
        "boundary": (
            "SEED released features with upstream LDS smoothing; every participant watched the same 15 films in "
            "every session, so participant hold-out is not stimulus hold-out. Training uses sessions 1-3 of the "
            "source participants; targets and validation participants are scored on session 1 only and never "
            "contribute any window, label or statistic to training. Per-recording standardisation uses the whole "
            "unlabelled recording of the scored participant (transductive, label-free). No temporal-input model."
        ),
    }
    if SMOKE:
        cfg["smoke"] = "SYNTHETIC DATA, code test only; results are meaningless"
    dump(PLAN, cfg)
    print("frozen", PLAN)


# ---------------------------------------------------------------- cells
def cell_list(cfg):
    pilot = (1, "dgcnn", 2024)
    cells = [(f, m, s) for f in range(1, 6) for m in cfg["models"] for s in cfg["optimization_seeds"]]
    rest = [c for c in cells if c != pilot]
    if SMOKE:  # smoke only: second cell is MLP so that --limit 2 exercises both models
        second = (1, "mlp", 2024)
        rest = [second] + [c for c in rest if c != second]
    return [pilot] + rest


def cell_key(cell):
    f, m, s = cell
    return f"fold{f}_{m}_seed{s}"


def make_model(name, cfg, dgcnn):
    if name == "dgcnn":
        with contextlib.redirect_stdout(io.StringIO()):
            return dgcnn.DGCNN(num_electrodes=62, in_channels=5, num_classes=3)
    a, b = cfg["training"]["mlp"]["hidden"]
    p = cfg["training"]["mlp"]["dropout"]
    return nn.Sequential(nn.Flatten(), nn.Linear(310, a), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(a, b), nn.ReLU(), nn.Dropout(p), nn.Linear(b, 3))


def unit_index(subject, trial):
    """Stable (participant, trial) unit ordering for per-trial arrays (as in E1)."""
    pairs = sorted(set(zip(subject.tolist(), trial.tolist())))
    lookup = {p: i for i, p in enumerate(pairs)}
    idx = np.array([lookup[p] for p in zip(subject.tolist(), trial.tolist())])
    return np.array(pairs, dtype=np.int16), idx


def per_trial(logits, y, idx, n_units):
    correct = np.bincount(idx, weights=(logits.argmax(1) == y).astype(np.float64), minlength=n_units)
    sums = np.zeros((n_units, logits.shape[1]))
    np.add.at(sums, idx, logits.astype(np.float64))
    return correct.astype(np.int32), sums


def predict(model, features, batch_size, device):
    """Same slicing as base.predict, with per-batch pacing."""
    model.eval()
    out = []
    for i in range(0, len(features), batch_size):
        tb = time.perf_counter()
        out.append(model(torch.from_numpy(features[i:i + batch_size]).to(device)).cpu().numpy())
        _pace(tb)
    return np.concatenate(out)


def build_pools(fold, labels):
    """Training arrays (all sessions of train participants) and scored pools (session 1 only)."""
    x, y, _, _ = load_recordings(train_recordings(fold), labels)
    parts = {}
    for part, subs in (("validation", fold["validation"]), ("target", fold["test"])):
        px, py, ps, pt = load_recordings(scored_recordings(subs), labels)
        units, idx = unit_index(ps, pt)
        count = np.bincount(idx, minlength=len(units)).astype(np.int32)
        ulab = np.array([py[idx == i][0] for i in range(len(units))], dtype=np.int8)
        assert all((py[idx == i] == ulab[i]).all() for i in range(len(units)))
        parts[part] = dict(x=px, y=py, idx=idx, units=units, count=count, label=ulab)
    return x, y, parts


def fit(cell, cfg, dgcnn, device):
    f, name, seed = cell
    key = cell_key(cell)
    folder = OUT / "cells" / key
    if (folder / "result.json").exists():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    fold = cfg["folds"][f - 1]
    tr = cfg["training"]
    x, y, parts = build_pools(fold, cfg["trial_labels"])

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
        batch_loss, batch_n = [], []
        for bx, by in loader:
            tb = time.perf_counter()
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(bx.to(device)), by.to(device))
            if reg is not None:
                loss = loss + reg(model)
            loss.backward()
            opt.step()
            batch_loss.append(loss.detach())  # no per-batch host sync; read back once per epoch
            batch_n.append(len(by))
            _pace(tb)
        vals = torch.stack(batch_loss).cpu().tolist()  # float32 values as python floats, as float(loss) gave
        if not all(np.isfinite(vals)):
            raise FloatingPointError(f"{key} epoch {epoch + 1}")
        total = 0.0
        for v, n in zip(vals, batch_n):
            total += v * n
        losses.append(total / len(y))
        with torch.no_grad():
            for p, v in parts.items():
                logits = predict(model, v["x"], tr["evaluation_batch_size"], device)
                c, sums = per_trial(logits, v["y"], v["idx"], len(v["units"]))
                rec[p]["correct"][epoch] = c
                rec[p]["logits"][epoch] = (sums / v["count"][:, None]).astype(np.float32)
    with torch.no_grad():
        train_acc = float(np.mean(predict(model, x, tr["evaluation_batch_size"], device).argmax(1) == y))
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
    seconds = time.monotonic() - t0
    row = {"cell": key, "fold": f, "model": name, "seed": seed, "epochs": epochs,
           "train_windows": int(len(y)),
           "train_window_accuracy_final": train_acc,
           "target_window_accuracy_epoch80": float(tv),
           "loss_first_last": [losses[0], losses[-1]],
           "seconds": round(seconds, 1),
           "seconds_per_epoch": round(seconds / epochs, 2),
           "gpu_duty": DUTY,
           "trace_sha256": base.sha(folder / "trace.npz")}
    dump(folder / "result.json", row)
    return row


def run(limit=None):
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    if not SMOKE:
        assert len(cfg["inputs_sha256"]) == 45, "plan must pin all 45 cache files"
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
    dgcnn = base.load_model_module(cfg)  # checks pinned DGCNN source/config hashes
    device = torch.device("cuda" if (torch.cuda.is_available() or not SMOKE) else "cpu")
    runtime = OUT / "runtime.json"
    if not runtime.exists():
        dump(runtime, {"runner_sha256": base.sha(Path(__file__)), "base_sha256": base.sha(Path(base.__file__)),
                       "torch": torch.__version__, "numpy": np.__version__,
                       "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})
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
        msg = {"completed": n_done, "total": N_CELLS, "cell": row["cell"], "seconds": row["seconds"],
               "train_acc": round(row["train_window_accuracy_final"], 4),
               "loss_first_last": [round(v, 4) for v in row["loss_first_last"]]}
        dump(OUT / "progress.json", msg)
        print(json.dumps(msg), flush=True)
    n_done = len(list((OUT / "cells").glob("*/result.json")))
    if n_done == N_CELLS:
        dump(OUT / "completion.json", {"status": "complete", "cells": N_CELLS})
        print("COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "run", "status"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--duty", type=float, default=0.5, help="GPU busy fraction (pacing only)")
    a = ap.parse_args()
    if not 0.0 < a.duty <= 1.0:
        raise SystemExit("--duty must be in (0, 1]")
    DUTY = a.duty
    if a.mode == "freeze":
        freeze()
    elif a.mode == "run":
        run(a.limit)
    else:
        print((OUT / "progress.json").read_text() if (OUT / "progress.json").exists() else "not started")
