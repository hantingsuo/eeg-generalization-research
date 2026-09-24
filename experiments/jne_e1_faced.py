"""E1/E5 on FACED: checkpoint-search retention on unseen participants, with and
without stimulus hold-out.

Written before the data arrived (2026-09-17) from the documented release layout:
  Processed_data/subXXX.pkl        (28 clips, 32 channels, 7500 samples @ 250 Hz)
  EEG_Features/DE/subXXX.pkl*      (28 clips, 32 channels, 30 one-second windows, 5 bands)
Clip index = video id 1..28; channels 31-32 are A1/A2 and are dropped (TorchEEG
convention). Every assumption is asserted in `audit`; nothing is fitted there.

Design decisions fixed before any FACED file was opened (see the plan for the rest):
* Participants that pass the structural audit are split by a fixed seed into
  60 training / 23 validation / 40 target (confirmation) participants. Target
  participants are never used in technical pilots.
* Primary scoring follows the SEED designs (window accuracy for selection and
  scoring); balanced window accuracy and clip accuracy are secondary. With 19
  scored clips per participant a clip-level primary endpoint would be too coarse.
* EEGNet input is decimated 250 -> 125 Hz per clip (zero-phase FIR, factor 2).
  The released signals are band-passed 0.05-47 Hz, below the new Nyquist
  frequency (62.5 Hz); the audit reports the power fraction above 47 Hz. The
  reason is memory: 60 training participants at 250 Hz do not fit on an 8 GB GPU.
* Stimulus hold-out: one clip per class, chosen by a fixed seed, is removed from
  the training and validation participants in the hold-out condition and is the
  scored set B* for targets in both conditions. Selection pools (validation clips
  and target calibration clips) exclude B* in both conditions, so the two
  conditions differ only in whether the source models saw B* clips (and in the
  amount of source training data, which is reported).

Modes:
  audit  --root DIR   structural checks + caches (no model, no labels used beyond mapping)
  freeze              write the frozen plan with split, rotations, hold-out and input hashes
  pilot               technical pilot on development participants only (not analysed)
  run                 all cells, resumable
Environment FACED_SMOKE=1 switches to a tiny synthetic layout for code testing.
"""
from pathlib import Path
import sys, os, re, json, time, argparse, pickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_selection_sensitivity_v1 as base  # noqa: E402
from experiments.jne_e1_seed_cross_subject import dump, per_trial, unit_index  # noqa: E402
from experiments.jne_e2_eegnet_seed import EEGNet  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

SMOKE = os.environ.get("FACED_SMOKE") == "1"
# v2: per-participant (label-free) standardisation instead of training-set statistics.
# v1 produced near-chance decoding on FACED, which makes retention ratios uninterpretable
# (the denominator is ~0). Per-participant standardisation is standard practice in
# cross-subject EEG and uses no labels, so the zero-target-label condition is preserved.
V2 = os.environ.get("FACED_V2") == "1"
TAG = "faced_smoke" if SMOKE else ("faced_v2" if V2 else "faced")
OUT = ROOT / f"results/jne_revision_2026-09/e1_{TAG}"
CACHE = ROOT / "results/jne_revision_2026-09/e1_faced/cache"  # audited caches are shared
AUDIT = ROOT / "results/jne_revision_2026-09/e1_faced/audit.json"  # one structural audit for both conditions
PLAN = ((OUT / "smoke_plan.json") if SMOKE else
        ROOT / ("plans/2026-09-20-jne-e1-faced-v2.json" if V2 else "plans/2026-09-18-jne-e1-faced.json"))
E1_PLAN = ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json"

N_CLIPS, N_CH, FS, SIG_FS, N_WIN = 28, 30, 250, 125, 30
CLASS_OF_CLIP = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4 + [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3
CLASS_NAMES = ["anger", "disgust", "fear", "sadness", "neutral", "amusement", "inspiration", "joy", "tenderness"]
SPLIT = (6, 3, 3) if SMOKE else (60, 23, 40)
EPOCHS = 80
BUDGETS = [1, 5, 10, 20, 40, 80]
SEEDS = [2024] if SMOKE else [2024, 2025, 2026]
MODELS = ["dgcnn", "mlp", "eegnet"]
# Fraction of wall time the GPU is kept busy. After every batch the runner waits for
# the GPU to finish and then sleeps (1 - DUTY) / DUTY times as long. Pure pacing:
# the sequence of computations is unchanged, so results are identical (verified on
# the synthetic layout). Added because this laptop loses power under sustained GPU load.
DUTY = 1.0


def _pace(t_start):
    if DUTY < 1.0:
        torch.cuda.synchronize()
        time.sleep((time.perf_counter() - t_start) * (1.0 - DUTY) / DUTY)
CONDITIONS = ["standard", "holdout"]


def _load_pkl(path):
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f, encoding="iso-8859-1"))


def _subject_files(folder):
    out = {}
    for p in folder.iterdir():
        m = re.fullmatch(r"sub(\d{3})\.pkl(\.pkl)?", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _data_dir(root, *parts):
    """The release unpacks as Processed_data/Processed_data/... ; accept both layouts."""
    for cand in (root.joinpath(*parts), root.joinpath(parts[0], *parts)):
        if cand.is_dir() and any(cand.glob("sub*.pkl*")):
            return cand
    raise FileNotFoundError(f"no sub*.pkl under {root.joinpath(*parts)} (or nested copy)")


def audit(root):
    from scipy.signal import decimate, welch
    root = Path(root).resolve()
    CACHE.mkdir(parents=True, exist_ok=True)
    sig_files = _subject_files(_data_dir(root, "Processed_data"))
    de_files = _subject_files(_data_dir(root, "EEG_Features", "DE"))
    report = {"root": str(root), "n_signal_files": len(sig_files), "n_de_files": len(de_files),
              "participants": {}, "excluded": {}}
    for s in sorted(set(sig_files) | set(de_files)):
        why = None
        if s not in sig_files or s not in de_files:
            why = "missing signal or DE file"
        else:
            sig = _load_pkl(sig_files[s])
            de = _load_pkl(de_files[s])
            if sig.shape != (N_CLIPS, 32, N_WIN * FS):
                why = f"signal shape {sig.shape}"
            elif de.shape != (N_CLIPS, 32, N_WIN, 5):
                why = f"DE shape {de.shape}"
            elif not (np.isfinite(sig).all() and np.isfinite(de).all()):
                why = "non-finite values"
        if why:
            report["excluded"][s] = why
            continue
        sig = sig[:, :N_CH].astype(np.float64)
        de = de[:, :N_CH].astype(np.float32)
        f, pxx = welch(sig, fs=FS, nperseg=FS, axis=-1)
        hi = float(pxx[..., f > 47].sum() / pxx.sum())
        dec = decimate(sig, 2, ftype="fir", zero_phase=True, axis=-1)[..., : N_WIN * SIG_FS]
        assert dec.shape == (N_CLIPS, N_CH, N_WIN * SIG_FS)
        sig_w = dec.reshape(N_CLIPS, N_CH, N_WIN, SIG_FS).transpose(0, 2, 1, 3).reshape(-1, N_CH, SIG_FS)
        de_w = de.transpose(0, 2, 1, 3).reshape(-1, N_CH, 5)
        clip = np.repeat(np.arange(1, N_CLIPS + 1, dtype=np.int16), N_WIN)
        y = np.array([CLASS_OF_CLIP[c - 1] for c in clip], dtype=np.int64)
        np.save(CACHE / f"sig_p{s:03d}.npy", sig_w.astype(np.float32))
        np.savez(CACHE / f"de_p{s:03d}.npz", x=de_w, y=y, trial=clip)
        report["participants"][s] = {
            "signal_abs_max": float(np.abs(sig).max()), "signal_std": float(sig.std()),
            "de_range": [float(de.min()), float(de.max())],
            "power_fraction_above_47Hz": hi,
        }
        print(s, report["participants"][s], flush=True)
    fr = [v["power_fraction_above_47Hz"] for v in report["participants"].values()]
    report["power_fraction_above_47Hz_max"] = max(fr) if fr else None
    dump(AUDIT, report)
    print("audited", len(report["participants"]), "participants; excluded", report["excluded"])


def freeze():
    if PLAN.exists():
        raise SystemExit(f"plan already frozen: {PLAN}")
    audit_rep = json.loads(AUDIT.read_text())
    ids = sorted(int(s) for s in audit_rep["participants"])
    need = sum(SPLIT)
    if len(ids) < need:
        raise SystemExit(f"{len(ids)} usable participants < {need}; re-plan before freezing")
    rng = np.random.default_rng(20260918)
    perm = [int(i) for i in rng.permutation(ids)]
    n_tr, n_va, n_ta = SPLIT
    split = {"train": sorted(perm[:n_tr]), "validation": sorted(perm[n_tr:n_tr + n_va]),
             "target": sorted(perm[n_tr + n_va:n_tr + n_va + n_ta]),
             "unused": sorted(perm[n_tr + n_va + n_ta:])}
    clips_by_class = {c: [i + 1 for i, k in enumerate(CLASS_OF_CLIP) if k == c] for c in range(9)}
    rng_h = np.random.default_rng(20260919)
    holdout = {c: int(rng_h.choice(v)) for c, v in clips_by_class.items()}
    rng_c = np.random.default_rng(20260920)
    cal_standard = {c: [int(x) for x in rng_c.permutation(v)] for c, v in clips_by_class.items()}
    cal_holdout = {c: [int(x) for x in rng_c.permutation([t for t in v if t != holdout[c]])]
                   for c, v in clips_by_class.items()}
    inputs = {}
    for s in split["train"] + split["validation"] + split["target"]:
        for p in (CACHE / f"sig_p{s:03d}.npy", CACHE / f"de_p{s:03d}.npz"):
            inputs[p.relative_to(ROOT).as_posix()] = base.sha(p)
    e1 = json.loads(E1_PLAN.read_text(encoding="utf-8"))
    cfg = {
        "protocol_id": f"jne_e1_{TAG}_v1",
        "design_status": ("Design written 2026-09-17 before FACED was available (experiments/jne_e1_faced.py); "
                          "frozen after the structural audit and before any model was fitted on FACED."),
        "audit_sha256": base.sha(AUDIT),
        "inputs_sha256_check": inputs,
        "classes": CLASS_NAMES, "class_of_clip": CLASS_OF_CLIP,
        "split_seed": 20260918, "split": split,
        "exclusion_rule": "structural audit only (missing/mis-shaped/non-finite files); no outcome-based exclusion",
        "models": MODELS, "conditions": CONDITIONS, "optimization_seeds": SEEDS,
        "training": {**e1["training"], "epochs": EPOCHS},
        "vendor_root": e1["vendor_root"], "dgcnn_source_sha256": e1["dgcnn_source_sha256"],
        "dgcnn_config_sha256": e1["dgcnn_config_sha256"],
        "inputs": {"dgcnn/mlp": "official DE, first 30 channels, 30 x 5 per window",
                   "eegnet": "official processed signal, first 30 channels, decimated to 125 Hz per clip, 1-s windows"},
        "normalization": "training participants only: per feature (DE) or per channel (signal); std floor 1e-6",
        "budgets": BUDGETS, "selection_metric": "window accuracy", "tie_rule": "earliest maximum",
        "stimulus_holdout": {"seed": 20260919, "clip_per_class": holdout,
                             "rule": ("holdout condition: B* clips removed from training and validation participants; "
                                      "both conditions score targets on B*, select on validation participants' non-B* "
                                      "clips (U) or on one non-B* clip per class of the target (A)")},
        "calibration": {
            "seed": 20260920,
            "standard": {"permutations": cal_standard, "rotations": 3,
                         "rule": "rotation r: C = permutations[c][r]; B = all other clips of the target"},
            "holdout": {"permutations": cal_holdout, "rotations": 2,
                        "rule": "rotation r: C = permutations[c][r] (never B*); B = B*"},
        },
        "primary_endpoints": [
            "standard condition, DGCNN: U80-U5, A80-A5, A80-U80 (as E1 SEED)",
            "stimulus effect, DGCNN: (A80-A5)_holdout - (A80-A5)_seen and (U80-U5)_holdout - (U80-U5)_seen on B*",
        ],
        "secondary": ["MLP and EEGNet", "balanced window accuracy", "clip accuracy", "seed dispersion",
                      "retention ratio and slope", "level differences on B* between conditions"],
        "inference": "participants (n = 40 targets); within-participant averaging over seeds and rotations; "
                     "participant bootstrap 10000 (seed 20260917); exact sign-flip is infeasible for n=40, use "
                     "100000 random sign flips (seed 20260917); Holm within each model's primary family",
        "boundary": ("Official FACED preprocessing (band-pass, ICA, re-reference) is applied per recording by the "
                     "provider and was not reconstructed. The hold-out condition also has fewer source clips "
                     "(19 vs 28 per participant); the stimulus contrast is therefore joint with training amount."),
        "audit_findings": ("All 123 participants passed the structural audit (no exclusions). The released processed "
                           "signals retain 50 Hz line noise: power above 47 Hz has median 3.2% across participants and "
                           "exceeds 10% in 15 (up to ~100% in sub023, sub003, sub056); power above 62.5 Hz is zero, "
                           "so decimation to 125 Hz removes nothing. DE bands end at 47 Hz and exclude the line "
                           "frequency. No notch filter or artefact rejection is added: such a step would be an "
                           "analysis choice made after inspecting the data. Large-amplitude participants (signal std "
                           "up to 5438 vs median 8.7; sub030, sub023) are retained by the structural-only rule."),
    }
    if V2:
        v1 = ROOT / "plans/2026-09-18-jne-e1-faced.json"
        cfg.update({
            "protocol_id": "jne_e1_faced_v2_per_participant_norm",
            "design_status": ("Frozen 2026-09-20 before any v2 model was fitted, after the v1 (training-set "
                              "normalisation) results were known. v1 is retained and reported; this is a second "
                              "condition, not a replacement."),
            "v1_plan": v1.relative_to(ROOT).as_posix(), "v1_plan_sha256": base.sha(v1),
            "change_from_v1": "input standardisation only",
            "normalization": ("per-participant, label-free: every participant's own windows give the mean and "
                              "standard deviation applied to that participant; no training-set statistics and no "
                              "labels are used, so the zero-target-label policy is unchanged. Splits, folds, "
                              "calibration rotations, stimulus hold-out, budgets, models, seeds and optimiser are "
                              "identical to v1."),
            "reason": ("v1 decoding was near chance on FACED (target window accuracy ~0.17; chance 0.111, majority "
                       "class 0.143), so apparent gains were below 1 pp and retention ratios had a near-zero "
                       "denominator. Per-participant standardisation is standard in cross-subject EEG emotion work. "
                       "The change is a preprocessing choice made because of a measurement problem, not because of "
                       "the direction of any retention result."),
            "predictions_written_before_running": [
                "P1: v2 raises target-participant accuracy well above the majority-class rate (0.143) for DGCNN and "
                "MLP; if it does not, retention ratios stay uninterpretable and are reported as such.",
                "P2: if retention tracks how informative the selection data are, the calibration policy retains a "
                "larger fraction in v2 than in v1 for DGCNN (v1: 0.11 [0.02, 0.19]).",
                "P3: the validation-pool curve (zero-label retention against number of validation participants) is "
                "higher in v2 than in v1 at the same pool sizes.",
                "P4: the stimulus hold-out contrast stays small for DGCNN, as in v1 (-0.26 pp [-0.91, 0.39]).",
            ],
            "reporting_rule": ("Both v1 and v2 are reported in full, with v1 as the no-per-participant-"
                               "normalisation condition. Whichever way the predictions come out, no condition is "
                               "dropped."),
        })
    dump(PLAN, cfg)
    print("frozen", PLAN)


def make_model(name, n_ch, dgcnn):
    if name == "dgcnn":
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            return dgcnn.DGCNN(num_electrodes=n_ch, in_channels=5, num_classes=9)
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(n_ch * 5, 128), nn.ReLU(), nn.Dropout(0.5),
                             nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 9))
    return EEGNet(chans=n_ch, samples=SIG_FS, classes=9, kern=SIG_FS // 2)


def _subject_standardise(g, model):
    """Label-free per-participant standardisation from that participant's own windows."""
    axes = (0, 2) if model == "eegnet" else (0,)
    shape = (1, N_CH, 1) if model == "eegnet" else (1, N_CH, 5)
    g64 = g.double()
    mu = g64.mean(dim=axes)
    sd = (g64.var(dim=axes, unbiased=False)).clamp_min(0).sqrt().clamp_min(1e-6)
    return (g - mu.float().view(shape)) / sd.float().view(shape)


def load_part(model, subjects, keep_clips, device, stats=None, mean=None, scale=None):
    """Load participants onto the GPU one at a time, restricted to keep_clips.

    Signals are kept in fp16. Standardisation (if mean/scale given) is applied per
    participant so that no full-size fp32 copy is ever materialised. With stats,
    float64 moments are accumulated instead and the raw chunks are returned.
    """
    xs, ys, ss, ts = [], [], [], []
    for s in subjects:
        with np.load(CACHE / f"de_p{s:03d}.npz") as z:
            y, t = z["y"], z["trial"]
            x = z["x"] if model != "eegnet" else None
        m = np.isin(t, keep_clips)
        if model == "eegnet":
            x = np.load(CACHE / f"sig_p{s:03d}.npy", mmap_mode="r")
        g = torch.from_numpy(np.ascontiguousarray(x[m])).to(device)
        if stats is not None:
            g64 = g.double()
            axes = (0, 2) if model == "eegnet" else (0,)
            stats["n"] += g.shape[0] * (g.shape[2] if model == "eegnet" else 1)
            stats["s1"] += g64.sum(dim=axes)
            stats["s2"] += (g64 * g64).sum(dim=axes)
            del g64
        if V2:
            g = _subject_standardise(g, model)
        elif mean is not None:
            g = (g - mean) / scale
        # fp16 only after standardisation: raw artefacts reach |x| ~ 2.8e5 > fp16 max (65504)
        xs.append(g.half() if model == "eegnet" and stats is None else g)
        ys.append(y[m]); ts.append(t[m]); ss.append(np.full(int(m.sum()), s, np.int16))
    return xs, np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


def fit(cfg, cond, model_name, seed, device, dgcnn, subjects=None, folder=None, epochs=None):
    folder = folder or OUT / "cells" / f"{cond}_{model_name}_seed{seed}"
    if (folder / "result.json").exists():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    tr = cfg["training"]
    epochs = epochs or tr["epochs"]
    split = subjects or cfg["split"]
    all_clips = list(range(1, N_CLIPS + 1))
    bstar = sorted(cfg["stimulus_holdout"]["clip_per_class"].values())
    source_clips = [c for c in all_clips if c not in bstar] if cond == "holdout" else all_clips
    shape = (1, N_CH, 1) if model_name == "eegnet" else (1, N_CH, 5)
    dims = N_CH if model_name == "eegnet" else (N_CH, 5)
    if V2:
        mean = scale = None
        chunks, y, _, _ = load_part(model_name, split["train"], source_clips, device)
        for i, g in enumerate(chunks):
            chunks[i] = g.half() if model_name == "eegnet" else g
    else:
        stats = {"n": 0, "s1": torch.zeros(dims, dtype=torch.float64, device=device),
                 "s2": torch.zeros(dims, dtype=torch.float64, device=device)}
        chunks, y, _, _ = load_part(model_name, split["train"], source_clips, device, stats)
        mu = stats["s1"] / stats["n"]
        sd = (stats["s2"] / stats["n"] - mu * mu).clamp_min(0).sqrt().clamp_min(1e-6)
        mean, scale = mu.float().view(shape), sd.float().view(shape)
        for i, g in enumerate(chunks):
            z = (g.float() - mean) / scale
            chunks[i] = z.half() if model_name == "eegnet" else z
    x = torch.cat(chunks)
    del chunks
    yt = torch.from_numpy(y).to(device)
    parts = {}
    for name, subs, clips in (("validation", split["validation"], source_clips),
                              ("target", split["target"], all_clips)):
        pchunks, py, ps, pt = load_part(model_name, subs, clips, device, mean=mean, scale=scale)
        if V2 and model_name == "eegnet":
            pchunks = [c.half() for c in pchunks]
        px = torch.cat(pchunks)
        del pchunks
        units, idx = unit_index(ps, pt)
        count = np.bincount(idx, minlength=len(units)).astype(np.int32)
        lab = np.array([py[idx == i][0] for i in range(len(units))], np.int8)
        parts[name] = dict(x=px, y=py, idx=idx, units=units, count=count, label=lab)

    base.seed_all(seed)
    model = make_model(model_name, N_CH, dgcnn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"],
                            eps=tr["adam_epsilon"])
    reg = dgcnn.NewSparseL2Regularization(tr["dgcnn_regularizer"]).to(device) if model_name == "dgcnn" else None
    gen = torch.Generator().manual_seed(seed)
    bs, ebs = tr["batch_size"], tr["evaluation_batch_size"]
    rec = {p: dict(correct=np.zeros((epochs, len(v["units"])), np.int32),
                   logits=np.zeros((epochs, len(v["units"]), 9), np.float32)) for p, v in parts.items()}
    losses = []
    t0 = time.monotonic()

    def predict(xx):
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(xx), ebs):
                tb = time.perf_counter()
                out.append(model(xx[i:i + ebs].float()))
                _pace(tb)
            return torch.cat(out).float().cpu().numpy()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            tb = time.perf_counter()
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x[b].float()), yt[b])
            if reg is not None:
                loss = loss + reg(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{folder.name} epoch {epoch + 1}")
            loss.backward()
            opt.step()
            if model_name == "eegnet":
                model.apply_max_norm()
            total += float(loss.detach()) * len(b)
            _pace(tb)
        losses.append(total / len(y))
        for p, v in parts.items():
            lg = predict(v["x"])
            c, sums = per_trial(lg, v["y"], v["idx"], len(v["units"]))
            rec[p]["correct"][epoch] = c
            rec[p]["logits"][epoch] = (sums / v["count"][:, None]).astype(np.float32)
    train_acc = float(np.mean(predict(x).argmax(1) == y))
    arrays = {"loss": np.array(losses)}
    for p, v in parts.items():
        for k in ("units", "count", "label"):
            arrays[f"{p}_{k}"] = v[k]
        arrays[f"{p}_correct"] = rec[p]["correct"]
        arrays[f"{p}_trial_logits"] = rec[p]["logits"]
    tmp = folder / "trace.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, folder / "trace.npz")
    row = {"cell": folder.name, "condition": cond, "model": model_name, "seed": seed, "epochs": epochs,
           "n_train_windows": int(len(y)), "train_window_accuracy_final": train_acc,
           "loss_first_last": [losses[0], losses[-1]], "seconds": round(time.monotonic() - t0, 1),
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20), "gpu_duty": DUTY,
           "normalization": "per_participant" if V2 else "training_set",
           "trace_sha256": base.sha(folder / "trace.npz")}
    dump(folder / "result.json", row)
    del x, parts
    torch.cuda.empty_cache()
    return row


def _setup(cfg):
    for rel, h in cfg.get("inputs_sha256_check", {}).items():
        if base.sha(ROOT / rel) != h:
            raise RuntimeError(f"input changed: {rel}")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return base.load_model_module(cfg), torch.device("cuda")


def pilot():
    """Technical pilot on development participants only; never analysed."""
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    dgcnn, device = _setup(cfg)
    tr_ids = cfg["split"]["train"]
    n_pt = max(1, len(tr_ids) // 6)
    dev = {"train": tr_ids[n_pt:], "validation": cfg["split"]["validation"], "target": tr_ids[:n_pt]}
    rows = []
    for m in MODELS:
        torch.cuda.reset_peak_memory_stats()
        rows.append(fit(cfg, "standard", m, 2024, device, dgcnn, subjects=dev,
                        folder=OUT / "pilot" / m, epochs=min(3, cfg["training"]["epochs"])))
    dump(OUT / "pilot" / "pilot_summary.json",
         {"note": "development participants only; target group untouched; not analysed", "rows": rows})
    print(json.dumps(rows, indent=1))


def run(models=None, limit=None):
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    rt = OUT / "runtime.json"
    if not rt.exists():
        dump(rt, {"runner_sha256": base.sha(Path(__file__)), "plan_sha256": base.sha(PLAN),
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    elif json.loads(rt.read_text())["runner_sha256"] != base.sha(Path(__file__)):
        raise RuntimeError("runner changed after cells were produced")
    dgcnn, device = _setup(cfg)
    # lighter models first; EEGNet (highest sustained GPU load) last
    cells = [(c, m, s) for m in MODELS for c in CONDITIONS for s in SEEDS]
    done = 0
    for c, m, s in cells:
        if (models and m not in models) or (limit is not None and done >= limit):
            continue
        torch.cuda.reset_peak_memory_stats()
        row = fit(cfg, c, m, s, device, dgcnn)
        if row:
            done += 1
            print(json.dumps({k: row[k] for k in ("cell", "seconds", "train_window_accuracy_final", "peak_gpu_mb")}),
                  flush=True)
    n = len(list((OUT / "cells").glob("*/result.json")))
    dump(OUT / "progress.json", {"completed": n, "total": len(cells)})
    if n == len(cells):
        dump(OUT / "completion.json", {"status": "complete", "cells": n})
        print("COMPLETE", flush=True)


def make_synthetic(root):
    """Tiny fake release for code testing only (FACED_SMOKE=1)."""
    rng = np.random.default_rng(0)
    root = Path(root)
    (root / "Processed_data").mkdir(parents=True, exist_ok=True)
    (root / "EEG_Features" / "DE").mkdir(parents=True, exist_ok=True)
    for s in range(sum(SPLIT) + 1):
        sig = rng.normal(size=(N_CLIPS, 32, N_WIN * FS)) * 10
        for c in range(N_CLIPS):
            sig[c, :4] += np.sin(np.arange(N_WIN * FS) / FS * 2 * np.pi * (4 + CLASS_OF_CLIP[c]))
        de = np.abs(rng.normal(size=(N_CLIPS, 32, N_WIN, 5)))
        for c in range(N_CLIPS):
            de[c, :, :, CLASS_OF_CLIP[c] % 5] += 1.0
        with open(root / "Processed_data" / f"sub{s:03d}.pkl", "wb") as f:
            pickle.dump(sig, f)
        with open(root / "EEG_Features" / "DE" / f"sub{s:03d}.pkl.pkl", "wb") as f:
            pickle.dump(de, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "freeze", "pilot", "run", "synthetic"])
    ap.add_argument("--root")
    ap.add_argument("--duty", type=float, default=1.0, help="GPU busy fraction (pacing only)")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    DUTY = a.duty
    if a.mode == "synthetic":
        assert SMOKE, "synthetic data only under FACED_SMOKE=1"
        make_synthetic(a.root)
    elif a.mode == "audit":
        audit(a.root)
    elif a.mode == "freeze":
        freeze()
    elif a.mode == "pilot":
        pilot()
    else:
        run(a.models, a.limit)
