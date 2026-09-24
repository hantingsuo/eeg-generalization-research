"""E2 input-route check: the E1 cross-subject design with a temporal-input model.

EEGNet-8,2 (Lawhern et al., 2018, J Neural Eng 15:056013), ported from the
Army Research Laboratory reference implementation (arl-eegmodels, CC0): temporal
conv F1=8 with kernel = fs/2, depthwise spatial conv D=2 with max-norm 1, separable
conv F2=16 with kernel 16, average pooling 4 and 8, ELU, dropout 0.5
(cross-subject setting in the paper), dense output with max-norm 0.25, Keras
BatchNorm defaults (momentum 0.99, eps 1e-3).

Input: SEED official Preprocessed_EEG (200 Hz; upstream down-sampling and
band-pass are release provenance), session 1, cut into non-overlapping 1-s windows.
Window counts per trial are asserted to equal the 1-s DE/LDS window counts, so the
temporal model sees the same windows as DGCNN/MLP in E1, as raw signal instead of
differential entropy.

Participant folds, calibration rotations, budgets, optimiser recipe and selection
rules are copied from the frozen E1 plan. Per-channel standardisation is fitted on
training participants only.

Modes: prepare | freeze | run | status
"""
from pathlib import Path
import sys, json, time, argparse, os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_selection_sensitivity_v1 as base  # noqa: E402
from experiments.jne_e1_seed_cross_subject import dump, per_trial, unit_index  # noqa: E402
from pcma.data.libeer_seed import resolve_seed_subject_file  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from scipy.io import loadmat, whosmat  # noqa: E402

E1_PLAN = ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json"
PLAN = ROOT / "plans/2026-09-17-jne-e2-eegnet-seed.json"
OUT = ROOT / "results/jne_revision_2026-09/e2_eegnet_seed"
CACHE = OUT / "cache"
RAW = ROOT / "data/SEED/SEED/SEED/SEED_EEG/Preprocessed_EEG"
DE_CACHE = ROOT / "results/revision_2026-09-11/seed_family_extension_v4/cache"
FS = 200
DUTY = 1.0  # GPU busy fraction; pacing only, never touches the computation


def _pace(t_start):
    """Sleep after a batch so the GPU idles part of the time (thermal/power limit).

    The work and its order are unchanged, so traces stay bit-identical; `verify`
    mode below re-fits a completed cell and checks that.
    """
    if DUTY < 1.0:
        torch.cuda.synchronize()
        time.sleep((time.perf_counter() - t_start) * (1.0 - DUTY) / DUTY)


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    e1 = json.loads(E1_PLAN.read_text(encoding="utf-8"))
    meta = {}
    for s in range(1, 16):
        dst = CACHE / f"p{s:02d}.npy"
        src = resolve_seed_subject_file(RAW, 1, s)
        names = sorted((n for n, _, _ in whosmat(src)), key=lambda n: int(n.rsplit("eeg", 1)[1]))
        assert len(names) == 15, (src, names)
        with np.load(DE_CACHE / f"seed_s1_p{s:02d}.npz", allow_pickle=False) as z:
            de_counts = np.bincount(z["trial"], minlength=16)[1:]
        if not dst.exists():
            blocks, trials, labels = [], [], []
            for t, name in enumerate(names, 1):
                x = loadmat(src, variable_names=[name])[name]
                assert x.shape[0] == 62 and np.isfinite(x).all()
                n = x.shape[1] // FS
                assert n == de_counts[t - 1], f"p{s} trial {t}: {n} signal windows vs {de_counts[t - 1]} DE windows"
                w = x[:, : n * FS].reshape(62, n, FS).transpose(1, 0, 2).astype(np.float32)
                blocks.append(w)
                trials.append(np.full(n, t, np.int16))
                labels.append(np.full(n, e1["trial_labels"][t - 1], np.int64))
            tmp = CACHE / f"p{s:02d}.tmp.npy"
            np.save(tmp, np.concatenate(blocks))
            os.replace(tmp, dst)
            np.savez(CACHE / f"p{s:02d}_index.npz", trial=np.concatenate(trials), y=np.concatenate(labels))
        arr = np.load(dst, mmap_mode="r")
        meta[s] = {"source": src.relative_to(ROOT).as_posix(), "windows": int(arr.shape[0]),
                   "shape": list(arr.shape), "abs_max": float(np.abs(arr).max())}
        print(s, meta[s], flush=True)
    dump(CACHE / "prepare_meta.json", meta)


def freeze():
    if PLAN.exists():
        raise SystemExit(f"plan already frozen: {PLAN}")
    e1 = json.loads(E1_PLAN.read_text(encoding="utf-8"))
    inputs = {}
    for s in range(1, 16):
        for f in (CACHE / f"p{s:02d}.npy", CACHE / f"p{s:02d}_index.npz"):
            inputs[f.relative_to(ROOT).as_posix()] = base.sha(f)
    cfg = {k: e1[k] for k in ("trial_labels", "folds", "fold_source", "budgets", "candidate_rule",
                              "selection_metric", "tie_rule", "calibration", "policies",
                              "primary_endpoints", "retention_definitions", "aggregation",
                              "optimization_seeds")}
    cfg.update({
        "protocol_id": "jne_e2_eegnet_seed_v1",
        "design_status": ("Frozen 2026-09-17 before any EEGNet outcome; E1 DGCNN/MLP were still running and "
                          "no E1 outcome had been inspected. Design copied from the E1 plan."),
        "e1_plan_sha256": base.sha(E1_PLAN),
        "models": ["eegnet"],
        "model": {"name": "EEGNet-8,2", "reference": "Lawhern et al. 2018 J Neural Eng 15:056013; arl-eegmodels (CC0)",
                  "F1": 8, "D": 2, "F2": 16, "kern_length": FS // 2, "separable_kernel": 16,
                  "dropout": 0.5, "depthwise_max_norm": 1.0, "dense_max_norm": 0.25,
                  "batchnorm": {"momentum_torch": 0.01, "eps": 1e-3}},
        "input": "SEED Preprocessed_EEG session 1, 62 channels, 200 Hz, non-overlapping 1-s windows aligned 1:1 with DE/LDS windows",
        "inputs_sha256": inputs,
        "normalization": "per-channel mean/std over all training-participant samples; std floor 1e-6",
        "training": e1["training"],
        "training_note": "same optimiser recipe as E1 (AdamW lr 0.0015, batch 64, 80 epochs), not EEGNet paper defaults, so the three models share one recipe",
        "boundary": ("Upstream down-sampling/filtering of the released preprocessed signal were not reconstructed. "
                     "Same films for all participants. Differences from DGCNN/MLP combine input route and architecture. "
                     "Released signals contain large-amplitude artefacts (per-participant |x| max 938 to 215,353 in "
                     "release units, see cache/prepare_meta.json); no clipping or artefact rejection is applied, "
                     "because any such rule chosen now would be an unregistered analysis choice."),
    })
    dump(PLAN, cfg)
    print("frozen", PLAN)


class EEGNet(nn.Module):
    def __init__(self, chans=62, samples=FS, classes=3, F1=8, D=2, F2=16, kern=FS // 2, sep=16, p=0.5):
        super().__init__()
        bn = dict(momentum=0.01, eps=1e-3)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding="same", bias=False),
            nn.BatchNorm2d(F1, **bn))
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(F1 * D, **bn), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p),
            nn.Conv2d(F1 * D, F1 * D, (1, sep), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2, **bn), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p),
            nn.Flatten())
        self.dense = nn.Linear(F2 * (samples // 4 // 8), classes)

    def forward(self, x):
        return self.dense(self.block2(self.depthwise(self.block1(x.unsqueeze(1)))))

    @torch.no_grad()
    def apply_max_norm(self):
        self.depthwise.weight.copy_(torch.renorm(self.depthwise.weight, 2, 0, 1.0))
        self.dense.weight.copy_(torch.renorm(self.dense.weight, 2, 0, 0.25))


def to_gpu(subjects, device, moments=None):
    """Move participants to the GPU one at a time; optionally accumulate float64
    per-channel first/second moments so no full-size float64 tensor is built."""
    xs, ys, ss, ts = [], [], [], []
    for s in subjects:
        arr = np.load(CACHE / f"p{s:02d}.npy", mmap_mode="r")
        g = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        if moments is not None:
            g64 = g.double()
            moments["n"] += g.shape[0] * g.shape[2]
            moments["s1"] += g64.sum(dim=(0, 2))
            moments["s2"] += (g64 * g64).sum(dim=(0, 2))
            del g64
        xs.append(g)
        with np.load(CACHE / f"p{s:02d}_index.npz") as z:
            ys.append(z["y"]); ts.append(z["trial"])
        ss.append(np.full(len(ys[-1]), s, np.int16))
    return torch.cat(xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


@torch.no_grad()
def predict_gpu(model, x, bs):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)]).float().cpu().numpy()


def fit(cell, cfg, device, folder=None, refit=False):
    f, _, seed = cell
    key = f"fold{f}_eegnet_seed{seed}"
    folder = folder or (OUT / "cells" / key)
    if (folder / "result.json").exists() and not refit:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    fold = cfg["folds"][f - 1]
    tr = cfg["training"]
    mom = {"n": 0, "s1": torch.zeros(62, dtype=torch.float64, device=device),
           "s2": torch.zeros(62, dtype=torch.float64, device=device)}
    x, y, _, _ = to_gpu(fold["train"], device, mom)
    mu = mom["s1"] / mom["n"]
    sd = (mom["s2"] / mom["n"] - mu * mu).clamp_min(0).sqrt()
    mean = mu.float().view(1, 62, 1)
    scale = sd.float().clamp_min(1e-6).view(1, 62, 1)
    x.sub_(mean).div_(scale)
    yt = torch.from_numpy(y).to(device)
    parts = {}
    for part, subs in (("validation", fold["validation"]), ("target", fold["test"])):
        px, py, ps, pt = to_gpu(subs, device)
        px.sub_(mean).div_(scale)
        units, idx = unit_index(ps, pt)
        count = np.bincount(idx, minlength=len(units)).astype(np.int32)
        lab = np.array([py[idx == i][0] for i in range(len(units))], np.int8)
        parts[part] = dict(x=px, y=py, idx=idx, units=units, count=count, label=lab)

    base.seed_all(seed)
    model = EEGNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"],
                            eps=tr["adam_epsilon"])
    gen = torch.Generator().manual_seed(seed)
    E, bs = tr["epochs"], tr["batch_size"]
    rec = {p: dict(correct=np.zeros((E, len(v["units"])), np.int32),
                   logits=np.zeros((E, len(v["units"]), 3), np.float32)) for p, v in parts.items()}
    losses = []
    t0 = time.monotonic()
    for epoch in range(E):
        model.train()
        order = torch.randperm(len(y), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), bs):
            t_batch = time.perf_counter()
            b = order[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x[b]), yt[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{key} epoch {epoch + 1}")
            loss.backward()
            opt.step()
            model.apply_max_norm()
            total += float(loss.detach()) * len(b)
            _pace(t_batch)
        losses.append(total / len(y))
        for p, v in parts.items():
            lg = predict_gpu(model, v["x"], tr["evaluation_batch_size"])
            c, sums = per_trial(lg, v["y"], v["idx"], len(v["units"]))
            rec[p]["correct"][epoch] = c
            rec[p]["logits"][epoch] = (sums / v["count"][:, None]).astype(np.float32)
    train_acc = float(np.mean(predict_gpu(model, x, tr["evaluation_batch_size"]).argmax(1) == y))
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
    row = {"cell": key, "fold": f, "model": "eegnet", "seed": seed, "epochs": E,
           "train_window_accuracy_final": train_acc, "loss_first_last": [losses[0], losses[-1]],
           "seconds": round(time.monotonic() - t0, 1), "trace_sha256": base.sha(folder / "trace.npz")}
    dump(folder / "result.json", row)
    del x, parts
    torch.cuda.empty_cache()
    return row


def _deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def verify(cell_key):
    """Re-fit a completed cell into a scratch folder and compare the trace bytes.

    Used to show that adding GPU pacing leaves the computation unchanged; writes
    nothing into cells/ and therefore does not need the runner hash lock.
    """
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    stored = json.loads((OUT / "cells" / cell_key / "result.json").read_text())
    f, seed = int(cell_key.split("_")[0][4:]), int(cell_key.split("seed")[-1])
    _deterministic()
    scratch = OUT / "verify" / f"{cell_key}_duty{DUTY}"
    row = fit((f, "eegnet", seed), cfg, torch.device("cuda"), folder=scratch, refit=True)
    same = row["trace_sha256"] == stored["trace_sha256"]
    print(json.dumps({"cell": cell_key, "duty": DUTY, "stored": stored["trace_sha256"][:16],
                      "refit": row["trace_sha256"][:16], "bit_identical": same,
                      "seconds_stored": stored["seconds"], "seconds_refit": row["seconds"]}), flush=True)
    if not same:
        raise SystemExit("trace differs - do not continue the run")


def run(limit=None):
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    for rel, h in cfg["inputs_sha256"].items():
        if base.sha(ROOT / rel) != h:
            raise RuntimeError(f"input changed since freeze: {rel}")
    fc = OUT / "frozen_manifest.json"
    if not fc.exists():
        fc.write_bytes(PLAN.read_bytes())
    elif fc.read_bytes() != PLAN.read_bytes():
        raise RuntimeError("plan differs from the copy used by earlier cells")
    rt = OUT / "runtime.json"
    if not rt.exists():
        dump(rt, {"runner_sha256": base.sha(Path(__file__)), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0)})
    else:
        prev = json.loads(rt.read_text())
        cur = base.sha(Path(__file__))
        if cur != prev["runner_sha256"] and cur not in prev.get("runner_sha256_accepted", {}):
            raise RuntimeError("runner changed after cells were produced; run `verify` and record "
                               "the bit-identity proof in runtime.json before continuing")
    _deterministic()
    device = torch.device("cuda")
    pilot = (1, "eegnet", 2024)
    cells = [pilot] + [(f, "eegnet", s) for f in range(1, 6) for s in cfg["optimization_seeds"]
                       if (f, "eegnet", s) != pilot]
    done = 0
    for cell in cells:
        if limit is not None and done >= limit:
            break
        row = fit(cell, cfg, device)
        if row is None:
            continue
        done += 1
        n = len(list((OUT / "cells").glob("*/result.json")))
        msg = {"completed": n, "total": 15, "cell": row["cell"], "seconds": row["seconds"],
               "train_acc": round(row["train_window_accuracy_final"], 4),
               "loss_first_last": [round(v, 4) for v in row["loss_first_last"]]}
        dump(OUT / "progress.json", msg)
        print(json.dumps(msg), flush=True)
    if len(list((OUT / "cells").glob("*/result.json"))) == 15:
        dump(OUT / "completion.json", {"status": "complete", "cells": 15})
        print("COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "freeze", "run", "status", "verify"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--duty", type=float, default=1.0, help="GPU busy fraction (pacing only)")
    ap.add_argument("--cell", help="cell key for verify, e.g. fold1_eegnet_seed2024")
    a = ap.parse_args()
    DUTY = a.duty
    {"prepare": prepare, "freeze": freeze, "status": lambda: print(
        (OUT / "progress.json").read_text() if (OUT / "progress.json").exists() else "not started"),
     "verify": lambda: verify(a.cell)}.get(a.mode, lambda: run(a.limit))()
