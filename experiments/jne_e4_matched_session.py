"""E4: does the session itself explain same- vs cross-session retention?

The same-session study (rotation_v2) and the cross-session study (v4) differed in
more than the session relation: training amount, pool sizes, validation session
and DGCNN standardisation all changed. Here one trajectory is scored two ways.

For participant p, rotation r and training session s, the model is trained on the
rotation's six training trials of session s and validated on its three validation
trials of session s -- exactly the rotation_v2 allocation. Every epoch it is scored
on the rotation's A and B trials in *every* session. SEED plays the same film
schedule in each session, so A/B in session s (same-session relation) and A/B in
the other sessions (cross-session relation) are the same film clips, the same pool
sizes and the same trajectory. Only the recording session differs.

Both models use training-trial standardisation (as v4 and E1). With optimisation
seed 2024 the MLP same-session scores should reproduce rotation_v2 exactly; this is
checked, not assumed.

Modes: freeze | run | status
"""
from pathlib import Path
import sys, json, time, argparse, os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_selection_sensitivity_v1 as base  # noqa: E402
from experiments.jne_e1_seed_cross_subject import dump, make_model, per_trial  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

PLAN = ROOT / "plans/2026-09-17-jne-e4-matched-session.json"
OUT = ROOT / "results/jne_revision_2026-09/e4_matched_session"
CACHE = ROOT / "results/revision_2026-09-11/seed_family_extension_v4/cache"
V2_PLAN = ROOT / "plans/2026-09-10-selection-rotation-v2.json"
E1_PLAN = ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json"


def freeze():
    if PLAN.exists():
        raise SystemExit(f"plan already frozen: {PLAN}")
    v2 = json.loads(V2_PLAN.read_text(encoding="utf-8"))
    e1 = json.loads(E1_PLAN.read_text(encoding="utf-8"))
    inputs = {}
    for s in range(1, 16):
        for q in (1, 2, 3):
            p = CACHE / f"seed_s{q}_p{s:02d}.npz"
            inputs[p.relative_to(ROOT).as_posix()] = base.sha(p)
    cfg = {
        "protocol_id": "jne_e4_matched_session_v1",
        "design_status": (
            "Frozen 2026-09-17 before any E4 outcome, after rotation_v2 and v4 results were known and while "
            "E1 was running (no E1 outcome inspected). Explanatory control, not confirmation."
        ),
        "question": "With trajectory, training amount, pool sizes, film clips and normalisation held fixed, "
                    "does scoring A/B in another session change apparent and retained checkpoint-search gains?",
        "inputs_sha256": inputs,
        "rotations": v2["rotations"],
        "rotation_source": "plans/2026-09-10-selection-rotation-v2.json (identical trial roles in every participant and session)",
        "training_sessions": [1, 2],
        "audit_sessions": [1, 2, 3],
        "subjects": list(range(1, 16)),
        "models": ["dgcnn", "mlp"],
        "optimization_seeds": [2024, 2025, 2026],
        "training": e1["training"],
        "vendor_root": e1["vendor_root"],
        "dgcnn_source_sha256": e1["dgcnn_source_sha256"],
        "dgcnn_config_sha256": e1["dgcnn_config_sha256"],
        "normalization": "per-feature mean/std from the six training trials; std floor 1e-6; both models",
        "budgets": [1, 5, 10, 20, 40, 80],
        "selection_metric": "window accuracy; earliest maximum; validation = session-s validation trials",
        "relations": {
            "same": "A and B from training session s (rotation_v2 relation)",
            "cross": "A and B from each other session q != s (same trial IDs = same film clips); the two cross sessions are averaged within cell",
        },
        "quantities": "S_K = mean of own-pool selected scores; C_K = mean of exchanged-pool scores (rotation_v2 definition), per relation",
        "primary_endpoints": [
            "(C80 - C5)_cross - (C80 - C5)_same, paired within participant",
            "(S80 - S5)_cross - (S80 - S5)_same, paired within participant",
        ],
        "secondary": [
            "each relation's S80-S5, C80-C5 and retention ratio/slope",
            "DGCNN standardisation effect: E4 same-relation DGCNN vs rotation_v2 DGCNN (seed 2024)",
            "reproduction check: E4 same-relation MLP seed 2024 vs rotation_v2 MLP, per cell",
            "seed dispersion",
        ],
        "aggregation": "average rotations, training sessions and seeds within participant; n=15; participant bootstrap 10000 seed 20260917; exact sign-flip + Holm over the two primary endpoints per model",
        "cells": "5 rotations x 2 models x 2 training sessions x 15 participants x 3 seeds = 900; all retained",
        "boundary": (
            "Validation remains in the training session for both relations, unlike v4. Pools are three trials; "
            "sessions differ in recording day and state, not in stimuli. Session labels in SEED are not "
            "counterbalanced orders. Explains, does not prove, the source of same/cross differences."
        ),
    }
    dump(PLAN, cfg)
    print("frozen", PLAN)


def cells(cfg):
    return [(seed, name, r, s, p) for seed in cfg["optimization_seeds"] for name in cfg["models"]
            for r in range(5) for s in cfg["training_sessions"] for p in cfg["subjects"]]


def key_of(cell):
    seed, name, r, s, p = cell
    return f"seed{seed}_{name}_r{r}_s{s}_p{p:02d}"


def load(session, subject, ids):
    with np.load(CACHE / f"seed_s{session}_p{subject:02d}.npz", allow_pickle=False) as z:
        m = np.isin(z["trial"], ids)
        return z["x"][m], z["y"][m], z["trial"][m]


def fit(cell, cfg, dgcnn, device):
    seed, name, r, s, p = cell
    key = key_of(cell)
    folder = OUT / "cells" / key
    if (folder / "result.json").exists():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    tr = cfg["training"]
    rot = cfg["rotations"][r]
    x, y, _ = load(s, p, rot["train"])
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    x = (x - mean) / scale
    vx, vy, vt = load(s, p, rot["validation"])
    audit = [load(q, p, sorted(rot["A"] + rot["B"])) + (q,) for q in cfg["audit_sessions"]]
    ax = np.concatenate([a[0] for a in audit])
    ay = np.concatenate([a[1] for a in audit])
    asess = np.concatenate([np.full(len(a[1]), a[3], np.int16) for a in audit])
    atrial = np.concatenate([a[2] for a in audit])
    pools = {}
    for name_p, (px, py, ps, pt) in (("validation", (vx, vy, np.full(len(vy), s, np.int16), vt)),
                                    ("audit", (ax, ay, asess, atrial))):
        pairs = sorted(set(zip(ps.tolist(), pt.tolist())))
        lookup = {u: i for i, u in enumerate(pairs)}
        idx = np.array([lookup[u] for u in zip(ps.tolist(), pt.tolist())])
        pools[name_p] = dict(x=(px - mean) / scale, y=py, idx=idx, units=np.array(pairs, np.int16),
                             count=np.bincount(idx, minlength=len(pairs)).astype(np.int32),
                             label=np.array([py[idx == i][0] for i in range(len(pairs))], np.int8))

    base.seed_all(seed)
    model = make_model(name, cfg, dgcnn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"],
                            eps=tr["adam_epsilon"])
    reg = dgcnn.NewSparseL2Regularization(tr["dgcnn_regularizer"]).to(device) if name == "dgcnn" else None
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=tr["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    E = tr["epochs"]
    rec = {k: dict(correct=np.zeros((E, len(v["units"])), np.int32),
                   logits=np.zeros((E, len(v["units"]), 3), np.float32)) for k, v in pools.items()}
    t0 = time.monotonic()
    for epoch in range(E):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(bx.to(device)), by.to(device))
            if reg is not None:
                loss = loss + reg(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{key} epoch {epoch + 1}")
            loss.backward()
            opt.step()
        with torch.no_grad():
            for k, v in pools.items():
                lg = base.predict(model, v["x"], tr["evaluation_batch_size"], device)
                c, sums = per_trial(lg, v["y"], v["idx"], len(v["units"]))
                rec[k]["correct"][epoch] = c
                rec[k]["logits"][epoch] = (sums / v["count"][:, None]).astype(np.float32)
    arrays = {}
    for k, v in pools.items():
        arrays[f"{k}_units"] = v["units"]
        arrays[f"{k}_count"] = v["count"]
        arrays[f"{k}_label"] = v["label"]
        arrays[f"{k}_correct"] = rec[k]["correct"]
        arrays[f"{k}_trial_logits"] = rec[k]["logits"]
    tmp = folder / "trace.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, folder / "trace.npz")
    row = {"cell": key, "seed": seed, "model": name, "rotation": r, "train_session": s, "subject": p,
           "seconds": round(time.monotonic() - t0, 2), "trace_sha256": base.sha(folder / "trace.npz")}
    dump(folder / "result.json", row)
    return row


def run():
    cfg = json.loads(PLAN.read_text(encoding="utf-8"))
    for rel, h in cfg["inputs_sha256"].items():
        if base.sha(ROOT / rel) != h:
            raise RuntimeError(f"input changed since freeze: {rel}")
    OUT.mkdir(parents=True, exist_ok=True)
    fc = OUT / "frozen_manifest.json"
    if not fc.exists():
        fc.write_bytes(PLAN.read_bytes())
    elif fc.read_bytes() != PLAN.read_bytes():
        raise RuntimeError("plan differs from the copy used by earlier cells")
    rt = OUT / "runtime.json"
    if not rt.exists():
        dump(rt, {"runner_sha256": base.sha(Path(__file__)), "torch": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0)})
    elif json.loads(rt.read_text())["runner_sha256"] != base.sha(Path(__file__)):
        raise RuntimeError("runner changed after cells were produced")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dgcnn = base.load_model_module(cfg)
    device = torch.device("cuda")
    todo = cells(cfg)
    t0 = time.monotonic()
    for i, cell in enumerate(todo, 1):
        row = fit(cell, cfg, dgcnn, device)
        if row is not None and i % 25 == 0:
            msg = {"index": i, "total": len(todo), "cell": row["cell"],
                   "elapsed_seconds": round(time.monotonic() - t0, 1)}
            dump(OUT / "progress.json", msg)
            print(json.dumps(msg), flush=True)
    n = len(list((OUT / "cells").glob("*/result.json")))
    dump(OUT / "progress.json", {"completed": n, "total": len(todo)})
    if n == len(todo):
        dump(OUT / "completion.json", {"status": "complete", "cells": n})
        print("COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "run", "status"])
    a = ap.parse_args()
    if a.mode == "freeze":
        freeze()
    elif a.mode == "run":
        run()
    else:
        pp = OUT / "progress.json"
        print(pp.read_text() if pp.exists() else "not started")
