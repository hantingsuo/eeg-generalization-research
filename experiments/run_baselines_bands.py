# experiments/run_baselines_bands.py
"""Per-band (filter-bank) baselines over P1-P4; compare to broadband Phase 1a.
Precomputes per-subject per-band Riemannian means once (reused across folds)."""
import json, os, time
import numpy as np
from pcma.data.competition import load_competition
from pcma.data.bandcov import to_band_covariances
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv, p4_leave_one_dep
from pcma.model.pipelines import fit_predict_noadapt_bands, fit_predict_ra_bands, subject_band_means
from pcma.eval.metrics import summarize

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

def _save(res):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "baselines_bands.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

def main():
    t0 = time.time()
    d = load_competition()
    C = to_band_covariances(d.X, sfreq=250.0)
    print(f"[t={time.time()-t0:.0f}s] band covariances {C.shape} done", flush=True)
    sbm = subject_band_means(C, d.subject)
    print(f"[t={time.time()-t0:.0f}s] subject band means precomputed", flush=True)
    res = {}

    def ev(Ctr, ytr, str_, Cte, yte, ste, pop_te):
        return {"noadapt": summarize(yte, fit_predict_noadapt_bands(Ctr, ytr, Cte), ste, pop_te),
                "ra": summarize(yte, fit_predict_ra_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm), ste, pop_te)}

    for name, splitter in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = splitter(d.subject, d.population)
        res[name] = ev(C[tr], d.y[tr], d.subject[tr], C[te], d.y[te], d.subject[te], d.population[te])
        print(f"[t={time.time()-t0:.0f}s] {name} noadapt {res[name]['noadapt']['accuracy']:.4f} ra {res[name]['ra']['accuracy']:.4f}", flush=True)
        _save(res)

    p3n, p3r = [], []
    for i, (tr, te) in enumerate(p3_mixed_cv(d.subject, n_splits=5)):
        rr = ev(C[tr], d.y[tr], d.subject[tr], C[te], d.y[te], d.subject[te], d.population[te])
        p3n.append(rr["noadapt"]["accuracy"]); p3r.append(rr["ra"]["accuracy"])
        print(f"[t={time.time()-t0:.0f}s] P3 fold {i+1}/5 noadapt {rr['noadapt']['accuracy']:.4f} ra {rr['ra']['accuracy']:.4f}", flush=True)
    res["P3_mixed_cv"] = {"noadapt": {"accuracy_mean": float(np.mean(p3n)), "accuracy_std": float(np.std(p3n))},
                          "ra": {"accuracy_mean": float(np.mean(p3r)), "accuracy_std": float(np.std(p3r))}}
    _save(res)
    print(f"[t={time.time()-t0:.0f}s] P3 noadapt {np.mean(p3n):.4f} ra {np.mean(p3r):.4f}", flush=True)

    p4 = []
    for i, (tr, te) in enumerate(p4_leave_one_dep(d.subject, d.population)):
        pred = fit_predict_ra_bands(C[tr], d.y[tr], d.subject[tr], C[te], d.subject[te], ref_means_per_band=sbm)
        p4.append(float((pred == d.y[te]).mean()))
        print(f"[t={time.time()-t0:.0f}s] P4 fold {i+1} acc {p4[-1]:.4f}", flush=True)
    res["P4_leave_one_dep"] = {"ra_mean": float(np.mean(p4)), "ra_worst": float(np.min(p4)), "ra_per_dep_acc": p4}

    xpop = np.mean([res["P1_HC2DEP"]["ra"]["accuracy"], res["P2_DEP2HC"]["ra"]["accuracy"]])
    gap = res["P3_mixed_cv"]["ra"]["accuracy_mean"] - xpop
    res["cross_population_gap_ra"] = float(gap)
    _save(res)
    print(f"[t={time.time()-t0:.0f}s] CROSS-POPULATION GAP (RA, per-band): {gap:.4f}", flush=True)

    lines = ["| Protocol | no-adapt | RA (per-band) |", "|---|---|---|",
             f"| P1 HC->DEP | {res['P1_HC2DEP']['noadapt']['accuracy']:.3f} | {res['P1_HC2DEP']['ra']['accuracy']:.3f} |",
             f"| P2 DEP->HC | {res['P2_DEP2HC']['noadapt']['accuracy']:.3f} | {res['P2_DEP2HC']['ra']['accuracy']:.3f} |",
             f"| P3 mixed CV | {res['P3_mixed_cv']['noadapt']['accuracy_mean']:.3f} | {res['P3_mixed_cv']['ra']['accuracy_mean']:.3f} |",
             f"\n**Cross-population gap (RA, per-band): {gap:.3f}**"]
    with open(os.path.join(OUT, "baselines_bands.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
