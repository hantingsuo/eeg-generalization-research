# experiments/run_rich_ensemble.py
"""Track B: RA vs rich-SVM vs rich-ensemble over P1-P3. Push accuracy. Honest reporting."""
import json, os, time
import numpy as np
from pcma.data.competition import load_competition
from pcma.data.bandcov import to_band_covariances
from pcma.data.features import extract_rich_features
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.model.pipelines import fit_predict_ra_bands, subject_band_means
from pcma.model.rich import fit_predict_rich, fit_predict_rich_ensemble
from pcma.eval.metrics import summarize
from pcma.eval.stats import bootstrap_paired_accuracy

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
LAD = ["ra", "rich_svm", "rich_ens"]

def _p(kind, Ftr, ytr, str_, Fte, ste, Ctr, Cte, sbm):
    if kind == "ra":       return fit_predict_ra_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm)
    if kind == "rich_svm": return fit_predict_rich(Ftr, ytr, str_, Fte, ste)
    if kind == "rich_ens": return fit_predict_rich_ensemble(Ftr, ytr, str_, Fte, ste, k=200)
    raise ValueError(kind)

def main():
    t0 = time.time()
    d = load_competition()
    F = extract_rich_features(d.X, sfreq=250.0)
    C = to_band_covariances(d.X, sfreq=250.0); sbm = subject_band_means(C, d.subject)
    print(f"[t={time.time()-t0:.0f}s] setup features {F.shape}", flush=True)
    res = {}
    for name, sp in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = sp(d.subject, d.population)
        preds = {k: _p(k, F[tr], d.y[tr], d.subject[tr], F[te], d.subject[te], C[tr], C[te], sbm) for k in LAD}
        res[name] = {k: summarize(d.y[te], preds[k], d.subject[te], d.population[te]) for k in LAD}
        res[name]["boot_ens_vs_svm"] = bootstrap_paired_accuracy(d.y[te], preds["rich_svm"], preds["rich_ens"])
        for k in LAD:
            m = res[name][k]
            print(f"[t={time.time()-t0:.0f}s] {name} {k:9s} acc {m['accuracy']:.4f} bal {m['balanced_accuracy']:.4f} pos_recall {m['recall_per_class'].get(1, float('nan')):.4f}", flush=True)
        bt = res[name]["boot_ens_vs_svm"]
        print(f"[t={time.time()-t0:.0f}s] {name} ENS-SVM {bt['mean_diff']:+.4f} CI[{bt['ci_low']:+.4f},{bt['ci_high']:+.4f}]", flush=True)
    fold = {k: [] for k in LAD}
    for i, (tr, te) in enumerate(p3_mixed_cv(d.subject, n_splits=5)):
        for k in LAD:
            p = _p(k, F[tr], d.y[tr], d.subject[tr], F[te], d.subject[te], C[tr], C[te], sbm)
            fold[k].append(float((p == d.y[te]).mean()))
        print(f"[t={time.time()-t0:.0f}s] P3 fold {i+1}/5 svm {fold['rich_svm'][-1]:.4f} ens {fold['rich_ens'][-1]:.4f}", flush=True)
    res["P3_mixed_cv"] = {k: {"acc_mean": float(np.mean(fold[k])), "acc_std": float(np.std(fold[k]))} for k in LAD}
    print(f"[t={time.time()-t0:.0f}s] P3 ra {np.mean(fold['ra']):.4f} svm {np.mean(fold['rich_svm']):.4f} ens {np.mean(fold['rich_ens']):.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rich_ensemble.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    lines = ["| Method | P1 HC->DEP | P2 DEP->HC | P3 mixed |", "|---|---|---|---|"]
    for k in LAD:
        lines.append(f"| {k} | {res['P1_HC2DEP'][k]['accuracy']:.3f} | {res['P2_DEP2HC'][k]['accuracy']:.3f} | {res['P3_mixed_cv'][k]['acc_mean']:.3f} |")
    with open(os.path.join(OUT, "rich_ensemble.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
