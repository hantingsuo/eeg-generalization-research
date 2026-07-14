# experiments/run_pcma_bands.py
"""PCMA ablation on the HC->DEP asymmetry. Ladder: no-adapt, RA, RA+global,
RA+conditional(1), RA+iterative(PCMA), RA+CORAL. Reports acc/balanced/macro-F1/
per-class recall/confusion + paired bootstrap (PCMA vs RA and CORAL vs RA).
Honest: no tuning to a target."""
import json, os, time
import numpy as np
from pcma.data.competition import load_competition
from pcma.data.bandcov import to_band_covariances
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.model.pipelines import (fit_predict_noadapt_bands, fit_predict_ra_bands,
                                  fit_predict_pcma_bands, fit_predict_coral_bands,
                                  subject_band_means)
from pcma.eval.metrics import summarize
from pcma.eval.stats import bootstrap_paired_accuracy, paired_wilcoxon

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
LADDER = ["noadapt", "ra", "ra_global", "ra_cond1", "pcma", "coral"]

def _predict(kind, Ctr, ytr, str_, Cte, ste, sbm):
    if kind == "noadapt": return fit_predict_noadapt_bands(Ctr, ytr, Cte)
    if kind == "ra":      return fit_predict_ra_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm)
    if kind == "ra_global": return fit_predict_pcma_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm, mode="global")
    if kind == "ra_cond1":  return fit_predict_pcma_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm, mode="conditional")
    if kind == "pcma":      return fit_predict_pcma_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm, mode="iterative")
    if kind == "coral":     return fit_predict_coral_bands(Ctr, ytr, str_, Cte, ste, ref_means_per_band=sbm)
    raise ValueError(kind)

def main():
    t0 = time.time()
    d = load_competition()
    C = to_band_covariances(d.X, sfreq=250.0)
    sbm = subject_band_means(C, d.subject)
    print(f"[t={time.time()-t0:.0f}s] setup done", flush=True)
    res = {}

    for name, splitter in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = splitter(d.subject, d.population)
        preds = {k: _predict(k, C[tr], d.y[tr], d.subject[tr], C[te], d.subject[te], sbm) for k in LADDER}
        res[name] = {k: summarize(d.y[te], preds[k], d.subject[te], d.population[te]) for k in LADDER}
        res[name]["boot_pcma_vs_ra"] = bootstrap_paired_accuracy(d.y[te], preds["ra"], preds["pcma"])
        res[name]["boot_coral_vs_ra"] = bootstrap_paired_accuracy(d.y[te], preds["ra"], preds["coral"])
        for k in LADDER:
            m = res[name][k]
            print(f"[t={time.time()-t0:.0f}s] {name} {k:9s} acc {m['accuracy']:.4f} bal {m['balanced_accuracy']:.4f} pos_recall {m['recall_per_class'].get(1, float('nan')):.4f}", flush=True)
        for tag in ("boot_pcma_vs_ra", "boot_coral_vs_ra"):
            bt = res[name][tag]
            print(f"[t={time.time()-t0:.0f}s] {name} {tag} diff {bt['mean_diff']:+.4f} CI[{bt['ci_low']:+.4f},{bt['ci_high']:+.4f}] frac>0 {bt['frac_b_gt_a']:.3f}", flush=True)

    fold = {k: [] for k in LADDER}
    for i, (tr, te) in enumerate(p3_mixed_cv(d.subject, n_splits=5)):
        for k in LADDER:
            p = _predict(k, C[tr], d.y[tr], d.subject[tr], C[te], d.subject[te], sbm)
            fold[k].append(float((p == d.y[te]).mean()))
        print(f"[t={time.time()-t0:.0f}s] P3 fold {i+1}/5 ra {fold['ra'][-1]:.4f} coral {fold['coral'][-1]:.4f}", flush=True)
    _, wp_pcma = paired_wilcoxon(fold["pcma"], fold["ra"])
    _, wp_coral = paired_wilcoxon(fold["coral"], fold["ra"])
    res["P3_mixed_cv"] = {k: {"acc_mean": float(np.mean(fold[k])), "acc_std": float(np.std(fold[k]))} for k in LADDER}
    res["P3_mixed_cv"]["wilcoxon_pcma_vs_ra_p"] = float(wp_pcma)
    res["P3_mixed_cv"]["wilcoxon_coral_vs_ra_p"] = float(wp_coral)
    print(f"[t={time.time()-t0:.0f}s] P3 ra {np.mean(fold['ra']):.4f} coral {np.mean(fold['coral']):.4f} pcma {np.mean(fold['pcma']):.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pcma_bands.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    def acc(name, k): return res[name][k]["accuracy"]
    lines = ["| Method | P1 HC->DEP | P2 DEP->HC | P3 mixed |", "|---|---|---|---|"]
    for k in LADDER:
        lines.append(f"| {k} | {acc('P1_HC2DEP',k):.3f} | {acc('P2_DEP2HC',k):.3f} | {res['P3_mixed_cv'][k]['acc_mean']:.3f} |")
    for tag, label in (("boot_pcma_vs_ra", "PCMA-RA"), ("boot_coral_vs_ra", "CORAL-RA")):
        b1 = res["P1_HC2DEP"][tag]
        lines.append(f"\n**P1 {label}: {b1['mean_diff']:+.3f} (95% CI [{b1['ci_low']:+.3f}, {b1['ci_high']:+.3f}], frac>0 {b1['frac_b_gt_a']:.2f})**")
    lines.append(f"\nP1 positive-class recall - RA {res['P1_HC2DEP']['ra']['recall_per_class'].get(1, float('nan')):.3f}, CORAL {res['P1_HC2DEP']['coral']['recall_per_class'].get(1, float('nan')):.3f}")
    with open(os.path.join(OUT, "pcma_bands.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
