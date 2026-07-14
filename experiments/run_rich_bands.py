# experiments/run_rich_bands.py
"""Track B: rich features + RBF-SVM vs RA over P1-P3. Does richer/more-discriminative
features lift HC->DEP and absolute accuracy? Honest: report as produced."""
import json, os, time
import numpy as np
from pcma.data.competition import load_competition
from pcma.data.bandcov import to_band_covariances
from pcma.data.features import extract_rich_features
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv
from pcma.model.pipelines import fit_predict_ra_bands, subject_band_means
from pcma.model.rich import fit_predict_rich
from pcma.eval.metrics import summarize
from pcma.eval.stats import bootstrap_paired_accuracy, paired_wilcoxon

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

def main():
    t0 = time.time()
    d = load_competition()
    F = extract_rich_features(d.X, sfreq=250.0)
    C = to_band_covariances(d.X, sfreq=250.0); sbm = subject_band_means(C, d.subject)
    print(f"[t={time.time()-t0:.0f}s] features {F.shape} + covariances done", flush=True)
    res = {}
    for name, sp in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = sp(d.subject, d.population)
        ra = fit_predict_ra_bands(C[tr], d.y[tr], d.subject[tr], C[te], d.subject[te], ref_means_per_band=sbm)
        rich = fit_predict_rich(F[tr], d.y[tr], d.subject[tr], F[te], d.subject[te])
        res[name] = {"ra": summarize(d.y[te], ra, d.subject[te], d.population[te]),
                     "rich": summarize(d.y[te], rich, d.subject[te], d.population[te]),
                     "boot_rich_vs_ra": bootstrap_paired_accuracy(d.y[te], ra, rich)}
        for k in ("ra", "rich"):
            m = res[name][k]
            print(f"[t={time.time()-t0:.0f}s] {name} {k:5s} acc {m['accuracy']:.4f} bal {m['balanced_accuracy']:.4f} pos_recall {m['recall_per_class'].get(1, float('nan')):.4f}", flush=True)
        bt = res[name]["boot_rich_vs_ra"]
        print(f"[t={time.time()-t0:.0f}s] {name} RICH-RA diff {bt['mean_diff']:+.4f} CI[{bt['ci_low']:+.4f},{bt['ci_high']:+.4f}] frac>0 {bt['frac_b_gt_a']:.3f}", flush=True)
    raf, rf = [], []
    for i, (tr, te) in enumerate(p3_mixed_cv(d.subject, n_splits=5)):
        ra = fit_predict_ra_bands(C[tr], d.y[tr], d.subject[tr], C[te], d.subject[te], ref_means_per_band=sbm)
        rich = fit_predict_rich(F[tr], d.y[tr], d.subject[tr], F[te], d.subject[te])
        raf.append(float((ra == d.y[te]).mean())); rf.append(float((rich == d.y[te]).mean()))
        print(f"[t={time.time()-t0:.0f}s] P3 fold {i+1}/5 ra {raf[-1]:.4f} rich {rf[-1]:.4f}", flush=True)
    _, wp = paired_wilcoxon(rf, raf)
    res["P3_mixed_cv"] = {"ra_mean": float(np.mean(raf)), "rich_mean": float(np.mean(rf)), "wilcoxon_rich_vs_ra_p": float(wp)}
    print(f"[t={time.time()-t0:.0f}s] P3 ra {np.mean(raf):.4f} rich {np.mean(rf):.4f} wilcoxon_p {wp:.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rich_bands.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    lines = ["| Method | P1 HC->DEP | P2 DEP->HC | P3 mixed |", "|---|---|---|---|",
             f"| RA (cov) | {res['P1_HC2DEP']['ra']['accuracy']:.3f} | {res['P2_DEP2HC']['ra']['accuracy']:.3f} | {res['P3_mixed_cv']['ra_mean']:.3f} |",
             f"| RICH (SVM) | {res['P1_HC2DEP']['rich']['accuracy']:.3f} | {res['P2_DEP2HC']['rich']['accuracy']:.3f} | {res['P3_mixed_cv']['rich_mean']:.3f} |"]
    b1 = res["P1_HC2DEP"]["boot_rich_vs_ra"]
    lines.append(f"\n**P1 RICH-RA: {b1['mean_diff']:+.3f} (95% CI [{b1['ci_low']:+.3f},{b1['ci_high']:+.3f}], frac>0 {b1['frac_b_gt_a']:.2f})**")
    lines.append(f"\nP1 positive recall - RA {res['P1_HC2DEP']['ra']['recall_per_class'].get(1, float('nan')):.3f} -> RICH {res['P1_HC2DEP']['rich']['recall_per_class'].get(1, float('nan')):.3f}")
    with open(os.path.join(OUT, "rich_bands.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
