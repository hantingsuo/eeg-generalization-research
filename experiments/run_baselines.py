# experiments/run_baselines.py
"""Run no-adapt and RA baselines over P1-P4, save results table.

Produces the cross-population accuracy-gap finding:
  gap = mean(P3 same-population CV acc) - mean(P1/P2 cross-population acc)
"""
import json, os
import numpy as np
from pcma.data.competition import load_competition, to_covariances
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv, p4_leave_one_dep
from pcma.model.pipelines import fit_predict_noadapt, fit_predict_ra
from pcma.eval.metrics import summarize

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

def _eval_single(Ctr, ytr, str_, Cte, yte, ste, pop_te):
    r = {}
    r["noadapt"] = summarize(yte, fit_predict_noadapt(Ctr, ytr, Cte), ste, pop_te)
    r["ra"]      = summarize(yte, fit_predict_ra(Ctr, ytr, str_, Cte, ste), ste, pop_te)
    return r

def main():
    d = load_competition()
    C = to_covariances(d.X)
    res = {}

    # P1 / P2: single split each
    for name, splitter in (("P1_HC2DEP", p1_hc_to_dep), ("P2_DEP2HC", p2_dep_to_hc)):
        tr, te = splitter(d.subject, d.population)
        res[name] = _eval_single(C[tr], d.y[tr], d.subject[tr],
                                  C[te], d.y[te], d.subject[te], d.population[te])
        print(name, "noadapt", round(res[name]["noadapt"]["accuracy"], 4),
                    "ra", round(res[name]["ra"]["accuracy"], 4))

    # P3: mixed cross-subject CV -> per-fold acc, then mean
    p3_noadapt, p3_ra = [], []
    for tr, te in p3_mixed_cv(d.subject, n_splits=5):
        rr = _eval_single(C[tr], d.y[tr], d.subject[tr],
                          C[te], d.y[te], d.subject[te], d.population[te])
        p3_noadapt.append(rr["noadapt"]["accuracy"]); p3_ra.append(rr["ra"]["accuracy"])
    res["P3_mixed_cv"] = {
        "noadapt": {"accuracy_mean": float(np.mean(p3_noadapt)), "accuracy_std": float(np.std(p3_noadapt)), "folds": p3_noadapt},
        "ra":      {"accuracy_mean": float(np.mean(p3_ra)),      "accuracy_std": float(np.std(p3_ra)),      "folds": p3_ra},
    }
    print("P3 noadapt", round(np.mean(p3_noadapt), 4), "ra", round(np.mean(p3_ra), 4))

    # P4: leave-one-DEP-out -> per-DEP-subject acc
    p4_ra = []
    for tr, te in p4_leave_one_dep(d.subject, d.population):
        pred = fit_predict_ra(C[tr], d.y[tr], d.subject[tr], C[te], d.subject[te])
        p4_ra.append(float((pred == d.y[te]).mean()))
    res["P4_leave_one_dep"] = {"ra_per_dep_acc": p4_ra,
                               "ra_mean": float(np.mean(p4_ra)),
                               "ra_worst": float(np.min(p4_ra))}
    print("P4 RA per-DEP mean", round(np.mean(p4_ra), 4), "worst", round(np.min(p4_ra), 4))

    # Headline: cross-population gap (RA)
    xpop = np.mean([res["P1_HC2DEP"]["ra"]["accuracy"], res["P2_DEP2HC"]["ra"]["accuracy"]])
    gap = res["P3_mixed_cv"]["ra"]["accuracy_mean"] - xpop
    res["cross_population_gap_ra"] = float(gap)
    print("CROSS-POPULATION GAP (RA):", round(gap, 4))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # markdown table
    lines = ["| Protocol | no-adapt acc | RA acc |", "|---|---|---|"]
    lines.append(f"| P1 HC->DEP | {res['P1_HC2DEP']['noadapt']['accuracy']:.3f} | {res['P1_HC2DEP']['ra']['accuracy']:.3f} |")
    lines.append(f"| P2 DEP->HC | {res['P2_DEP2HC']['noadapt']['accuracy']:.3f} | {res['P2_DEP2HC']['ra']['accuracy']:.3f} |")
    lines.append(f"| P3 mixed CV | {res['P3_mixed_cv']['noadapt']['accuracy_mean']:.3f} | {res['P3_mixed_cv']['ra']['accuracy_mean']:.3f} |")
    lines.append(f"\n**Cross-population gap (RA): {gap:.3f}**")
    with open(os.path.join(OUT, "baselines.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
