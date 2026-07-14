# experiments/diagnose_pcma.py
"""Diagnose why PCMA (conditional mean alignment) was inert on HC->DEP (P1).
Measures target confidence distribution and how many predictions change under
global / conditional alignment (gate off vs on). Confirms/refutes the fixed-point."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pcma.data.competition import load_competition
from pcma.data.bandcov import to_band_covariances
from pcma.data.splits import p1_hc_to_dep
from pcma.model.pipelines import (band_tangent_features, subject_band_means,
                                  conditional_mean_align, global_mean_align)

def main():
    d = load_competition()
    C = to_band_covariances(d.X, sfreq=250.0)
    sbm = subject_band_means(C, d.subject)
    tr, te = p1_hc_to_dep(d.subject, d.population)
    Xtr, Xte = band_tangent_features(C[tr], C[te], d.subject[tr], d.subject[te], sbm, recenter=True)
    ytr = d.y[tr]; yte = d.y[te]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    yt = clf.predict(Xte)
    conf = clf.predict_proba(Xte).max(axis=1)
    print("target n:", len(yt), "RA-pred acc:", round(float((yt == yte).mean()), 4), flush=True)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        counts = {int(c): int(((yt == c) & (conf >= t)).sum()) for c in np.unique(ytr)}
        print(f"conf>={t}: frac {np.mean(conf >= t):.3f}  confident-per-pseudo-class {counts}", flush=True)
    yt_g = clf.predict(global_mean_align(Xtr, Xte))
    print("GLOBAL align: labels changed vs RA-pred:", int((yt_g != yt).sum()), flush=True)
    Xa_off = conditional_mean_align(Xtr, ytr, Xte, yt, conf, conf_thresh=0.0, min_count=1)
    print("COND gate OFF: labels changed:", int((clf.predict(Xa_off) != yt).sum()),
          "mean|dX|:", round(float(np.abs(Xa_off - Xte).mean()), 5), flush=True)
    Xa_on = conditional_mean_align(Xtr, ytr, Xte, yt, conf, conf_thresh=0.6, min_count=10)
    print("COND gate ON(0.6/10): labels changed:", int((clf.predict(Xa_on) != yt).sum()),
          "mean|dX|:", round(float(np.abs(Xa_on - Xte).mean()), 5), flush=True)

if __name__ == "__main__":
    main()
