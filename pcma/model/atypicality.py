"""Subject-level emotional EEG atypicality metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pcma.model.rich import per_subject_zscore


def _as_array(x):
    return np.asarray(x)


def _ordered_unique(x):
    return np.asarray(list(dict.fromkeys(np.asarray(x).tolist())))


def _reference_mask(subject, population, target_subject, healthy_label="HC"):
    subject = _as_array(subject)
    population = _as_array(population)
    target_pop = np.unique(population[subject == target_subject])
    if len(target_pop) != 1:
        raise ValueError(f"{target_subject}: population is not constant")
    mask = population == healthy_label
    if target_pop[0] == healthy_label:
        mask = mask & (subject != target_subject)
    if mask.sum() == 0:
        raise ValueError(f"{target_subject}: no healthy reference rows")
    return mask


def healthy_manifold_distance(F, subject, population, healthy_label="HC", max_components=20):
    """Per-subject distance to a leave-one-HC-out healthy PCA manifold."""
    F = np.asarray(F, dtype=float)
    subject = _as_array(subject)
    population = _as_array(population)
    rows = []
    for sid in _ordered_unique(subject):
        target = subject == sid
        ref = _reference_mask(subject, population, sid, healthy_label)
        scaler = StandardScaler().fit(F[ref])
        Xref = scaler.transform(F[ref])
        Xt = scaler.transform(F[target])
        n_comp = min(max_components, Xref.shape[0] - 1, Xref.shape[1])
        if n_comp < 1:
            dist = float(np.mean(np.sum(Xt * Xt, axis=1)))
        else:
            pca = PCA(n_components=n_comp, whiten=True, random_state=0).fit(Xref)
            Z = pca.transform(Xt)
            dist = float(np.mean(np.sum(Z * Z, axis=1) / n_comp))
        rows.append({"subject": str(sid), "manifold_distance": dist})
    return rows


def healthy_classifier_difficulty(F, y, subject, population, healthy_label="HC", C=1.0, classifier="svm_rbf"):
    """Train emotion classifier on healthy controls and score each subject."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(y)
    subject = _as_array(subject)
    population = _as_array(population)
    rows = []
    for sid in _ordered_unique(subject):
        target = subject == sid
        ref = _reference_mask(subject, population, sid, healthy_label)
        if len(np.unique(y[ref])) < 2:
            raise ValueError("healthy reference has fewer than two emotion classes")
        Xref = per_subject_zscore(F[ref], subject[ref])
        Xt = per_subject_zscore(F[target], subject[target])
        scaler = StandardScaler().fit(Xref)
        if classifier == "svm_rbf":
            clf = SVC(C=C, gamma="scale", kernel="rbf", probability=True, random_state=0)
        elif classifier == "logreg":
            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=0)
        else:
            raise ValueError(f"unknown healthy classifier: {classifier}")
        clf.fit(scaler.transform(Xref), y[ref])
        proba = clf.predict_proba(scaler.transform(Xt))
        pred = clf.classes_[proba.argmax(axis=1)]
        rows.append(
            {
                "subject": str(sid),
                "classifier_error": float(np.mean(pred != y[target])),
                "low_confidence": float(np.mean(1.0 - np.max(proba, axis=1))),
            }
        )
    return rows


def _hc_scale(values, population, healthy_label):
    values = np.asarray(values, dtype=float)
    population = _as_array(population)
    hc = population == healthy_label
    mu = float(np.mean(values[hc]))
    sd = float(np.std(values[hc], ddof=1)) if hc.sum() > 1 else 1.0
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return (values - mu) / sd, mu, sd


def subject_atypicality(
    F,
    y,
    subject,
    population,
    healthy_label="HC",
    max_components=20,
    classifier="svm_rbf",
    include_classifier=True,
):
    """Compute subject-level components and HC-standardized composite atypicality."""
    subject = _as_array(subject)
    population = _as_array(population)
    md = {r["subject"]: r["manifold_distance"] for r in healthy_manifold_distance(F, subject, population, healthy_label, max_components)}
    y_arr = None if y is None else np.asarray(y)
    can_score_classifier = include_classifier and y_arr is not None and len(np.unique(y_arr)) >= 2
    cd = {}
    skipped_components = []
    if can_score_classifier:
        cd = {
            r["subject"]: r
            for r in healthy_classifier_difficulty(F, y_arr, subject, population, healthy_label, classifier=classifier)
        }
    else:
        skipped_components.extend(["classifier_error", "low_confidence"])
    rows = []
    for sid in _ordered_unique(subject):
        pop = np.unique(population[subject == sid])
        if len(pop) != 1:
            raise ValueError(f"{sid}: population is not constant")
        rows.append(
            {
                "subject": str(sid),
                "population": str(pop[0]),
                "n_trials": int(np.sum(subject == sid)),
                "manifold_distance": float(md[str(sid)]),
            }
        )
        if cd:
            rows[-1]["classifier_error"] = float(cd[str(sid)]["classifier_error"])
            rows[-1]["low_confidence"] = float(cd[str(sid)]["low_confidence"])
    comps = ["manifold_distance", "classifier_error", "low_confidence"]
    comps = [comp for comp in comps if comp in rows[0]]
    pop_subj = np.asarray([r["population"] for r in rows])
    scaling = {}
    zcols = []
    for comp in comps:
        z, mu, sd = _hc_scale([r[comp] for r in rows], pop_subj, healthy_label)
        scaling[comp] = {"healthy_mean": mu, "healthy_sd": sd}
        zcols.append(z)
        for r, value in zip(rows, z):
            r[f"{comp}_z_hc"] = float(value)
    composite = np.mean(np.vstack(zcols), axis=0)
    for r, value in zip(rows, composite):
        r["atypicality"] = float(value)
    return {
        "subjects": rows,
        "components": comps,
        "skipped_components": skipped_components,
        "healthy_label": healthy_label,
        "scaling": scaling,
        "classifier": classifier if cd else None,
    }


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def bootstrap_mean_diff(x, y, n_boot=5000, seed=0):
    """Bootstrap mean(x)-mean(y)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        diffs[i] = xb.mean() - yb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(np.mean(diffs)), "ci_low": float(lo), "ci_high": float(hi)}


def permutation_group_pvalue(x, y, n_perm=5000, seed=0, alternative="greater"):
    """Permutation p-value for mean(x)-mean(y)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    nx = len(x)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[:nx].mean() - perm[nx:].mean())
        if alternative == "greater":
            hits += diff >= observed
        elif alternative == "less":
            hits += diff <= observed
        else:
            hits += abs(diff) >= abs(observed)
    return {"observed_diff": observed, "p_value": float((hits + 1) / (n_perm + 1)), "alternative": alternative}


def group_contrast(rows, score="atypicality", target_label="DEP", reference_label="HC", n_perm=5000, seed=0):
    target = np.asarray([r[score] for r in rows if r["population"] == target_label], dtype=float)
    ref = np.asarray([r[score] for r in rows if r["population"] == reference_label], dtype=float)
    if len(target) == 0 or len(ref) == 0:
        raise ValueError("both target and reference groups are required")
    u = mannwhitneyu(target, ref, alternative="greater")
    return {
        "score": score,
        "target_label": target_label,
        "reference_label": reference_label,
        "n_target": int(len(target)),
        "n_reference": int(len(ref)),
        "target_mean": float(target.mean()),
        "reference_mean": float(ref.mean()),
        "target_median": float(np.median(target)),
        "reference_median": float(np.median(ref)),
        "mannwhitney_u": float(u.statistic),
        "mannwhitney_p_greater": float(u.pvalue),
        "cliffs_delta": cliffs_delta(target, ref),
        "bootstrap_mean_diff": bootstrap_mean_diff(target, ref, seed=seed),
        "permutation_mean_diff": permutation_group_pvalue(target, ref, seed=seed, alternative="greater", n_perm=n_perm),
    }


def _corr_statistic(x, y, method):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(method)


def bootstrap_correlation_ci(x, y, method="spearman", n_boot=2000, seed=0):
    """Paired bootstrap confidence interval for a correlation coefficient."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        stats[i] = _corr_statistic(x[idx], y[idx], method)
    finite = stats[np.isfinite(stats)]
    if len(finite) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot_finite": 0}
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot_finite": int(len(finite))}


def permutation_correlation(x, y, method="spearman", n_perm=5000, seed=0, alternative="two-sided"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stat = _corr_statistic(x, y, method)
    if not np.isfinite(stat):
        return {"statistic": stat, "p_value": float("nan"), "alternative": alternative}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        perm_stat = _corr_statistic(x, yp, method)
        if not np.isfinite(perm_stat):
            continue
        if alternative == "greater":
            hits += perm_stat >= stat
        elif alternative == "less":
            hits += perm_stat <= stat
        else:
            hits += abs(perm_stat) >= abs(stat)
    return {"statistic": stat, "p_value": float((hits + 1) / (n_perm + 1)), "alternative": alternative}


def severity_correlations(
    rows,
    severity_by_subject,
    score="atypicality",
    n_perm=5000,
    seed=0,
    population_filter=None,
    n_boot=2000,
):
    x, y = [], []
    for r in rows:
        if population_filter is not None and r.get("population") != population_filter:
            continue
        sid = r["subject"]
        if score in r and sid in severity_by_subject and severity_by_subject[sid] is not None:
            x.append(r[score])
            y.append(float(severity_by_subject[sid]))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("at least three subjects with severity scores are required")
    sp = spearmanr(x, y)
    pr = pearsonr(x, y)
    return {
        "score": score,
        "population_filter": population_filter,
        "n": int(len(x)),
        "spearman_r": float(sp.statistic),
        "spearman_p_asymptotic": float(sp.pvalue),
        "spearman_bootstrap_ci": bootstrap_correlation_ci(x, y, "spearman", n_boot=n_boot, seed=seed),
        "pearson_r": float(pr.statistic),
        "pearson_p_asymptotic": float(pr.pvalue),
        "pearson_bootstrap_ci": bootstrap_correlation_ci(x, y, "pearson", n_boot=n_boot, seed=seed + 1000),
        "spearman_permutation": permutation_correlation(x, y, "spearman", n_perm=n_perm, seed=seed),
        "pearson_permutation": permutation_correlation(x, y, "pearson", n_perm=n_perm, seed=seed),
    }
