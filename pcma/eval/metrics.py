# pcma/eval/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix

def summarize(y_true, y_pred, subject, population=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); subject = np.asarray(subject)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    subj_acc = [ (y_true[subject == s] == y_pred[subject == s]).mean()
                 for s in np.unique(subject) ]
    out["worst_subject_acc"] = float(np.min(subj_acc))
    out["mean_subject_acc"] = float(np.mean(subj_acc))
    if population is not None:
        population = np.asarray(population)
        for p in np.unique(population):
            mask = population == p
            out[f"acc_{p}"] = float(accuracy_score(y_true[mask], y_pred[mask]))
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out["recall_per_class"] = {int(l): float(r) for l, r in zip(labels, rec)}
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out
