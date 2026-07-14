# tests/test_metrics.py
import numpy as np
from pcma.eval.metrics import summarize

def test_summarize_known_values():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])   # 4/6 correct
    subject = np.array([0, 0, 0, 1, 1, 1])
    m = summarize(y_true, y_pred, subject)
    assert abs(m["accuracy"] - 4/6) < 1e-9
    assert 0.0 <= m["macro_f1"] <= 1.0
    # subject 0: 2/3 correct; subject 1: 2/3 correct → worst = 2/3
    assert abs(m["worst_subject_acc"] - 2/3) < 1e-9

def test_summarize_has_balanced_recall_confusion():
    import numpy as np
    from pcma.eval.metrics import summarize
    y_true = np.array([0, 0, 1, 1, 1, 1]); y_pred = np.array([0, 0, 1, 1, 0, 0])
    subject = np.array([0, 0, 1, 1, 1, 1])
    m = summarize(y_true, y_pred, subject)
    assert "balanced_accuracy" in m
    assert m["recall_per_class"][1] == 0.5   # 2 of 4 positives recalled
    assert m["recall_per_class"][0] == 1.0
    assert np.array(m["confusion"]).shape == (2, 2)
