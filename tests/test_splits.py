# tests/test_splits.py
import numpy as np
from pcma.data.splits import p1_hc_to_dep, p2_dep_to_hc, p3_mixed_cv, p4_leave_one_dep

subject   = np.repeat(np.arange(6), 4)                      # 6 subjects, 4 trials each
population = np.array(["HC"]*16 + ["DEP"]*8)                # subj 0-3 HC, 4-5 DEP

def _no_subject_leak(tr, te):
    assert set(subject[tr]).isdisjoint(set(subject[te]))

def test_p1_trains_hc_tests_dep():
    tr, te = p1_hc_to_dep(subject, population)
    assert set(population[tr]) == {"HC"}
    assert set(population[te]) == {"DEP"}
    _no_subject_leak(tr, te)

def test_p2_trains_dep_tests_hc():
    tr, te = p2_dep_to_hc(subject, population)
    assert set(population[tr]) == {"DEP"}
    assert set(population[te]) == {"HC"}
    _no_subject_leak(tr, te)

def test_p3_mixed_cv_no_leak():
    folds = list(p3_mixed_cv(subject, n_splits=3))
    assert len(folds) == 3
    for tr, te in folds:
        _no_subject_leak(tr, te)

def test_p4_leave_one_dep_out():
    folds = list(p4_leave_one_dep(subject, population))
    # one fold per DEP subject; each test fold is exactly one DEP subject
    assert len(folds) == 2
    for tr, te in folds:
        assert len(set(subject[te])) == 1
        assert set(population[te]) == {"DEP"}
        _no_subject_leak(tr, te)
