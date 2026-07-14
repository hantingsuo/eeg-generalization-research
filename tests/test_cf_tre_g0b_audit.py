from __future__ import annotations

from experiments.audit_cf_tre_g0b_matrix import subject_equal_mean


def test_subject_equal_mean_averages_sessions_before_subjects() -> None:
    rows = []
    for subject in range(1, 16):
        for session in range(1, 4):
            rows.append({"subject": subject, "session": session, "accuracy": subject / 15})
    expected = sum(subject / 15 for subject in range(1, 16)) / 15
    assert subject_equal_mean(rows, "accuracy") == expected
