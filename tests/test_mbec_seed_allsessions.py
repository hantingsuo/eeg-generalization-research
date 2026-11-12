"""Unit tests for experiments/mbec_seed_allsessions.py (SEED-AS). Synthetic data only; no SEED cache is read."""
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ["SEEDAS_SMOKE"] = "1"  # must be set before import: keeps every path inside the smoke directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import mbec_seed_allsessions as m  # noqa: E402

E1 = json.loads((ROOT / "plans/2026-09-17-jne-e1-seed-cross-subject.json").read_text(encoding="utf-8"))
FOLDS = E1["folds"]


def test_standardise_zero_mean_unit_std_per_feature_per_recording():
    rng = np.random.default_rng(1)
    for scale, shift in ((1.0, 0.0), (7.0, 30.0), (0.01, -2.0)):
        x = (rng.normal(size=(400, 62, 5)) * scale + shift).astype(np.float32)
        z = m.standardise_recording(x)
        assert z.shape == x.shape and z.dtype == np.float32
        np.testing.assert_allclose(z.mean(0, dtype=np.float64), 0.0, atol=1e-4)
        np.testing.assert_allclose(z.std(0, dtype=np.float64), 1.0, atol=1e-4)


def test_standardise_uses_no_labels_and_no_other_recording():
    assert list(inspect.signature(m.standardise_recording).parameters) == ["x"]
    rng = np.random.default_rng(2)
    a = rng.normal(size=(300, 62, 5)).astype(np.float32)
    b = (rng.normal(size=(300, 62, 5)) * 5 + 9).astype(np.float32)
    za = m.standardise_recording(a)
    m.standardise_recording(b)  # no state shared across calls
    np.testing.assert_array_equal(za, m.standardise_recording(a))


def test_standardise_std_floor():
    x = np.ones((50, 62, 5), np.float32) * 3.0  # constant feature: std 0 -> floor 1e-6, output 0 not NaN
    z = m.standardise_recording(x)
    assert np.isfinite(z).all() and np.abs(z).max() == 0.0


def test_standardise_matches_faced_reference():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    x = (rng.normal(size=(500, 62, 5)) * 3 + 4).astype(np.float32)
    g = torch.from_numpy(x)
    g64 = g.double()
    mu = g64.mean(dim=0)
    sd = g64.var(dim=0, unbiased=False).clamp_min(0).sqrt().clamp_min(1e-6)
    ref = ((g - mu.float().view(1, 62, 5)) / sd.float().view(1, 62, 5)).numpy()
    np.testing.assert_allclose(m.standardise_recording(x), ref, atol=1e-5)


@pytest.mark.parametrize("fold", FOLDS)
def test_train_set_is_nine_participants_times_three_sessions(fold):
    recs = m.train_recordings(fold)
    assert len(recs) == 27
    assert sorted({s for s, _ in recs}) == [1, 2, 3]
    assert {p for _, p in recs} == set(fold["train"]) and len(set(fold["train"])) == 9
    forbidden = set(fold["validation"]) | set(fold["test"])
    assert not ({p for _, p in recs} & forbidden)
    assert len(set(recs)) == 27


def test_fold_training_windows():
    cfg = {"folds": FOLDS}
    n = sum(len(m.load_recording(s, p)[1]) for s, p in m.train_recordings(cfg["folds"][0]))
    assert n == 91638 == 27 * 3394


def test_scored_pools_are_session1_only_with_45_units(monkeypatch):
    requested = []
    real = m.load_recording

    def spy(session, subject):
        requested.append((session, subject))
        return real(session, subject)

    monkeypatch.setattr(m, "load_recording", spy)
    fold = FOLDS[0]
    x, y, parts = m.build_pools(fold, E1["trial_labels"])
    assert len(x) == 91638
    for name, subs in (("validation", fold["validation"]), ("target", fold["test"])):
        v = parts[name]
        assert v["units"].shape == (45, 2) and len(v["count"]) == 45 and len(v["label"]) == 45
        assert sorted(set(v["units"][:, 0].tolist())) == sorted(subs)
        assert sorted(set(v["units"][:, 1].tolist())) == list(range(1, 16))
        assert len(v["y"]) == 3394 * 3  # one session per scored participant
        for p in subs:
            assert [r for r in requested if r[1] == p] == [(1, p)]  # only session 1, loaded once
    assert {q for s, q in requested if s in (2, 3)} == set(fold["train"])
    assert {q for s, q in requested} == set(fold["train"]) | set(fold["validation"]) | set(fold["test"])


def test_each_scored_recording_standardised_on_its_own():
    _, _, parts = m.build_pools(FOLDS[0], E1["trial_labels"])
    for name in ("validation", "target"):
        for block in np.split(parts[name]["x"], 3):  # one session-1 recording per participant
            np.testing.assert_allclose(block.mean(0, dtype=np.float64), 0.0, atol=1e-4)
            np.testing.assert_allclose(block.std(0, dtype=np.float64), 1.0, atol=1e-4)


def test_freeze_refuses_when_predictions_empty(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    monkeypatch.setattr(m, "SMOKE", False)
    monkeypatch.setattr(m, "PREDICTIONS", [])
    monkeypatch.setattr(m, "PLAN", plan)
    with pytest.raises(SystemExit) as e:
        m.freeze()
    assert "PREDICTIONS is empty" in str(e.value)
    assert not plan.exists()


def test_module_constant_predictions_is_filled_list():
    # filled in by the project lead on 2026-11-12, before freeze
    assert isinstance(m.PREDICTIONS, list) and len(m.PREDICTIONS) == 5


def test_smoke_plan_fields(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    monkeypatch.setattr(m, "PLAN", plan)
    m.freeze()
    cfg = json.loads(plan.read_text(encoding="utf-8"))
    assert cfg["predictions_written_before_running"] == list(m.PREDICTIONS)
    assert cfg["exposure"].startswith("E1 (session-1 training, source-statistics standardisation)")
    assert "only in items (2) and (4)" in cfg["differs_from_e1"]
    assert cfg["models"] == ["dgcnn", "mlp"] and cfg["folds"] == FOLDS
    assert cfg["training_windows_per_fold"] == [91638] * 5
    assert cfg["calibration"] == E1["calibration"] and cfg["trial_labels"] == E1["trial_labels"]
    assert len(cfg["primary_endpoints"]) == 3
