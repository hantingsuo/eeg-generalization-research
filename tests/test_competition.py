# tests/test_competition.py
import numpy as np

from pcma.data.competition import (
    aggregate_by_video,
    infer_video_ids,
    load_competition,
    to_covariances,
    verify_video_grouping,
)


def test_load_shapes(tmp_path):
    X = np.random.default_rng(1).standard_normal((80, 30, 500)).astype(np.float32)
    y = np.array(([0] * 20 + [1] * 20) * 2)
    subject_ids = np.array(["HC1003"] * 40 + ["DEP1003"] * 40)  # strings
    groups = np.array(["HC"] * 40 + ["DEP"] * 40)               # string groups
    p = tmp_path / "train_data.npz"
    np.savez(p, X=X, y=y, subject_ids=subject_ids, groups=groups)

    data = load_competition(str(p))
    assert data.X.shape == (80, 30, 500)
    assert set(np.unique(data.y)) <= {0, 1}
    assert data.n_subjects == 2
    assert data.population.shape == (80,)
    assert set(np.unique(data.population)) == {"HC", "DEP"}
    assert data.video_id.shape == (80,)


def test_numeric_groups_map_to_population(tmp_path):
    # Real data encodes groups as int64: 0=HC, 1=DEP
    X = np.random.default_rng(3).standard_normal((80, 30, 200)).astype(np.float32)
    y = np.array(([0] * 20 + [1] * 20) * 2)
    subject_ids = np.array(["HC1003"] * 40 + ["DEP1008"] * 40)
    groups = np.array([0] * 40 + [1] * 40, dtype=np.int64)
    p = tmp_path / "train_data.npz"
    np.savez(p, X=X, y=y, subject_ids=subject_ids, groups=groups)

    data = load_competition(str(p))
    assert list(data.population[:2]) == ["HC", "HC"]
    assert list(data.population[-2:]) == ["DEP", "DEP"]
    # subject ids kept as strings, not cast to int
    assert data.subject.dtype.kind in ("U", "S")


def test_infer_video_ids_group_every_five_by_subject_and_label():
    subject = np.array(["S1"] * 20 + ["S1"] * 20 + ["S2"] * 20 + ["S2"] * 20)
    y = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20)
    video_id = infer_video_ids(subject, y)

    assert len(np.unique(video_id)) == 16
    assert video_id[:5].tolist() == ["S1_y0_v0"] * 5
    assert video_id[15:20].tolist() == ["S1_y0_v3"] * 5
    assert video_id[20:25].tolist() == ["S1_y1_v0"] * 5
    audit = verify_video_grouping(subject, y, video_id)
    assert audit["per_video_counts_unique"] == [5]
    assert audit["per_video_labels_constant"]
    assert audit["per_video_subject_constant"]


def test_aggregate_by_video_means_values_and_keeps_metadata_order():
    subject = np.array(["S1"] * 10 + ["S2"] * 10)
    y = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)
    population = np.array(["HC"] * 10 + ["DEP"] * 10)
    video_id = infer_video_ids(subject, y)
    values = np.column_stack([np.arange(20), np.arange(20) * 10.0])

    V, yv, sv, popv, vidv = aggregate_by_video(values, y, subject, population, video_id)

    assert V.shape == (4, 2)
    assert np.allclose(V[:, 0], [2, 7, 12, 17])
    assert yv.tolist() == [0, 1, 0, 1]
    assert sv.tolist() == ["S1", "S1", "S2", "S2"]
    assert popv.tolist() == ["HC", "HC", "DEP", "DEP"]
    assert vidv.tolist() == ["S1_y0_v0", "S1_y1_v0", "S2_y0_v0", "S2_y1_v0"]


def test_covariances_are_spd():
    X = np.random.default_rng(2).standard_normal((5, 30, 400))
    C = to_covariances(X)
    assert C.shape == (5, 30, 30)
    for Ci in C:
        assert np.allclose(Ci, Ci.T, atol=1e-8)
        assert np.linalg.eigvalsh(Ci).min() > 0
