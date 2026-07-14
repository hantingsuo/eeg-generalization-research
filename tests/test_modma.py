import numpy as np
from openpyxl import Workbook

from pcma.data.modma import (
    bandpower_de_features,
    bandpower_feature_names,
    load_modma_feature_cache,
    load_modma_metadata,
    save_modma_feature_cache,
    ModmaFeatureData,
)


def test_load_modma_feature_cache(tmp_path):
    path = tmp_path / "modma_features.npz"
    np.savez(
        path,
        F=np.ones((4, 3)),
        y_emotion=np.array([0, 1, 0, 1]),
        subject=np.array(["S1", "S1", "S2", "S2"]),
        population=np.array(["HC", "HC", "MDD", "MDD"]),
        phq9_subject=np.array(["S1", "S2"]),
        phq9=np.array([2, 18]),
        event=np.array(["face", "face", "face", "face"]),
    )
    data = load_modma_feature_cache(path)
    assert data.F.shape == (4, 3)
    assert data.phq9_by_subject == {"S1": 2.0, "S2": 18.0}
    assert data.event.tolist() == ["face"] * 4


def test_load_modma_metadata_joins_xlsx_to_raw_files(tmp_path):
    root = tmp_path
    (root / "02010002erp test.raw").write_bytes(b"raw")
    (root / "02020008_erp test.raw").write_bytes(b"raw")
    wb = Workbook()
    ws = wb.active
    ws.append(["subject id", "type", "age", "gender", "education(years)", "PHQ-9", "GAD-7", "PSQI"])
    ws.append(["02010002", "MDD", 18, "F", 12, 23, 18, 12])
    ws.append(["02020008", "HC", 20, "M", 16, 1, 0, 2])
    wb.save(root / "subjects_information_test.xlsx")

    rows = load_modma_metadata(root)
    assert [r.subject_id for r in rows] == ["02010002", "02020008"]
    assert rows[0].population == "MDD"
    assert rows[0].raw_path.name.startswith("02010002")
    assert rows[0].scales["PHQ-9"] == 23.0
    assert rows[1].population == "HC"


def test_bandpower_de_features_are_finite_and_named():
    sfreq = 250.0
    t = np.arange(250) / sfreq
    epoch = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)])
    names = bandpower_feature_names(["E1", "E2"])
    feat = bandpower_de_features(epoch, sfreq)
    assert feat.shape == (len(names),)
    assert np.isfinite(feat).all()
    assert names[0] == "de_delta_E1"
    assert names[-1] == "rel_power_gamma_E2"


def test_save_and_load_modma_feature_cache_with_scales(tmp_path):
    path = tmp_path / "cache.npz"
    data = ModmaFeatureData(
        F=np.ones((2, 4), dtype=np.float32),
        y_emotion=np.array(["fcue", "hcue"]),
        subject=np.array(["02010002", "02020008"]),
        population=np.array(["MDD", "HC"]),
        phq9_by_subject={"02010002": 23.0, "02020008": 1.0},
        event=np.array(["fcue", "hcue"]),
        scales_by_subject={
            "02010002": {"PHQ-9": 23.0, "GAD-7": 18.0, "PSQI": 12.0},
            "02020008": {"PHQ-9": 1.0, "GAD-7": 0.0, "PSQI": 2.0},
        },
        feature_names=["a", "b", "c", "d"],
        source="unit-test",
        config={"erp_v2": {"cluster_channels": ["E55", "E62"]}},
    )
    save_modma_feature_cache(path, data)
    loaded = load_modma_feature_cache(path)
    assert loaded.F.shape == (2, 4)
    assert loaded.y_emotion.tolist() == ["fcue", "hcue"]
    assert loaded.scales_by_subject["02010002"]["GAD-7"] == 18.0
    assert loaded.feature_names == ["a", "b", "c", "d"]
    assert loaded.config == {"erp_v2": {"cluster_channels": ["E55", "E62"]}}
