import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from experiments.audit_cf_tre_g0a_split_artifacts import audit_split_artifacts
from experiments.build_cf_tre_g0a_split_artifacts import build_split_artifacts


def _seed_label_file(tmp_path: Path) -> Path:
    path = tmp_path / "label.mat"
    labels = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]])
    savemat(path, {"label": labels})
    return path


def test_builder_and_production_independent_auditor_agree(tmp_path):
    label_file = _seed_label_file(tmp_path)
    output_dir = tmp_path / "artifacts"
    paths = build_split_artifacts(label_file, output_dir)
    assert all(path.is_file() for path in paths.values())
    artifact_manifest = json.loads(paths["artifact_manifest"].read_text(encoding="utf-8"))
    assert artifact_manifest["contains_predictions"] is False
    assert artifact_manifest["contains_accuracy"] is False

    report = audit_split_artifacts(output_dir, label_file)
    assert report["status"] == "PASS"
    assert report["checks_passed"] == report["checks_total"] == 7
    assert report["independence"]["imports_production_split_module"] is False


def test_independent_auditor_rejects_assignment_tampering(tmp_path):
    label_file = _seed_label_file(tmp_path)
    output_dir = tmp_path / "artifacts"
    paths = build_split_artifacts(label_file, output_dir)
    seed = json.loads(paths["seed"].read_text(encoding="utf-8"))
    seed["units"][0]["test_trials"][0] = seed["units"][0]["train_trials"][0]
    paths["seed"].write_text(json.dumps(seed), encoding="utf-8")

    report = audit_split_artifacts(output_dir, label_file)
    assert report["status"] == "FAIL"
    assert any("mismatch" in error for error in report["errors"])

