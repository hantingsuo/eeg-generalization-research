import json

import numpy as np
import pytest

from experiments.run_postreview_lr_rf_batch import audit_complete_cell, schedule
from experiments.run_postreview_lr_rf_subject_session import (
    build_models,
    model_configurations,
)


def test_postreview_schedule_is_complete_and_unique():
    cells = schedule()
    assert len(cells) == 90
    assert len({cell["cell_id"] for cell in cells}) == 90
    assert {
        (cell["dataset"], cell["session"], cell["subject"]) for cell in cells
    } == {
        (dataset, session, subject)
        for dataset in ("seed", "seediv")
        for session in range(1, 4)
        for subject in range(1, 16)
    }


def test_postreview_models_match_frozen_configuration():
    models = build_models()
    configs = model_configurations()
    logistic = models["logistic_regression_all555"]
    assert logistic.named_steps["model"].get_params()["C"] == configs[
        "logistic_regression_all555"
    ]["C"]
    forest = models["random_forest_all555"]
    assert forest.get_params()["n_estimators"] == configs[
        "random_forest_all555"
    ]["n_estimators"]
    assert forest.get_params()["max_features"] == "sqrt"


def test_resume_audit_rejects_partial_probability_payload(tmp_path):
    output = tmp_path / "cell.json"
    predictions = tmp_path / "cell.npz"
    output.write_text(
        json.dumps(
            {
                "protocol": "cf_tre_postreview_lr_rf_v1",
                "status": "PASS",
                "stage": "post_review_descriptive_control",
                "post_review": True,
                "confirmatory_gate_member": False,
                "dataset": "seed",
                "session": 1,
                "subject": 1,
                "test_contacted": True,
                "test_evaluation_count": 1,
                "feature_space": "engineered_all555",
                "methods": {
                    method: {"configuration": configuration}
                    for method, configuration in model_configurations().items()
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        predictions,
        validation_labels=np.asarray([0]),
        validation_trials=np.asarray([1]),
        test_labels=np.asarray([0]),
        test_trials=np.asarray([1]),
    )
    with pytest.raises(ValueError, match="missing arrays"):
        audit_complete_cell(
            output,
            predictions,
            {"dataset": "seed", "session": 1, "subject": 1},
        )
