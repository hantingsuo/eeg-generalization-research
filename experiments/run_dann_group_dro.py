"""G1: DANN + group-DRO evaluation over P1/P2/P3."""
from __future__ import annotations

from run_novelty_common import evaluate_method, load_features, write_json_md


def main():
    data, F, t0 = load_features()
    from pcma.model.dann import DANNConfig, fit_predict_dann_proba

    cfg = DANNConfig(epochs=60, patience=12, lambda_max=0.5, seed=0)

    def predictor(F_all, d, tr, te):
        return fit_predict_dann_proba(
            F_all[tr],
            d.y[tr],
            d.subject[tr],
            F_all[te],
            d.subject[te],
            group_tr=d.population[tr],
            config=cfg,
        )

    result = evaluate_method("g1_dann_group_dro", predictor, data, F, t0=t0)
    write_json_md(result, "novelty_g1_dann_group_dro")


if __name__ == "__main__":
    main()
