"""G3: clinical-prior classifier evaluation over P1/P2/P3."""
from __future__ import annotations

from pcma.model.clinical_prior import fit_predict_clinical_prior_proba

from run_novelty_common import evaluate_method, load_features, write_json_md


def main():
    data, F, t0 = load_features()

    def predictor(F_all, d, tr, te):
        return fit_predict_clinical_prior_proba(
            F_all[tr],
            d.y[tr],
            d.subject[tr],
            F_all[te],
            d.subject[te],
        )

    result = evaluate_method("g3_clinical_prior", predictor, data, F, t0=t0)
    write_json_md(result, "novelty_g3_clinical_prior")


if __name__ == "__main__":
    main()
