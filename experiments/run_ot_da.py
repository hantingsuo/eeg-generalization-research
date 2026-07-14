"""G2: optimal-transport domain adaptation evaluation over P1/P2/P3."""
from __future__ import annotations

from pcma.model.ot_da import fit_predict_ot_da_proba

from run_novelty_common import evaluate_method, load_features, write_json_md


def main():
    data, F, t0 = load_features()
    variants = {}
    for variant in ("sinkhorn", "classaware"):
        method_name = f"g2_ot_{variant}"

        def predictor(F_all, d, tr, te, variant=variant):
            return fit_predict_ot_da_proba(
                F_all[tr],
                d.y[tr],
                d.subject[tr],
                F_all[te],
                d.subject[te],
                variant=variant,
                reg=0.05,
                alpha=0.5,
                cost_dim=50,
            )

        variants[method_name] = evaluate_method(method_name, predictor, data, F, t0=t0)
    result = {"method": "g2_ot_da", "variants": variants}
    result["retained"] = any(v["retained"] for v in variants.values())
    result["retention_reason"] = "keep: at least one OT variant passed" if result["retained"] else "drop: no OT variant passed"
    write_json_md(result, "novelty_g2_ot_da")


if __name__ == "__main__":
    main()
