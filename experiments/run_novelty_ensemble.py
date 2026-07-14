"""G4: soft-vote ensemble of rich-SVM with retained novelty methods only."""
from __future__ import annotations

import json

import numpy as np

from pcma.model.novelty_eval import align_proba, pred_from_proba

from run_novelty_common import OUT, evaluate_method, load_features, rich_proba, write_json_md


def _retained_methods():
    specs = []
    g1 = OUT / "novelty_g1_dann_group_dro.json"
    if g1.exists():
        res = json.loads(g1.read_text(encoding="utf-8"))
        if res.get("retained"):
            specs.append(("g1_dann_group_dro", "dann"))
    g2 = OUT / "novelty_g2_ot_da.json"
    if g2.exists():
        res = json.loads(g2.read_text(encoding="utf-8"))
        for name, variant in res.get("variants", {}).items():
            if variant.get("retained"):
                specs.append((name, name.replace("g2_ot_", "")))
    g3 = OUT / "novelty_g3_clinical_prior.json"
    if g3.exists():
        res = json.loads(g3.read_text(encoding="utf-8"))
        if res.get("retained"):
            specs.append(("g3_clinical_prior", "clinical"))
    return specs


def _method_proba(kind, F_all, d, tr, te):
    if kind == "dann":
        from pcma.model.dann import DANNConfig, fit_predict_dann_proba

        pred, proba, classes = fit_predict_dann_proba(
            F_all[tr],
            d.y[tr],
            d.subject[tr],
            F_all[te],
            d.subject[te],
            group_tr=d.population[tr],
            config=DANNConfig(epochs=60, patience=12, lambda_max=0.5, seed=0),
        )
    elif kind in ("sinkhorn", "classaware"):
        from pcma.model.ot_da import fit_predict_ot_da_proba

        pred, proba, classes = fit_predict_ot_da_proba(
            F_all[tr],
            d.y[tr],
            d.subject[tr],
            F_all[te],
            d.subject[te],
            variant=kind,
            reg=0.05,
            alpha=0.5,
            cost_dim=50,
        )
    elif kind == "clinical":
        from pcma.model.clinical_prior import fit_predict_clinical_prior_proba

        pred, proba, classes = fit_predict_clinical_prior_proba(
            F_all[tr],
            d.y[tr],
            d.subject[tr],
            F_all[te],
            d.subject[te],
        )
    else:
        raise ValueError(kind)
    return align_proba(proba, classes, target_classes=(0, 1))


def main():
    specs = _retained_methods()
    if not specs:
        OUT.mkdir(exist_ok=True)
        result = {
            "method": "g4_retained_soft_vote",
            "retained_inputs": [],
            "retained": False,
            "retention_reason": "drop: no G1/G2/G3 method passed the fixed gate; ensemble not run",
        }
        (OUT / "novelty_g4_ensemble.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (OUT / "novelty_g4_ensemble.md").write_text(
            "# novelty_g4_ensemble\n\nNo G1/G2/G3 method passed the fixed keep gate; no ensemble was run.\n",
            encoding="utf-8",
        )
        print("No retained methods; G4 ensemble not run.", flush=True)
        return

    data, F, t0 = load_features()

    def predictor(F_all, d, tr, te):
        _, rich_p, _ = rich_proba(F_all, d, tr, te)
        all_p = [rich_p]
        for _, kind in specs:
            all_p.append(_method_proba(kind, F_all, d, tr, te))
        proba = np.mean(all_p, axis=0)
        return pred_from_proba(proba, classes=(0, 1)), proba, np.asarray([0, 1])

    result = evaluate_method("g4_retained_soft_vote", predictor, data, F, t0=t0)
    result["retained_inputs"] = [name for name, _ in specs]
    write_json_md(result, "novelty_g4_ensemble")


if __name__ == "__main__":
    main()
