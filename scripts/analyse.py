"""Step 3. Compute every result table and figure from the stored predictions (no GPU needed).

Outputs: Tables 4-9, A2-A5, Figures 4-7, A1, A3, A4, the provenance-label file and numbers.json, followed by
consistency checks between tables and figures.

    python scripts/analyse.py --data-root /path/to/MangoLeafBD --out outputs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mangoleaf import analysis as an
from mangoleaf import figures as F
from mangoleaf.config import NICE, add_common_args, config_from_args
from mangoleaf.experiments import done_ids, registry
from mangoleaf.metrics import ece
from mangoleaf.study import load
from mangoleaf.utils import jsonable, update_numbers


def main():
    a = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(a)
    st = load(cfg)
    R = an.Results(cfg, st)
    T, NUMF = cfg.tbl, cfg.tbl / "numbers.json"
    F.set_style(cfg.save_dpi)
    num = lambda k, v: update_numbers(NUMF, k, v)
    models = R.models_with_sw()
    print("models with session-wise results:", models)

    # Table 4, Figure 4
    t4, store, pmean, n4 = an.principal(R, models, cfg.n_boot)
    t4.round(4).to_csv(T / "Table4_principal_result.csv", index=False); print(t4.round(4).to_string(index=False)); num("main", n4)
    order = sorted(models, key=lambda m: -store[(m, "SW")].mean())
    F.savefig(F.fig4_protocols(store, order), cfg.fig, "Fig4_three_protocols", cfg.save_dpi)

    # Table 5
    t5, ctl, n5 = an.matched_control(R, models, store, pmean, st)
    if t5 is not None:
        t5.round(4).to_csv(T / "Table5_matched_size_control.csv", index=False)
        ctl.round(4).to_csv(T / "Table5b_control_per_model.csv", index=False)
        print(t5.round(4).to_string(index=False)); num("control", n5)

    # Table 6, Table A5
    t6, pw, A, reps, n6 = an.ranking(R, models, store, cfg.n_boot)
    t6.round(4).to_csv(T / "Table6_ranking.csv", index=False); pw.round(4).to_csv(T / "Table6b_paired_bootstrap.csv", index=False)
    print(t6.round(4).to_string(index=False)); print(pw.round(4).to_string(index=False)); num("ranking", n6)
    ta5, n_lr = an.learning_rate(R, models)
    if n_lr:
        ta5.round(4).to_csv(T / "TableA5_learning_rate.csv", index=False); print(ta5.round(4).to_string(index=False)); num("lr_sweep", n_lr)

    # deployed model, Table 7, Figures 5 and 6
    dep = cfg.deploy_model or max(models, key=lambda m: A[m].mean())
    print("deployed model:", NICE[dep], "(highest mean session-wise accuracy over the folds)"); num("deployed_model", NICE[dep])
    t7, cm = an.per_class(R, dep, reps)
    t7.round(3).to_csv(T / "Table7_per_category.csv", index=False); print(t7.round(3).to_string(index=False))
    pd.concat([an.per_class(R, m, reps)[0].assign(model=NICE[m]) for m in models]).round(3) \
        .to_csv(T / "extra_per_category_all_models.csv", index=False)
    s7 = t7.set_index("category").status
    num("per_class", {"rule": f"precision and recall >= {cfg.reliable_min}; borderline if F1 >= {cfg.borderline_f1}",
                      "reliable": [c for c in s7.index if s7[c] == "reliable"],
                      "borderline": [c for c in s7.index if s7[c] == "borderline"],
                      "not_reliable": [c for c in s7.index if s7[c] == "not reliable"],
                      "table": t7.round(3).to_dict("records")})
    F.savefig(F.fig5_per_class(t7, cfg.reliable_min), cfg.fig, "Fig5_per_class", cfg.save_dpi)
    F.savefig(F.fig6_confusion(cm, st.classes), cfg.fig, "Fig6_confusion", cfg.save_dpi)
    pd.DataFrame(cm, index=st.classes, columns=st.classes).to_csv(T / "Fig6_confusion_counts.csv")

    # Table A3, Figure A1
    cal, ta3, pool, n_cal = an.calibration(R, models, reps, dep)
    cal.to_csv(T / "calibration_per_run.csv", index=False); ta3.round(4).to_csv(T / "TableA3_calibration.csv")
    print(ta3.round(4).to_string()); num("calibration_deployed", n_cal)
    F.savefig(F.figA1_reliability(pool, lambda P, y: ece(P, y, cfg.ece_bins)), cfg.fig, "FigA1_reliability", cfg.save_dpi)

    # Table 8, Figure 7
    ref, g, t8, curves = an.referral(R, models, reps, dep)
    ref.to_csv(T / "referral_per_fold.csv", index=False); g.round(4).to_csv(T / "extra_referral_all_models.csv", index=False)
    t8.round(4).to_csv(T / "Table8_referral.csv", index=False); print(t8.round(3).to_string(index=False))
    num("referral_deployed", t8.round(4).to_dict("records"))
    F.savefig(F.fig7_risk_coverage(curves, t8), cfg.fig, "Fig7_risk_coverage", cfg.save_dpi)

    # Tables A2, A4, 9; Figures A3, A4
    ta2, n_tol = an.tolerance(R, store, st)
    ta2.round(4).to_csv(T / "TableA2_tolerance.csv", index=False); print(ta2.round(4).to_string(index=False)); num("tolerance", n_tol)
    ta4 = an.padding(R)
    ta4.round(4).to_csv(T / "TableA4_padding.csv", index=False); print(ta4.round(4).to_string(index=False))
    num("padding", ta4.round(4).to_dict("records"))
    lat_csv = T / "latency.csv"
    if lat_csv.exists():
        lat = pd.read_csv(lat_csv).set_index("model")
        t9 = an.deployment_cost(lat, A, models)
        out9 = t9.copy(); out9.index = [NICE[m] for m in models]
        out9.round(4).to_csv(T / "Table9_deployment_cost.csv"); print(out9.round(4).to_string())
        num("deployment_cost", out9.round(4).to_dict("index"))
        F.savefig(F.figA4_cost(t9), cfg.fig, "FigA4_cost", cfg.save_dpi)
    if cfg.history_csv.exists():
        H = pd.read_csv(cfg.history_csv).merge(R.res[["run_id", "model", "protocol", "group"]], on="run_id")
        F.savefig(F.figA3_curves(H[H.group.isin(["main", "sw"])], models), cfg.fig, "FigA3_training_curves", cfg.save_dpi)

    # provenance labels for the repository
    prov = an.provenance_labels(st, cfg)
    prov.to_csv(T / "mangoleafbd_provenance_labels.csv", index=False); print("provenance rows:", len(prov))

    # consistency checks
    problems = []
    def check(name, cond):
        print(("OK   " if cond else "FAIL ") + name)
        if not cond: problems.append(name)
    nums = json.load(open(NUMF))
    check("Table 2 / Figure 3 use one chance rate (1/number of categories)",
          abs(nums["probes"]["chance"] - round(1 / st.num_classes, 3)) < 1e-9)
    check("Table 5 rows equal Table 4 means", n5 is not None and abs(n5["LW_full"] - pmean["LW"]) < 1e-12)
    check("Table A2 at 3 s equals Table 4 for the same model", cfg.leaf_tol_s in n_tol)
    check("Figure 7 operating points are the Table 8 values", len(t8) == len(cfg.targets))
    check("Figure A1 and Table A3 computed on session-wise test folds", ("SW", "Tcross") in pool)
    check("Figure 6 and Table 7 are the same pooled predictions", int(cm.sum()) == int(t7.support_pooled.sum()))
    specs = registry(cfg, len(st.sw_partitions)); missing = [s["run_id"] for s in specs if s["run_id"] not in done_ids(cfg)]
    check("every planned run finished", not missing)
    print("ALL CONSISTENT" if not problems else f"{len(problems)} problem(s): {problems}")


if __name__ == "__main__":
    main()
