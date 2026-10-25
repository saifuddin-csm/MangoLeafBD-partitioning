"""Analyses of the stored predictions. One function per table of the article; each writes its CSV to
<out>/tables, and the values quoted in the text are collected in numbers.json.

Table numbering follows the article: T4 principal result, T5 matched-size control, T6 ranking and paired
bootstrap, T7 per-category reliability, T8 referral, T9 deployment cost, A2-A5 supplementary tables.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import FAMILY, NICE
from .experiments import run_id
from .metrics import at_threshold, boot_mean, ci, ece, fit_T, risk_coverage, temper, threshold_for


class Results:
    """Loads the stored predictions (<out>/preds/*.npz) of every finished run."""
    def __init__(self, cfg, study):
        self.cfg, self.study = cfg, study
        self.res = pd.read_csv(cfg.results_csv) if cfg.results_csv.exists() else pd.DataFrame(columns=["run_id"])
        if "lr" in self.res: self.res["lr"] = self.res["lr"].astype(float)
        self._cache = {}
        nparts = len(study.sw_partitions)
        self.sw_reps = [f"p{p}f{k}" for p in range(nparts) for k in range(cfg.n_folds)]
        self.seed_reps = [f"s{s}" for s in cfg.seeds]

    def rid(self, model, proto, rep, lr=None, resize="squash"):
        return run_id(model, proto, rep, self.cfg.lr if lr is None else lr, resize, self.cfg.lr)

    def get(self, rid):
        if rid not in self._cache:
            f = self.cfg.pred / (rid.replace("|", "__") + ".npz")
            self._cache[rid] = dict(np.load(f, allow_pickle=True)) if f.exists() else None
        return self._cache[rid]

    def run(self, model, proto, rep, **kw):
        return self.get(self.rid(model, proto, rep, **kw))

    def avail(self, model, proto, reps):
        return [r for r in reps if self.run(model, proto, r) is not None]

    def acc(self, model, proto, rep, **kw):
        r = self.run(model, proto, rep, **kw)
        return accuracy_score(r["y_test"], r["P_test"].argmax(1))

    @staticmethod
    def partner(rep):
        p, k = rep[1:].split("f"); return f"p{p}f{1 - int(k)}"

    def models_with_sw(self):
        return [m for m in self.cfg.models if self.avail(m, "SW", self.sw_reps)]


# ---------------------------------------------------------------- Table 4 (principal result)
def principal(R, models, n_boot):
    rows, store = [], {}
    for m in models:
        r = {"model": NICE[m], "family": FAMILY[m]}
        for proto, reps in (("RP", R.seed_reps), ("LW", R.seed_reps), ("SW", R.sw_reps)):
            a = np.array([R.acc(m, proto, x) for x in R.avail(m, proto, reps)])
            store[(m, proto)] = a
            r[f"{proto}_mean"], r[f"{proto}_sd"], r[f"{proto}_n"] = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)
        r["SW_min"], r["SW_max"] = store[(m, "SW")].min(), store[(m, "SW")].max()
        bs = boot_mean([((R.run(m, "SW", x)["P_test"].argmax(1) == R.run(m, "SW", x)["y_test"]),
                         R.run(m, "SW", x)["leaf_test"]) for x in R.avail(m, "SW", R.sw_reps)], B=n_boot)
        r["SW_ci_lo"], r["SW_ci_hi"] = ci(bs)
        r["RP_minus_SW"] = r["RP_mean"] - r["SW_mean"]; r["LW_minus_SW"] = r["LW_mean"] - r["SW_mean"]
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("SW_mean", ascending=False).reset_index(drop=True)
    mean_row = {"model": "Mean", "family": ""}
    for c in t.columns[2:]: mean_row[c] = t[c].mean()
    t = pd.concat([t, pd.DataFrame([mean_row])], ignore_index=True)
    pmean = {p: float(t.iloc[-1][f"{p}_mean"]) for p in ("RP", "LW", "SW")}
    num = {"mean": pmean, "RP_minus_SW": float(t.iloc[-1]["RP_minus_SW"]), "LW_minus_SW": float(t.iloc[-1]["LW_minus_SW"]),
           "sd_range": {p: [float(t.iloc[:-1][f"{p}_sd"].min()), float(t.iloc[:-1][f"{p}_sd"].max())] for p in ("RP", "LW", "SW")},
           "per_model": t.iloc[:-1].set_index("model").round(4).to_dict("index"),
           "n_sw_folds": len(R.sw_reps), "n_seeds": len(R.seed_reps)}
    return t, store, pmean, num


# ---------------------------------------------------------------- Table 5 (matched-size control, Eq. 9)
def matched_control(R, models, store, pmean, study):
    ctl = []
    for m in models:
        lwm = [R.acc(m, "LWm", x) for x in R.avail(m, "LWm", R.seed_reps)]
        if lwm:
            ctl.append({"model": NICE[m], "LW_full": store[(m, "LW")].mean(), "LW_matched": np.mean(lwm),
                        "SW": store[(m, "SW")].mean()})
    ctl = pd.DataFrame(ctl)
    if len(ctl) != len(models):
        return None, ctl, None
    pt = study.partitions_table
    lw_full, lw_m, sw = ctl.LW_full.mean(), ctl.LW_matched.mean(), ctl.SW.mean()
    assert abs(lw_full - pmean["LW"]) < 1e-12 and abs(sw - pmean["SW"]) < 1e-12   # Table 5 == Table 4
    loss = lw_full - sw; size_eff = lw_full - lw_m; part_eff = lw_m - sw
    t = pd.DataFrame([
        {"condition": "Leaf-wise, full training portion", "training_images": round(float(pt[pt.protocol == "LW"].train.mean())), "accuracy": lw_full},
        {"condition": "Leaf-wise, matched training portion", "training_images": study.n_match, "accuracy": lw_m},
        {"condition": "Session-wise", "training_images": round(float(pt[pt.protocol == "SW"].train.mean())), "accuracy": sw},
        {"condition": "Attributable to training-set size", "training_images": "", "accuracy": size_eff},
        {"condition": "Attributable to the partition", "training_images": "", "accuracy": part_eff}])
    num = {"LW_full": lw_full, "LW_matched": lw_m, "SW": sw, "loss": loss, "size": size_eff, "partition": part_eff,
           "partition_pct": 100 * part_eff / loss,
           "reduction_pct": 100 * (1 - study.n_match / pt[pt.protocol == "LW"].train.mean())}
    return t, ctl, num


# ---------------------------------------------------------------- Table 6 (ranking, paired bootstrap)
def ranking(R, models, store, n_boot):
    reps = [x for x in R.sw_reps if all(R.run(m, "SW", x) is not None for m in models)]
    A = pd.DataFrame({m: [R.acc(m, "SW", x) for x in reps] for m in models}, index=reps)
    ranks = A.rank(axis=1, ascending=False, method="average")
    lw_mean = {m: store[(m, "LW")].mean() for m in models}
    lw_rank = pd.Series(lw_mean).rank(ascending=False, method="min")
    t = pd.DataFrame({"model": [NICE[m] for m in models],
                      "LW_accuracy": [lw_mean[m] for m in models], "LW_rank": [lw_rank[m] for m in models],
                      "SW_accuracy": [A[m].mean() for m in models], "SW_sd": [A[m].std(ddof=1) for m in models],
                      "SW_mean_rank": [ranks[m].mean() for m in models],
                      "SW_best_in_folds": [int((ranks[m] == 1).sum()) for m in models]}).sort_values("SW_accuracy", ascending=False)
    spread = {"LW": max(lw_mean.values()) - min(lw_mean.values()), "SW": float(A.mean().max() - A.mean().min())}

    def paired(m1, m2, proto, rr):
        per = []
        for x in rr:
            a, b = R.run(m1, proto, x), R.run(m2, proto, x)
            assert (a["key_test"] == b["key_test"]).all()      # identical test images
            per.append(((a["P_test"].argmax(1) == a["y_test"]).astype(float) -
                        (b["P_test"].argmax(1) == b["y_test"]).astype(float), a["leaf_test"]))
        d = np.array([v.mean() for v, _ in per]); lo, hi = ci(boot_mean(per, B=n_boot))
        return {"delta": d.mean(), "ci_lo": lo, "ci_hi": hi, "folds_first_better": int((d > 0).sum()),
                "folds_second_better": int((d < 0).sum()), "n": len(d), "ci_excludes_0": not (lo <= 0 <= hi)}
    pw = []
    for proto, rr in (("SW", reps), ("LW", R.seed_reps)):
        for m1, m2 in itertools.combinations(sorted(models, key=lambda m: -A[m].mean()), 2):
            pw.append({"protocol": proto, "pair": f"{NICE[m1]} - {NICE[m2]}", **paired(m1, m2, proto, rr)})
    PW = pd.DataFrame(pw, columns=["protocol", "pair", "delta", "ci_lo", "ci_hi", "folds_first_better",
                                   "folds_second_better", "n", "ci_excludes_0"])
    num = {"spread": spread, "table": t.round(4).to_dict("records"),
           "paired_SW": PW[PW.protocol == "SW"].round(4).to_dict("records"), "n_folds": len(reps)}
    return t, PW, A, reps, num


# ---------------------------------------------------------------- Table A5 (learning rate)
def learning_rate(R, models):
    rows = []
    for m in models:
        for lr in [R.cfg.lr] + R.cfg.lr_sweep:
            rr = [R.run(m, "SW", f"p0f{k}", lr=lr) for k in range(R.cfg.n_folds)]
            if any(r is None for r in rr): continue
            ids = [R.rid(m, "SW", f"p0f{k}", lr=lr) for k in range(R.cfg.n_folds)]
            rows.append({"model": NICE[m], "lr": lr, "val_acc": R.res[R.res.run_id.isin(ids)].val_acc.mean(),
                         "SW_accuracy_p0": np.mean([accuracy_score(r["y_test"], r["P_test"].argmax(1)) for r in rr])})
    t = pd.DataFrame(rows)
    if not len(t):
        return t, None
    t["rank_within_lr"] = t.groupby("lr")["SW_accuracy_p0"].rank(ascending=False, method="min")
    best = t.loc[t.groupby("model")["val_acc"].idxmax()]
    num = {"table": t.round(4).to_dict("records"), "best_lr_by_val": dict(zip(best.model, best.lr)),
           "top_model_per_lr": t[t.rank_within_lr == 1].groupby("lr").model.apply(list).to_dict()}
    return t, num


# ---------------------------------------------------------------- Table 7 (per-category reliability)
def per_class(R, m, reps):
    cfg, st = R.cfg, R.study; nc = st.num_classes
    yy = np.concatenate([R.run(m, "SW", x)["y_test"] for x in reps])
    pp = np.concatenate([R.run(m, "SW", x)["P_test"].argmax(1) for x in reps])
    pr, rc, f1, sup = precision_recall_fscore_support(yy, pp, labels=range(nc), zero_division=0)
    fold_f1 = np.array([precision_recall_fscore_support(R.run(m, "SW", x)["y_test"], R.run(m, "SW", x)["P_test"].argmax(1),
                                                        labels=range(nc), zero_division=0)[2] for x in reps])
    t = pd.DataFrame({"category": st.classes, "precision": pr, "recall": rc, "f1": f1,
                      "f1_sd_folds": fold_f1.std(0, ddof=1) if len(reps) > 1 else 0,
                      "f1_min_fold": fold_f1.min(0), "support_pooled": sup})
    t["status"] = np.where((t.precision >= cfg.reliable_min) & (t.recall >= cfg.reliable_min), "reliable",
                           np.where(t.f1 >= cfg.borderline_f1, "borderline", "not reliable"))
    return t.sort_values("f1", ascending=False), confusion_matrix(yy, pp, labels=range(nc))


# ---------------------------------------------------------------- Table A3 (calibration under session shift)
def calibration(R, models, reps, dep):
    rows, pool = [], {}
    bins = R.cfg.ece_bins
    for m in models:
        for x in reps:
            r, q = R.run(m, "SW", x), R.run(m, "SW", R.partner(x))
            if q is None: continue
            Tv, Tc = fit_T(r["P_val"], r["y_val"]), fit_T(q["P_test"], q["y_test"])   # neither uses the test fold
            P, yy = r["P_test"], r["y_test"]
            rows.append({"model": NICE[m], "protocol": "SW", "rep": x, "T_val": Tv, "T_cross": Tc,
                         "acc": accuracy_score(yy, P.argmax(1)), "mean_conf": P.max(1).mean(),
                         "ECE_uncal": ece(P, yy, bins)[0], "ECE_Tval": ece(temper(P, Tv), yy, bins)[0],
                         "ECE_Tcross": ece(temper(P, Tc), yy, bins)[0]})
            if m == dep:
                for k, PP in (("uncal", P), ("Tval", temper(P, Tv)), ("Tcross", temper(P, Tc))):
                    pool.setdefault(("SW", k), []).append((PP, yy))
        for x in R.avail(m, "LW", R.seed_reps):
            r = R.run(m, "LW", x); Tv = fit_T(r["P_val"], r["y_val"]); P, yy = r["P_test"], r["y_test"]
            rows.append({"model": NICE[m], "protocol": "LW", "rep": x, "T_val": Tv, "T_cross": np.nan,
                         "acc": accuracy_score(yy, P.argmax(1)), "mean_conf": P.max(1).mean(),
                         "ECE_uncal": ece(P, yy, bins)[0], "ECE_Tval": ece(temper(P, Tv), yy, bins)[0], "ECE_Tcross": np.nan})
            if m == dep:
                for k, PP in (("uncal", P), ("Tval", temper(P, Tv))):
                    pool.setdefault(("LW", k), []).append((PP, yy))
    cal = pd.DataFrame(rows)
    ta3 = cal.groupby(["model", "protocol"])[["acc", "mean_conf", "T_val", "T_cross", "ECE_uncal", "ECE_Tval",
                                              "ECE_Tcross"]].agg(["mean", "std"])
    num = cal[cal.model == NICE[dep]].groupby("protocol").mean(numeric_only=True).round(4).to_dict("index")
    return cal, ta3, pool, num


# ---------------------------------------------------------------- Table 8 (referral, cross-fitted)
def referral(R, models, reps, dep):
    rows = []
    for m in models:
        for x in reps:
            a, b = R.run(m, "SW", x), R.run(m, "SW", R.partner(x))
            if b is None: continue
            T = fit_T(b["P_test"], b["y_test"])                     # temperature and threshold from the partner fold
            Pa, Pb = temper(a["P_test"], T), temper(b["P_test"], T)
            for tg in R.cfg.targets:
                th = threshold_for(Pb, b["y_test"], tg)
                cov, acc = at_threshold(Pa, a["y_test"], th)
                cov_o, _ = at_threshold(Pa, a["y_test"], threshold_for(Pa, a["y_test"], tg))
                rows.append({"model": NICE[m], "rep": x, "target": tg, "T": T, "theta": th, "coverage": cov,
                             "accuracy": acc, "met": bool(acc >= tg), "coverage_oracle": cov_o})
    ref = pd.DataFrame(rows)
    g = ref.groupby(["model", "target"]).agg(theta=("theta", "mean"), coverage=("coverage", "mean"),
                                             coverage_sd=("coverage", "std"), accuracy=("accuracy", "mean"),
                                             accuracy_sd=("accuracy", "std"), folds_met=("met", "sum"),
                                             n=("met", "size"), coverage_oracle=("coverage_oracle", "mean")).reset_index()
    g["referred_pct"] = 100 * (1 - g.coverage)
    t8 = g[g.model == NICE[dep]].sort_values("target", ascending=False)
    curves = []
    for x in reps:
        r = R.run(dep, "SW", x); cov, acc, _ = risk_coverage(r["P_test"], r["y_test"]); curves.append((cov, acc))
    return ref, g, t8, curves


# ---------------------------------------------------------------- Tables A2, A4, 9
def tolerance(R, store, study):
    cfg = R.cfg; acc_tau = {}
    for t in [cfg.leaf_tol_s] + cfg.tol_retrain:
        proto = "LW" if t == cfg.leaf_tol_s else f"LWtau{t}"
        a = [R.acc(cfg.tol_model, proto, x) for x in R.avail(cfg.tol_model, proto, R.seed_reps)]
        if a: acc_tau[t] = (np.mean(a), np.std(a, ddof=1) if len(a) > 1 else 0.0, len(a))
    t = study.tol_table.copy()
    t["LW_accuracy"] = t.tau_s.map(lambda x: acc_tau.get(x, (np.nan,))[0])
    t["LW_sd"] = t.tau_s.map(lambda x: acc_tau.get(x, (np.nan, np.nan))[1])
    if cfg.leaf_tol_s in acc_tau and (cfg.tol_model, "LW") in store:   # Table A2 at 3 s equals Table 4
        assert abs(acc_tau[cfg.leaf_tol_s][0] - store[(cfg.tol_model, "LW")].mean()) < 1e-12
    return t, {int(k): round(v[0], 4) for k, v in acc_tau.items()}


def padding(R):
    cfg = R.cfg; rows = []
    for rz in ("squash", "pad"):
        rr = [R.run(cfg.pad_model, "SW", f"p0f{k}", resize=rz) for k in range(cfg.n_folds)]
        if all(r is not None for r in rr):
            rows.append({"model": NICE[cfg.pad_model], "resizing": rz, "protocol": "SW (partition 0)",
                         "accuracy": np.mean([accuracy_score(r["y_test"], r["P_test"].argmax(1)) for r in rr]),
                         "macro_f1": np.mean([f1_score(r["y_test"], r["P_test"].argmax(1), average="macro") for r in rr])})
    return pd.DataFrame(rows)


def deployment_cost(lat, A, models):
    t = lat.loc[models, ["params_m", "fp32_mb", "fp16_mb", "latency_ms_fp32_bs1"]].copy()
    t["SW_accuracy"] = [A[m].mean() for m in models]
    return t


# ---------------------------------------------------------------- provenance labels for the repository
def provenance_labels(study, cfg):
    full, df = study.full, study.df
    prov = full[["cls", "file", "ts", "Make", "Model", "w", "h", "leaf", "session"]].copy()
    prov.columns = ["category", "file", "capture_time", "device_make", "device_model", "width", "height",
                    "leaf_group", "session"]
    prov["in_three_protocol_study"] = prov.category.isin(study.classes)
    key = prov.category + "/" + prov.file
    for p, fo in enumerate(study.sw_partitions):
        prov[f"sw_partition{p}_test_fold"] = key.map(dict(zip(study.keys, fo))).astype("Int64")
    for s in cfg.seeds:
        for proto in ("RP", "LW"):
            tr, va, te = study.splits[(proto, f"s{s}")]
            lab = np.array([""] * len(df), dtype=object); lab[tr], lab[va], lab[te] = "train", "val", "test"
            prov[f"{proto}_seed{s}"] = key.map(dict(zip(study.keys, lab)))
    return prov
