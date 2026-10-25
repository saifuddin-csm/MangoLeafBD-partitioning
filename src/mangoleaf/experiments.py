"""Registry of all 103 training runs and their resumable execution.

Each run is identified by "model|protocol|repetition[|lrX][|pad]". Finished runs are recorded in
results_rev.csv and skipped when the script is started again, so a long job can be split across sessions.
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from .config import FAMILY
from .data import ImageStore
from .models import ModelZoo
from .utils import append_row, device


def run_id(model, proto, rep, lr, resize, base_lr):
    return f"{model}|{proto}|{rep}" + (f"|lr{lr:g}" if lr != base_lr else "") + \
           (f"|{resize}" if resize != "squash" else "")


def registry(cfg, n_partitions):
    """All planned runs: SW (10 folds), RP and LW (3 seeds), matched-size control, leaf-tolerance
    sensitivity, learning-rate comparison and aspect-ratio comparison."""
    specs = []

    def spec(group, model, proto, rep, seed, lr=cfg.lr, resize="squash", tau=cfg.leaf_tol_s, ckpt=False):
        specs.append(dict(run_id=run_id(model, proto, rep, lr, resize, cfg.lr), group=group, model=model,
                          protocol=proto, rep=rep, seed=seed, lr=lr, resize=resize, tau=tau, ckpt=ckpt))
    for m in cfg.models:
        for p in range(n_partitions):
            for k in range(cfg.n_folds):
                spec("sw", m, "SW", f"p{p}f{k}", 42 + p, ckpt=(p == 0))
    for m in cfg.models:
        for s in cfg.seeds:
            spec("main", m, "RP", f"s{s}", s); spec("main", m, "LW", f"s{s}", s)
    for m in cfg.models:
        for s in cfg.seeds:
            spec("control", m, "LWm", f"s{s}", s)
    if cfg.tol_model in cfg.models:
        for t in cfg.tol_retrain:
            for s in cfg.seeds:
                spec("tolerance", cfg.tol_model, f"LWtau{t}", f"s{s}", s, tau=t)
    for m in cfg.models:
        for lr in cfg.lr_sweep:
            for k in range(cfg.n_folds):
                spec("lr", m, "SW", f"p0f{k}", 42, lr=lr)
    if cfg.pad_model in cfg.models:
        for k in range(cfg.n_folds):
            spec("pad", cfg.pad_model, "SW", f"p0f{k}", 42, resize="pad")
    assert len({s["run_id"] for s in specs}) == len(specs)
    return specs


def done_ids(cfg):
    return set(pd.read_csv(cfg.results_csv).run_id) if cfg.results_csv.exists() else set()


def run_all(cfg, study, groups, time_budget_min=None):
    dev = device()
    specs = registry(cfg, len(study.sw_partitions))
    print(pd.Series([s["group"] for s in specs]).value_counts().to_string(), "\ntotal", len(specs))
    zoo = ModelZoo(cfg, study.num_classes, dev)
    zoo.fetch(cfg.models)
    todo = [s for s in specs if s["group"] in groups and s["run_id"] not in done_ids(cfg)]
    print("to run:", len(todo))
    if not todo:
        return specs
    store = ImageStore(study.full.path if cfg.cache_images else (), cache=cfg.cache_images)
    Y, LEAF, SESS, KEYS = study.Y, study.LEAF, study.SESS, study.keys
    t_start = time.time()
    for i, s in enumerate(todo, 1):
        if time_budget_min and (time.time() - t_start) / 60 > time_budget_min:
            print("time budget reached; start the script again to continue"); break
        split = study.splits[(s["protocol"], s["rep"])]
        tr, va, te = split
        print(f"\n[{i}/{len(todo)}] {s['run_id']}  (train {len(tr)}, test {len(te)})", flush=True)
        t0 = time.time()
        m, o = zoo.train_one(s["model"], study, split, store, s["seed"], s["lr"], s["resize"])
        if s["ckpt"]:
            zoo.save_ckpt(m, s["run_id"])
        del m
        if dev == "cuda": torch.cuda.empty_cache()
        np.savez_compressed(cfg.pred / (s["run_id"].replace("|", "__") + ".npz"),
                            P_test=o["P_test"].astype(np.float32), P_test_plain=o["P_test_plain"].astype(np.float32),
                            P_val=o["P_val"].astype(np.float32), y_test=Y[te], y_val=Y[va],
                            key_test=np.array(KEYS[te], dtype="U"), key_val=np.array(KEYS[va], dtype="U"),
                            leaf_test=LEAF[te], sess_test=SESS[te], n_train=len(tr))
        yh = o["P_test"].argmax(1)
        row = {k: s[k] for k in ("run_id", "group", "model", "protocol", "rep", "seed", "lr", "resize", "tau")}
        row.update({"family": FAMILY[s["model"]], "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                    "minutes": round((time.time() - t0) / 60, 2), "best_epoch": o["best_epoch"],
                    "val_acc": round(o["val_acc"], 4), "accuracy": accuracy_score(Y[te], yh),
                    "macro_f1": f1_score(Y[te], yh, average="macro"), "mcc": matthews_corrcoef(Y[te], yh),
                    "accuracy_no_tta": accuracy_score(Y[te], o["P_test_plain"].argmax(1))})
        append_row(cfg.results_csv, row, "run_id")
        h = pd.DataFrame(o["hist"]); h.insert(0, "run_id", s["run_id"])
        if cfg.history_csv.exists():
            h = pd.concat([pd.read_csv(cfg.history_csv), h], ignore_index=True)
        h.drop_duplicates(["run_id", "epoch"], keep="last").to_csv(cfg.history_csv, index=False)
        print(f"  -> acc {row['accuracy']:.4f}  F1 {row['macro_f1']:.4f}  [{row['minutes']} min]", flush=True)
    return specs
