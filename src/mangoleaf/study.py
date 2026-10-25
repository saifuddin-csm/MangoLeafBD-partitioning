"""Builds (once) and reloads everything that precedes training: corpus, provenance, working set and all
partitions. Saved to <out>/work/study.pkl so that training and analysis use identical splits."""
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import provenance as pv
from .partitions import Partitioner
from .utils import update_numbers


@dataclass
class Study:
    full: pd.DataFrame            # every image with metadata, leaf group and session
    df: pd.DataFrame              # working set of the three-protocol study
    classes: list
    splits: dict                  # (protocol, repetition) -> (train, val, test) index arrays into df
    sw_partitions: list           # fold assignment per SW partition
    partitions_table: pd.DataFrame
    t3: pd.DataFrame
    n_match: int
    tol_table: pd.DataFrame
    tau_star: int
    t1: pd.DataFrame
    audit: dict
    working_info: dict
    keys: np.ndarray = field(default=None)   # "category/file" identifier of every working-set image

    @property
    def Y(self): return self.df.label.values
    @property
    def LEAF(self): return self.df.leaf.values
    @property
    def SESS(self): return self.df.session.values
    @property
    def num_classes(self): return len(self.classes)


def prepare(cfg, verbose=True) -> Study:
    full, n_ts = pv.build_corpus(cfg)
    tol, tau_star = pv.tolerance_table(full, cfg)
    if not cfg.dry_run and tau_star != cfg.leaf_tol_s:
        raise AssertionError("Eq. (2) no longer selects the configured leaf tolerance")
    t1 = pv.composition_table(full, cfg)
    a1 = pv.audit(full, n_ts)
    df, classes, info = pv.working_set(full, cfg)

    P = Partitioner(df, classes, cfg)
    splits, parts, tries = P.build_splits()
    ptab, t3 = P.verify(splits)
    n_match = P.add_control_splits(splits, ptab)
    keys = np.array([f"{c}/{f}" for c, f in zip(df.cls, df.file)], dtype="U")

    st = Study(full, df, classes, splits, parts, ptab, t3, n_match, tol, tau_star, t1, a1, info, keys)

    # tables and numbers available before training
    t1.to_csv(cfg.tbl / "Table1_composition.csv")
    pd.Series(a1).to_csv(cfg.tbl / "TableA1_audit.csv", header=["value"])
    ptab.to_csv(cfg.tbl / "partitions_all.csv", index=False)
    t3.to_csv(cfg.tbl / "Table3_partitions.csv")
    num = cfg.tbl / "numbers.json"
    update_numbers(num, "audit", a1)
    update_numbers(num, "working_set", info)
    update_numbers(num, "partitions", t3.to_dict("index"))
    update_numbers(num, "matched_size_n", n_match)
    pickle.dump(st, open(cfg.work / "study.pkl", "wb"))
    if verbose:
        print(t1.to_string()); print(tol.to_string(index=False)); print("tau_l chosen by Eq. (2):", tau_star)
        print(f"{len(parts)} distinct session-disjoint partitions after {tries} draws")
        print("VERIFIED: LW kappa_l = 0; SW kappa_s = kappa_l = 0; SW validation leaf-disjoint; "
              "every category in every SW test fold")
        print(t3.to_string()); print("matched-size n:", n_match)
    return st


def load(cfg) -> Study:
    f = cfg.work / "study.pkl"
    if not f.exists():
        return prepare(cfg)
    return pickle.load(open(f, "rb"))
