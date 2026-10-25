"""Random (RP), leaf-wise (LW) and session-wise (SW) partitions, the contamination measure of Eq. (3),
Algorithm A1 (session-disjoint fold construction) and the verification asserted before any training."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .provenance import gap_groups


def kappa(tr, te, u):
    """Eq. (3): proportion of test images whose group u also occurs in training."""
    return float(np.isin(u[te], np.unique(u[tr])).mean())


class Partitioner:
    def __init__(self, df: pd.DataFrame, classes, cfg):
        self.df, self.classes, self.cfg = df, list(classes), cfg
        self.Y, self.LEAF, self.SESS = df.label.values, df.leaf.values, df.session.values
        self.IDX = np.arange(len(df))

    # ---- RP and LW
    def split_rp(self, seed):
        c = self.cfg
        tr, te = train_test_split(self.IDX, test_size=c.test_frac, stratify=self.Y, random_state=seed)
        tr, va = train_test_split(tr, test_size=c.val_frac, stratify=self.Y[tr], random_state=seed)
        return tr, va, te

    def split_lw(self, seed, leaf=None):
        c = self.cfg; leaf = self.LEAF if leaf is None else leaf
        tr, te = next(StratifiedGroupKFold(round(1 / c.test_frac), shuffle=True, random_state=seed)
                      .split(self.IDX, self.Y, groups=leaf))
        a, b = next(StratifiedGroupKFold(round(1 / c.val_frac), shuffle=True, random_state=seed)
                    .split(tr, self.Y[tr], groups=leaf[tr]))
        return tr[a], tr[b], te

    def leafwise_val(self, tr, seed):
        """Validation for SW is drawn leaf-wise inside the training sessions."""
        k = max(2, round(1 / self.cfg.val_frac))
        a, b = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed)
                    .split(tr, self.Y[tr], groups=self.LEAF[tr]))
        return tr[a], tr[b]

    # ---- Algorithm A1
    def session_partition(self, rng=None):
        """rng=None: deterministic order (sessions by size, categories alphabetical). Otherwise category
        order, session order and the initial fold order are randomised, giving another valid partition."""
        nf = self.cfg.n_folds
        assign, load = {}, np.zeros(nf)
        cls_order = list(self.classes) if rng is None else list(rng.permutation(self.classes))
        for c in cls_order:
            s = self.df[self.df.cls == c].groupby("session").size()
            s = s.sort_values(ascending=False) if rng is None else s.iloc[rng.permutation(len(s))]
            order = list(np.argsort(load, kind="stable"))
            if rng is not None and np.all(load == load[0]):
                order = list(rng.permutation(nf))
            for i, (sid, sz) in enumerate(s.items()):
                f = int(order[i]) if i < nf else int(np.argmin(load))
                assign[sid] = f; load[f] += sz
        return np.array([assign[s] for s in self.SESS])

    def _sig(self, fo):
        return frozenset(frozenset(np.unique(self.SESS[fo == k]).tolist()) for k in range(self.cfg.n_folds))

    def draw_sw_partitions(self, max_tries=20000, seed=2026):
        """Distinct session-disjoint partitions in which every category appears in every fold and each fold
        holds at least 35% of the images."""
        parts, sigs, rng, tries = [], set(), np.random.default_rng(seed), 0
        n_cls = len(self.classes); nf = self.cfg.n_folds
        cand = self.session_partition(None)
        while len(parts) < self.cfg.n_sw_partitions and tries < max_tries:
            tries += 1
            share = np.bincount(cand, minlength=nf) / len(cand)
            ok = (all(np.unique(self.Y[cand == k]).size == n_cls for k in range(nf))
                  and share.min() >= 0.35 and self._sig(cand) not in sigs)
            if ok:
                parts.append(cand); sigs.add(self._sig(cand))
            cand = self.session_partition(rng)
        if not parts:
            raise RuntimeError("no valid session-disjoint partition found")
        return parts, tries

    # ---- all splits used in the study
    def build_splits(self):
        c = self.cfg
        splits = {}
        for s in c.seeds:
            splits[("RP", f"s{s}")] = self.split_rp(s)
            splits[("LW", f"s{s}")] = self.split_lw(s)
        parts, tries = self.draw_sw_partitions()
        for p, fo in enumerate(parts):
            for k in range(c.n_folds):
                te = self.IDX[fo == k]; trf = self.IDX[fo != k]
                tr, va = self.leafwise_val(trf, 42 + p)
                splits[("SW", f"p{p}f{k}")] = (tr, va, te)
        return splits, parts, tries

    def verify(self, splits):
        """Partition table and the assertions of Section 3.4 (raise if any partition is contaminated)."""
        rows = []
        for (proto, rep), (tr, va, te) in splits.items():
            rows.append({"protocol": proto, "rep": rep, "train": len(tr), "val": len(va), "test": len(te),
                         "kappa_leaf": kappa(tr, te, self.LEAF), "kappa_session": kappa(tr, te, self.SESS),
                         "kappa_leaf_val": kappa(tr, va, self.LEAF),
                         "classes_in_test": int(np.unique(self.Y[te]).size)})
        parts = pd.DataFrame(rows)
        lw, sw = parts[parts.protocol == "LW"], parts[parts.protocol == "SW"]
        assert (lw.kappa_leaf == 0).all(), "LW partition shares leaf groups"
        assert (sw.kappa_session == 0).all() and (sw.kappa_leaf == 0).all(), "SW partition shares sessions"
        assert (sw.kappa_leaf_val == 0).all(), "SW validation not leaf-wise"
        assert (sw.classes_in_test == len(self.classes)).all(), "a category is missing from an SW test fold"
        t3 = parts.groupby("protocol").agg(
            train=("train", "mean"), train_min=("train", "min"), train_max=("train", "max"),
            val=("val", "mean"), test=("test", "mean"), test_min=("test", "min"), test_max=("test", "max"),
            kappa_leaf=("kappa_leaf", "mean"), kappa_session=("kappa_session", "mean"),
            n_splits=("rep", "count")).reindex(["RP", "LW", "SW"]).round(3)
        return parts, t3

    def add_control_splits(self, splits, parts):
        """Matched-size control (LW training portion reduced to the median SW training size) and the
        leaf-tolerance sensitivity splits (LW with leaf groups recomputed at another tau)."""
        c = self.cfg
        n_match = int(np.median(parts[parts.protocol == "SW"].train))
        for s in c.seeds:
            tr, va, te = splits[("LW", f"s{s}")]
            trm, _ = train_test_split(tr, train_size=n_match, stratify=self.Y[tr], random_state=s)
            splits[("LWm", f"s{s}")] = (trm, va, te)
        for t in c.tol_retrain:
            leaf_t = gap_groups(self.df, t)
            for s in c.seeds:
                splits[(f"LWtau{t}", f"s{s}")] = self.split_lw(s, leaf_t)
        return n_match
