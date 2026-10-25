"""Cluster bootstrap (Section 3.8), expected calibration error (Eq. 4), temperature scaling (Eq. 5) and
coverage / selective accuracy of the referral threshold (Eqs. 6-8)."""
import numpy as np


def grp_cn(correct, groups):
    u, inv = np.unique(groups, return_inverse=True)
    return (np.bincount(inv, weights=correct.astype(float), minlength=len(u)),
            np.bincount(inv, minlength=len(u)).astype(float))


def boot_mean(per_fold, B=2000, seed=0):
    """per_fold: list of (values, leaf groups). Leaf groups are resampled within each fold and the
    resampled means are averaged over folds -> bootstrap distribution of the mean over folds."""
    rng = np.random.default_rng(seed); acc = np.zeros(B)
    for v, g in per_fold:
        c, n = grp_cn(v, g); ix = rng.integers(0, len(c), (B, len(c)))
        acc += c[ix].sum(1) / n[ix].sum(1)
    return acc / len(per_fold)


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def ece(P, y, bins=15):
    """Eq. (4). Returns the ECE and the per-bin mean confidence, accuracy and count."""
    conf, pred = P.max(1), P.argmax(1); hit = (pred == y).astype(float)
    ed = np.linspace(0, 1, bins + 1); e, xs, ys, ns = 0.0, [], [], []
    for i in range(bins):
        k = (conf > ed[i]) & (conf <= ed[i + 1])
        if k.sum() == 0: continue
        e += k.mean() * abs(hit[k].mean() - conf[k].mean())
        xs.append(conf[k].mean()); ys.append(hit[k].mean()); ns.append(int(k.sum()))
    return float(e), np.array(xs), np.array(ys), np.array(ns)


def temper(P, T):
    """Eq. (5), using the log of the (test-time averaged) probability as the logit."""
    lg = np.log(np.clip(P, 1e-12, 1)) / T; lg -= lg.max(1, keepdims=True)
    e = np.exp(lg); return e / e.sum(1, keepdims=True)


def fit_T(P, y):
    """Temperature minimising the negative log likelihood over the range 0.25 to 5."""
    grid = np.arange(0.25, 5.001, 0.01)
    nll = [-np.log(np.clip(temper(P, T)[np.arange(len(y)), y], 1e-12, 1)).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


def risk_coverage(P, y):
    conf = P.max(1); o = np.argsort(-conf, kind="stable"); hit = (P.argmax(1) == y)[o]
    n = np.arange(1, len(y) + 1)
    return n / len(y), np.cumsum(hit) / n, conf[o]


def threshold_for(P, y, target):
    """Eq. (8): the threshold with the largest coverage whose selective accuracy still meets the target."""
    cov, acc, thr = risk_coverage(P, y); ok = np.where(acc >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 1.0 + 1e-9


def at_threshold(P, y, th):
    """Eqs. (6) and (7): coverage and selective accuracy at threshold th."""
    sel = P.max(1) >= th
    return float(sel.mean()), (float((P.argmax(1)[sel] == y[sel]).mean()) if sel.any() else float("nan"))
