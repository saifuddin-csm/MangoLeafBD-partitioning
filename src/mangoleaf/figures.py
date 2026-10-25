"""All figures of the article. Journal style: no grid, no in-figure titles, 600 dpi PNG and vector PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

from .config import NICE, SHORT

PCOL = {"RP": "#C0504D", "LW": "#E39B4B", "SW": "#2F5597"}
MCOL = {"vgg16": "#2F5597", "resnet50": "#C0504D",
        "vit_base_patch16_224": "#6E9B3A", "swin_tiny_patch4_window7_224": "#8064A2"}
GREY = "#7F7F7F"


def set_style(dpi=600):
    plt.style.use("default")
    plt.rcParams.update({
        "axes.grid": False, "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "savefig.dpi": dpi, "savefig.bbox": "tight", "figure.dpi": 110,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def savefig(fig, folder, name, dpi=600):
    for a in fig.get_axes(): a.grid(False)
    fig.savefig(folder / f"{name}.png", dpi=dpi); fig.savefig(folder / f"{name}.pdf")
    plt.close(fig); print("  figure:", name)


def _panel(ax, lab):
    ax.text(0.0, 1.04, lab, transform=ax.transAxes, fontweight="bold")


def fig2_corpus(full, t1, t3):
    """Figure 2: composition and acquisition structure of the corpus."""
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.9))
    cats = list(t1.index[:-1]); xs = np.arange(len(cats))
    ax = axes[0, 0]
    ax.bar(xs, t1.loc[cats, "images"].astype(float), width=0.65,
           color=[PCOL["SW"] if t1.loc[c, "in_three_protocol_study"] == "yes" else "#BFBFBF" for c in cats])
    top = float(t1.loc[cats, "images"].max())
    for x_, c in zip(xs, cats):
        ax.text(x_, top * 1.02, f"{int(t1.loc[c, 'sessions'])}", ha="center", va="bottom", fontsize=7)
    ax.text(-0.35, top * 1.12, "Sessions:", fontsize=7, color=GREY, va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[c] for c in cats], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Images"); ax.set_ylim(0, 1.24 * top); _panel(ax, "(a)")

    ax = axes[0, 1]; gsz = full.groupby("leaf").size()
    ax.hist(gsz.values, bins=np.arange(0.5, min(int(gsz.max()), 30) + 1.5), color=PCOL["SW"],
            edgecolor="white", lw=0.4)
    ax.set_yscale("log"); ax.set_xlabel("Images per leaf group"); ax.set_ylabel("Leaf groups (log scale)")
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
    _panel(ax, "(b)")

    ax = axes[1, 0]; pr = ["RP", "LW", "SW"]; xs3 = np.arange(3); w = 0.36
    kl = [100 * t3.loc[p, "kappa_leaf"] for p in pr]; ks = [100 * t3.loc[p, "kappa_session"] for p in pr]
    ax.bar(xs3 - w / 2, kl, w, color="#E39B4B", label=r"Leaf contamination $\kappa_\ell$")
    ax.bar(xs3 + w / 2, ks, w, color="#2F5597", label=r"Session contamination $\kappa_s$")
    for i in range(3):
        ax.text(i - w / 2, kl[i] + 2, f"{kl[i]:.1f}", ha="center", fontsize=6.5)
        ax.text(i + w / 2, ks[i] + 2, f"{ks[i]:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs3); ax.set_xticklabels(["Random", "Leaf-wise", "Session-wise"])
    ax.set_ylim(0, 135); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Test images contaminated (%)"); ax.legend(loc="upper center", ncol=2, fontsize=6.5)
    _panel(ax, "(c)")

    ax = axes[1, 1]; cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cats):
        sub = full[full.cls == c]
        for j, (_, s_) in enumerate(sub.groupby("session")):
            ax.scatter(s_.ts, np.full(len(s_), i), s=9, color=cmap(j % 10), linewidths=0)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([SHORT[c] for c in cats], fontsize=7)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=30, labelsize=6.5); ax.set_xlabel("Capture date (2021)")
    _panel(ax, "(d)")
    plt.tight_layout(h_pad=1.6, w_pad=1.6)
    return fig


def fig3_probes(t2, chance):
    """Figure 3: category predicted from acquisition metadata alone (values of Table 2)."""
    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    yy = np.arange(len(t2))[::-1]
    ax.barh(yy, t2.accuracy, color=PCOL["SW"], height=0.6)
    ax.axvline(chance, ls="--", lw=0.9, color=GREY)
    ax.text(chance + 0.015, -0.75, f"chance = 1/{round(1 / chance)} = {chance:.3f}", fontsize=6.5,
            color=GREY, va="center")
    for y_, v in zip(yy, t2.accuracy): ax.text(v + 0.01, y_, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(yy); ax.set_yticklabels(t2.feature_set)
    ax.set_ylim(-1.1, len(t2) - 0.5)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Cross-validated accuracy (no image content)")
    return fig


def fig4_protocols(store, order):
    """Figure 4: test accuracy of each model under the three protocols."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0)); w = 0.26; rng = np.random.default_rng(0)
    for i, proto in enumerate(("RP", "LW", "SW")):
        xs = np.arange(len(order)) + (i - 1) * w
        ax.bar(xs, [store[(m, proto)].mean() for m in order], w * 0.92, color=PCOL[proto], alpha=0.35,
               edgecolor=PCOL[proto], linewidth=0.8,
               label={"RP": "Random (RP)", "LW": "Leaf-wise (LW)", "SW": "Session-wise (SW)"}[proto])
        for x_, m in zip(xs, order):
            v = store[(m, proto)]
            ax.scatter(x_ + rng.uniform(-w * 0.25, w * 0.25, len(v)), v, s=7, color=PCOL[proto], zorder=3, linewidths=0)
            ax.text(x_, max(v) + 0.008, f"{v.mean():.3f}", ha="center", fontsize=6.5)
    lo = min(store[(m, "SW")].min() for m in order)
    ax.set_ylim(max(0, np.floor((lo - 0.05) * 20) / 20), 1.03)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels([NICE[m] for m in order])
    ax.set_ylabel("Test accuracy"); ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    return fig


def fig5_per_class(t8, reliable_min):
    fig, ax = plt.subplots(figsize=(6.6, 2.7)); xs = np.arange(len(t8)); w3 = 0.26
    ordc = list(t8.category); t = t8.set_index("category")
    for i, (met, col) in enumerate((("precision", "#2F5597"), ("recall", "#9DC3E6"), ("f1", "#E39B4B"))):
        ax.bar(xs + (i - 1) * w3, t.loc[ordc, met], w3 * 0.92, color=col,
               label={"precision": "Precision", "recall": "Recall", "f1": "F$_1$"}[met])
    ax.axhline(reliable_min, ls="--", lw=0.8, color=GREY)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[c] for c in ordc], rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    return fig


def fig6_confusion(cm, classes):
    n = len(classes); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.tick_params(which="minor", length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    for i in range(n):
        for k in range(n):
            ax.text(k, i, f"{cmn[i, k]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, k] > .5 else "#404040")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[c] for c in classes], rotation=40, ha="right"); ax.set_yticklabels([SHORT[c] for c in classes])
    ax.set_xlabel("Predicted category"); ax.set_ylabel("True category")
    return fig


def fig7_risk_coverage(curves_cov_acc, t9):
    """Figure 7: accuracy against coverage; markers are the cross-fitted operating points of Table 8."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0)); grid_c = np.linspace(0.02, 1, 99); curves = []
    for cov, acc in curves_cov_acc:
        ax.plot(cov, acc, lw=0.6, color="#BFBFBF"); curves.append(np.interp(grid_c, cov, acc))
    ax.plot(grid_c, np.mean(curves, 0), lw=1.8, color=PCOL["SW"],
            label=f"Mean of {len(curves)} session-wise folds")
    for tg, col in ((0.95, "#C0504D"), (0.90, "#6E9B3A")):
        rr = t9[np.isclose(t9.target, tg)].iloc[0]
        ax.errorbar(rr.coverage, rr.accuracy, xerr=rr.coverage_sd, yerr=rr.accuracy_sd, fmt="o", ms=4,
                    color=col, capsize=2, lw=0.9, zorder=4,
                    label=f"Target {tg:.2f}: coverage {rr.coverage:.3f}, accuracy {rr.accuracy:.3f}")
        ax.axhline(tg, ls=":", lw=0.7, color=col)
    ax.set_xlabel("Coverage (proportion of leaves diagnosed automatically)")
    ax.set_ylabel("Accuracy on diagnosed leaves"); ax.set_xlim(0, 1.01)
    ax.set_ylim(max(0.5, np.floor(min(np.min(c) for c in curves) * 20) / 20), 1.005)
    ax.legend(loc="lower left", fontsize=6.5)
    return fig


def figA1_reliability(pool, ece_fn):
    """Figure A1: reliability diagrams pooled over repetitions (ECE per repetition is in Table A3)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    styles = {"uncal": ("Uncalibrated", "#C0504D", "o"),
              "Tval": ("Temperature from in-session validation", "#E39B4B", "s"),
              "Tcross": ("Temperature from a held-out session fold", "#2F5597", "^")}
    for ax, proto, lab in zip(axes, ("LW", "SW"), ("(a) Leaf-wise test (sessions seen in training)",
                                                   "(b) Session-wise test (unseen sessions)")):
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color=GREY)
        for k in ("uncal", "Tval", "Tcross"):
            if (proto, k) not in pool: continue
            P = np.concatenate([a for a, _ in pool[(proto, k)]]); y = np.concatenate([b for _, b in pool[(proto, k)]])
            _, xs, ys, ns = ece_fn(P, y); keep = ns >= 5
            ax.plot(xs[keep], ys[keep], marker=styles[k][2], ms=3.5, lw=1.1, color=styles[k][1], label=styles[k][0])
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.set_xlabel("Mean predicted confidence")
        ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontsize=7.5, va="bottom")
        ax.legend(loc="lower right", fontsize=6.5)
    axes[0].set_ylabel("Observed accuracy")
    return fig


def figA2_region(ab):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bp = ax.boxplot([ab.full, ab.centre_only, ab.centre_removed], widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=1), flierprops=dict(ms=2, mec=GREY))
    for b in bp["boxes"]: b.set(facecolor="#DCE6F2", edgecolor=PCOL["SW"], lw=0.8)
    for i, v in enumerate(ab[["full", "centre_only", "centre_removed"]].mean().values, 1):
        ax.scatter(i, v, marker="D", s=14, color="#C0504D", zorder=4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Full image", "Centre only", "Centre removed"])
    ax.set_ylabel("Probability of the true category"); ax.set_ylim(-0.02, 1.02)
    return fig


def figA3_curves(H, models):
    fig, axes = plt.subplots(2, len(models), figsize=(1.7 * len(models) + 0.4, 3.4), sharex=True,
                             sharey="row", squeeze=False)
    for j, m in enumerate(models):
        for proto in ("RP", "LW", "SW"):
            h = H[(H.model == m) & (H.protocol == proto)].groupby("epoch")[["train_loss", "val_acc"]].mean()
            if not len(h): continue
            axes[0, j].plot(h.index, h.train_loss, lw=1.1, color=PCOL[proto], label=proto)
            axes[1, j].plot(h.index, h.val_acc, lw=1.1, color=PCOL[proto], label=proto)
        axes[0, j].text(0.5, 1.03, NICE[m], transform=axes[0, j].transAxes, ha="center", fontsize=7.5)
        axes[1, j].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training loss"); axes[1, 0].set_ylabel("Validation accuracy"); axes[0, -1].legend(fontsize=6.5)
    plt.tight_layout()
    return fig


def figA4_cost(cost):
    """Figure A4. cost: rows indexed by timm model name with params_m, latency_ms_fp32_bs1, SW_accuracy."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for m, r in cost.iterrows():
        ax.scatter(r.params_m, r.SW_accuracy, s=12 * r.latency_ms_fp32_bs1, color=MCOL[m], alpha=0.75, linewidths=0)
        ax.annotate(f"{NICE[m]}\n{r.latency_ms_fp32_bs1:.1f} ms", (r.params_m, r.SW_accuracy), xytext=(9, -1),
                    textcoords="offset points", fontsize=7, va="center")
    ax.set_xscale("log"); ax.set_xticks([20, 30, 50, 100, 150])
    ax.xaxis.set_major_formatter(mt.ScalarFormatter()); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xlim(17, 230)
    lo, hi = cost.SW_accuracy.min(), cost.SW_accuracy.max(); pad = max(0.005, (hi - lo) * 0.2)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Parameters (millions, log scale)"); ax.set_ylabel("Session-wise accuracy")
    return fig
