"""Matplotlib performance graphs for the CineMatch model comparison.

make_figures(results) -> {file_stem: Figure}   (used by the Streamlit app)
save_figures(results, out_dir)                  (used by run_evaluation.py; writes PNG files)
"""
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from evaluation import POPULARITY, CONTENT, COLLAB, RANDOM, MODELS, ALL_MODELS, metric_directions, ACCURACY_METRICS

# categorical slots 1-3 of the validated palette (blue, orange, aqua) + a neutral grey for the baseline
COLORS = {POPULARITY: "#2a78d6", CONTENT: "#eb6834", COLLAB: "#1baf7a", RANDOM: "#8a8a85"}
SHORT = {POPULARITY: "Popularity", CONTENT: "Content (genres)", COLLAB: "Collaborative", RANDOM: "Random baseline"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7, "axes.axisbelow": True,
    "legend.frameon": False, "font.size": 9, "figure.dpi": 100, "savefig.dpi": 160,
})


def _fmt(v):
    return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"


def _legend(fig, models, y=0.0):
    handles = [Patch(color=COLORS[m], label=SHORT[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), bbox_to_anchor=(.5, y))


def _grouped(ax, summary, metrics, models, title, ylabel=None, hint=None):
    n = len(models); w = .8 / n; x = np.arange(len(metrics))
    s = summary.set_index("Model")
    for j, m in enumerate(models):
        vals = [s.loc[m, c] for c in metrics]
        bars = ax.bar(x + (j - (n - 1) / 2) * w, vals, w * .9, color=COLORS[m], label=SHORT[m])
        for b, v in zip(bars, vals):
            ax.annotate(_fmt(v), (b.get_x() + b.get_width() / 2, max(v, 0)), ha="center", va="bottom",
                        xytext=(0, 2), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title(title + (f"  ({hint})" if hint else ""))
    if ylabel: ax.set_ylabel(ylabel)
    ax.margins(y=.15); ax.grid(axis="x", visible=False)
    if min(summary[metrics].min()) < 0: ax.axhline(0, color=INK2, lw=.8)


def _fig(w, h):
    fig = Figure(figsize=(w, h), constrained_layout=True)
    return fig


def make_figures(res):
    S, k, cv, meta = res["summary"], res["k"], res["curves"], res["meta"]
    allm, mods = ALL_MODELS, MODELS
    figs = {}

    # 01 rating-error metrics
    fig = _fig(10, 4.2); axs = fig.subplots(1, 2)
    _grouped(axs[0], S, ["RMSE"], allm, "RMSE", hint="lower is better")
    _grouped(axs[1], S, ["MAE", "Median AE"], allm, "MAE & median absolute error", hint="lower is better")
    axs[0].set_xticks([]); axs[0].set_xticklabels([])
    _legend(fig, allm, -.07); figs["01_rating_error_RMSE_MAE"] = fig

    # 02 rating quality
    fig = _fig(10, 4.2); axs = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
    _grouped(axs[0], S, ["R2"], allm, "R² (variance explained)", hint="higher is better")
    axs[0].set_xticks([])
    _grouped(axs[1], S, ["Within ±0.5", "Within ±1.0"], allm, "Share of predictions close to the true rating", hint="higher is better")
    _legend(fig, allm, -.07); figs["02_rating_R2_and_accuracy_bands"] = fig

    # 03 error distribution
    fig = _fig(11, 4.4); axs = fig.subplots(1, 2)
    bins = np.linspace(-4.5, 4.5, 37)
    for m in allm:
        y, p = res["residuals"][m]
        axs[0].hist(p - y, bins=bins, histtype="step", lw=1.8, density=True, color=COLORS[m], label=SHORT[m])
    axs[0].axvline(0, color=INK2, lw=.8); axs[0].set_xlabel("Prediction error (predicted − actual rating)")
    axs[0].set_ylabel("Density"); axs[0].set_title("Distribution of prediction errors")
    data = [np.abs(res["residuals"][m][1] - res["residuals"][m][0]) for m in allm]
    bp = axs[1].boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=.55,
                        medianprops={"color": INK, "lw": 1.5})
    for patch, m in zip(bp["boxes"], allm):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(.85); patch.set_edgecolor(COLORS[m])
    axs[1].set_xticks(range(1, len(allm) + 1)); axs[1].set_xticklabels([SHORT[m].replace(" ", "\n", 1) for m in allm])
    axs[1].set_ylabel("Absolute error"); axs[1].set_title("Absolute error spread (box = middle 50%)")
    axs[1].grid(axis="x", visible=False)
    _legend(fig, allm, -.07); figs["03_error_distribution"] = fig

    # 04 calibration by true rating
    L = res["rating_levels"]
    fig = _fig(11, 4.4); axs = fig.subplots(1, 2)
    lv = np.sort(L["True rating"].unique())
    axs[0].plot([lv.min(), lv.max()], [lv.min(), lv.max()], color=INK2, ls="--", lw=1, label="Perfect prediction")
    for m in allm:
        d = L[L.Model == m].sort_values("True rating")
        axs[0].plot(d["True rating"], d["Mean prediction"], marker="o", ms=5, lw=2, color=COLORS[m], label=SHORT[m])
    axs[0].set_xlabel("Actual rating"); axs[0].set_ylabel("Mean predicted rating")
    axs[0].set_title("Calibration: predicted vs actual rating")
    axs[0].legend(loc="upper left", fontsize=7)
    w = .8 / len(allm)
    for j, m in enumerate(allm):
        d = L[L.Model == m].sort_values("True rating")
        axs[1].bar(np.arange(len(d)) + (j - (len(allm) - 1) / 2) * w, d["Mean error"], w * .9, color=COLORS[m])
    axs[1].set_xticks(np.arange(len(lv))); axs[1].set_xticklabels([f"{v:g}" for v in lv])
    axs[1].axhline(0, color=INK2, lw=.8); axs[1].set_xlabel("Actual rating"); axs[1].set_ylabel("Mean error (pred − actual)")
    axs[1].set_title("Bias by rating level (over-/under-prediction)"); axs[1].grid(axis="x", visible=False)
    figs["04_calibration_and_bias"] = fig

    # 05 confusion matrices
    fig = _fig(13, 3.9); axs = fig.subplots(1, len(allm))
    for ax, m in zip(axs, allm):
        cmx = res["confusion"][m]; pct = cmx / cmx.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cmx[i, j]:,}\n{pct[i, j]:.0%}", ha="center", va="center",
                        color="white" if pct[i, j] > .55 else INK, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Dislike", "Like"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Dislike", "Like"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(SHORT[m])
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.suptitle(f"Confusion matrices (like = rating ≥ {meta['like_threshold']:g}; predicted like = score ≥ {meta['pred_threshold']:g}; % of each actual class)",
                 x=.01, ha="left", fontsize=10, color=INK2)
    figs["05_confusion_matrices"] = fig

    # 06 like/dislike classification metrics
    fig = _fig(11, 4.4); ax = fig.subplots()
    _grouped(ax, S, ["Accuracy", "Precision (like)", "Recall (like)", "F1 (like)", "Specificity", "ROC-AUC", "PR-AUC"],
             allm, "Like / dislike classification metrics", hint="higher is better")
    ax.set_ylim(0, 1.12); _legend(fig, allm, -.06); figs["06_classification_metrics"] = fig

    # 07 ROC + PR curves
    fig = _fig(11, 4.6); axs = fig.subplots(1, 2)
    axs[0].plot([0, 1], [0, 1], color=INK2, ls="--", lw=1)
    for m in allm:
        fpr, tpr = res["roc"][m]
        axs[0].plot(fpr, tpr, lw=2, color=COLORS[m], label=f"{SHORT[m]} (AUC {S.set_index('Model').loc[m, 'ROC-AUC']:.3f})")
        rec, prec = res["pr"][m]
        axs[1].plot(rec, prec, lw=2, color=COLORS[m], label=f"{SHORT[m]} (AP {S.set_index('Model').loc[m, 'PR-AUC']:.3f})")
    axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].set_title("ROC curve")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision–recall curve"); axs[1].set_ylim(0, 1.02)
    for ax in axs: ax.legend(loc="lower right" if ax is axs[0] else "lower left", fontsize=7)
    figs["07_ROC_and_precision_recall_curves"] = fig

    # 08 top-K ranking metrics at K
    fig = _fig(11, 4.4); ax = fig.subplots()
    cols = [f"Precision@{k}", f"Recall@{k}", f"F1@{k}", f"NDCG@{k}", f"MAP@{k}", f"HitRate@{k}", f"MRR@{k}"]
    _grouped(ax, S, cols, allm, f"Top-{k} ranking metrics", hint="higher is better")
    _legend(fig, allm, -.06); figs["08_ranking_metrics_at_K"] = fig

    # 09 ranking curves vs K
    fig = _fig(14, 6.4); axs = fig.subplots(2, 4).ravel()
    for ax, met in zip(axs, ["Precision", "Recall", "F1", "NDCG", "MAP", "HitRate", "MRR"]):
        for m in allm:
            d = cv[cv.Model == m]
            ax.plot(d.K, d[met], lw=2, color=COLORS[m])
        ax.axvline(k, color=INK2, lw=.8, ls=":"); ax.set_title(f"{met}@K"); ax.set_xlabel("K (list length)")
        ax.set_xticks([1, 5, 10, 15, 20])
    for m in allm:
        d = cv[cv.Model == m]; axs[7].plot(d.Recall, d.Precision, lw=2, color=COLORS[m])
    axs[7].set_xlabel("Recall"); axs[7].set_ylabel("Precision"); axs[7].set_title("Precision–recall trade-off over K")
    _legend(fig, allm, -.04); figs["09_ranking_curves_vs_K"] = fig

    # 10 beyond accuracy
    fig = _fig(12, 4.2); axs = fig.subplots(1, 4)
    for ax, (c, t) in zip(axs, [("Catalogue coverage", "Catalogue coverage"), ("Novelty", "Novelty (bits)"),
                                ("Diversity", "List diversity"), ("Long-tail share", "Long-tail share")]):
        _grouped(ax, S, [c], allm, t); ax.set_xticks([])
    fig.suptitle("Beyond accuracy (higher = broader / more surprising; not a quality score on its own)",
                 x=.01, ha="left", fontsize=10, color=INK2)
    _legend(fig, allm, -.07); figs["10_beyond_accuracy"] = fig

    # 11 efficiency
    fig = _fig(9, 4.0); axs = fig.subplots(1, 2)
    _grouped(axs[0], S, ["Fit time (s)"], allm, "Fit time (s)", hint="lower is faster")
    _grouped(axs[1], S, ["Scoring time (s)"], allm, "Time to score every user × film (s)", hint="lower is faster")
    for ax in axs: ax.set_xticks([])
    _legend(fig, allm, -.07); figs["11_efficiency_time"] = fig

    # 12 segment RMSE
    seg = res["segments"]
    kinds = list(seg.Segment.unique())
    fig = _fig(12, 4.4); axs = fig.subplots(1, len(kinds))
    for ax, kind in zip(np.atleast_1d(axs), kinds):
        d = seg[seg.Segment == kind]; buckets = list(dict.fromkeys(d.Bucket))
        w = .8 / len(allm)
        for j, m in enumerate(allm):
            dm = d[d.Model == m].set_index("Bucket").reindex(buckets)
            ax.bar(np.arange(len(buckets)) + (j - (len(allm) - 1) / 2) * w, dm.RMSE, w * .9, color=COLORS[m])
        n = d[d.Model == allm[0]].set_index("Bucket").reindex(buckets).n
        ax.set_xticks(np.arange(len(buckets))); ax.set_xticklabels([f"{b}\n(n={int(v):,})" if v == v else b for b, v in zip(buckets, n)], fontsize=8)
        ax.set_ylabel("RMSE"); ax.set_title(f"RMSE by {kind.lower()}"); ax.grid(axis="x", visible=False)
    _legend(fig, allm, -.07); figs["12_RMSE_by_user_and_item_segment"] = fig

    # 13 radar (relative strengths, three project models)
    d = metric_directions(k); s = S[S.Model.isin(mods)].set_index("Model")
    rad = ACCURACY_METRICS(k) + ["Catalogue coverage", "Diversity"]
    norm = {}
    for c in rad:
        lo, hi = s[c].min(), s[c].max()
        v = (s[c] - lo) / (hi - lo) if hi > lo else s[c] * 0 + .5
        norm[c] = v if d[c] else 1 - v
    ang = np.linspace(0, 2 * np.pi, len(rad), endpoint=False).tolist(); ang += ang[:1]
    fig = Figure(figsize=(7.4, 7.2), constrained_layout=True); ax = fig.add_subplot(111, polar=True)
    for m in mods:
        vals = [float(norm[c][m]) for c in rad]; vals += vals[:1]
        ax.plot(ang, vals, lw=2, color=COLORS[m]); ax.fill(ang, vals, color=COLORS[m], alpha=.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([c.replace("Catalogue ", "Cat. ") for c in rad], fontsize=8)
    ax.set_yticks([.25, .5, .75, 1]); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    ax.set_title("Relative strengths (1 = best of the three, 0 = worst; error metrics inverted)", loc="left")
    _legend(fig, mods, -.05); figs["13_radar_relative_strengths"] = fig

    # 14 heatmap of all metrics
    allc = [c for c in d if c in s.columns]
    norm_all = np.array([[(((s.loc[m, c] - s[c].min()) / (s[c].max() - s[c].min())) if s[c].max() > s[c].min() else .5)
                          if d[c] else (1 - (s.loc[m, c] - s[c].min()) / (s[c].max() - s[c].min()) if s[c].max() > s[c].min() else .5)
                          for m in mods] for c in allc])
    fig = Figure(figsize=(7.5, .36 * len(allc) + 1.6), constrained_layout=True); ax = fig.add_subplot(111)
    ax.imshow(norm_all, cmap="Blues", vmin=0, vmax=1.15, aspect="auto"); ax.grid(False)
    for i, c in enumerate(allc):
        for j, m in enumerate(mods):
            ax.text(j, i, _fmt(s.loc[m, c]), ha="center", va="center", fontsize=8,
                    color="white" if norm_all[i, j] > .6 else INK)
    ax.set_xticks(range(len(mods))); ax.set_xticklabels([SHORT[m] for m in mods]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(allc))); ax.set_yticklabels([f"{c} {'↑' if d[c] else '↓'}" for c in allc])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("All metrics at a glance (darker = better among the three models; ↑ higher / ↓ lower is better)", pad=28)
    figs["14_all_metrics_heatmap"] = fig

    # 15 overall ranking
    rk = res["ranking"]
    fig = _fig(9, 3.8); axs = fig.subplots(1, 2)
    for ax, col, t, hint in [(axs[0], "Average rank", "Average rank over core accuracy metrics", "1 = best"),
                             (axs[1], "Metrics won", "Metrics won (out of 10)", "more is better")]:
        b = ax.barh([SHORT[m] for m in rk.Model][::-1], rk[col][::-1], color=[COLORS[m] for m in rk.Model][::-1])
        for bar, v in zip(b, rk[col][::-1]): ax.annotate(f"{v:.1f}" if col == "Average rank" else f"{int(v)}", (v, bar.get_y() + bar.get_height() / 2),
                                                         xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_title(f"{t}  ({hint})", fontsize=9); ax.grid(axis="y", visible=False); ax.margins(x=.15)
    figs["15_overall_ranking"] = fig

    # 16 error vs ranking
    fig = _fig(7.5, 5.2); ax = fig.subplots()
    for m in allm:
        r = S[S.Model == m].iloc[0]
        ax.scatter(r.RMSE, r[f"NDCG@{k}"], s=60 + 1800 * r[f"Precision@{k}"], color=COLORS[m], alpha=.9, edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(SHORT[m], (r.RMSE, r[f"NDCG@{k}"]), xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("RMSE (lower is better →  better to the left)"); ax.set_ylabel(f"NDCG@{k} (higher is better)")
    ax.set_title(f"Rating error vs ranking quality (bubble size = Precision@{k})"); ax.invert_xaxis()
    figs["16_error_vs_ranking_quality"] = fig
    return figs


def save_figures(res, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    figs = make_figures(res)
    for stem, fig in figs.items():
        fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    return sorted(p.name for p in out.glob("*.png"))
