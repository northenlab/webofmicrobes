# revision_r1_pr_calibration.py
"""PR-AUC, learning-free baselines and calibration for Reviewer 1 (mSystems revision).

Comment 2 (PR-AUC): per-isolate PR-AUC (average precision) with its random-ranking
value (the isolate's prevalence of consumed pairs) for
  * the Fig. 1b hold-out (all models and dimensions; saved held-out scores),
  * the Fig. 3 hierarchical hold-out (BPR vs the naive baseline; the fig3_pr run, whose
    ROC-AUC columns must equal the Fig. 3 results),
  * the 12 held-out isolates of Supplementary Fig. 3 (SI Table 2 values).
Comment 5 (what BPR buys): learning-free baselines on the Fig. 1b hold-out splits
(compound popularity, and genus prevalence with popularity as fallback).
Comment 3 (calibration): BPR scores (32 dimensions) of the Fig. 1b hold-out pairs mapped
to P(consumed) by Platt scaling or isotonic regression, cross-fitted over isolates;
reliability over ten equal-width bins and the unbinned slope of the label on the
prediction, compared with the logistic function of the raw score.

Writes CSV/JSON summaries and Supplementary Figs. 6 (PR-AUC) and 7 (calibration).

Usage:
    python scripts/revision_r1_pr_calibration.py --lambda0-dir <manuscript_figures/msystems_revision/lambda0>
"""

import sys
import json
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_curve, average_precision_score

from core.types import AssertionType
from loaders.loaders import prepare_training_data, load_exometabolite
import scripts.revision_fig1b_benchmark as fig1b
from scripts.revision_r1_metrics import (per_isolate, crossfit_probabilities, naive_probabilities,
                                         calibration_stats, reliability, baseline_scores)

logger = logging.getLogger(__name__)

BPR = "Ordinal-BPR [lambda_ord=0]"  # "BPR" in the revised manuscript
MODELS = [BPR, "WR-MF", "SVD", "LMF"]
STYLE = {m: fig1b.MODEL_STYLES[m] for m in MODELS}
DIM = 32
LEVELS = ["isolate", "genus", "family", "order"]
TEAL, ORANGE = "#237c7c", "#d98a34"  # Fig. 3 colours of BPR and the naive baseline
GREY, DARK = "#9a9a9a", "#4d4d4d"
BASELINE_STYLE = {"popularity": {"label": "popularity (no learning)", "color": DARK, "linestyle": "-."},
                  "genus": {"label": "genus prevalence (no learning)", "color": DARK, "linestyle": ":"}}


# ----------------------------------------------------------------------------- Fig. 1b hold-out
def fig1b_per_isolate(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, dim, repeat), g in scores.groupby(["model", "dim", "repeat"], observed=True):
        iso = per_isolate((g.label == "consumed").to_numpy(int), g.score.to_numpy(float), g.microbe.to_numpy())
        rows.append(iso.assign(model=model, dim=dim, repeat=repeat))
    return pd.concat(rows, ignore_index=True)


def check_against_results(per_iso: pd.DataFrame, results_csv: Path):
    """Mean per-isolate ROC-AUC must equal the Protocol B rows of the Fig. 1b results."""
    res = pd.read_csv(results_csv)
    res = res[res.protocol == "B_holdout"].set_index(["model", "dim", "repeat"]).roc_auc
    mine = per_iso.groupby(["model", "dim", "repeat"]).roc_auc.mean()
    diff = (mine - res.loc[mine.index]).abs().max()
    if diff > 5e-7 + 1e-9:
        raise ValueError(f"per-isolate ROC-AUC differs from {results_csv.name} (max |diff| {diff:.2e})")
    logger.info(f"Per-isolate ROC-AUC reproduces {results_csv.name} (max |diff| {diff:.1e})")


def pooled_pr(results_csv: Path) -> dict:
    """PR-AUC over all withheld pairs pooled across isolates (also rewards between-isolate score scale)."""
    res = pd.read_csv(results_csv)
    res = res[(res.protocol == "B_holdout") & (res.dim == DIM) & res.model.isin(MODELS)]
    g = res.groupby("model").pr_auc_pooled
    return {m: {"mean": float(g.mean()[m]), "sd": float(g.std()[m])} for m in MODELS}


def summarise_by_repeat(per_iso: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rep = per_iso.groupby(keys + ["repeat"])[["roc_auc", "pr_auc", "prevalence"]].mean()
    g = rep.groupby(keys)
    return g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()


def fig1b_baselines(td, labels, scores: pd.DataFrame, repeats: list[int]) -> pd.DataFrame:
    genus_of = {i: td.microbe_taxonomy[m].genus if isinstance(td.microbe_taxonomy[m].genus, str) else None
                for m, i in td.microbe_to_idx.items()}
    rows = []
    for repeat in repeats:
        hb_pos, hb_kn, withheld = fig1b.holdout_split(td, labels, seed=repeat)
        saved = scores[(scores.model == BPR) & (scores.dim == DIM) & (scores.repeat == repeat)]
        mic = saved.microbe.astype(str).map(td.microbe_to_idx).to_numpy()
        met = saved.compound.astype(str).map(td.metabolite_to_idx).to_numpy()
        if not np.array_equal(np.c_[mic, met], withheld):
            raise ValueError(f"repeat {repeat}: held-out pairs differ from the saved Fig. 1b scores")
        y = (labels[withheld[:, 0], withheld[:, 1]] == fig1b.LABELS.index("consumed")).astype(int)
        for name, bs in baseline_scores(hb_pos, hb_kn, td.n_microbes, td.n_metabolites, genus_of).items():
            iso = per_isolate(y, bs[withheld[:, 0], withheld[:, 1]], withheld[:, 0])
            rows.append(iso.assign(baseline=name, repeat=repeat))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- calibration
def calibration(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    bpr = scores[(scores.model == BPR) & (scores.dim == DIM)]
    stats, pooled = [], {"raw": [], "platt": [], "isotonic": []}
    ys = []
    for repeat, g in bpr.groupby("repeat"):
        y = (g.label == "consumed").to_numpy(int)
        s = g.score.to_numpy(float)
        iso_ids = g.microbe.astype(str).to_numpy()
        probs = {"raw": naive_probabilities(s),
                 "platt": crossfit_probabilities(y, s, iso_ids, seed=repeat, method="platt"),
                 "isotonic": crossfit_probabilities(y, s, iso_ids, seed=repeat, method="isotonic")}
        for method, p in probs.items():
            stats.append({"repeat": repeat, "method": method, **calibration_stats(y, p)})
            pooled[method].append(p)
        ys.append(y)
    y_all = np.concatenate(ys)
    bins = pd.concat([reliability(y_all, np.concatenate(p)).assign(method=m) for m, p in pooled.items()],
                     ignore_index=True)
    probs_all = {m: np.concatenate(p) for m, p in pooled.items()}
    return pd.DataFrame(stats), bins, {"y": y_all, **probs_all}


# ----------------------------------------------------------------------------- Fig. 3 and Fig. S3
def fig3_pr(fig3_pr_csv: Path, fig3_csv: Path) -> tuple[pd.DataFrame, dict]:
    new, ref = pd.read_csv(fig3_pr_csv), pd.read_csv(fig3_csv)
    keys = ["level", "group", "isolate"]
    merged = ref.merge(new, on=keys, suffixes=("", "_pr"), validate="one_to_one")
    if len(merged) != len(ref) or len(new) != len(ref):
        raise ValueError("Fig. 3 PR run does not contain the same hold-outs as Fig. 3")
    for col in ["auc_bpr_all", "auc_bpr_scoreable", "auc_baseline_scoreable", "coverage"]:
        a, b = merged[col], merged[f"{col}_pr"]
        if not a.isna().equals(b.isna()) or (a - b).abs().max() > 1e-12:
            raise ValueError(f"Fig. 3 PR run differs from Fig. 3 in {col}")
    logger.info("Fig. 3 PR run reproduces every Fig. 3 ROC-AUC")
    out = {}
    for level in LEVELS:
        sub = new[new.level == level]
        paired = sub.dropna(subset=["ap_bpr_scoreable", "ap_baseline_scoreable"])
        out[level] = {
            "n_all": int(len(sub)), "n_paired": int(len(paired)),
            "median_ap_bpr_all": float(sub.ap_bpr_all.median()),
            "median_prevalence_all": float(sub.prevalence.median()),
            "median_ap_bpr_scoreable": float(paired.ap_bpr_scoreable.median()),
            "median_ap_baseline_scoreable": float(paired.ap_baseline_scoreable.median()),
            "median_prevalence_scoreable": float(paired.prevalence_scoreable.median()),
            "median_delta": float((paired.ap_bpr_scoreable - paired.ap_baseline_scoreable).median()),
            "wilcoxon_p": float(wilcoxon(paired.ap_bpr_scoreable, paired.ap_baseline_scoreable).pvalue),
            "frac_bpr_all_above_prevalence": float((sub.ap_bpr_all > sub.prevalence).mean()),
        }
    return new, out


def figS3_pr(table_csv: Path) -> tuple[pd.DataFrame, dict]:
    tab = pd.read_csv(table_csv)
    t = tab[["isolate_id", "genus", "prevalence", "clean_holdout_pr_auc_mean", "clean_holdout_roc_auc_mean"]].rename(
        columns={"clean_holdout_pr_auc_mean": "pr_auc", "clean_holdout_roc_auc_mean": "roc_auc"})
    return t, {"mean_pr_auc": float(t.pr_auc.mean()), "mean_prevalence": float(t.prevalence.mean()),
               "min_pr_auc_minus_prevalence": float((t.pr_auc - t.prevalence).min()),
               "n_pr_auc_above_prevalence": int((t.pr_auc > t.prevalence).sum()), "n": int(len(t))}


# ----------------------------------------------------------------------------- figures
RECALL_GRID = np.linspace(0, 1, 101)


def mean_pr_curve(g: pd.DataFrame) -> np.ndarray:
    """Precision at each recall level (interpolated: best precision at that recall or higher),
    averaged over isolates; matches the per-isolate PR-AUC of panel a."""
    curves = []
    for _, h in g.groupby("microbe", observed=True):
        y, sc = (h.label == "consumed").to_numpy(int), h.score.to_numpy(float)
        if 0 < y.sum() < len(y):
            precision, recall, _ = precision_recall_curve(y, sc)
            best = np.maximum.accumulate(precision)  # sklearn orders by decreasing recall: max over recall >= r
            curves.append(np.interp(RECALL_GRID, recall[::-1], best[::-1]))
    return np.mean(curves, axis=0)


def _style(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_figS6(by_dim, base_summary, scores, fig3_rows, s3, out_stem: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    # a: Fig. 1b hold-out PR-AUC by dimension
    ax = axes[0, 0]
    for m in MODELS:
        d = by_dim[by_dim.model == m].sort_values("dim")
        st = STYLE[m]
        ax.errorbar(d.dim, d.pr_auc_mean, yerr=d.pr_auc_sd, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], linewidth=2, markersize=9, capsize=3, label=st["label"])
    for name, st in BASELINE_STYLE.items():
        v = base_summary.set_index("baseline").loc[name, "pr_auc_mean"]
        ax.axhline(v, color=st["color"], linestyle=st["linestyle"], linewidth=1.5, label=st["label"])
    prev = by_dim.prevalence_mean.mean()
    ax.axhline(prev, color=GREY, linestyle="--", linewidth=1.5, label="random ranking (prevalence)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(by_dim.dim.unique()))
    ax.set_xticklabels(sorted(by_dim.dim.unique()))
    ax.set_xlabel("Embedding dimension", fontsize=13)
    ax.set_ylabel("Hold-out PR-AUC (mean per isolate)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left", ncol=2)
    ax.set_title("Fig. 1b hold-out", fontsize=13)
    _style(ax, "a")
    # b: precision-recall curves at 32 dimensions, averaged over isolates (repeat 0)
    ax = axes[0, 1]
    rep0 = scores[(scores.dim == DIM) & (scores.repeat == 0)]
    for m in MODELS:
        st = STYLE[m]
        pr = by_dim.set_index(["model", "dim"]).loc[(m, DIM), "pr_auc_mean"]
        ax.plot(RECALL_GRID, mean_pr_curve(rep0[rep0.model == m]), color=st["color"], linestyle=st["linestyle"],
                linewidth=2, label=f"{st['label']} (PR-AUC = {pr:.2f})")
    prev = by_dim.prevalence_mean.mean()
    ax.axhline(prev, color=GREY, linestyle="--", linewidth=1.5, label=f"random ranking ({prev:.2f})")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.set_title(f"Fig. 1b hold-out, {DIM} dimensions, averaged over isolates (repeat 1)", fontsize=13)
    _style(ax, "b")
    # c: Fig. 3 hierarchical hold-out, PR-AUC on scoreable compounds
    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    for k, level in enumerate(LEVELS):
        sub = fig3_rows[fig3_rows.level == level].dropna(subset=["ap_bpr_scoreable", "ap_baseline_scoreable"])
        for offset, col, color in [(-0.2, "ap_bpr_scoreable", TEAL), (0.2, "ap_baseline_scoreable", ORANGE)]:
            vals = sub[col].to_numpy()
            ax.boxplot(vals, positions=[k + offset], widths=0.3, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color},
                       medianprops={"color": "black", "linewidth": 2}, whiskerprops={"color": color},
                       capprops={"color": color})
            ax.scatter(k + offset + rng.uniform(-0.1, 0.1, len(vals)), vals, s=5, color=color, alpha=0.5,
                       linewidths=0)
        ax.plot([k - 0.38, k + 0.38], [sub.prevalence_scoreable.median()] * 2, color=DARK, linestyle="--",
                linewidth=1.5)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"{lv.capitalize()}\n(n = {(fig3_rows.level == lv).sum()})" for lv in LEVELS], fontsize=11)
    ax.set_xlabel("Hold-out level", fontsize=13)
    ax.set_ylabel("PR-AUC on scoreable metabolites", fontsize=13)
    ax.set_ylim(0, 1.02)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=TEAL, alpha=0.5), plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, alpha=0.5),
               plt.Line2D([0], [0], color=DARK, linestyle="--")]
    ax.legend(handles, ["BPR", "naive baseline", "random ranking (median prevalence)"],
              frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_title("Fig. 3 hierarchical hold-out", fontsize=13)
    _style(ax, "c")
    # d: 12 held-out isolates
    ax = axes[1, 1]
    t = s3.sort_values("pr_auc").reset_index(drop=True)
    yy = np.arange(len(t))
    ax.hlines(yy, t.prevalence, t.pr_auc, color="#cccccc", linewidth=2)
    ax.scatter(t.prevalence, yy, s=60, facecolors="white", edgecolors=DARK, linewidths=1.5, zorder=3,
               label="random ranking (prevalence)")
    ax.scatter(t.pr_auc, yy, s=110, color=STYLE[BPR]["color"], marker="*", zorder=4, label="BPR PR-AUC")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{g} {i}" for g, i in zip(t.genus, t.isolate_id)], fontsize=10, style="italic")
    ax.set_xlim(0.4, 1.01)
    ax.set_xlabel("PR-AUC (mean over 20 models)", fontsize=13)
    ax.legend(frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("12 held-out isolates (Supplementary Fig. 3)", fontsize=13)
    _style(ax, "d")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figS7(bins: pd.DataFrame, stats: pd.DataFrame, probs: dict, out_stem: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.5, label="perfect calibration")
    summary = stats.groupby("method")[["ols_slope", "ols_intercept", "brier"]].mean()
    for method, color, marker, label in [("raw", DARK, "s", "logistic function of the raw BPR score"),
                                         ("platt", STYLE[BPR]["color"], "*", "Platt-scaled BPR score (cross-fitted)")]:
        b = bins[bins.method == method]
        s = summary.loc[method]
        ax.errorbar(b.mean_predicted, b.observed, yerr=[b.observed - b.ci_low, b.ci_high - b.observed],
                    color=color, marker=marker, markersize=10 if marker == "*" else 7, linewidth=1.8, capsize=3,
                    label=f"{label}\n(slope {s.ols_slope:.2f}, intercept {s.ols_intercept:.2f}, Brier {s.brier:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of consumption (bin mean)", fontsize=13)
    ax.set_ylabel("Observed fraction consumed", fontsize=13)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    _style(ax, "a")
    ax = axes[1]
    y, p = probs["y"], probs["platt"]
    edges = np.linspace(0, 1, 21)
    ax.hist(p[y == 0], bins=edges, histtype="stepfilled", color=fig1b.MODEL_STYLES["WR-MF"]["color"], alpha=0.25)
    ax.hist(p[y == 0], bins=edges, histtype="step", color=fig1b.MODEL_STYLES["WR-MF"]["color"], linewidth=1.8,
            label=f"held-out neutral pairs (n = {(y == 0).sum():,})")
    ax.hist(p[y == 1], bins=edges, histtype="stepfilled", color=STYLE[BPR]["color"], alpha=0.25)
    ax.hist(p[y == 1], bins=edges, histtype="step", color=STYLE[BPR]["color"], linewidth=1.8,
            label=f"held-out consumed pairs (n = {(y == 1).sum():,})")
    ax.set_xlabel("Platt-scaled probability of consumption", fontsize=13)
    ax.set_ylabel("Number of held-out pairs", fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    _style(ax, "b")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(lambda0_dir: Path):
    out = lambda0_dir / "r1" / "pr_calibration"
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(lambda0_dir / "fig1b" / "fig1b_heldout_scores.parquet")
    scores = scores[scores.model.isin(MODELS)].copy()
    scores["model"] = scores.model.astype(str)

    per_iso = fig1b_per_isolate(scores)
    check_against_results(per_iso, lambda0_dir / "fig1b" / "fig1b_benchmark_results.csv")
    by_dim = summarise_by_repeat(per_iso, ["model", "dim"])
    by_dim.to_csv(out / "fig1b_pr_by_dim.csv", index=False, float_format="%.6f")

    df = load_exometabolite()
    td = prepare_training_data(df, preference_type=AssertionType.CONSUMED, include_secreted=False)
    labels = fig1b.build_label_matrix(df, td, rule="methods")
    base = fig1b_baselines(td, labels, scores, sorted(scores.repeat.unique()))
    base_summary = summarise_by_repeat(base, ["baseline"])
    base_summary.to_csv(out / "fig1b_baselines.csv", index=False, float_format="%.6f")

    stats, bins, probs = calibration(scores)
    stats.to_csv(out / "calibration_by_repeat.csv", index=False, float_format="%.6f")
    bins.to_csv(out / "calibration_bins.csv", index=False, float_format="%.6f")

    fig3_rows, fig3_summary = fig3_pr(lambda0_dir / "r1" / "fig3_pr" / "fig3_holdout_results.csv",
                                      lambda0_dir / "fig3" / "fig3_holdout_results.csv")
    s3, s3_summary = figS3_pr(lambda0_dir / "figS3" / "table_S_validation_12_isolates.csv")
    s3.to_csv(out / "figS3_pr.csv", index=False, float_format="%.6f")

    cal = stats.groupby("method").agg(["mean", "std"])
    cal.columns = [f"{a}_{b}" for a, b in cal.columns]
    summary = {
        "fig1b_by_dim": by_dim.to_dict(orient="records"),
        "fig1b_baselines": base_summary.to_dict(orient="records"),
        "fig1b_pooled_pr_auc_dim32": pooled_pr(lambda0_dir / "fig1b" / "fig1b_benchmark_results.csv"),
        "calibration": cal.drop(columns=["repeat_mean", "repeat_std"]).reset_index().to_dict(orient="records"),
        "fig3": fig3_summary,
        "figS3": s3_summary,
    }
    (out / "r1_pr_calibration_summary.json").write_text(json.dumps(summary, indent=2))
    plot_figS6(by_dim, base_summary, scores, fig3_rows, s3, out / "figS6_pr_auc")
    plot_figS7(bins, stats, probs, out / "figS7_calibration")
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="PR-AUC, baselines and calibration for Reviewer 1")
    parser.add_argument("--lambda0-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.lambda0_dir)
