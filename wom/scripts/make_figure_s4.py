# make_figure_s4.py
"""Redraw Supplementary Fig. 4 from its two data sheets in the Supplementary Tables.

The script that drew the published Fig. S4 is not in this repository. This one reproduces the
four panels from the sheets "Data for Figure S4a and S4b" and "Data for Figure S4c and S4d", so the
figure can be regenerated if the data or a label change. Before drawing, the counts and
correlations are compared with the values printed on the published figure (EXPECTED); a mismatch
stops the script unless --no-check is given.

  a  Prompt 1: BPR score vs. observed consumption for 22 metabolites (Pseudomonas fluorescens).
  b  Prompt 1: BPR score vs. depletion time (24 h = not consumed within 24 h).
  c  Prompt 2: genus-level BPR score vs. observed consumption for 48 genus-compound pairs
     (a genus consumes a compound if more than half of its isolates did).
  d  Prompt 2: percentage of correct predictions by compound and by genus.

A pair is predicted to be consumed if its BPR score is above THRESHOLD (4.0). Permutation P values
use 10,000 permutations and P = (k + 1) / (n + 1); they vary in the last digit with the random seed.

Usage:
    python scripts/make_figure_s4.py --workbook <Supplementary Tables.xlsx> --output-dir <dir>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

THRESHOLD = 4.0
N_PERMUTATIONS = 10_000
GREEN, PINK = "#eafbe9", "#fdf1f3"
COUNT_GREEN, COUNT_RED = "#8fbf8f", "#d39a9a"
TIME_COLORS = {3: "#6cb9ea", 6: "#1fa87b", 12: "#e9b430", 24: "#e07f2c"}
SHORT = {"S-methyl-5'-thioadenosine": "5'-MTA", "3',5'-cyclic AMP": "cAMP", "nicotinamide": "Nicotinamide",
         "arginine": "Arginine", "adenosine": "Adenosine", "uridine": "Uridine"}
# values printed on the published figure (revised model)
EXPECTED = {
    "a": {"TP": 16, "FN": 1, "TN": 4, "FP": 1, "r": 0.680, "accuracy": 0.909},
    "b": {"pearson_r": -0.712, "spearman_rho": -0.698},
    "c": {"TP": 19, "FN": 3, "TN": 16, "FP": 10, "r": 0.476, "accuracy": 0.729},
}


def read_sheets(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import openpyxl
    wb = openpyxl.load_workbook(workbook, data_only=True, read_only=True)
    rows, started = [], False
    for r in wb["Data for Figure S4a and S4b"].iter_rows(values_only=True):
        if r and r[0] == "metabolite":
            started = True
            continue
        if started:
            if not r or r[0] is None:  # the block of metabolites ends at the first blank row
                break
            rows.append(r[:4])
    a = pd.DataFrame(rows, columns=["metabolite", "consumed", "time_h", "score"])
    a = a[a.time_h.notna() & a.score.map(lambda v: isinstance(v, (int, float)))].reset_index(drop=True)
    a[["consumed", "time_h", "score"]] = a[["consumed", "time_h", "score"]].astype(float)

    rows, started = [], False
    for r in wb["Data for Figure S4c and S4d"].iter_rows(values_only=True):
        if r and r[0] == "Isolate":
            started = True
            continue
        if started and r and r[0] and isinstance(r[2], (int, float)):
            rows.append(r[:5])
    c = pd.DataFrame(rows, columns=["isolate", "compound", "score", "wom_score", "consumed"])
    c[["score", "wom_score", "consumed"]] = c[["score", "wom_score", "consumed"]].astype(float)
    c["genus"] = c.isolate.str.split().str[0]
    return a, c


def genus_level(c: pd.DataFrame) -> pd.DataFrame:
    """One row per genus and compound: consumed if more than half of the genus's isolates consumed it."""
    g = c.groupby(["genus", "compound"], sort=False)
    assert (g.score.nunique() == 1).all(), "isolates of a genus must share the genus-level BPR score"
    return g.agg(score=("score", "first"), consumed=("consumed", lambda v: float(v.mean() > 0.5))).reset_index()


def classification(score: pd.Series, consumed: pd.Series, seed: int = 0) -> dict:
    pred, obs = (score > THRESHOLD).to_numpy(), consumed.astype(bool).to_numpy()
    rng = np.random.default_rng(seed)
    r = pearsonr(score, consumed).statistic
    acc = float((pred == obs).mean())
    perm_r = np.array([abs(pearsonr(score, rng.permutation(consumed.to_numpy())).statistic) for _ in range(N_PERMUTATIONS)])
    perm_acc = np.array([(pred == rng.permutation(obs)).mean() for _ in range(N_PERMUTATIONS)])
    return {"TP": int((pred & obs).sum()), "FN": int((~pred & obs).sum()), "TN": int((~pred & ~obs).sum()),
            "FP": int((pred & ~obs).sum()), "r": float(r), "accuracy": acc,
            "r_p": float(((perm_r >= abs(r)).sum() + 1) / (N_PERMUTATIONS + 1)),
            "accuracy_p": float(((perm_acc >= acc).sum() + 1) / (N_PERMUTATIONS + 1))}


def time_correlation(a: pd.DataFrame, seed: int = 0) -> dict:
    """Pearson r of score with depletion time and its permutation P value; Spearman rho as a point estimate."""
    rng = np.random.default_rng(seed)
    r = pearsonr(a.score, a.time_h).statistic
    perm = np.array([abs(pearsonr(a.score, rng.permutation(a.time_h.to_numpy())).statistic) for _ in range(N_PERMUTATIONS)])
    return {"pearson_r": float(r), "pearson_p": float(((perm >= abs(r)).sum() + 1) / (N_PERMUTATIONS + 1)),
            "spearman_rho": float(spearmanr(a.score, a.time_h).statistic)}


def check(stats: dict):
    """Counts, correlations and accuracies are exact; permutation P values depend on the seed and are not compared."""
    for panel, expected in EXPECTED.items():
        for key, value in expected.items():
            got = stats[panel][key]
            ok = got == value if isinstance(value, int) else abs(got - value) < 0.0006
            if not ok:
                raise ValueError(f"panel {panel}: {key} = {got}, but the published figure shows {value}")
    logger.info("Counts and correlations match the values printed on the published figure")


def panel_classification(ax, score: pd.Series, consumed: pd.Series, st: dict, letter: str):
    pred, obs = score > THRESHOLD, consumed.astype(bool)
    x0, x1 = -3.5, 12
    for (xa, xb), (ya, yb), color in [((THRESHOLD, x1), (0.5, 1.5), GREEN), ((x0, THRESHOLD), (-0.5, 0.5), GREEN),
                                      ((x0, THRESHOLD), (0.5, 1.5), PINK), ((THRESHOLD, x1), (-0.5, 0.5), PINK)]:
        ax.fill_between([xa, xb], ya, yb, color=color, zorder=0, linewidth=0)
    ax.axvline(THRESHOLD, color="#b0b0b0", linestyle="--", linewidth=2)
    ax.axhline(0.5, color="#b0b0b0", linestyle="--", linewidth=2)
    groups = [("True Positive", pred & obs, "o", True), ("True Negative", ~pred & ~obs, "s", True),
              ("False Positive", pred & ~obs, "o", False), ("False Negative", ~pred & obs, "s", False)]
    for label, sel, marker, filled in groups:
        ax.scatter(score[sel], consumed[sel], marker=marker, s=42, linewidths=1.2, edgecolors="black",
                   facecolors="#2b2b2b" if filled else "white", zorder=3)
        ax.scatter([], [], marker=marker, s=42, linewidths=1.2, edgecolors="black",
                   facecolors="#2b2b2b" if filled else "white", label=label)
    for key, x, y, color in [("FN", 0.25, 1.13, COUNT_RED), ("TP", 8, 1.13, COUNT_GREEN),
                             ("TN", 0.25, 0.13, COUNT_GREEN), ("FP", 8, 0.13, COUNT_RED)]:
        ax.text(x, y, str(st[key]), fontsize=22, fontweight="bold", color=color, ha="center", va="center")
    ax.text(0.97, 0.03, f"r = {st['r']:.3f}, p = {st['r_p']:.4f}\nAccuracy: {100 * st['accuracy']:.1f}%, p = {st['accuracy_p']:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, family="monospace", color="#555555")
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not consumed", "Consumed"])
    ax.set_xlabel("BPR Model Score", fontsize=11)
    ax.set_ylabel("Experimental observations", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc="center left", bbox_to_anchor=(0.0, 0.5), handletextpad=0.3)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.18, 1.02, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")


def panel_time(ax, a: pd.DataFrame, st: dict):
    for t, color in TIME_COLORS.items():
        sel = a.time_h == t
        if t == 24:
            ax.scatter(a.time_h[sel], a.score[sel], marker="x", s=90, color=color, linewidths=1.3, zorder=3)
        else:
            ax.scatter(a.time_h[sel], a.score[sel], marker="^", s=75, color=color, edgecolors="black",
                       linewidths=1.1, zorder=3)
        ax.axvline(t, color="#bbbbbb", linestyle=":", linewidth=1.2, zorder=1)
    slope, intercept = np.polyfit(a.time_h, a.score, 1)
    ax.plot([3, 24], [intercept + 3 * slope, intercept + 24 * slope], color="#777777", linestyle="--", linewidth=2)
    ax.text(0.98, 0.97, f"r = {st['pearson_r']:.3f}\np = {st['pearson_p']:.4f}\nρ = {st['spearman_rho']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, family="monospace", color="#555555")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black") for c in TIME_COLORS.values()]
    ax.legend(handles, ["Consumed in 3 h", "Consumed in 6 h", "Consumed in 12 h", "Not consumed"], frameon=False,
              fontsize=9.5, loc="lower right", handletextpad=0.3, borderaxespad=0.1)
    ax.set_xticks(list(TIME_COLORS))
    ax.set_xlim(2, 25)
    ax.set_ylim(-3, 12)
    ax.set_xlabel("Depletion time (hours)", fontsize=11)
    ax.set_ylabel("BPR Model Score", fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.18, 1.02, "b", transform=ax.transAxes, fontsize=14, fontweight="bold")


def panel_bars(ax, correct: pd.Series, title: str, italic: bool):
    correct = correct.sort_values(ascending=False, kind="stable")
    shades = [str(round(0.12 + 0.42 * (100 - v) / 50, 3)) for v in correct]
    ax.bar(range(len(correct)), correct, width=0.62, color=shades, edgecolor="black", linewidth=0.8)
    ax.axhline(50, color="#aaaaaa", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(correct)))
    ax.set_xticklabels(correct.index, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.5,
                       style="italic" if italic else "normal")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Correct predictions (%)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main(workbook: Path, output_dir: Path, do_check: bool):
    a, c = read_sheets(workbook)
    g = genus_level(c)
    stats = {"a": classification(a.score, a.consumed), "b": time_correlation(a),
             "c": classification(g.score, g.consumed)}
    logger.info(f"panel a: n = {len(a)}, {stats['a']}")
    logger.info(f"panel b: {stats['b']}")
    logger.info(f"panel c: n = {len(g)} genus-compound pairs, {stats['c']}")
    if do_check:
        check(stats)
    g["correct"] = (g.score > THRESHOLD) == g.consumed.astype(bool)
    by_compound = 100 * g.groupby("compound", sort=False).correct.mean().rename(index=SHORT)
    by_genus = 100 * g.groupby("genus", sort=False).correct.mean()

    fig = plt.figure(figsize=(10.24, 7.9))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.82, 0.82], hspace=0.34, wspace=0.75,
                            left=0.13, right=0.985, top=0.96, bottom=0.12)
    panel_classification(fig.add_subplot(grid[0, :2]), a.score, a.consumed, stats["a"], "a")
    panel_time(fig.add_subplot(grid[0, 2:]), a, stats["b"])
    panel_classification(fig.add_subplot(grid[1, :2]), g.score, g.consumed, stats["c"], "c")
    left = fig.add_subplot(grid[1, 2])
    panel_bars(left, by_compound, "By compound", italic=False)
    left.text(-0.15, 1.02, "d", transform=left.transAxes, fontsize=14, fontweight="bold")
    panel_bars(fig.add_subplot(grid[1, 3]), by_genus, "By genus", italic=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"figS4.{ext}", dpi=200)
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'figS4.png'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Redraw Supplementary Fig. 4 from the Supplementary Tables")
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-check", action="store_true",
                        help="draw even if the statistics differ from those on the published figure")
    args = parser.parse_args()
    main(args.workbook, args.output_dir, not args.no_check)
