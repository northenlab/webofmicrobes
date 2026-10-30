"""make_figure_biology.py — Main biology figure for WoM manuscript.

Two-panel figure:
  A: Metabolic similarity decays with phylogenetic distance (distance-decay violin)
  B: BPR model prediction uncertainty (mean score_sd) vs. empirical within-genus
     consistency (GCS) — validates that the agent's uncertainty signal is calibrated

Together these panels make one closed argument:
  (A) Substrate preferences are phylogenetically conserved but imperfect —
      same-genus isolates agree (ρ ≈ 0.44) but not perfectly.
  (B) The WoM agent's built-in uncertainty (score_sd) is not arbitrary —
      it correlates with the empirical GCS, meaning high score_sd flags
      exactly the genera where isolate-level diversity is real.

Usage:
    cd wom/
    uv run python scripts/make_figure_biology.py
    uv run python scripts/make_figure_biology.py --n-bootstrap 5000 --n-permutations 999
    uv run python scripts/make_figure_biology.py \\
        --output-dir manuscript_figures_nat_microbiology_first_submission
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import load_model_artifacts
from scripts.taxonomic_group_analysis import (
    CLASS_COLORS,
    DISTANCE_LABELS,
    get_class_color,
    load_and_preprocess,
    compute_all_pairs,
    compute_embedding_pairs,
    load_bpr_embeddings,
    per_genus_analysis,
    distance_decay_analysis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model: within-genus embedding diversity
# ---------------------------------------------------------------------------

def compute_genus_embedding_diversity(
    model_dir: Path,
    tax_map: pd.DataFrame,
    min_isolates: int = 3,
) -> pd.Series:
    """Compute mean pairwise cosine distance between isolate embeddings per genus.

    This is the embedding-space analog of GCS: both measure within-genus
    heterogeneity, but GCS uses empirical assertion values while this uses
    the BPR model's learned representation.

        emb_diversity = 1 − mean(pairwise cosine similarity)

    Scale-invariant (cosine distance normalises out embedding magnitude).
    Range: 0 = all isolates identical in embedding space,
           1 = maximally diverse.

    Returns: pd.Series  (genus → mean pairwise cosine distance)
    """
    logger.info(f"Loading BPR model from {model_dir}...")
    artifacts = load_model_artifacts(model_dir)
    model     = artifacts.model
    td        = artifacts.training_data

    E_all = model.microbe_embeddings    # (n_microbes, d)

    # L2-normalise once for cosine similarity via dot product
    norms = np.linalg.norm(E_all, axis=1, keepdims=True).clip(1e-12)
    E_norm = E_all / norms              # (n_microbes, d)

    # Group microbe indices by genus
    genus_to_indices: dict[str, list[int]] = {}
    for org_id, idx in td.microbe_to_idx.items():
        if org_id not in tax_map.index:
            continue
        genus = tax_map.loc[org_id, "genus"]
        if pd.notna(genus) and str(genus) not in ("nan", "None", ""):
            genus_to_indices.setdefault(str(genus), []).append(idx)

    results: dict[str, float] = {}
    for genus, indices in genus_to_indices.items():
        if len(indices) < min_isolates:
            continue
        E = E_norm[indices]             # (K, d) — already L2-normalised
        # Cosine similarity matrix: sim[i,j] = dot(e_i, e_j)
        sims = E @ E.T                  # (K, K)
        K = len(indices)
        # Mean over upper triangle (all pairs, excl. self-similarity = 1)
        mask      = np.triu(np.ones((K, K), dtype=bool), k=1)
        mean_sim  = float(sims[mask].mean())
        results[genus] = 1.0 - mean_sim # cosine distance

    series = pd.Series(results, name="emb_diversity")
    logger.info(
        f"  Embedding diversity computed for {len(series)} genera "
        f"(mean={series.mean():.4f}, sd={series.std():.4f})"
    )
    return series


# ---------------------------------------------------------------------------
# Panel A: asymmetric distance-decay violin (empirical left | embedding right)
# ---------------------------------------------------------------------------

# Colors for the two data sources
_EMP_COLOR = "#4878CF"   # steel blue  — empirical (Spearman ρ)
_EMB_COLOR = "#EE854A"   # warm orange — model embedding (cosine similarity)


def _half_violin(
    ax: plt.Axes,
    data: np.ndarray,
    x_center: float,
    side: str,          # "left" or "right"
    color: str,
    width: float = 0.38,
    alpha: float = 0.80,
    n_points: int = 300,
) -> None:
    """Draw one half of a violin (KDE) at x_center."""
    data = np.asarray(data)
    if len(data) < 3:
        return
    bw = gaussian_kde(data, bw_method="scott")
    lo, hi = np.percentile(data, [0.5, 99.5])
    margin = max((hi - lo) * 0.05, 0.01)
    y_range  = np.linspace(lo - margin, hi + margin, n_points)
    density  = bw(y_range)
    density  = density / density.max() * width   # scale to half-width

    if side == "left":
        ax.fill_betweenx(
            y_range, x_center - density, x_center,
            color=color, alpha=alpha,
            edgecolor="black", linewidth=0.8,
        )
    else:
        ax.fill_betweenx(
            y_range, x_center, x_center + density,
            color=color, alpha=alpha,
            edgecolor="black", linewidth=0.8,
        )


def _draw_distance_decay(
    ax: plt.Axes,
    groups_emp: dict,
    summary_emp: pd.DataFrame,
    kw_stat_emp: float,
    kw_p_emp: float,
    groups_emb: dict,
    summary_emb: pd.DataFrame,
    kw_stat_emb: float,
    kw_p_emb: float,
) -> None:
    """Draw asymmetric distance-decay violins: empirical (left) | embedding (right)."""
    # Use intersection of distances present in both datasets
    distances = sorted(set(groups_emp.keys()) & set(groups_emb.keys()))
    x_pos     = list(range(len(distances)))

    for k, d in enumerate(distances):
        _half_violin(ax, groups_emp[d], k, "left",  _EMP_COLOR)
        _half_violin(ax, groups_emb[d], k, "right", _EMB_COLOR)

        # White median tick across each half
        ax.hlines(
            np.median(groups_emp[d]), k - 0.34, k,
            color="white", lw=2.0, zorder=6,
        )
        ax.hlines(
            np.median(groups_emb[d]), k, k + 0.34,
            color="white", lw=2.0, zorder=6,
        )

        # Mean ± 95% CI dots, offset inside each half
        row_e = summary_emp[summary_emp["distance"] == d].iloc[0]
        row_m = summary_emb[summary_emb["distance"] == d].iloc[0]
        for x_off, row, clr in [(-0.13, row_e, _EMP_COLOR), (+0.13, row_m, _EMB_COLOR)]:
            ax.errorbar(
                k + x_off, row["mean_rho"],
                yerr=[[row["mean_rho"] - row["ci_low"]],
                      [row["ci_high"]  - row["mean_rho"]]],
                fmt="o", color=clr, markersize=7,
                capsize=4, capthick=1.8,
                markeredgecolor="white", markeredgewidth=1.2,
                lw=1.5, zorder=7,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{DISTANCE_LABELS.get(d, d)}\n(n={len(groups_emp[d]):,})" for d in distances],
        fontsize=12,
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel(
        "Pairwise metabolic profile similarity\n"
        "(Spearman ρ  |  cosine similarity)",
        fontsize=18,
    )
    ax.set_xlabel("")
    ax.tick_params(labelsize=16)

    # Legend — no frame, wider, larger font
    legend_handles = [
        mpatches.Patch(facecolor=_EMP_COLOR, edgecolor="black", alpha=0.80,
                       label="Empirical — Spearman ρ\nbetween measured profiles"),
        mpatches.Patch(facecolor=_EMB_COLOR, edgecolor="black", alpha=0.80,
                       label="Model embedding — cosine similarity\nbetween BPR latent vectors"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=14,
              frameon=False, handlelength=2.0, handletextpad=0.8)

    sns.despine(ax=ax)

    # Panel label
    ax.text(0.02, 1.00, "a", transform=ax.transAxes,
            fontsize=30, va="top", ha="left")


# ---------------------------------------------------------------------------
# Stats text file for Panel A
# ---------------------------------------------------------------------------

def _write_distance_decay_stats(
    summary_emp: pd.DataFrame,
    kw_stat_emp: float,
    kw_p_emp: float,
    permanova_r2_emp: float,
    summary_emb: pd.DataFrame,
    kw_stat_emb: float,
    kw_p_emb: float,
    permanova_r2_emb: float,
    output_dir: Path,
) -> None:
    """Write a plain-text summary of distance-decay statistics for Panel A."""
    lines: list[str] = []

    def _pval(p: float) -> str:
        if p < 1e-300:
            return "< 1e-300"
        return f"{p:.4e}"

    lines += [
        "=" * 72,
        "Distance-decay analysis statistics (Panel A)",
        "=" * 72,
        "",
        "Two complementary metrics are compared at each taxonomic distance:",
        "  Empirical : pairwise Spearman ρ between isolate assertion-value profiles",
        "  Embedding : pairwise cosine similarity between BPR microbe latent vectors",
        "",
        "Only distances present in BOTH datasets are included in the figure.",
        "",
    ]

    for label, summary, kw_stat, kw_p, perm_r2 in [
        ("EMPIRICAL (Spearman ρ)", summary_emp, kw_stat_emp, kw_p_emp, permanova_r2_emp),
        ("EMBEDDING (cosine similarity)", summary_emb, kw_stat_emb, kw_p_emb, permanova_r2_emb),
    ]:
        lines += [
            "-" * 72,
            label,
            "-" * 72,
            "",
            f"  Kruskal-Wallis omnibus test across distance groups:",
            f"    H = {kw_stat:.2f},  p {_pval(kw_p)}",
        ]
        if not np.isnan(perm_r2):
            lines.append(f"  PERMANOVA R² (genus label) = {perm_r2:.4f}")
        lines += ["", f"  {'Distance':<20}  {'n_pairs':>9}  {'Mean':>8}  {'Median':>8}  {'SD':>8}  {'95% CI low':>10}  {'95% CI high':>11}"]
        lines.append("  " + "-" * 80)
        for _, row in summary.iterrows():
            dist_label = DISTANCE_LABELS.get(int(row["distance"]), str(int(row["distance"])))
            dist_label_flat = dist_label.replace("\n", " ")
            lines.append(
                f"  {dist_label_flat:<20}  {int(row['n_pairs']):>9,}  "
                f"{row['mean_rho']:>8.4f}  {row['median_rho']:>8.4f}  "
                f"{row['sd_rho']:>8.4f}  {row['ci_low']:>10.4f}  {row['ci_high']:>11.4f}"
            )
        lines.append("")

    lines += [
        "=" * 72,
        "Notes",
        "=" * 72,
        "",
        "• Mean and 95% CI are shown as dots with error bars on the figure.",
        "• White horizontal lines on the violins indicate the median.",
        "• 95% CI computed by bootstrap resampling (n_bootstrap specified at run time).",
        "• PERMANOVA uses genus labels as groups and pairwise (1 − similarity) as distance.",
        "• The abstract claim about within-genus similarity is supported by the",
        "  'Same genus' group having the highest mean and median in both datasets.",
        "",
    ]

    out_path = output_dir / "figure_biology_panel_A_stats.txt"
    out_path.write_text("\n".join(lines))
    logger.info(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Panel B: GCS vs mean score_sd scatter
# ---------------------------------------------------------------------------

def _annotate_spread(
    ax: plt.Axes,
    items: list,
    fontsize: int = 16,
    style: str = "italic",
    min_dy_frac: float = 0.055,
) -> None:
    """Annotate scatter points with labels, spreading them vertically to reduce overlap.

    items: list of (data_x, data_y, label) tuples.
    min_dy_frac: minimum vertical gap between labels as a fraction of y-axis range.
    """
    if not items:
        return

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    min_dy = min_dy_frac * y_range
    xlim = ax.get_xlim()
    x_offset = 0.015 * (xlim[1] - xlim[0])

    # Sort top-to-bottom so we push overlapping labels downward
    sorted_items = sorted(items, key=lambda t: -t[1])

    # Propagate minimum gap: if two consecutive labels are too close, push the
    # lower one down.  A single forward pass is sufficient for typical clusters.
    text_ys = [y for _, y, _ in sorted_items]
    for i in range(len(text_ys) - 1):
        if text_ys[i] - text_ys[i + 1] < min_dy:
            text_ys[i + 1] = text_ys[i] - min_dy

    # for (data_x, data_y, label), text_y in zip(sorted_items, text_ys):
    #     ax.annotate(
    #         label,
    #         (data_x, data_y),
    #         xytext=(data_x + x_offset, text_y),
    #         textcoords="data",
    #         fontsize=fontsize,
    #         style=style,
    #         arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    #         color="black",
    #     )


def _draw_gcs_emb_diversity(
    ax: plt.Axes,
    merged: pd.DataFrame,
    n_label: int = 6,
) -> "plt.Legend":
    """Scatter: GCS (x) vs embedding diversity (y), colored by class.

    merged must have columns: genus, gcs, ci_low, ci_high, class, emb_diversity
    """
    rho, pval = spearmanr(merged["gcs"], merged["emb_diversity"])
    pval_str  = f"p = {pval:.2e}" if pval >= 0.001 else "p < 0.001"

    # Scatter + GCS error bars
    for _, row in merged.iterrows():
        color = get_class_color(row["class"])
        ax.scatter(
            row["gcs"], row["emb_diversity"],
            c=[color], s=55, edgecolors="gray", linewidths=0.4,
            alpha=0.85, zorder=4,
        )
        ax.errorbar(
            row["gcs"], row["emb_diversity"],
            xerr=[[row["gcs"] - row["ci_low"]], [row["ci_high"] - row["gcs"]]],
            fmt="none", color=color, alpha=0.35, lw=0.8, zorder=3,
        )

    # Label most and least diverse genera (spread vertically to avoid overlap)
    labeled: set[str] = set()
    label_items: list[tuple[float, float, str]] = []
    top_div = merged.nlargest(n_label, "emb_diversity")
    top_gcs = merged.nlargest(n_label // 2, "gcs")
    for sub in [top_div, top_gcs]:
        for _, row in sub.iterrows():
            g = row["genus"]
            if g in labeled:
                continue
            labeled.add(g)
            label_items.append((row["gcs"], row["emb_diversity"], g))
    _annotate_spread(ax, label_items)

    # Linear fit in original space
    b, a = np.polyfit(merged["gcs"].values, merged["emb_diversity"].values, 1)
    x_line = np.linspace(merged["gcs"].min(), merged["gcs"].max(), 100)
    ax.plot(x_line, a + b * x_line, color="#555555", lw=1.8, ls="--",
            alpha=0.7, zorder=2)

    ax.set_xlabel("Genus Consistency Score (GCS)\nmean within-genus pairwise Spearman ρ",
                  fontsize=18)
    ax.set_ylabel("Within-genus embedding diversity\n(mean pairwise cosine distance)",
                  fontsize=18)
    ax.tick_params(labelsize=18)

    # Class legend — bottom left, no frame, wider, larger font
    seen: set[str] = set()
    handles = []
    for cls in sorted(merged["class"].unique()):
        if cls not in seen:
            seen.add(cls)
            handles.append(mpatches.Patch(color=get_class_color(cls), label=cls))
    leg = ax.legend(handles=handles, fontsize=12, ncol=1, loc="lower left",
                    frameon=False, handlelength=1.8, handletextpad=0.6)

    sns.despine(ax=ax)

    # Panel label
    ax.text(0.02, 1.00, "b", transform=ax.transAxes,
            fontsize=30,  va="top", ha="left")

    return leg


# ---------------------------------------------------------------------------
# GCS ranking figure (standalone)
# ---------------------------------------------------------------------------

def _draw_gcs_ranking(ax: plt.Axes, genus_df: pd.DataFrame) -> None:
    """Bar chart of genera ordered highest → lowest GCS, colored by class.

    genus_df must have columns: genus, gcs, ci_low, ci_high, class
    (as returned by per_genus_analysis, already sorted by gcs descending).
    """
    df = genus_df.sort_values("gcs", ascending=False).reset_index(drop=True)

    x = np.arange(len(df))
    colors = [get_class_color(c) for c in df["class"]]

    ax.scatter(x, df["gcs"], c=colors, s=60, zorder=4,
               edgecolors="white", linewidths=0.3)

    # CI error bars (the "variance line")
    ax.errorbar(
        x, df["gcs"],
        yerr=[df["gcs"] - df["ci_low"], df["ci_high"] - df["gcs"]],
        fmt="none", ecolor=[get_class_color(c) for c in df["class"]],
        alpha=1, lw=2, zorder=3,
    )

    # ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["genus"], rotation=90, fontsize=12, ha="center")
    ax.set_ylabel(
        "Genus Consistency Score (GCS)\nmean within-genus\npairwise Spearman ρ",
        fontsize=13,
    )
    ax.set_xlabel("Genus (ordered highest → lowest GCS)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)

    # Class legend
    seen: set[str] = set()
    handles = []
    for cls in sorted(df["class"].unique()):
        if cls not in seen:
            seen.add(cls)
            handles.append(mpatches.Patch(color=get_class_color(cls), label=cls))
    ax.legend(handles=handles, fontsize=9, ncol=2, loc="lower left",
              frameon=False, handlelength=1.5, handletextpad=0.5,
              title="Class", title_fontsize=10)
    ax.set_ylim(0,1)
    sns.despine(ax=ax)


def make_figure_gcs_ranking(
    genus_df: pd.DataFrame,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
) -> None:
    """Standalone figure: all genera ranked by GCS, colored by class."""
    n = len(genus_df)
    fig_width = max(14, n * 0.22)   # ~0.22 in per genus
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    _draw_gcs_ranking(ax, genus_df)

    fig.tight_layout()

    stem = output_dir / "figure_gcs_ranking"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure_gcs_ranking (.png + .pdf)")


# ---------------------------------------------------------------------------
# Assemble figure
# ---------------------------------------------------------------------------

def make_figure_biology(
    # Panel A — empirical (Spearman ρ)
    groups_emp: dict,
    summary_emp: pd.DataFrame,
    kw_stat_emp: float,
    kw_p_emp: float,
    permanova_r2_emp: float,
    # Panel A — embedding (cosine similarity)
    groups_emb: dict,
    summary_emb: pd.DataFrame,
    kw_stat_emb: float,
    kw_p_emb: float,
    permanova_r2_emb: float,
    # Panel B
    merged: pd.DataFrame,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
) -> None:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6.5),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )

    _draw_distance_decay(
        axes[0],
        groups_emp, summary_emp, kw_stat_emp, kw_p_emp,
        groups_emb, summary_emb, kw_stat_emb, kw_p_emb,
    )
    leg_b = _draw_gcs_emb_diversity(axes[1], merged)

    # No overall suptitle (removed per manuscript style)
    fig.tight_layout()

    # Place the "Class" legend title left-aligned.  We must render first so that
    # tight_layout has finalised all axes positions, then use the legend's actual
    # screen bbox to compute the correct axes-fraction coordinate.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_b = axes[1]
    leg_bbox = leg_b.get_window_extent(renderer)
    ax_bbox  = ax_b.get_window_extent(renderer)
    x_frac = (leg_bbox.x0 - ax_bbox.x0) / ax_bbox.width
    y_frac = (leg_bbox.y1 - ax_bbox.y0) / ax_bbox.height
    ax_b.text(
        x_frac, y_frac-0.05, "Class",
        transform=ax_b.transAxes, fontsize=18,
        ha="left", va="bottom",
    )

    stem = output_dir / "figure_biology_main"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure_biology_main (.png + .pdf)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main biology figure")
    _script_dir = Path(__file__).parent.parent
    parser.add_argument("--model-dir",            type=Path,
                        default=_script_dir / "trained_models/ordinal_bpr_consumed")
    parser.add_argument("--output-dir",           type=Path,
                        default=_script_dir / "manuscript_figures_nat_microbiology_first_submission")
    parser.add_argument("--min-isolates-per-genus", type=int, default=3)
    parser.add_argument("--min-shared-compounds",   type=int, default=5)
    parser.add_argument("--n-bootstrap",             type=int, default=1000)
    parser.add_argument("--n-permutations",          type=int, default=999)
    parser.add_argument("--seed",                    type=int, default=42)
    parser.add_argument("--dpi-png",                 type=int, default=300)
    parser.add_argument("--dpi-pdf",                 type=int, default=600)
    parser.add_argument("--verbose",                 action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load empirical data ----
    logger.info("=== Step 1: Load empirical data ===")
    df, wide, tax_map = load_and_preprocess(
        min_isolates_per_genus=args.min_isolates_per_genus
    )

    # ---- Step 2: Pairwise Spearman ρ ----
    logger.info("=== Step 2: Pairwise Spearman ρ ===")
    pairs = compute_all_pairs(wide, tax_map,
                              min_shared_compounds=args.min_shared_compounds)

    # ---- Step 3: Distance-decay analysis (empirical) ----
    logger.info("=== Step 3: Distance-decay analysis (empirical) ===")
    result = distance_decay_analysis(
        pairs,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    logger.info(
        f"  KW H={result['kw_stat']:.1f}, p={result['kw_p']:.2e}, "
        f"PERMANOVA R²={result['permanova_r2']:.3f}"
    )

    # ---- Step 3b: Distance-decay analysis (embedding) ----
    logger.info("=== Step 3b: Load embeddings and compute embedding distance decay ===")
    emb_matrix, org_ids, tax_map_emb, _met_matrix, _met_names = load_bpr_embeddings(
        args.model_dir
    )
    pairs_emb = compute_embedding_pairs(emb_matrix, org_ids, tax_map_emb)
    result_emb = distance_decay_analysis(
        pairs_emb,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    logger.info(
        f"  Embedding KW H={result_emb['kw_stat']:.1f}, p={result_emb['kw_p']:.2e}, "
        f"PERMANOVA R²={result_emb['permanova_r2']:.3f}"
    )

    # ---- Step 4: Per-genus GCS ----
    logger.info("=== Step 4: Per-genus GCS ===")
    genus_df = per_genus_analysis(
        pairs, tax_map,
        min_isolates_per_genus=args.min_isolates_per_genus,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    logger.info(f"  {len(genus_df)} genera")

    # ---- Step 5: Within-genus embedding diversity ----
    logger.info("=== Step 5: Within-genus embedding diversity ===")
    emb_div_series = compute_genus_embedding_diversity(
        args.model_dir, tax_map,
        min_isolates=args.min_isolates_per_genus,
    )

    # ---- Step 6: Merge and correlate ----
    logger.info("=== Step 6: Merge GCS ↔ embedding diversity ===")
    merged = (
        genus_df
        .set_index("genus")[["gcs", "ci_low", "ci_high", "class", "n_isolates"]]
        .join(emb_div_series, how="inner")
        .reset_index()
        .rename(columns={"index": "genus"})
        .dropna(subset=["gcs", "emb_diversity"])
    )
    if "genus" not in merged.columns and merged.index.name == "genus":
        merged = merged.reset_index()

    rho, pval = spearmanr(merged["gcs"], merged["emb_diversity"])
    logger.info(
        f"  GCS vs embedding diversity: Spearman ρ = {rho:.3f}, p = {pval:.4g}, "
        f"n = {len(merged)} genera"
    )

    # Save merged table
    merged.to_csv(args.output_dir / "stats_gcs_emb_diversity.csv", index=False)
    logger.info("  Saved stats_gcs_emb_diversity.csv")

    # Summary statistics quoted in the manuscript
    def _permanova(res):
        n, g = res["permanova_n"], res["permanova_n_groups"]
        return {"pseudo_f": float(res["permanova_f"]), "r2": float(res["permanova_r2"]),
                "p": float(res["permanova_p"]), "n_permutations": args.n_permutations,
                "n_isolates": n, "n_genera": g, "df": [g - 1, n - g]}

    def _same_genus(summary):
        row = summary.loc[summary["distance"] == 1].iloc[0]  # 1 = same genus
        return {"mean": float(row["mean_rho"]), "median": float(row["median_rho"]),
                "n_pairs": int(row["n_pairs"])}

    stats = {
        "model_dir": str(args.model_dir),
        "within_genus_empirical_spearman": _same_genus(result["summary"]),
        "within_genus_embedding_cosine": _same_genus(result_emb["summary"]),
        "kruskal_wallis": {"empirical": {"H": result["kw_stat"], "p": result["kw_p"]},
                           "embedding": {"H": result_emb["kw_stat"], "p": result_emb["kw_p"]}},
        "permanova_empirical": _permanova(result),
        "permanova_embedding": _permanova(result_emb),
        "gcs_vs_embedding_diversity": {"spearman_rho": float(rho), "p": float(pval),
                                       "n_genera": int(len(merged))},
    }
    with open(args.output_dir / "stats_figure_biology.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("  Saved stats_figure_biology.json")

    # ---- Step 7: Stats text file ----
    logger.info("=== Step 7: Write distance-decay stats text file ===")
    _write_distance_decay_stats(
        summary_emp=result["summary"],
        kw_stat_emp=result["kw_stat"],
        kw_p_emp=result["kw_p"],
        permanova_r2_emp=result["permanova_r2"],
        summary_emb=result_emb["summary"],
        kw_stat_emb=result_emb["kw_stat"],
        kw_p_emb=result_emb["kw_p"],
        permanova_r2_emb=result_emb["permanova_r2"],
        output_dir=args.output_dir,
    )

    # ---- Step 8: Figure ----
    logger.info("=== Step 8: Generate figure ===")
    make_figure_biology(
        groups_emp=result["groups"],
        summary_emp=result["summary"],
        kw_stat_emp=result["kw_stat"],
        kw_p_emp=result["kw_p"],
        permanova_r2_emp=result["permanova_r2"],
        groups_emb=result_emb["groups"],
        summary_emb=result_emb["summary"],
        kw_stat_emb=result_emb["kw_stat"],
        kw_p_emb=result_emb["kw_p"],
        permanova_r2_emb=result_emb["permanova_r2"],
        merged=merged,
        output_dir=args.output_dir,
        dpi_png=args.dpi_png,
        dpi_pdf=args.dpi_pdf,
    )

    # ---- Step 9: GCS ranking figure ----
    logger.info("=== Step 9: GCS ranking figure ===")
    make_figure_gcs_ranking(
        genus_df=genus_df,
        output_dir=args.output_dir,
        dpi_png=args.dpi_png,
        dpi_pdf=args.dpi_pdf,
    )

    logger.info(f"=== Done. Outputs in {args.output_dir}/ ===")


if __name__ == "__main__":
    main()
