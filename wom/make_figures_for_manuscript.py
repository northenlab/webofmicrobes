# make_figures_for_manuscript.py
"""Generate publication figures from pre-computed results.

Expects the following files in the input directory:
  - core_results_consumed.csv (from benchmark.py --core-only)
  - unified_holdout_results_consumed.csv (from benchmark.py --holdout-only)
  - taxon_profiles_consumed.parquet (from taxon_metabolite_profiles.py)
  - metabolite_categories_consumed.csv (from taxon_metabolite_profiles.py)

Also needs the raw data for taxonomy and embeddings:
  - cache/exometabolite.parquet

Usage:
    python make_figures_for_manuscript.py --results-dir figures/ --output-dir manuscript_figures/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from model import BPRModel, prepare_training_data

logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# Taxonomy ordering: class → order → family → genus
# We'll build this dynamically from the data, but define class-level colors
CLASS_COLORS = {
    "Actinomycetes": "#1f77b4",
    "Bacilli": "#ff7f0e",
    "Alphaproteobacteria": "#2ca02c",
    "Betaproteobacteria": "#9467bd",
    "Gammaproteobacteria": "#8c564b",
    "Flavobacteriia": "#e377c2",
    "Sphingobacteriia": "#7f7f7f",
    "Cytophagia": "#bcbd22",
    "Cyanophyceae": "#17becf",
}

DEFAULT_CLASS_COLOR = "#aaaaaa"


def get_class_color(class_name: str) -> str:
    """Get color for a taxonomic class."""
    for key, color in CLASS_COLORS.items():
        if key.lower() in str(class_name).lower():
            return color
    return DEFAULT_CLASS_COLOR


def build_taxonomy_table(df: pd.DataFrame, microbe_to_idx: dict) -> pd.DataFrame:
    """Build a taxonomy lookup table for all organisms.

    Returns DataFrame with columns: organism_id, genus, family, order, class, phylum
    """
    tax_levels = ["genus", "family", "order", "class", "phylum"]
    rows = []

    for org_id in microbe_to_idx:
        org_rows = df[df["organism_id"] == org_id]
        if len(org_rows) == 0:
            continue

        row = {"organism_id": org_id}
        for level in tax_levels:
            if level in org_rows.columns:
                vals = org_rows[level].dropna().unique()
                row[level] = vals[0] if len(vals) > 0 else "Unknown"
            else:
                row[level] = "Unknown"
        rows.append(row)

    return pd.DataFrame(rows)


def get_taxonomic_genus_order(tax_table: pd.DataFrame) -> list[str]:
    """Order genera by class → order → family → genus alphabetically.

    Returns list of genus names in taxonomic order.
    """
    sort_cols = []
    for col in ["class", "order", "family", "genus"]:
        if col in tax_table.columns:
            sort_cols.append(col)

    genus_tax = tax_table.drop_duplicates(subset=["genus"]).copy()
    genus_tax = genus_tax[genus_tax["genus"] != "Unknown"]
    genus_tax = genus_tax.sort_values(sort_cols)

    return genus_tax["genus"].tolist()


# =============================================================================
# Figure 1: Four-Panel Benchmark (Model Comparison + Holdout)
# =============================================================================

def figure_1_benchmark(
    core_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    output_dir: Path,
):
    """Figure 1: Four-panel benchmark figure.

    Top row: BPR vs WR-MF vs SVD leave-one-out (AUC and Mean Percentile).
    Bottom row: Taxonomic holdout with raw vs distinctive scores (Mean Percentile and AUC).
    """
    has_score_type = "score_type" in holdout_results.columns
    if not has_score_type:
        logger.warning("Holdout results missing 'score_type' column. "
                       "Run benchmark with distinctive scoring enabled.")
        return

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    panel_labels = ['a.', 'b.', 'c.', 'd.']

    # -------------------------------------------------------------------------
    # Top row: Model comparison (panels a, b)
    # -------------------------------------------------------------------------
    style = {
        "BPR": {"color": "#d62728", "marker": "s", "label": "BPR"},
        "WR-MF": {"color": "#1f77b4", "marker": "o", "label": "WR-MF"},
        "SVD": {"color": "#2ca02c", "marker": "^", "label": "SVD"},
    }

    # Panel a: AUC vs embedding dimension
    if type(axes) is np.ndarray and axes.ndim == 2:
        ax = axes[0, 0]
    else:
        ax = axes
    for method in ["BPR", "WR-MF", "SVD"]:
        method_df = core_results[core_results["method"] == method]
        if len(method_df) == 0:
            continue
        summary = method_df.groupby("dim")["auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["dim"], summary["mean"], yerr=summary["std"],
            label=style[method]["label"],
            marker=style[method]["marker"],
            color=style[method]["color"],
            capsize=4, linewidth=2, markersize=6,
        )
    ax.set_xlabel("Embedding dimension", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.legend(frameon=False, fancybox=False, edgecolor="gray", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.0)

    # # Panel b: Mean Percentile vs embedding dimension
    # ax = axes[0, 1]
    # for method in ["BPR", "WR-MF", "SVD"]:
    #     method_df = core_results[core_results["method"] == method]
    #     if len(method_df) == 0:
    #         continue
    #     summary = method_df.groupby("dim")["mean_percentile"].agg(["mean", "std"]).reset_index()
    #     ax.errorbar(
    #         summary["dim"], summary["mean"], yerr=summary["std"],
    #         label=style[method]["label"],
    #         marker=style[method]["marker"],
    #         color=style[method]["color"],
    #         capsize=4, linewidth=2, markersize=6,
    #     )
    # ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    # ax.set_xlabel("Embedding dimension", fontsize=12)
    # ax.set_ylabel("Mean percentile of positives", fontsize=12)
    # ax.set_xscale("log", base=2)
    # ax.grid(True, alpha=0.3)

    # # -------------------------------------------------------------------------
    # # Bottom row: Holdout benchmark (panels c, d)
    # # -------------------------------------------------------------------------
    # level_order = ["order", "family", "genus", "isolate"]
    # present_levels = [l for l in level_order if l in holdout_results["holdout_level"].values]

    # level_labels = {
    #     "order": "Hold out\norder",
    #     "family": "Hold out\nfamily",
    #     "genus": "Hold out\ngenus",
    #     "isolate": "Hold out\nisolate",
    # }

    # colors = {"raw": "#377eb8", "distinctive": "#e41a1c"}

    # for panel_idx, (ax, metric, ylabel) in enumerate([
    #     (axes[1, 0], "mean_percentile", "Mean percentile (lower = better)"),
    #     (axes[1, 1], "auc", "AUC"),
    # ]):
    #     positions = []
    #     data_boxes = []
    #     box_colors = []
    #     tick_positions = []
    #     tick_labels = []

    #     for i, level in enumerate(present_levels):
    #         center = i * 3
    #         for j, score_type in enumerate(["raw", "distinctive"]):
    #             vals = holdout_results[
    #                 (holdout_results["holdout_level"] == level) &
    #                 (holdout_results["score_type"] == score_type)
    #             ][metric].values

    #             if len(vals) == 0:
    #                 continue

    #             pos = center + j - 0.5
    #             positions.append(pos)
    #             data_boxes.append(vals)
    #             box_colors.append(colors[score_type])

    #         tick_positions.append(center)
    #         tick_labels.append(level_labels.get(level, level))

    #     if data_boxes:
    #         bp = ax.boxplot(
    #             data_boxes, positions=positions, widths=0.8,
    #             patch_artist=True, flierprops=dict(markersize=3),
    #         )
    #         for patch, color in zip(bp["boxes"], box_colors):
    #             patch.set_facecolor(color)
    #             patch.set_alpha(0.6)

    #     ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    #     ax.set_xticks(tick_positions)
    #     ax.set_xticklabels(tick_labels, fontsize=14)
    #     ax.tick_params(axis='y', which='major', labelsize=14)
    #     ax.set_ylabel(ylabel, fontsize=16)
    #     ax.grid(True, alpha=0.3, axis="y")

    #     # Only show legend on the right panel
    #     if panel_idx > 0:
    #         ax.legend(
    #             handles=[
    #                 Patch(facecolor=colors["raw"], alpha=0.6, label="Raw score"),
    #                 Patch(facecolor=colors["distinctive"], alpha=0.6, label="Distinctive score"),
    #             ],
    #             loc="best",
    #             fontsize=14,
    #             frameon=False,
    #             fancybox=False,
    #             edgecolor="gray",
    #         )

    #     # Remove top and right spines
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel labels and final formatting
    # -------------------------------------------------------------------------
    # for ax, label in zip(axes.flat, panel_labels):
    #     ax.text(0.075, 1.02, label, transform=ax.transAxes,
    #             fontsize=24, va='top', ha='right')

    # Increase tick label fontsize on top row
    if type(axes) is np.ndarray and axes.ndim == 2:
        for ax in axes[0, :]:
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    else:
        axes.tick_params(axis='both', which='major', labelsize=12)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure1_benchmark.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure1_benchmark")


# =============================================================================
# Figure 2: t-SNE of Microbe Embeddings
# =============================================================================

def figure_2_tsne(
    df: pd.DataFrame,
    output_dir: Path,
    embedding_dim: int = 32,
    perplexity: float = 30.0,
    random_state: int = 42,
):
    """Figure 2: t-SNE of microbe embeddings colored by class, shaped by genus."""
    data = prepare_training_data(df, "consumed")
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    microbe_to_idx = data["microbe_to_idx"]
    microbe_taxonomy = data["microbe_taxonomy"]
    idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}

    model = BPRModel(embedding_dim=embedding_dim)
    model.fit(positives, n_microbes, n_metabolites, verbose=False)

    embeddings = model.microbe_embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    coords = tsne.fit_transform(embeddings)

    tax_table = build_taxonomy_table(df, microbe_to_idx)

    classes = sorted(tax_table["class"].unique())
    genera = sorted(tax_table["genus"].unique())

    class_color_map = {c: get_class_color(c) for c in classes}
    # Assign markers per genus WITHIN each class to avoid reuse
    marker_set = [
        "o", "s", "^", "v", "<", ">", "D", "p", "*", "h",
        "d", "P", "X", "8",
    ]

    genus_marker_map = {}
    genus_fill_map = {}  # True = filled, False = open

    for cls in classes:
        cls_genera = sorted(
            tax_table[tax_table["class"] == cls]["genus"].unique()
        )
        for i, genus in enumerate(cls_genera):
            marker_idx = i % len(marker_set)
            is_filled = i < len(marker_set)  # first pass filled, second pass open
            genus_marker_map[genus] = marker_set[marker_idx]
            genus_fill_map[genus] = is_filled
    # Wider figure to accommodate legend on the right
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx in range(n_microbes):
        org_id = idx_to_microbe[idx]
        org_tax = tax_table[tax_table["organism_id"] == org_id]

        if len(org_tax) == 0:
            continue

        org_class = org_tax.iloc[0]["class"]
        org_genus = org_tax.iloc[0]["genus"]

        color = class_color_map.get(org_class, DEFAULT_CLASS_COLOR)
        marker = genus_marker_map.get(org_genus, "o")
        filled = genus_fill_map.get(org_genus, True)

        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            facecolors=color if filled else "none",
            edgecolors=color,
            marker=marker, s=80, alpha=1,
            linewidths=1.5,
        )

    # Build legend grouped by class
    legend_handles = []
    for cls in classes:
        cls_genera = sorted(
            tax_table[tax_table["class"] == cls]["genus"].unique()
        )
        color = class_color_map[cls]

        legend_handles.append(
            Patch(facecolor=color, edgecolor="gray", label=cls)
        )

        for genus in cls_genera:
            marker = genus_marker_map.get(genus, "o")
            filled = genus_fill_map.get(genus, True)
            handle = ax.scatter(
                [], [],
                facecolors=color if filled else "none",
                edgecolors=color,
                marker=marker, s=40,
                linewidths=1.5,
                label=f"  {genus}",
            )
            legend_handles.append(handle)

    # Place legend outside plot to the right
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=11,
        ncol=3,
        frameon=False,
        fancybox=False,
        edgecolor="gray",
        handletextpad=0.3,
        columnspacing=0.8,
        labelspacing=0.3,
    )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    # set tick label fontsize
    ax.tick_params(axis='both', which='major', labelsize=12)

    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"figure2_tsne.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close()
    logger.info("Saved figure2_tsne")


# =============================================================================
# Figure 3: Heatmap
# =============================================================================

def figure_3_heatmap(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    metabolite_categories: pd.DataFrame,
    output_dir: Path,
    min_isolates: int = 1,
    exclude_universal: bool = True,
    max_metabolites: int = 60,
):
    """Figure 3: Genus × metabolite heatmap of distinctive preferences.

    Genera ordered by taxonomy. Class color bar on top.
    Only shows the most distinctive metabolites.
    """
    data = prepare_training_data(df, "consumed")
    microbe_to_idx = data["microbe_to_idx"]
    tax_table = build_taxonomy_table(df, microbe_to_idx)

    genus_profiles = profiles[
        (profiles["level"] == "genus") &
        (profiles["n_isolates"] >= min_isolates)
    ].copy()

    if exclude_universal and metabolite_categories is not None:
        niche_mets = metabolite_categories[
            metabolite_categories["category"] != "universal"
        ]["metabolite_idx"].values
        genus_profiles = genus_profiles[genus_profiles["metabolite_idx"].isin(niche_mets)]

    matrix = genus_profiles.pivot_table(
        index="metabolite_name",
        columns="taxon",
        values="distinctive_score",
        aggfunc="mean",
    )

    # Filter to the most distinctive metabolites:
    # Keep those with the highest variance across genera (most discriminating)
    met_variance = matrix.var(axis=1).sort_values(ascending=False)
    top_metabolites = met_variance.head(max_metabolites).index
    matrix = matrix.loc[top_metabolites]

    # Order columns by taxonomy
    taxonomic_order = get_taxonomic_genus_order(tax_table)
    present_genera = [g for g in taxonomic_order if g in matrix.columns]
    missing = [g for g in matrix.columns if g not in present_genera]
    if missing:
        present_genera.extend(sorted(missing))

    matrix = matrix[present_genera]

    # Order rows by clustering
    if matrix.shape[0] > 2:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        row_dist = pdist(matrix.fillna(0).values)
        row_linkage = linkage(row_dist, method="ward")
        row_order = leaves_list(row_linkage)
        metabolite_order = [matrix.index[i] for i in row_order]
    else:
        metabolite_order = list(matrix.index)

    ordered_matrix = matrix.loc[metabolite_order, present_genera]

    # Class info for color bar
    genus_to_class = {}
    for _, row in tax_table.iterrows():
        genus_to_class[row["genus"]] = row.get("class", "Unknown")

    class_colors_for_genera = [
        get_class_color(genus_to_class.get(g, "Unknown"))
        for g in present_genera
    ]

    n_met = len(metabolite_order)
    n_gen = len(present_genera)

    fig_height = max(12, n_met * 0.28)
    fig_width = max(14, n_gen * 0.4 + 4)

    fig = plt.figure(figsize=(fig_width, fig_height))

    left_margin = 0.25
    right_margin = 0.02
    bottom_margin = 0.10
    top_margin = 0.08

    plot_width = 1.0 - left_margin - right_margin
    plot_height = 1.0 - bottom_margin - top_margin

    # Class color bar — tight against the heatmap
    class_bar_height = 0.012
    ax_class = fig.add_axes([
        left_margin,
        1.0 - top_margin + 0.005,
        plot_width,
        class_bar_height,
    ])
    for i, color in enumerate(class_colors_for_genera):
        ax_class.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.9)
    ax_class.set_xlim(-0.5, n_gen - 0.5)
    ax_class.set_yticks([])
    ax_class.set_xticks([])

    # Heatmap
    ax_heat = fig.add_axes([
        left_margin,
        bottom_margin,
        plot_width,
        plot_height,
    ])

    vmax = max(abs(ordered_matrix.min().min()), abs(ordered_matrix.max().max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax_heat.imshow(
        ordered_matrix.values,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )

    ax_heat.set_xticks(range(n_gen))
    ax_heat.set_xticklabels(present_genera, rotation=90, fontsize=20, style="italic")
    ax_heat.set_yticks(range(n_met))
    ax_heat.set_yticklabels(metabolite_order, fontsize=18)

    # Colorbar — top left corner, horizontal
    cbar_width = plot_width * 0.25
    cbar_height = 0.012
    cbar_left = left_margin - 0.125
    cbar_bottom = 1.0 - top_margin + class_bar_height + 0.065

    ax_cb = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cbar.set_label("Distinctive score", fontsize=24, labelpad=5)
    cbar.ax.tick_params(labelsize=24)

    # Class legend — top right corner
    unique_classes = sorted(set(genus_to_class.get(g, "Unknown") for g in present_genera))
    legend_handles = [
        Patch(facecolor=get_class_color(c), edgecolor="gray", label=c)
        for c in unique_classes
    ]

    # Invisible axes for legend placement in top right
    ax_legend = fig.add_axes([
        left_margin + plot_width * 0.15,
        cbar_bottom - 0.01,
        plot_width * 0.65,
        0.01,
    ])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=legend_handles,
        loc="center left",
        fontsize=24,
        ncol=int(np.ceil(len(unique_classes) / 2)),
        frameon=False,
        fancybox=False,
        edgecolor="gray",
        handletextpad=0.3,
        columnspacing=0.8,
    )

    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"figure3_heatmap.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close()
    logger.info("Saved figure3_heatmap")


def figure_3_heatmap_seaborn(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    metabolite_categories: pd.DataFrame,
    output_dir: Path,
    min_isolates: int = 1,
    exclude_universal: bool = True,
    max_metabolites: int = 60,
):
    """Figure 3 (seaborn): Genus × metabolite clustermap with taxonomy color bar."""
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    # --- Data preparation (same as figure_3_heatmap) ---
    data = prepare_training_data(df, "consumed")
    microbe_to_idx = data["microbe_to_idx"]
    tax_table = build_taxonomy_table(df, microbe_to_idx)

    genus_profiles = profiles[
        (profiles["level"] == "genus") &
        (profiles["n_isolates"] >= min_isolates)
    ].copy()

    if exclude_universal and metabolite_categories is not None:
        niche_mets = metabolite_categories[
            metabolite_categories["category"] != "universal"
        ]["metabolite_idx"].values
        genus_profiles = genus_profiles[genus_profiles["metabolite_idx"].isin(niche_mets)]

    matrix = genus_profiles.pivot_table(
        index="metabolite_name",
        columns="taxon",
        values="distinctive_score",
        aggfunc="mean",
    )

    met_variance = matrix.var(axis=1).sort_values(ascending=False)
    top_metabolites = met_variance.head(max_metabolites).index
    matrix = matrix.loc[top_metabolites]

    # --- Taxonomy color mapping ---
    genus_to_class = {}
    for _, row in tax_table.iterrows():
        genus_to_class[row["genus"]] = row.get("class", "Unknown")

    col_colors = pd.Series(
        [get_class_color(genus_to_class.get(g, "Unknown")) for g in matrix.columns],
        index=matrix.columns,
        name="",
    )

    # --- Clustering & normalization ---
    fill_matrix = matrix.fillna(0)

    row_linkage = linkage(pdist(fill_matrix.values, metric="euclidean"), method="ward")
    col_linkage = linkage(pdist(fill_matrix.values.T, metric="euclidean"), method="ward")

    vmax = max(abs(fill_matrix.min().min()), abs(fill_matrix.max().max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # --- Figure sizing ---
    n_met = fill_matrix.shape[0]
    n_gen = fill_matrix.shape[1]
    fig_height = max(12, n_met * 0.28)
    fig_width = max(14, n_gen * 0.4 + 4)

    # --- Clustermap ---
    g = sns.clustermap(
        fill_matrix,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        col_colors=col_colors,
        cmap="RdBu_r",
        norm=norm,
        figsize=(fig_width, fig_height),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.95, 0.03, 0.08),
    )
    # Tick labels on heatmap
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=20, style="italic"
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=18
    )

    # Remove default axis labels or set to manuscript size
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # Colorbar
    g.cax.set_ylabel("Distinctive\nscore", fontsize=20)
    g.cax.tick_params(labelsize=18)

    # Class color strip label
    g.ax_col_colors.set_ylabel("Class", fontsize=22, rotation=0, ha="right", va="center")
    g.ax_col_colors.yaxis.set_tick_params(labelsize=18)

    # Legend
    unique_classes = sorted(set(genus_to_class.get(g, "Unknown") for g in matrix.columns))
    legend_handles = [
        Patch(facecolor=get_class_color(c), edgecolor="gray", label=c)
        for c in unique_classes
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 1.2),
        # two columns
        ncol=int(np.ceil(len(unique_classes) / 2)),
        fontsize=22,
        # ncol=len(unique_classes),
        frameon=False,
    )

    # Dendrogram line width
    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        for c in ax.collections:
            c.set_linewidth(1.5)
            

    for ext in ["png", "pdf"]:
        g.savefig(output_dir / f"figure3_heatmap_biclustering.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure3_heatmap (seaborn)")


def figure_5_raw_score_heatmap(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    metabolite_categories: pd.DataFrame,
    output_dir: Path,
    min_isolates: int = 1,
    exclude_universal: bool = True,
    max_metabolites: int = 60,
):
    """Figure 5: Genus × metabolite clustermap of RAW scores.

    Shows the metabolites present in the most genera.
    """
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    # --- Data preparation ---
    data = prepare_training_data(df, "consumed")
    microbe_to_idx = data["microbe_to_idx"]
    tax_table = build_taxonomy_table(df, microbe_to_idx)

    genus_profiles = profiles[
        (profiles["level"] == "genus") &
        (profiles["n_isolates"] >= min_isolates)
    ].copy()

    if exclude_universal and metabolite_categories is not None:
        niche_mets = metabolite_categories[
            metabolite_categories["category"] != "universal"
        ]["metabolite_idx"].values
        genus_profiles = genus_profiles[genus_profiles["metabolite_idx"].isin(niche_mets)]
    matrix = genus_profiles.pivot_table(
        index="metabolite_name",
        columns="taxon",
        values="raw_score",
        aggfunc="mean",
    )

    # Select metabolites seen in the most genera (most non-NaN columns)
    genera_count = matrix.notna().sum(axis=1).sort_values(ascending=False)
    top_metabolites = genera_count.head(max_metabolites).index
    matrix = matrix.loc[top_metabolites]

    # --- Taxonomy color mapping ---
    genus_to_class = {}
    for _, row in tax_table.iterrows():
        genus_to_class[row["genus"]] = row.get("class", "Unknown")

    col_colors = pd.Series(
        [get_class_color(genus_to_class.get(g, "Unknown")) for g in matrix.columns],
        index=matrix.columns,
        name="",
    )

    # --- Clustering & normalization ---
    fill_matrix = matrix.fillna(0)

    row_linkage = linkage(pdist(fill_matrix.values, metric="euclidean"), method="ward")
    col_linkage = linkage(pdist(fill_matrix.values.T, metric="euclidean"), method="ward")

    vmax = max(abs(fill_matrix.min().min()), abs(fill_matrix.max().max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # --- Figure sizing ---
    n_met = fill_matrix.shape[0]
    n_gen = fill_matrix.shape[1]
    fig_height = max(12, n_met * 0.28)
    fig_width = max(14, n_gen * 0.4 + 4)

    # --- Clustermap ---
    g = sns.clustermap(
        fill_matrix,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        col_colors=col_colors,
        cmap="RdBu_r",
        norm=norm,
        figsize=(fig_width, fig_height),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.95, 0.03, 0.08),
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=20, style="italic"
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=18
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    g.cax.set_ylabel("Raw\nscore", fontsize=20)
    g.cax.tick_params(labelsize=18)

    g.ax_col_colors.set_ylabel("Class", fontsize=22, rotation=0, ha="right", va="center")
    g.ax_col_colors.yaxis.set_tick_params(labelsize=18)

    unique_classes = sorted(set(genus_to_class.get(g, "Unknown") for g in matrix.columns))
    legend_handles = [
        Patch(facecolor=get_class_color(c), edgecolor="gray", label=c)
        for c in unique_classes
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=int(np.ceil(len(unique_classes) / 2)),
        fontsize=22,
        frameon=False,
    )

    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        for c in ax.collections:
            c.set_linewidth(1.5)

    for ext in ["png", "pdf"]:
        g.savefig(output_dir / f"figure5_raw_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure5_raw_heatmap")


def figure_6_rank_clustermap(
    df: pd.DataFrame,
    output_dir: Path,
    embedding_dim: int = 32,
):
    """Figure 6: Clustermap of consumption status by predicted rank.

    Each row is an isolate, each column is a rank position (1 = highest
    predicted preference). Blue = consumed, salmon = neutral. Demonstrates
    that the model consistently ranks consumed metabolites higher.
    """
    import seaborn as sns

    # --- Train model and score all organism-metabolite pairs ---
    data = prepare_training_data(df, "consumed")
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    microbe_to_idx = data["microbe_to_idx"]
    metabolite_to_idx = data["metabolite_to_idx"]

    model = BPRModel(embedding_dim=embedding_dim)
    model.fit(positives, n_microbes, n_metabolites, verbose=False)

    # Score every valid organism-metabolite pair
    mask = (
        df["organism_id"].isin(microbe_to_idx) &
        df["representative_compound"].isin(metabolite_to_idx)
    )
    valid = df[mask].copy()

    m_indices = valid["organism_id"].map(microbe_to_idx).values
    c_indices = valid["representative_compound"].map(metabolite_to_idx).values

    m_emb = model.microbe_embeddings[m_indices]
    c_emb = model.metabolite_embeddings[c_indices]
    valid_scores = (m_emb * c_emb).sum(axis=1)

    df = df.copy()
    df["score"] = None
    df.loc[mask, "score"] = valid_scores

    # --- Build rank matrix ---
    organisms_groups = [
        (org, group)
        for org, group in df.dropna(subset=["score"]).groupby("organism_id")
        if group["assertion"].nunique() > 1
    ]

    rows = []
    for org, group in organisms_groups:
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        consumed = (group["assertion"] == "consumed").astype(int)
        consumed.index = np.arange(1, len(consumed) + 1)
        consumed.name = org
        rows.append(consumed)

    rank_matrix = pd.DataFrame(rows).fillna(0).astype(int)
    rank_matrix.columns.name = "Rank"
    rank_matrix.index.name = "Organism"

    # Drop columns where no organism consumes the metabolite at that rank
    rank_matrix_filtered = rank_matrix.loc[:, rank_matrix.sum(axis=0) > 0]

    # --- Plot ---
    g = sns.clustermap(
        rank_matrix_filtered,
        cmap=["salmon", "steelblue"],
        figsize=(12, 10),
        col_cluster=False,
        yticklabels=False,
        xticklabels=10,
        cbar_kws={"ticks": [0, 1]},
        cbar_pos=(0.8, 0.87, 0.03, 0.061),
        dendrogram_ratio=(0.15, 0.0),
    )

    # Colorbar styling
    g.cax.tick_params(labelsize=16)
    g.cax.set_yticklabels(["Neutral", "Consumed"], fontsize=16)
    for spine in g.cax.spines.values():
        spine.set_visible(True)
    g.cax.set_facecolor("white")

    # Dendrogram line width
    for c in g.ax_row_dendrogram.collections:
        c.set_linewidth(1.5)

    # Axis labels
    g.ax_heatmap.set_xlabel("Rank (1 = highest predicted preference)", fontsize=24)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=18)
    g.ax_heatmap.set_ylabel("Bacterial isolate", labelpad=-850, fontsize=24)

    for ext in ["png", "pdf"]:
        g.savefig(output_dir / f"figure6_rank_clustermap.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure6_rank_clustermap")


def figure_7_embedding_distance(
    df: pd.DataFrame,
    output_dir: Path,
    model_path: Path = Path('./cache/bpr_consumed'),
):
    """Figure 7: Embedding distance distributions by taxonomic level.

    Four-panel histogram showing cosine distances between organism embeddings,
    split by whether pairs share the same genus, family, order, or class.
    """
    wrapper = load_model(model_path, model_class=BPRModel)

    mask = (
        df['organism_id'].isin(wrapper.microbe_to_idx) &
        df['representative_compound'].isin(wrapper.metabolite_to_idx)
    )
    valid = df[mask].copy()
    valid = valid.drop_duplicates(subset=['organism_id'])
    m_indices = valid['organism_id'].map(wrapper.microbe_to_idx).values

    m_emb = wrapper.model.microbe_embeddings[m_indices]

    dist_matrix = pairwise_distances(m_emb, metric='cosine')
    dist_df = pd.DataFrame(
        dist_matrix,
        index=valid['organism_id'],
        columns=valid['organism_id'],
    )
    dist_df.index.name = 'organism_id_1'
    dist_df = dist_df.melt(
        ignore_index=False,
        var_name='organism_id_2',
        value_name='embedding_distance',
    )
    dist_df.reset_index(inplace=True)

    cols = ['organism_id_1', 'organism_id_2']
    dist_df = dist_df[dist_df[cols[0]] != dist_df[cols[1]]]
    idx = np.sort(dist_df[cols].values, axis=1)
    dist_df[cols] = idx
    dist_df = dist_df.drop_duplicates(subset=cols)

    merged = dist_df.merge(valid.add_suffix('_1'), on='organism_id_1', how='left')
    merged = merged.merge(valid.add_suffix('_2'), on='organism_id_2', how='left')

    levels = ['genus', 'family', 'order', 'class']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, sharex=True)
    axes = axes.flatten()
    edges = np.linspace(
        merged['embedding_distance'].min(),
        merged['embedding_distance'].max(),
        51,
    )

    for ax, level in zip(axes, levels):
        col1 = f'{level}_1'
        col2 = f'{level}_2'
        same = merged[merged[col1] == merged[col2]]
        diff = merged[merged[col1] != merged[col2]]
        median_same = same['embedding_distance'].median()
        median_diff = diff['embedding_distance'].median()
        p_value = ttest_ind(
            same['embedding_distance'],
            diff['embedding_distance'],
            equal_var=False,
        ).pvalue

        ax.hist(
            same['embedding_distance'], bins=edges,
            alpha=0.7, label=f'same {level}', density=True,
        )
        ax.hist(
            diff['embedding_distance'], bins=edges,
            alpha=0.7, label=f'different {level}', density=True,
        )
        ax.axvline(median_same, color='blue', linestyle='dashed', linewidth=3)
        ax.axvline(median_diff, color='orange', linestyle='dashed', linewidth=3)
        ax.legend(fontsize=16, frameon=False)
        ax.text(
            0.8, 0.6, f'p = {p_value:.2e}',
            transform=ax.transAxes, fontsize=16, ha='center',
        )

    axes[-1].set_xlabel('Embedding distance (cosine)', fontsize=24)
    axes[-2].set_xlabel('Embedding distance (cosine)', fontsize=24)
    axes[0].set_ylabel('Density', fontsize=24)
    axes[2].set_ylabel('Density', fontsize=24)

    panel_labels = ['a.', 'b.', 'c.', 'd.']
    for a, label in zip(axes, panel_labels):
        a.tick_params(axis='both', which='major', labelsize=20)
        for spine in a.spines.values():
            spine.set_linewidth(1.5)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.text(
            0.1, 1.0, label, transform=a.transAxes,
            fontsize=36, va='top', ha='right',
        )

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            output_dir / f"figure7_embedding_distance.{ext}",
            dpi=300, bbox_inches="tight",
        )
    plt.close()
    logger.info("Saved figure7_embedding_distance")
# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate publication figures from pre-computed results"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("cache/exometabolite.parquet"),
        help="Path to exometabolite parquet file",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("figures"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript_figures"),
        help="Output directory for publication figures",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension (for t-SNE retraining)",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Which figures to generate (1-7)",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} rows")

    # Figure 1: Combined benchmark (model comparison + holdout)
    if 1 in args.figures:
        core_path = args.results_dir / "core_results_consumed.csv"
        holdout_path = args.results_dir / "unified_holdout_results_consumed.csv"

        if core_path.exists() and holdout_path.exists():
            logger.info("Generating Figure 1: Combined Benchmark")
            core_results = pd.read_csv(core_path)
            holdout_results = pd.read_csv(holdout_path)
            figure_1_benchmark(core_results, holdout_results, args.output_dir)
        else:
            missing = []
            if not core_path.exists():
                missing.append(str(core_path))
            if not holdout_path.exists():
                missing.append(str(holdout_path))
            logger.warning(f"Missing {', '.join(missing)}, skipping Figure 1")

    # Figure 2: t-SNE
    if 2 in args.figures:
        logger.info("Generating Figure 2: t-SNE")
        figure_2_tsne(df, args.output_dir, embedding_dim=args.embedding_dim)

    # Figure 3: Heatmap
    if 3 in args.figures:
        profiles_path = args.results_dir / "taxon_profiles_consumed.parquet"
        categories_path = args.results_dir / "metabolite_categories_consumed.csv"

        if profiles_path.exists() and categories_path.exists():
            logger.info("Generating Figure 3: Heatmap")
            profiles = pd.read_parquet(profiles_path)
            categories = pd.read_csv(categories_path)
            figure_3_heatmap(
                df, profiles, categories, args.output_dir,
                min_isolates=1,
                exclude_universal=True,
            )
        else:
            logger.warning(f"Missing profiles or categories, skipping Figure 3")
    if 4 in args.figures:
        profiles_path = args.results_dir / "taxon_profiles_consumed.parquet"
        categories_path = args.results_dir / "metabolite_categories_consumed.csv"

        if profiles_path.exists() and categories_path.exists():
            logger.info("Generating Figure 3 (seaborn version)")
            profiles = pd.read_parquet(profiles_path)
            categories = pd.read_csv(categories_path)
            figure_3_heatmap_seaborn(
                df, profiles, categories, args.output_dir,
                min_isolates=1,
                exclude_universal=True,
            )
        else:
            logger.warning("Missing profiles or categories, skipping Figure 3 seaborn")
    if 5 in args.figures:
        profiles_path = args.results_dir / "taxon_profiles_consumed.parquet"
        categories_path = args.results_dir / "metabolite_categories_consumed.csv"

        if profiles_path.exists() and categories_path.exists():
            logger.info("Generating Figure 5: Raw score heatmap")
            profiles = pd.read_parquet(profiles_path)
            categories = pd.read_csv(categories_path)
            figure_5_raw_score_heatmap(
                df, profiles, categories, args.output_dir,
                min_isolates=1,
                exclude_universal=True,
            )
        else:
            logger.warning("Missing profiles or categories, skipping Figure 5")
    if 6 in args.figures:
        logger.info("Generating Figure 6: Rank clustermap")
        figure_6_rank_clustermap(df, args.output_dir, embedding_dim=args.embedding_dim)
    if 7 in args.figures:
        logger.info("Generating Figure 7: Embedding distance by taxonomy")
        model_path = Path('./cache/bpr_consumed')
        if model_path.exists():
            figure_7_embedding_distance(df, args.output_dir, model_path=model_path)
        else:
            logger.warning(f"Missing model at {model_path}, skipping Figure 7")