"""make_figure3_heatmap.py — Figure 3 heatmap for the WoM manuscript.

Generates a single figure: genus × metabolite clustermap of raw BPR scores,
compounds ranked by how many isolates were experimentally tested.

Usage:
    cd wom/
    python scripts/make_figure3_heatmap.py
    python scripts/make_figure3_heatmap.py --model-dir trained_models/ordinal_bpr_consumed
    python scripts/make_figure3_heatmap.py --max-metabolites 80 --max-genera 60
"""

import re
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from core.io import load_model_artifacts
from core.types import AssertionType
from loaders.loaders import prepare_training_data, load_exometabolite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy class colors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Culture condition colors
# ---------------------------------------------------------------------------

ANAEROBIC_COLOR = "#2196F3"   # blue  — CZ-Biohub (nature_2021, anaerobic)
AEROBIC_COLOR   = "#FF9800"   # orange — Berkeley Lab (aerobic)
MIXED_COLOR     = "#9C27B0"   # purple — genera present in both collections
UNKNOWN_COND    = "#aaaaaa"

CLASS_COLORS = {
    "Actinomycetes":       "#1f77b4",
    "Bacilli":             "#ff7f0e",
    "Alphaproteobacteria": "#2ca02c",
    "Betaproteobacteria":  "#9467bd",
    "Gammaproteobacteria": "#8c564b",
    "Flavobacteriia":      "#e377c2",
    "Sphingobacteriia":    "#7f7f7f",
    "Cytophagia":          "#bcbd22",
    "Cyanophyceae":        "#17becf",
    "Actinobacteria":      "#3498db",
    "Bacteroidia":         "#e74c3c",
    "Chitinophagia":       "#f39c12",
    "Clostridia":          "#16a085",
    "Coriobacteriia":      "#9b59b6",
    "Deltaproteobacteria": "#34495e",
    "Erysipelotrichia":    "#e67e22",
    "Fusobacteriia":       "#1abc9c",
    "Negativicutes":       "#d35400",
    "Tissierellia":        "#c0392b",
    "Verrucomicrobiae":    "#27ae60",
}
DEFAULT_CLASS_COLOR = "#aaaaaa"


def get_class_color(class_name: str) -> str:
    for key, color in CLASS_COLORS.items():
        if key.lower() in str(class_name).lower():
            return color
    return DEFAULT_CLASS_COLOR


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def strip_stereo_prefix(name: str) -> str:
    """Remove D-/L- stereochemistry prefixes from a compound name.

    Handles leading prefixes ("D-glucose" → "glucose") and mid-string
    occurrences after a hyphen ("N-acetyl-D-glutamate" → "N-acetyl-glutamate").
    """
    name = re.sub(r"^[DL]-", "", name)
    name = re.sub(r"-[DL]-", "-", name)
    return name


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def build_genus_matrix(
    df: pd.DataFrame,
    model,
    training_data,
    max_metabolites: int = 60,
    max_genera: int | None = None,
) -> pd.DataFrame:
    """Build genus × metabolite matrix of raw BPR scores.

    Rows   = top ``max_metabolites`` compounds ranked by number of isolates
             that were experimentally tested (any assertion including neutral).
    Columns = genera that have at least one consumed compound in training data,
              optionally capped at ``max_genera`` (most-measured genera first).
    """
    microbe_to_idx = training_data.microbe_to_idx
    idx_to_metabolite = training_data.idx_to_metabolite
    microbe_taxonomy = training_data.microbe_taxonomy

    # Genera with at least one consumed training positive
    genera_with_positives: set[str] = set()
    for microbe_id, microbe_idx in microbe_to_idx.items():
        if training_data.positives.get(microbe_idx):
            tax = microbe_taxonomy.get(microbe_id)
            if tax and tax.genus:
                genera_with_positives.add(tax.genus)

    # Map each isolate index → genus
    idx_to_genus: dict[int, str] = {}
    for microbe_id, microbe_idx in microbe_to_idx.items():
        tax = microbe_taxonomy.get(microbe_id)
        if tax and tax.genus and tax.genus in genera_with_positives:
            idx_to_genus[microbe_idx] = tax.genus

    # Compute genus-mean raw scores: microbe_emb @ metabolite_emb.T averaged per genus
    genus_scores: dict[str, list[np.ndarray]] = {}
    for microbe_idx, genus in idx_to_genus.items():
        scores = model.predict_scores(microbe_idx)
        genus_scores.setdefault(genus, []).append(scores)

    genus_mean: dict[str, np.ndarray] = {
        g: np.stack(vecs).mean(axis=0) for g, vecs in genus_scores.items()
    }

    n_metabolites = training_data.n_metabolites
    metabolite_names = [idx_to_metabolite[i] for i in range(n_metabolites)]

    matrix = pd.DataFrame(genus_mean, index=metabolite_names)
    matrix.index.name = "metabolite"
    matrix.columns.name = "genus"

    # --- Select rows: top max_metabolites by isolates experimentally tested ---
    isolate_counts = (
        df.dropna(subset=["assertion_value"])
        .groupby("representative_compound")["organism_id"]
        .nunique()
    )
    top_metabolites = (
        isolate_counts
        .reindex(matrix.index)
        .sort_values(ascending=False)
        .head(max_metabolites)
        .index
    )
    matrix = matrix.loc[top_metabolites]

    # --- Optionally cap columns: most-measured genera ---
    if max_genera is not None and len(matrix.columns) > max_genera:
        genus_isolate_counts = (
            df[df["genus"].isin(matrix.columns)]
            .groupby("genus")["organism_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        top_genera = genus_isolate_counts.head(max_genera).index
        matrix = matrix[[g for g in matrix.columns if g in top_genera]]

    return matrix


def build_col_colors(genera: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Two color tracks per genus: taxonomy class and culture condition.

    Culture condition:
        nature_2021_wom_format.tsv → CZ-Biohub → anaerobic (blue)
        all other wom_* files      → Berkeley Lab → aerobic (orange)
        genera present in both     → mixed (purple)
    """
    # --- Class track ---
    genus_to_class: dict[str, str] = {}
    if "genus" in df.columns and "class" in df.columns:
        for _, row in df[["genus", "class"]].dropna().drop_duplicates().iterrows():
            genus_to_class[row["genus"]] = row["class"]
    class_colors = [get_class_color(genus_to_class.get(g, "Unknown")) for g in genera]

    # --- Culture condition track ---
    czb_genera = set(
        df[df["source_file"] == "nature_2021_wom_format.tsv"]["genus"].dropna().unique()
    )
    lbnl_sources = [s for s in df["source_file"].unique() if "nature" not in s]
    lbnl_genera = set(df[df["source_file"].isin(lbnl_sources)]["genus"].dropna().unique())

    culture_colors = []
    for g in genera:
        in_czb  = g in czb_genera
        in_lbnl = g in lbnl_genera
        if in_czb and in_lbnl:
            culture_colors.append(MIXED_COLOR)
        elif in_czb:
            culture_colors.append(ANAEROBIC_COLOR)
        elif in_lbnl:
            culture_colors.append(AEROBIC_COLOR)
        else:
            culture_colors.append(UNKNOWN_COND)

    return pd.DataFrame(
        {"Class": class_colors, "Culture condition": culture_colors},
        index=genera,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _render_clustermap(
    matrix: pd.DataFrame,
    col_colors_df: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: Path,
    figure_name: str,
    figsize: tuple[float, float] | None = None,
    width_scale: float = 1.0,
    legend_anchor_y: float = 1.08,
    cbar_pos: tuple[float, float, float, float] = (0.02, 0.95, 0.03, 0.08),
    dendrogram_linewidth: float = 1.0,
) -> None:
    """Render and save a single genus × metabolite clustermap (genera as rows).

    If figsize is None, size is computed from the data: 0.4 in per genus (height)
    and 0.28 in per metabolite (width), matching the original per-item dimensions.
    """
    fill_matrix = matrix.fillna(0)  # metabolites × genera

    # Compute linkages before transposing
    met_linkage   = linkage(pdist(fill_matrix.values,   metric="euclidean"), method="ward")
    genus_linkage = linkage(pdist(fill_matrix.values.T, metric="euclidean"), method="ward")

    # Transpose so genera are rows, metabolites are columns
    fill_matrix_T = fill_matrix.T

    n_gen = fill_matrix_T.shape[0]
    n_met = fill_matrix_T.shape[1]
    if figsize is None:
        figsize = (max(10, n_met * 0.28 + 4) * width_scale * 2, max(8, n_gen * 0.4 + 4))

    vmax = max(abs(fill_matrix_T.values.min()), abs(fill_matrix_T.values.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    g = sns.clustermap(
        fill_matrix_T,
        row_linkage=genus_linkage,
        col_linkage=met_linkage,
        row_colors=col_colors_df,
        cmap="RdBu_r",
        norm=norm,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=cbar_pos,
        tree_kws={"linewidths": dendrogram_linewidth},
    )

    # x-axis = metabolites (strip D-/L- stereo prefixes)
    xlabels = [strip_stereo_prefix(t.get_text()) for t in g.ax_heatmap.get_xticklabels()]
    g.ax_heatmap.set_xticklabels(xlabels, rotation=90, fontsize=32)
    # y-axis = genera (italic)
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=24, style="italic"
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    g.cax.set_ylabel("Raw\nscore", fontsize=36)
    g.cax.tick_params(labelsize=32)
    g.ax_row_colors.xaxis.set_tick_params(labelsize=32)

    # --- Legends ---
    tax_table = (
        df[["genus", "class"]].dropna().drop_duplicates()
        if "class" in df.columns else pd.DataFrame(columns=["genus", "class"])
    )
    genus_to_class = dict(zip(tax_table["genus"], tax_table["class"]))
    unique_classes = sorted(set(genus_to_class.get(gn, "Unknown") for gn in matrix.columns))

    class_handles = [
        Patch(facecolor=get_class_color(c), edgecolor="gray", label=c)
        for c in unique_classes
    ]

    # Cap at 4 columns so the class legend doesn't spread into the culture legend
    n_class_cols = min(int(np.ceil(len(unique_classes) / 3)), 6)

    class_leg = g.ax_col_dendrogram.legend(
        handles=class_handles,
        title="Taxonomy class",
        title_fontsize=36,
        loc="upper left",
        bbox_to_anchor=(0.10, legend_anchor_y),
        bbox_transform=g.fig.transFigure,
        ncol=n_class_cols,
        fontsize=32,
        frameon=False,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    g.ax_col_dendrogram.add_artist(class_leg)

    # Determine which culture condition labels are actually present
    czb_genera  = set(df[df["source_file"] == "nature_2021_wom_format.tsv"]["genus"].dropna().unique())
    lbnl_sources = [s for s in df["source_file"].unique() if "nature" not in s]
    lbnl_genera = set(df[df["source_file"].isin(lbnl_sources)]["genus"].dropna().unique())

    culture_handles = []
    for genus in matrix.columns:
        in_czb  = genus in czb_genera
        in_lbnl = genus in lbnl_genera
        if in_czb and not in_lbnl:
            culture_handles.append(("anaerobic", ANAEROBIC_COLOR))
        elif in_lbnl and not in_czb:
            culture_handles.append(("aerobic", AEROBIC_COLOR))
        elif in_czb and in_lbnl:
            culture_handles.append(("mixed", MIXED_COLOR))
    seen: set[str] = set()
    unique_culture = [
        Patch(facecolor=color, edgecolor="gray", label=label)
        for label, color in culture_handles
        if label not in seen and not seen.add(label)  # type: ignore[func-returns-value]
    ]

    # Place culture condition to the right of the taxonomy legend, just below it —
    # both legends sit above the figure top in the header white space.
    cond_leg = g.ax_col_dendrogram.legend(
        handles=unique_culture,
        title="Culture condition",
        title_fontsize=36,
        loc="upper right",
        bbox_to_anchor=(1.0, legend_anchor_y - 0.08),
        bbox_transform=g.fig.transFigure,
        ncol=1,
        fontsize=32,
        frameon=False,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    # bbox_extra_artists ensures add_artist legends are included in tight bbox
    dpi_by_ext = {"png": 300, "pdf": 600}
    for ext in ["png", "pdf"]:
        g.savefig(output_dir / f"{figure_name}.{ext}", dpi=dpi_by_ext[ext],
                  bbox_inches="tight", bbox_extra_artists=[class_leg, cond_leg])
    plt.close()
    logger.info(f"Saved {figure_name}")


def make_figure3(
    df: pd.DataFrame,
    model,
    training_data,
    output_dir: Path,
    max_metabolites: int = 60,
    max_genera: int | None = None,
    figure_name: str = "figure3_heatmap",
):
    """Genus × metabolite clustermap of raw BPR scores (all genera combined)."""
    matrix = build_genus_matrix(
        df, model, training_data,
        max_metabolites=max_metabolites,
        max_genera=max_genera,
    )
    logger.info(f"Matrix: {matrix.shape[0]} metabolites × {matrix.shape[1]} genera")

    col_colors_df = build_col_colors(list(matrix.columns), df)
    _render_clustermap(
        matrix,
        col_colors_df,
        df,
        output_dir,
        figure_name=figure_name,
        dendrogram_linewidth=3.0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Generate Figure 3 heatmap")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("trained_models/ordinal_bpr_consumed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript_figures_nat_microbiology_first_submission"),
    )
    parser.add_argument("--max-metabolites", type=int, default=60)
    parser.add_argument("--max-genera",      type=int, default=None)
    parser.add_argument("--figure-name",     type=str, default="figure3_heatmap")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {args.model_dir}")
    artifacts = load_model_artifacts(args.model_dir)
    model         = artifacts.model
    training_data = artifacts.training_data
    logger.info(
        f"Loaded {type(model).__name__} — "
        f"{training_data.n_microbes} microbes, {training_data.n_metabolites} metabolites"
    )

    logger.info("Loading exometabolite data")
    df = load_exometabolite()

    make_figure3(
        df=df,
        model=model,
        training_data=training_data,
        output_dir=args.output_dir,
        max_metabolites=args.max_metabolites,
        max_genera=args.max_genera,
        figure_name=args.figure_name,
    )
