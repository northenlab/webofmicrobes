# taxon_metabolite_profiles.py
"""Compute taxonomically resolved metabolite preference profiles.

For each taxon at each level, produces:
  - raw_score: taxon_embedding @ metabolite_embedding
  - global_score: global_mean_embedding @ metabolite_embedding (popularity)
  - distinctive_score: residual after projecting out the global direction

Outputs:
  - Full scores table (long format)
  - "Distinctive metabolites" per taxon (top/bottom N by distinctive_score)
  - Summary heatmaps at each taxonomic level
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import BPRModel, prepare_training_data

logger = logging.getLogger(__name__)


def compute_global_direction(microbe_embeddings: np.ndarray) -> np.ndarray:
    """Compute the unit vector representing the 'average organism'.

    Metabolites with high projection onto this direction are universally
    preferred. Metabolites orthogonal to it are niche-specific.
    """
    global_mean = microbe_embeddings.mean(axis=0)
    norm = np.linalg.norm(global_mean)
    if norm < 1e-10:
        return global_mean
    return global_mean / norm


def project_out(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component of vector along direction (unit vector)."""
    return vector - np.dot(vector, direction) * direction


def compute_taxon_profiles(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
    levels: list[str] = None,
) -> pd.DataFrame:
    """Compute metabolite preference profiles for every taxon at every level.

    Args:
        df: Exometabolite dataframe
        preference_type: "consumed" or "secreted"
        embedding_dim: BPR embedding dimension
        levels: Taxonomic levels to compute profiles for

    Returns:
        DataFrame with columns:
            level, taxon, n_isolates, metabolite_idx, metabolite_name,
            raw_score, global_score, distinctive_score, prevalence
    """
    if levels is None:
        levels = ["class", "order", "family", "genus"]

    data = prepare_training_data(df, preference_type)
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    microbe_to_idx = data["microbe_to_idx"]
    microbe_taxonomy = data["microbe_taxonomy"]
    metabolite_to_idx = data["metabolite_to_idx"]
    idx_to_metabolite = {i: m for m, i in metabolite_to_idx.items()}

    # Train model
    logger.info(f"Training BPR (dim={embedding_dim})...")
    model = BPRModel(embedding_dim=embedding_dim)
    model.fit(positives, n_microbes, n_metabolites, verbose=False)

    microbe_embeddings = model.microbe_embeddings
    metabolite_embeddings = model.metabolite_embeddings

    # Global direction
    global_dir = compute_global_direction(microbe_embeddings)
    global_scores = metabolite_embeddings @ global_dir

    # Compute prevalence for each metabolite
    metabolite_prevalence = np.zeros(n_metabolites)
    for pos_set in positives.values():
        for m in pos_set:
            metabolite_prevalence[m] += 1
    metabolite_prevalence /= n_microbes

    # Group microbes by taxon at each level
    logger.info("Computing taxon profiles...")
    results = []

    for level in levels:
        taxon_groups = {}
        for org_id, idx in microbe_to_idx.items():
            tax = microbe_taxonomy.get(org_id, {})
            taxon = tax.get(level)
            if taxon and pd.notna(taxon):
                if taxon not in taxon_groups:
                    taxon_groups[taxon] = []
                taxon_groups[taxon].append(idx)

        logger.info(f"  {level}: {len(taxon_groups)} taxa")

        for taxon_name, member_indices in taxon_groups.items():
            # Taxon embedding = mean of member embeddings
            taxon_embedding = microbe_embeddings[member_indices].mean(axis=0)

            # Raw scores
            raw_scores = metabolite_embeddings @ taxon_embedding

            # Distinctive scores: project out global direction from taxon embedding
            taxon_distinctive = project_out(taxon_embedding, global_dir)
            distinctive_scores = metabolite_embeddings @ taxon_distinctive

            # Also compute observed prevalence within this taxon
            taxon_positives = np.zeros(n_metabolites)
            for idx in member_indices:
                for m in positives.get(idx, set()):
                    taxon_positives[m] += 1
            taxon_prevalence = taxon_positives / len(member_indices)

            for met_idx in range(n_metabolites):
                results.append({
                    "level": level,
                    "taxon": taxon_name,
                    "n_isolates": len(member_indices),
                    "metabolite_idx": met_idx,
                    "metabolite_name": idx_to_metabolite[met_idx],
                    "raw_score": raw_scores[met_idx],
                    "global_score": global_scores[met_idx],
                    "distinctive_score": distinctive_scores[met_idx],
                    "global_prevalence": metabolite_prevalence[met_idx],
                    "taxon_prevalence": taxon_prevalence[met_idx],
                })

    return pd.DataFrame(results)


def extract_distinctive_metabolites(
    profiles_df: pd.DataFrame,
    top_n: int = 10,
    min_isolates: int = 3,
) -> pd.DataFrame:
    """Extract the most distinctive metabolites for each taxon.

    For each taxon, returns the top_n metabolites by distinctive_score
    (uniquely preferred) and bottom top_n (uniquely avoided).

    Args:
        profiles_df: Output of compute_taxon_profiles
        top_n: Number of top/bottom metabolites per taxon
        min_isolates: Minimum isolates in a taxon to include

    Returns:
        DataFrame with distinctive metabolites per taxon
    """
    filtered = profiles_df[profiles_df["n_isolates"] >= min_isolates].copy()

    results = []

    for (level, taxon), group in filtered.groupby(["level", "taxon"]):
        sorted_group = group.sort_values("distinctive_score", ascending=False)

        # Top N: uniquely preferred
        top = sorted_group.head(top_n).copy()
        top["direction"] = "preferred"
        top["rank"] = range(1, len(top) + 1)
        results.append(top)

        # Bottom N: uniquely avoided
        bottom = sorted_group.tail(top_n).copy()
        bottom["direction"] = "avoided"
        bottom["rank"] = range(1, len(bottom) + 1)
        results.append(bottom)

    return pd.concat(results, ignore_index=True)


def compute_metabolite_categories(
    profiles_df: pd.DataFrame,
    popularity_threshold: float = 0.7,
    distinctive_threshold: float = None,
) -> pd.DataFrame:
    """Categorize metabolites into universal, niche-specific, and rare.

    Categories:
      - universal: global_prevalence > popularity_threshold
      - niche: at least one taxon has |distinctive_score| > threshold
      - rare: low prevalence, low distinctive scores everywhere

    Args:
        profiles_df: Output of compute_taxon_profiles
        popularity_threshold: Prevalence above which a metabolite is "universal"
        distinctive_threshold: If None, uses 1 std above mean |distinctive_score|

    Returns:
        DataFrame with one row per metabolite and its category
    """
    # Work at one level (genus) to determine categories
    genus_profiles = profiles_df[profiles_df["level"] == "genus"].copy()

    if len(genus_profiles) == 0:
        genus_profiles = profiles_df.copy()

    # Per-metabolite summary
    met_summary = genus_profiles.groupby(["metabolite_idx", "metabolite_name"]).agg(
        global_prevalence=("global_prevalence", "first"),
        max_distinctive=("distinctive_score", "max"),
        min_distinctive=("distinctive_score", "min"),
        std_distinctive=("distinctive_score", "std"),
        max_taxon_prevalence=("taxon_prevalence", "max"),
    ).reset_index()

    met_summary["abs_max_distinctive"] = met_summary[["max_distinctive", "min_distinctive"]].abs().max(axis=1)

    if distinctive_threshold is None:
        distinctive_threshold = met_summary["abs_max_distinctive"].mean() + met_summary["abs_max_distinctive"].std()

    def categorize(row):
        if row["global_prevalence"] >= popularity_threshold:
            return "universal"
        elif row["abs_max_distinctive"] >= distinctive_threshold:
            return "niche_specific"
        else:
            return "rare"

    met_summary["category"] = met_summary.apply(categorize, axis=1)

    logger.info(f"Metabolite categories: {met_summary['category'].value_counts().to_dict()}")

    return met_summary


def build_heatmap_matrix(
    profiles_df: pd.DataFrame,
    level: str = "family",
    score_type: str = "distinctive_score",
    min_isolates: int = 3,
    exclude_universal: bool = True,
    metabolite_categories: pd.DataFrame = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a taxon × metabolite matrix for heatmap plotting.

    Args:
        profiles_df: Output of compute_taxon_profiles
        level: Taxonomic level for columns
        score_type: Which score to use ("raw_score", "distinctive_score")
        min_isolates: Minimum isolates per taxon
        exclude_universal: Whether to exclude universal metabolites
        metabolite_categories: Output of compute_metabolite_categories

    Returns:
        (matrix_df, taxon_order, metabolite_order)
    """
    level_df = profiles_df[
        (profiles_df["level"] == level) &
        (profiles_df["n_isolates"] >= min_isolates)
    ].copy()

    if exclude_universal and metabolite_categories is not None:
        niche_mets = metabolite_categories[
            metabolite_categories["category"] != "universal"
        ]["metabolite_idx"].values
        level_df = level_df[level_df["metabolite_idx"].isin(niche_mets)]

    # Pivot to matrix
    matrix = level_df.pivot_table(
        index="metabolite_name",
        columns="taxon",
        values=score_type,
        aggfunc="mean",
    )

    # Order by clustering
    if matrix.shape[0] > 1 and matrix.shape[1] > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        # Cluster metabolites (rows)
        if matrix.shape[0] > 2:
            row_dist = pdist(matrix.fillna(0).values)
            row_linkage = linkage(row_dist, method="ward")
            row_order = leaves_list(row_linkage)
            metabolite_order = [matrix.index[i] for i in row_order]
        else:
            metabolite_order = list(matrix.index)

        # Cluster taxa (columns)
        if matrix.shape[1] > 2:
            col_dist = pdist(matrix.fillna(0).values.T)
            col_linkage = linkage(col_dist, method="ward")
            col_order = leaves_list(col_linkage)
            taxon_order = [matrix.columns[i] for i in col_order]
        else:
            taxon_order = list(matrix.columns)
    else:
        metabolite_order = list(matrix.index)
        taxon_order = list(matrix.columns)

    return matrix, taxon_order, metabolite_order


def plot_preference_heatmap(
    matrix: pd.DataFrame,
    taxon_order: list[str],
    metabolite_order: list[str],
    output_path: Path,
    title: str = "Metabolite Preferences",
    score_type: str = "distinctive_score",
):
    """Plot a taxon × metabolite heatmap.

    Args:
        matrix: Pivot table from build_heatmap_matrix
        taxon_order: Column order
        metabolite_order: Row order
        output_path: Path to save figure
        title: Figure title
        score_type: For colorbar label
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ordered_matrix = matrix.loc[metabolite_order, taxon_order]

    fig_height = max(6, len(metabolite_order) * 0.25)
    fig_width = max(8, len(taxon_order) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Center colormap at 0 for distinctive scores
    if score_type == "distinctive_score":
        vmax = max(abs(ordered_matrix.min().min()), abs(ordered_matrix.max().max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = "RdBu_r"
    else:
        norm = None
        cmap = "YlOrRd"

    im = ax.imshow(
        ordered_matrix.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(taxon_order)))
    ax.set_xticklabels(taxon_order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(metabolite_order)))
    ax.set_yticklabels(metabolite_order, fontsize=7)
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    label = "Distinctive score" if score_type == "distinctive_score" else "Raw score"
    cbar.set_label(label)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(
            output_path.with_suffix(f".{ext}"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.close()
    logger.info(f"Saved {output_path.stem}.png/pdf")


def run_taxon_profiles(
    df: pd.DataFrame,
    output_dir: Path,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
):
    """Run full taxon profile analysis and generate outputs.

    Produces:
      - taxon_profiles_{type}.parquet: Full scores table
      - distinctive_metabolites_{type}.csv: Top/bottom metabolites per taxon
      - metabolite_categories_{type}.csv: Universal/niche/rare classification
      - heatmap_{level}_{type}.png/pdf: Heatmaps at each level
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute profiles
    profiles = compute_taxon_profiles(
        df,
        preference_type=preference_type,
        embedding_dim=embedding_dim,
    )
    profiles.to_parquet(output_dir / f"taxon_profiles_{preference_type}.parquet")
    logger.info(f"Saved taxon_profiles_{preference_type}.parquet ({len(profiles)} rows)")

    # Categorize metabolites
    categories = compute_metabolite_categories(profiles)
    categories.to_csv(output_dir / f"metabolite_categories_{preference_type}.csv", index=False)

    # Extract distinctive metabolites
    distinctive = extract_distinctive_metabolites(profiles, top_n=10, min_isolates=3)
    distinctive.to_csv(output_dir / f"distinctive_metabolites_{preference_type}.csv", index=False)
    logger.info(f"Saved distinctive_metabolites_{preference_type}.csv ({len(distinctive)} rows)")

    # Generate heatmaps at each level
    for level in ["class", "order", "family", "genus"]:
        for score_type, suffix, exclude in [
            ("distinctive_score", "distinctive", True),
            ("raw_score", "raw", False),
        ]:
            matrix, taxon_order, met_order = build_heatmap_matrix(
                profiles,
                level=level,
                score_type=score_type,
                min_isolates=3,
                exclude_universal=exclude,
                metabolite_categories=categories,
            )

            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                logger.warning(f"  {level}/{suffix}: matrix too small ({matrix.shape}), skipping")
                continue

            plot_preference_heatmap(
                matrix, taxon_order, met_order,
                output_path=output_dir / f"heatmap_{level}_{suffix}_{preference_type}",
                title=f"{level.capitalize()}-level {suffix} preferences ({preference_type})",
                score_type=score_type,
            )

    return profiles, categories, distinctive


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
        description="Compute taxonomically resolved metabolite preference profiles"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("cache/exometabolite.parquet"),
        help="Path to exometabolite parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory",
    )
    parser.add_argument(
        "--preference-type",
        choices=["consumed", "secreted"],
        default="consumed",
        help="Which preference type to analyze",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="BPR embedding dimension",
    )

    args = parser.parse_args()

    logger.info(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} rows")

    run_taxon_profiles(
        df,
        output_dir=args.output_dir,
        preference_type=args.preference_type,
        embedding_dim=args.embedding_dim,
    )