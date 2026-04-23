# make_figures_for_manuscript_ordinal.py
"""Generate publication figures using the Ordinal BPR model.

Same as make_figures_for_manuscript.py but defaults to ordinal_bpr_consumed /
ordinal_bpr_secreted and includes Ordinal-BPR in cross-model benchmarks.

Usage:
    python make_figures_for_manuscript_ordinal.py --output-dir manuscript_figures/
    python make_figures_for_manuscript_ordinal.py --model-dir ../trained_models/ordinal_bpr_consumed --output-dir manuscript_figures/
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_score, recall_score

from core.io import load_model_artifacts
from core.training import train_model
from core.types import ModelArtifacts, AssertionType, TrainingData
from core.models import BPRModel, OrdinalBPRModel
from loaders.loaders import prepare_training_data, load_exometabolite
from loaders.config import VALIDATION_DATA_FILE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Held-out validation isolates
# VALIDATION_DATA_FILE is the single source of truth — it contains both the
# taxonomy and the measurement values for every held-out isolate.  To add a
# new isolate, add a row to that TSV; no code change is needed here.
# Do NOT add held-out isolates to exometabolite_data/phylogeny.tsv or any
# wom_*.tsv source file, as those feed model training.
# ---------------------------------------------------------------------------

# Consistent style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})
from scripts.taxonomic_group_analysis import CLASS_COLORS

def compute_model_metrics(model, training_data):
    """Compute evaluation metrics for a model on training data.
    
    Returns:
        dict with 'mean_percentile' and 'auc' metrics
    """
    from sklearn.metrics import roc_auc_score
    
    n_microbes = training_data.n_microbes
    positives = training_data.positives
    
    percentile_ranks = []
    auc_scores = []
    
    for m_idx in range(n_microbes):
        if m_idx not in positives or len(positives[m_idx]) == 0:
            continue
        
        # Get scores for this microbe
        scores = model.predict_scores(m_idx)
        
        # Handle invalid scores
        if np.all(np.isnan(scores)) or np.all(np.isinf(scores)):
            continue
        
        # Replace nan/inf with 0
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute ranks (higher score = better rank)
        # argsort gives indices that would sort the array
        # argsort again gives the rank (0 = lowest, n-1 = highest)
        ranks = np.argsort(np.argsort(-scores))  # 0-indexed ranks
        
        # Get percentiles for positive items
        pos_indices = list(positives[m_idx])
        if any(idx >= len(scores) for idx in pos_indices):
            continue  # Skip if indices out of bounds
        
        pos_ranks = ranks[pos_indices]
        pos_percentiles = 100 * (1 - pos_ranks / len(scores))
        
        percentile_ranks.extend(pos_percentiles)
        
        # Compute AUC
        y_true = np.zeros(len(scores))
        y_true[pos_indices] = 1
        
        # Only compute AUC if we have both positives and negatives
        if np.sum(y_true) > 0 and np.sum(1 - y_true) > 0:
            try:
                # Ensure scores are finite for AUC computation
                if np.all(np.isfinite(scores)):
                    auc = roc_auc_score(y_true, scores)
                    auc_scores.append(auc)
            except:
                pass
    
    mean_perc = float(np.mean(percentile_ranks)) if percentile_ranks else 50.0
    mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.5
    
    return {
        'mean_percentile': mean_perc,
        'auc': mean_auc,
    }


def compute_model_metrics_detailed(model, training_data):
    """Compute per-microbe metrics and summary statistics for plotting."""
    from sklearn.metrics import roc_auc_score

    n_microbes = training_data.n_microbes
    positives = training_data.positives

    per_microbe_auc = []
    per_microbe_percentile = []

    for m_idx in range(n_microbes):
        pos_set = positives.get(m_idx)
        if not pos_set:
            continue

        scores = model.predict_scores(m_idx)
        if np.all(np.isnan(scores)) or np.all(np.isinf(scores)):
            continue

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        pos_indices = np.array(list(pos_set), dtype=int)
        if np.any(pos_indices >= len(scores)):
            continue

        ranks = np.argsort(np.argsort(-scores))
        pos_ranks = ranks[pos_indices]
        pos_percentiles = 100 * (1 - pos_ranks / len(scores))
        per_microbe_percentile.append(float(np.mean(pos_percentiles)))

        y_true = np.zeros(len(scores), dtype=int)
        y_true[pos_indices] = 1
        if y_true.sum() > 0 and (1 - y_true).sum() > 0 and np.all(np.isfinite(scores)):
            try:
                per_microbe_auc.append(float(roc_auc_score(y_true, scores)))
            except Exception:
                pass

    auc_arr = np.array(per_microbe_auc, dtype=float)
    perc_arr = np.array(per_microbe_percentile, dtype=float)

    return {
        "mean_auc": float(np.mean(auc_arr)) if len(auc_arr) > 0 else 0.5,
        "std_auc": float(np.std(auc_arr)) if len(auc_arr) > 0 else 0.0,
        "mean_percentile": float(np.mean(perc_arr)) if len(perc_arr) > 0 else 50.0,
        "std_percentile": float(np.std(perc_arr)) if len(perc_arr) > 0 else 0.0,
        "n_auc": int(len(auc_arr)),
        "n_percentile": int(len(perc_arr)),
    }


def leave_one_out_split(
    positives: dict[int, set[int]],
    seed: int = 42,
) -> tuple[dict[int, set[int]], list[tuple[int, int]]]:
    """Leave-one-out split: hold out one positive per microbe when possible."""
    rng = np.random.default_rng(seed)

    train_positives = {}
    test_pairs = []

    for microbe_idx, pos_set in positives.items():
        if len(pos_set) < 2:
            train_positives[microbe_idx] = pos_set.copy()
            continue

        test_item = int(rng.choice(list(pos_set)))
        train_positives[microbe_idx] = pos_set - {test_item}
        test_pairs.append((microbe_idx, test_item))

    return train_positives, test_pairs


def compute_loo_auc(
    predicted_scores: np.ndarray,
    test_item: int,
    exclude_from_ranking: set[int],
) -> float:
    """Compute AUC for a single held-out positive against all valid negatives."""
    n_items = len(predicted_scores)
    negative_scores = [
        predicted_scores[i]
        for i in range(n_items)
        if i != test_item and i not in exclude_from_ranking
    ]

    if len(negative_scores) == 0:
        return 0.5

    pos_score = predicted_scores[test_item]
    n_below = sum(1 for score in negative_scores if score < pos_score)
    n_ties = sum(1 for score in negative_scores if score == pos_score)
    return float((n_below + 0.5 * n_ties) / len(negative_scores))


def run_figure1_oos_benchmark(
    dimensions: list[int],
    n_repeats: int,
    assertion_type: AssertionType = AssertionType.CONSUMED,
) -> pd.DataFrame:
    """Run out-of-sample leave-one-out benchmark for 4 models."""
    model_map = {
        "Ordinal-BPR": "ordinal_bpr",
        "WR-MF": "wrmf",
        "SVD": "svd",
        "LMF": "lmf",
    }

    # Match training settings used in this codebase for fair comparison.
    model_kwargs = {
        "ordinal_bpr": {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01, "hard_neg_ratio": 0.5, "lambda_ord": 0.1},
        "wrmf": {"iterations": 50, "regularization": 0.1},
        "svd": {},
        "lmf": {"iterations": 30, "learning_rate": 1.0, "regularization": 0.1},
    }

    logger.info(f"Loading {assertion_type.name.lower()} data for out-of-sample benchmark")
    df = load_exometabolite()
    training_data = prepare_training_data(
        df,
        preference_type=assertion_type,
        include_secreted=False,
    )

    positives = training_data.positives
    n_microbes = training_data.n_microbes
    n_metabolites = training_data.n_metabolites

    results = []
    for repeat in range(n_repeats):
        logger.info(f"Figure 1 benchmark repeat {repeat + 1}/{n_repeats}")
        train_pos, test_pairs = leave_one_out_split(positives, seed=repeat)

        split_training_data = TrainingData(
            n_microbes=training_data.n_microbes,
            n_metabolites=training_data.n_metabolites,
            microbe_to_idx=training_data.microbe_to_idx,
            metabolite_to_idx=training_data.metabolite_to_idx,
            positives=train_pos,
            microbe_taxonomy=training_data.microbe_taxonomy,
            preference_type=training_data.preference_type,
        )

        for dim in dimensions:
            for method_name, model_type in model_map.items():
                try:
                    artifacts = train_model(
                        training_data=split_training_data,
                        model_type=model_type,
                        embedding_dim=dim,
                        verbose=False,
                        **model_kwargs[model_type],
                    )
                    model = artifacts.model

                    auc_scores = []
                    for microbe_idx, test_item in test_pairs:
                        train_items_for_microbe = train_pos.get(microbe_idx, set())
                        scores = model.predict_scores(microbe_idx)

                        auc_scores.append(
                            compute_loo_auc(scores, test_item, train_items_for_microbe)
                        )

                    mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.5

                    results.append(
                        {
                            "method": method_name,
                            "dim": dim,
                            "repeat": repeat,
                            "auc": mean_auc,
                            "n_microbes": len(test_pairs),
                            "n_microbes_total": n_microbes,
                            "n_metabolites": n_metabolites,
                        }
                    )
                    logger.info(f"  {method_name} dim={dim}: AUC={mean_auc:.3f}")
                except Exception as exc:
                    logger.warning(f"  {method_name} dim={dim} failed: {exc}")

    return pd.DataFrame(results)





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


def _compute_global_direction(microbe_embeddings: np.ndarray) -> np.ndarray:
    """Compute unit vector for global embedding direction."""
    global_mean = microbe_embeddings.mean(axis=0)
    norm = np.linalg.norm(global_mean)
    if norm < 1e-10:
        return global_mean
    return global_mean / norm


def _project_out(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove vector component along direction."""
    return vector - np.dot(vector, direction) * direction


def compute_genus_profiles_from_bpr(
    model,
    training_data,
) -> pd.DataFrame:
    """Compute genus-level metabolite profiles from a trained BPR model.

    Returns long-format DataFrame matching legacy taxon profiles schema.
    """
    microbe_embeddings = model.microbe_embeddings
    metabolite_embeddings = model.metabolite_embeddings

    idx_to_metabolite = {idx: metabolite for metabolite, idx in training_data.metabolite_to_idx.items()}
    n_microbes = training_data.n_microbes
    n_metabolites = training_data.n_metabolites

    global_dir = _compute_global_direction(microbe_embeddings)
    global_scores = metabolite_embeddings @ global_dir

    metabolite_prevalence = np.zeros(n_metabolites)
    for pos_set in training_data.positives.values():
        for met_idx in pos_set:
            metabolite_prevalence[met_idx] += 1
    metabolite_prevalence /= max(1, n_microbes)

    genus_to_indices = {}
    for microbe_id, microbe_idx in training_data.microbe_to_idx.items():
        taxonomy = training_data.microbe_taxonomy.get(microbe_id)
        genus = taxonomy.genus if taxonomy is not None else None
        if genus is None or pd.isna(genus):
            continue
        genus_to_indices.setdefault(genus, []).append(microbe_idx)

    results = []
    for genus, member_indices in genus_to_indices.items():
        taxon_embedding = microbe_embeddings[member_indices].mean(axis=0)
        raw_scores = metabolite_embeddings @ taxon_embedding

        taxon_distinctive = _project_out(taxon_embedding, global_dir)
        distinctive_scores = metabolite_embeddings @ taxon_distinctive

        taxon_positives = np.zeros(n_metabolites)
        for microbe_idx in member_indices:
            for met_idx in training_data.positives.get(microbe_idx, set()):
                taxon_positives[met_idx] += 1
        taxon_prevalence = taxon_positives / len(member_indices)

        for met_idx in range(n_metabolites):
            results.append(
                {
                    "level": "genus",
                    "taxon": genus,
                    "n_isolates": len(member_indices),
                    "metabolite_idx": met_idx,
                    "metabolite_name": idx_to_metabolite[met_idx],
                    "raw_score": float(raw_scores[met_idx]),
                    "global_score": float(global_scores[met_idx]),
                    "distinctive_score": float(distinctive_scores[met_idx]),
                    "global_prevalence": float(metabolite_prevalence[met_idx]),
                    "taxon_prevalence": float(taxon_prevalence[met_idx]),
                }
            )

    return pd.DataFrame(results)


def compute_metabolite_categories_from_profiles(
    profiles_df: pd.DataFrame,
    popularity_threshold: float = 0.7,
) -> pd.DataFrame:
    """Categorize metabolites as universal / niche_specific / rare."""
    genus_profiles = profiles_df[profiles_df["level"] == "genus"].copy()
    if genus_profiles.empty:
        genus_profiles = profiles_df.copy()

    met_summary = genus_profiles.groupby(["metabolite_idx", "metabolite_name"]).agg(
        global_prevalence=("global_prevalence", "first"),
        max_distinctive=("distinctive_score", "max"),
        min_distinctive=("distinctive_score", "min"),
        std_distinctive=("distinctive_score", "std"),
        max_taxon_prevalence=("taxon_prevalence", "max"),
    ).reset_index()

    met_summary["abs_max_distinctive"] = met_summary[["max_distinctive", "min_distinctive"]].abs().max(axis=1)
    distinctive_threshold = met_summary["abs_max_distinctive"].mean() + met_summary["abs_max_distinctive"].std()

    def categorize(row):
        if row["global_prevalence"] >= popularity_threshold:
            return "universal"
        if row["abs_max_distinctive"] >= distinctive_threshold:
            return "niche_specific"
        return "rare"

    met_summary["category"] = met_summary.apply(categorize, axis=1)
    return met_summary


# =============================================================================
# Figure 1: Four-Panel Benchmark (Model Comparison + Holdout)
# =============================================================================

def figure_1_model_comparison(
    output_dir: Path,
    figure1_dims: list[int] | None = None,
    figure1_repeats: int = 5,
    figure1_recompute: bool = False,
    assertion_type: AssertionType = AssertionType.CONSUMED,
):
    """Figure 1: Original-style AUC benchmark with dense dimension sweep."""
    if figure1_dims is None:
        figure1_dims = [4, 8, 16, 32, 64, 128]

    model_styles = {
        "ordinal_bpr": {"label": "Ordinal-BPR", "color": "#e377c2", "marker": "*"},
        "wrmf": {"label": "WR-MF", "color": "#1f77b4", "marker": "o"},
        "svd": {"label": "SVD", "color": "#2ca02c", "marker": "^"},
        "lmf": {"label": "LMF", "color": "#8c564b", "marker": "P"},
    }
    model_order = ["ordinal_bpr", "wrmf", "svd", "lmf"]

    atype_name = assertion_type.name.lower()
    cache_csv = output_dir / f"figure1_core_results_{atype_name}_4models_ordinal.csv"
    if cache_csv.exists() and not figure1_recompute:
        logger.info(f"Loading cached Figure 1 benchmark results from {cache_csv}")
        core_results = pd.read_csv(cache_csv)
    else:
        logger.info("Running out-of-sample Figure 1 benchmark (this may take several minutes)")
        core_results = run_figure1_oos_benchmark(figure1_dims, figure1_repeats, assertion_type)
        core_results.to_csv(cache_csv, index=False)
        logger.info(f"Saved Figure 1 benchmark results to {cache_csv}")

    if core_results.empty:
        logger.error(f"No {atype_name} models were available for Figure 1")
        return None

    metrics_df = core_results.groupby(["method", "dim"], as_index=False)["auc"].agg(["mean", "std"]).reset_index()
    metrics_df = metrics_df.rename(columns={"mean": "auc", "std": "auc_std"})

    label_to_key = {v["label"]: k for k, v in model_styles.items()}
    metrics_df["model_key"] = metrics_df["method"].map(label_to_key)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    for model_key in model_order:
        model_df = metrics_df[metrics_df["model_key"] == model_key].sort_values("dim")
        if len(model_df) == 0:
            continue

        style = model_styles[model_key]
        ax.errorbar(
            model_df["dim"],
            model_df["auc"],
            yerr=model_df["auc_std"],
            label=style["label"],
            marker='o',#style["marker"],
            color=style["color"],
            linestyle="-",
            capsize=8,
            linewidth=4,
            markersize=8,
        )

    ax.set_xlabel("Embedding dimension", fontsize=20)
    ax.set_ylabel("AUC", fontsize=20)
    ax.legend(frameon=False, fancybox=False, edgecolor="gray", fontsize=20, loc="lower left")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    unique_dims = sorted(metrics_df["dim"].unique())
    if unique_dims:
        ax.set_xticks(unique_dims)
        ax.set_xticklabels([fr"$2^{{{int(np.log2(d))}}}$" if d > 0 and np.log2(d).is_integer() else str(d) for d in unique_dims])

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure1_benchmark.{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    logger.info("Saved figure1_benchmark")
    return metrics_df


# =============================================================================
# Figure 2: t-SNE of Microbe Embeddings
# =============================================================================

def figure_2_tsne(
    model,
    training_data,
    output_dir: Path,
    embedding_dim: int = 32,
    perplexity: float = 30.0,
    random_state: int = 42,
):
    """Figure 2: t-SNE of microbe embeddings colored by class, shaped by genus."""
    n_microbes = training_data.n_microbes
    microbe_taxonomy = training_data.microbe_taxonomy
    idx_to_microbe = training_data.idx_to_microbe

    embeddings = model.microbe_embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    coords = tsne.fit_transform(embeddings)

    # Build simple taxonomy for available microbes
    tax_data = []
    for idx in range(n_microbes):
        org_id = idx_to_microbe[idx]
        if org_id in microbe_taxonomy:
            tax = microbe_taxonomy[org_id]
            tax_data.append({
                'organism_id': org_id,
                'genus': tax.genus or 'Unknown',
                'class': tax.class_ or 'Unknown',
            })
    
    if not tax_data:
        logger.warning("No taxonomy data available for Figure 2")
        return
    
    tax_table = pd.DataFrame(tax_data)

    # Filter out NaN values before sorting to avoid type comparison errors
    classes = sorted([c for c in tax_table["class"].unique() if pd.notna(c)])
    genera = sorted([g for g in tax_table["genus"].unique() if pd.notna(g)])

    class_color_map = {c: get_class_color(c) for c in classes}
    # Assign markers per genus
    marker_set = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h", "d", "P", "X", "8"]

    genus_marker_map = {}
    genus_fill_map = {}

    for i, genus in enumerate(genera):
        marker_idx = i % len(marker_set)
        is_filled = (i // len(marker_set)) == 0
        genus_marker_map[genus] = marker_set[marker_idx]
        genus_fill_map[genus] = is_filled

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx in range(n_microbes):
        org_id = idx_to_microbe[idx]
        org_tax_rows = tax_table[tax_table["organism_id"] == org_id]

        if len(org_tax_rows) == 0:
            continue

        org_class = org_tax_rows.iloc[0]["class"]
        org_genus = org_tax_rows.iloc[0]["genus"]

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
        cls_genera = sorted([
            g for g in tax_table[tax_table["class"] == cls]["genus"].unique()
            if pd.notna(g)
        ])
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
            dpi=600, bbox_inches="tight",
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
    assertion_type: AssertionType = AssertionType.CONSUMED,
):
    """Figure 3: Genus × metabolite heatmap of distinctive preferences.

    Genera ordered by taxonomy. Class color bar on top.
    Only shows the most distinctive metabolites.
    """
    data = prepare_training_data(df, assertion_type, include_secreted=False)
    microbe_to_idx = data.microbe_to_idx
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
            dpi=600, bbox_inches="tight",
        )
    plt.close()
    logger.info("Saved figure3_heatmap")


def _build_genus_col_colors(genera: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame of per-genus color tracks for seaborn clustermap col_colors.

    Returns a DataFrame indexed by genus with columns:
        - "Class"      : taxonomy class color
        - "Collection" : CZ-Biohub (blue) vs Berkeley Lab (orange) vs mixed (purple)
        - "Oxygen"     : aerobic (green) vs anaerobic (red) from GenomeSPOT
    """
    import loaders.genome_spot as genome_spot
    from loaders.config import PROCESSED_DATA_DIR

    genome_spot.CACHE_DIR = str(PROCESSED_DATA_DIR)

    # --- Class track ---
    # genus → class from df taxonomy columns
    genus_to_class: dict[str, str] = {}
    if "genus" in df.columns and "class" in df.columns:
        for _, row in df[["genus", "class"]].dropna().drop_duplicates().iterrows():
            genus_to_class[row["genus"]] = row["class"]

    class_colors = [get_class_color(genus_to_class.get(g, "Unknown")) for g in genera]

    # --- Collection track ---
    CZB_COLOR = "#2196F3"   # blue  — CZ-Biohub
    LBNL_COLOR = "#FF9800"  # orange — Berkeley Lab
    MIXED_COLOR = "#9C27B0" # purple — both
    UNKNOWN_COL = "#aaaaaa"

    # nature_2021_wom_format.tsv = gut microbes → CZ-Biohub
    # wom_*.tsv / wom_*.csv = soil/environmental microbes → Berkeley Lab (LBNL)
    lbnl_sources = [s for s in df["source_file"].unique() if "nature" not in s]
    czb_genera = set(df[df["source_file"] == "nature_2021_wom_format.tsv"]["genus"].dropna().unique())
    lbnl_genera = set(df[df["source_file"].isin(lbnl_sources)]["genus"].dropna().unique())

    collection_colors = []
    for g in genera:
        in_czb = g in czb_genera
        in_lbnl = g in lbnl_genera
        if in_czb and in_lbnl:
            collection_colors.append(MIXED_COLOR)
        elif in_czb:
            collection_colors.append(CZB_COLOR)
        elif in_lbnl:
            collection_colors.append(LBNL_COLOR)
        else:
            collection_colors.append(UNKNOWN_COL)

    # --- Oxygen track (GenomeSPOT) ---
    AEROBIC_COLOR = "#4CAF50"   # green
    ANAEROBIC_COLOR = "#F44336" # red
    UNKNOWN_OXY = "#aaaaaa"

    oxygen_colors = []
    for g in genera:
        try:
            result = genome_spot.lookup(g, level="genus", condition="oxygen", stat="mean")
            if "results" in result and result["results"]:
                val = result["results"][0].get("value")
                if val is not None:
                    oxygen_colors.append(AEROBIC_COLOR if val >= 0.5 else ANAEROBIC_COLOR)
                else:
                    oxygen_colors.append(UNKNOWN_OXY)
            else:
                oxygen_colors.append(UNKNOWN_OXY)
        except Exception:
            oxygen_colors.append(UNKNOWN_OXY)

    return pd.DataFrame(
        {"Class": class_colors, "Collection": collection_colors, "Oxygen": oxygen_colors},
        index=genera,
    )


def figure_3_heatmap_seaborn(
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    metabolite_categories: pd.DataFrame,
    output_dir: Path,
    min_isolates: int = 1,
    exclude_universal: bool = True,
    max_metabolites: int = 60,
    assertion_type: AssertionType = AssertionType.CONSUMED,
    score_col: str = "distinctive_score",
    figure_name: str = "figure3_heatmap_biclustering",
    compound_selection: str = "model_score",
):
    """Figure 3 (seaborn): Genus × metabolite clustermap with multi-track color bars.

    Args:
        score_col: 'distinctive_score' or 'raw_score'
        compound_selection: How to rank and pick the top max_metabolites rows.
            'model_score' — distinctive: highest variance across genera;
                            raw: most genera with non-NaN scores.
            'most_measured' — compounds tested in the most isolates in the
                              experimental data (any assertion, including neutral).
        figure_name: Output filename stem (no extension).

    Color tracks (top of heatmap):
        - Class:      taxonomy class
        - Collection: CZ-Biohub (blue) vs Berkeley Lab (orange) vs both (purple)
        - Oxygen:     aerobic (green) vs anaerobic (red) from GenomeSPOT predictions
    """
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    # --- Data preparation ---
    data = prepare_training_data(df, assertion_type, include_secreted=False)
    microbe_to_idx = data.microbe_to_idx
    tax_table = build_taxonomy_table(df, microbe_to_idx)

    # Genera that have training positives (at least one consumed compound)
    genera_with_positives: set[str] = set()
    for microbe_id, microbe_idx in microbe_to_idx.items():
        if data.positives.get(microbe_idx):
            taxonomy = data.microbe_taxonomy.get(microbe_id)
            if taxonomy and taxonomy.genus:
                genera_with_positives.add(taxonomy.genus)

    genus_profiles = profiles[
        (profiles["level"] == "genus") &
        (profiles["n_isolates"] >= min_isolates) &
        (profiles["taxon"].isin(genera_with_positives))
    ].copy()

    if exclude_universal and metabolite_categories is not None:
        niche_mets = metabolite_categories[
            metabolite_categories["category"] != "universal"
        ]["metabolite_idx"].values
        genus_profiles = genus_profiles[genus_profiles["metabolite_idx"].isin(niche_mets)]

    matrix = genus_profiles.pivot_table(
        index="metabolite_name",
        columns="taxon",
        values=score_col,
        aggfunc="mean",
    )

    # Drop genera where all scores are zero/NaN (no real signal)
    nonzero_cols = matrix.columns[matrix.abs().sum(axis=0) > 1e-9]
    matrix = matrix[nonzero_cols]

    if compound_selection == "most_measured":
        # Rank by number of isolates with any experimental assertion (any value, incl. neutral)
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
    elif score_col == "distinctive_score":
        # Highest variance of distinctive scores across genera (most discriminating)
        met_variance = matrix.var(axis=1).sort_values(ascending=False)
        top_metabolites = met_variance.head(max_metabolites).index
    else:
        # Raw scores: most genera with non-NaN scores
        genera_count = matrix.notna().sum(axis=1).sort_values(ascending=False)
        top_metabolites = genera_count.head(max_metabolites).index
    matrix = matrix.loc[top_metabolites]

    # --- Multi-track column colors ---
    col_colors_df = _build_genus_col_colors(list(matrix.columns), df)

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
        col_colors=col_colors_df,
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

    score_label = "Distinctive\nscore" if score_col == "distinctive_score" else "Raw\nscore"
    g.cax.set_ylabel(score_label, fontsize=26)
    g.cax.tick_params(labelsize=22)

    # Color track labels (seaborn uses DataFrame column names automatically)
    g.ax_col_colors.yaxis.set_tick_params(labelsize=22)

    # --- Legend: class colors ---
    genus_to_class = {row["genus"]: row.get("class", "Unknown") for _, row in tax_table.iterrows()}
    unique_classes = sorted(set(genus_to_class.get(g, "Unknown") for g in matrix.columns))
    class_handles = [
        Patch(facecolor=get_class_color(c), edgecolor="gray", label=c)
        for c in unique_classes
    ]

    # Collection legend patches
    collection_handles = [
        Patch(facecolor="#2196F3", edgecolor="gray", label="CZ-Biohub"),
        Patch(facecolor="#FF9800", edgecolor="gray", label="Berkeley Lab"),
        Patch(facecolor="#9C27B0", edgecolor="gray", label="Both"),
    ]

    # Oxygen legend patches
    oxygen_handles = [
        Patch(facecolor="#4CAF50", edgecolor="gray", label="Aerobic"),
        Patch(facecolor="#F44336", edgecolor="gray", label="Anaerobic"),
        Patch(facecolor="#aaaaaa", edgecolor="gray", label="Unknown"),
    ]

    # Three legend boxes in figure coordinates (bbox_transform=g.fig.transFigure).
    # Collection and Oxygen: 1 column × 3 rows, anchored at the right.
    # Taxonomy class: ceil(n/3) columns so it is also exactly 3 rows tall, on the left.
    # add_artist() preserves each legend after the next legend() call replaces ax._legend.
    n_class_cols = int(np.ceil(len(unique_classes) / 3))

    class_leg = g.ax_col_dendrogram.legend(
        handles=class_handles,
        title="Taxonomy class",
        title_fontsize=22,
        loc="upper left",
        bbox_to_anchor=(0.20, 1.15),
        bbox_transform=g.fig.transFigure,
        ncol=n_class_cols,
        fontsize=20,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    g.ax_col_dendrogram.add_artist(class_leg)

    coll_leg = g.ax_col_dendrogram.legend(
        handles=collection_handles,
        title="Collection",
        title_fontsize=22,
        loc="upper right",
        bbox_to_anchor=(0.88, 1.15),
        bbox_transform=g.fig.transFigure,
        ncol=1,
        fontsize=20,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )
    g.ax_col_dendrogram.add_artist(coll_leg)

    g.ax_col_dendrogram.legend(
        handles=oxygen_handles,
        title="Oxygen",
        title_fontsize=22,
        loc="upper right",
        bbox_to_anchor=(1.03, 1.15),
        bbox_transform=g.fig.transFigure,
        ncol=1,
        fontsize=20,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        for c in ax.collections:
            c.set_linewidth(1.5)

    for ext in ["png", "pdf"]:
        g.savefig(output_dir / f"{figure_name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {figure_name} (score_col={score_col})")

def quantify_genus_conservation(
    df: pd.DataFrame,
    output_dir: Path,
    embedding_dim: int = 32,
    min_genus_size: int = 2,
    n_permutations: int = 999,
):
    """Quantify whether substrate preferences are conserved at the genus level.

    Computes:
      1. Within-genus vs. between-genus cosine similarity (with Mann-Whitney test)
      2. PERMANOVA (genus explains variance in embedding space)
      3. Silhouette score (how well organisms cluster by genus)
      4. Within-genus rank correlation of metabolite preferences

    Args:
        df: Exometabolite dataframe
        output_dir: Where to save results
        embedding_dim: BPR embedding dimension
        min_genus_size: Minimum isolates per genus to include
        n_permutations: Permutations for PERMANOVA
    """
    from scipy.spatial.distance import pdist, squareform, cosine
    from scipy.stats import mannwhitneyu, spearmanr
    from sklearn.metrics import silhouette_score

    data = prepare_training_data(df, "consumed")
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    microbe_to_idx = data["microbe_to_idx"]
    metabolite_to_idx = data["metabolite_to_idx"]
    idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}

    model = BPRModel(embedding_dim=embedding_dim)
    model.fit(positives, n_microbes, n_metabolites, verbose=False)

    embeddings = model.microbe_embeddings
    tax_table = build_taxonomy_table(df, microbe_to_idx)

    # Map each organism index to its genus
    idx_to_genus = {}
    for _, row in tax_table.iterrows():
        org_id = row["organism_id"]
        if org_id in microbe_to_idx:
            idx_to_genus[microbe_to_idx[org_id]] = row["genus"]

    # Filter to genera with enough members
    from collections import Counter
    genus_counts = Counter(idx_to_genus.values())
    valid_genera = {g for g, c in genus_counts.items() if c >= min_genus_size and g != "Unknown"}

    valid_indices = [i for i in range(n_microbes) if idx_to_genus.get(i) in valid_genera]
    valid_labels = [idx_to_genus[i] for i in valid_indices]
    valid_embeddings = embeddings[valid_indices]

    logger.info(f"Genus conservation analysis: {len(valid_indices)} isolates in "
                f"{len(valid_genera)} genera (min size {min_genus_size})")

    # -------------------------------------------------------------------------
    # 1. Within-genus vs. between-genus cosine similarity
    # -------------------------------------------------------------------------
    n = len(valid_indices)
    cos_sim_matrix = 1 - squareform(pdist(valid_embeddings, metric="cosine"))

    within_sims = []
    between_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cos_sim_matrix[i, j]
            if valid_labels[i] == valid_labels[j]:
                within_sims.append(sim)
            else:
                between_sims.append(sim)

    within_sims = np.array(within_sims)
    between_sims = np.array(between_sims)

    mw_stat, mw_pval = mannwhitneyu(within_sims, between_sims, alternative="greater")

    logger.info(f"  Cosine similarity — within-genus: {within_sims.mean():.3f} ± {within_sims.std():.3f}, "
                f"between-genus: {between_sims.mean():.3f} ± {between_sims.std():.3f}, "
                f"Mann-Whitney p = {mw_pval:.2e}")

    # -------------------------------------------------------------------------
    # 2. PERMANOVA (pseudo-F from distance matrix, genus as grouping)
    # -------------------------------------------------------------------------
    dist_matrix = squareform(pdist(valid_embeddings, metric="euclidean"))
    labels_arr = np.array(valid_labels)
    unique_genera = list(valid_genera)

    def pseudo_f(dists, labels):
        """Compute PERMANOVA pseudo-F statistic."""
        n = len(labels)
        unique = np.unique(labels)
        a = len(unique)

        # Total sum of squares
        ss_total = (dists ** 2).sum() / (2 * n)

        # Within-group sum of squares
        ss_within = 0
        for g in unique:
            mask = labels == g
            n_g = mask.sum()
            if n_g > 1:
                group_dists = dists[np.ix_(mask, mask)]
                ss_within += (group_dists ** 2).sum() / (2 * n_g)

        ss_between = ss_total - ss_within
        f_stat = (ss_between / (a - 1)) / (ss_within / (n - a))
        return f_stat

    observed_f = pseudo_f(dist_matrix, labels_arr)

    # Permutation test
    perm_fs = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels_arr)
        perm_fs.append(pseudo_f(dist_matrix, perm_labels))
    perm_fs = np.array(perm_fs)

    permanova_p = (np.sum(perm_fs >= observed_f) + 1) / (n_permutations + 1)

    # R² (effect size)
    ss_total = (dist_matrix ** 2).sum() / (2 * n)
    ss_within = 0
    for g in np.unique(labels_arr):
        mask = labels_arr == g
        n_g = mask.sum()
        if n_g > 1:
            group_dists = dist_matrix[np.ix_(mask, mask)]
            ss_within += (group_dists ** 2).sum() / (2 * n_g)
    r_squared = 1 - ss_within / ss_total

    logger.info(f"  PERMANOVA — pseudo-F = {observed_f:.2f}, R² = {r_squared:.3f}, "
                f"p = {permanova_p:.4f} ({n_permutations} permutations)")

    # -------------------------------------------------------------------------
    # 3. Silhouette score
    # -------------------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    numeric_labels = le.fit_transform(valid_labels)
    sil_score = silhouette_score(valid_embeddings, numeric_labels, metric="cosine")

    logger.info(f"  Silhouette score (cosine): {sil_score:.3f}")

    # -------------------------------------------------------------------------
    # 4. Within-genus Spearman rank correlation of metabolite scores
    # -------------------------------------------------------------------------
    met_embeddings = model.metabolite_embeddings
    score_matrix = valid_embeddings @ met_embeddings.T  # (n_isolates, n_metabolites)

    within_rhos = []
    between_rhos = []
    rng2 = np.random.default_rng(123)

    for i in range(n):
        for j in range(i + 1, n):
            if valid_labels[i] == valid_labels[j]:
                rho, _ = spearmanr(score_matrix[i], score_matrix[j])
                within_rhos.append(rho)

    # Sample an equal number of between-genus pairs for comparison
    n_between_sample = min(len(within_rhos) * 5, n * (n - 1) // 2)
    between_pairs = []
    while len(between_pairs) < n_between_sample:
        i, j = rng2.integers(0, n, size=2)
        if i != j and valid_labels[i] != valid_labels[j]:
            rho, _ = spearmanr(score_matrix[i], score_matrix[j])
            between_rhos.append(rho)
            between_pairs.append((i, j))

    within_rhos = np.array(within_rhos)
    between_rhos = np.array(between_rhos)

    rho_mw_stat, rho_mw_pval = mannwhitneyu(within_rhos, between_rhos, alternative="greater")

    logger.info(f"  Spearman rank correlation — within-genus: {within_rhos.mean():.3f} ± {within_rhos.std():.3f}, "
                f"between-genus: {between_rhos.mean():.3f} ± {between_rhos.std():.3f}, "
                f"Mann-Whitney p = {rho_mw_pval:.2e}")

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------
    summary = {
        "n_isolates": len(valid_indices),
        "n_genera": len(valid_genera),
        "min_genus_size": min_genus_size,
        "cosine_sim_within_mean": within_sims.mean(),
        "cosine_sim_within_std": within_sims.std(),
        "cosine_sim_between_mean": between_sims.mean(),
        "cosine_sim_between_std": between_sims.std(),
        "cosine_sim_mann_whitney_p": mw_pval,
        "permanova_pseudo_f": observed_f,
        "permanova_r_squared": r_squared,
        "permanova_p": permanova_p,
        "permanova_n_permutations": n_permutations,
        "silhouette_score_cosine": sil_score,
        "spearman_within_mean": within_rhos.mean(),
        "spearman_within_std": within_rhos.std(),
        "spearman_between_mean": between_rhos.mean(),
        "spearman_between_std": between_rhos.std(),
        "spearman_mann_whitney_p": rho_mw_pval,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "genus_conservation_stats.csv", index=False)
    logger.info(f"Saved genus_conservation_stats.csv")

    return summary

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
        g.savefig(output_dir / f"figure5_raw_heatmap.{ext}", dpi=600, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure5_raw_heatmap")


def figure_6_rank_clustermap(
    model,
    training_data,
    output_dir: Path,
    df: pd.DataFrame | None = None,
    embedding_dim: int = 32,
):
    """Figure 6: Clustermap of consumption status by predicted rank.

    Each row is an isolate, each column is a rank position (1 = highest
    predicted preference). Blue = consumed, salmon = neutral.
    Demonstrates that consumed metabolites tend to appear at lower ranks.
    """
    import seaborn as sns

    n_microbes = training_data.n_microbes
    n_metabolites = training_data.n_metabolites
    positives = training_data.positives
    idx_to_microbe = training_data.idx_to_microbe

    # Build rank matrix across the full model metabolite space:
    # rows = isolates, cols = rank positions, values = consumed(1)/not-consumed(0)
    rows = []
    for microbe_idx in range(n_microbes):
        pos_set = positives.get(microbe_idx, set())
        if len(pos_set) == 0 or len(pos_set) >= n_metabolites:
            continue

        scores = model.predict_scores(microbe_idx)
        ranked_metabolite_indices = np.argsort(-scores)

        consumed_by_rank = np.isin(ranked_metabolite_indices, list(pos_set)).astype(int)
        consumed_series = pd.Series(consumed_by_rank, index=np.arange(1, n_metabolites + 1), name=idx_to_microbe[microbe_idx])
        rows.append(consumed_series)

    if not rows:
        logger.warning("No valid microbes with mixed consumed/non-consumed profiles for Figure 6")
        return

    rank_matrix = pd.DataFrame(rows).fillna(0).astype(int)
    rank_matrix.columns.name = "Rank"
    rank_matrix.index.name = "Organism"

    # Match legacy behavior: drop rank columns where no organism is consumed.
    rank_matrix_filtered = rank_matrix.loc[:, rank_matrix.sum(axis=0) > 0]

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

    g.cax.tick_params(labelsize=16)
    g.cax.set_yticklabels(["Neutral", "Consumed"], fontsize=16)
    for spine in g.cax.spines.values():
        spine.set_visible(True)
    g.cax.set_facecolor("white")

    for coll in g.ax_row_dendrogram.collections:
        coll.set_linewidth(1.5)

    g.ax_heatmap.set_xlabel("Rank (1 = highest predicted preference)", fontsize=24)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=18)
    g.ax_heatmap.set_ylabel("Bacterial isolate", labelpad=-850, fontsize=24)

    for ext in ["png", "pdf"]:
        g.savefig(output_dir / f"figure6_rank_clustermap.{ext}", dpi=600, bbox_inches="tight")
    plt.close()

    logger.info("Saved figure6_rank_clustermap")


def figure_7_embedding_distance(
    model,
    training_data,
    output_dir: Path
):
    """Figure 7: Embedding distance distributions by taxonomic level.

    Four-panel histogram showing cosine distances between organism embeddings.
    """
    # Get embeddings
    m_emb = model.microbe_embeddings
    microbe_taxonomy = training_data.microbe_taxonomy
    idx_to_microbe = training_data.idx_to_microbe

    # Compute pairwise distance matrix
    dist_matrix = pairwise_distances(m_emb, metric='cosine')
    
    # Build taxonomy dataframe for all microbes with valid taxonomy
    tax_data = []
    for idx in range(len(idx_to_microbe)):
        org_id = idx_to_microbe[idx]
        if org_id in microbe_taxonomy:
            tax = microbe_taxonomy[org_id]
            tax_data.append({
                'organism_id': org_id,
                'genus': tax.genus or 'Unknown',
                'family': tax.family or 'Unknown',
                'order': tax.order or 'Unknown',
                'class': tax.class_ or 'Unknown',
            })
    
    if not tax_data:
        logger.warning("No taxonomy data for Figure 7")
        return
    
    valid = pd.DataFrame(tax_data)
    
    # Convert distance matrix to DataFrame
    dist_df = pd.DataFrame(
        dist_matrix,
        index=[idx_to_microbe[i] for i in range(len(idx_to_microbe))],
        columns=[idx_to_microbe[i] for i in range(len(idx_to_microbe))],
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
        delta = median_diff - median_same

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
            0.8, 0.6, f'Δmedian = {delta:.3f}',
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
            dpi=600, bbox_inches="tight",
        )
    plt.close()
    logger.info("Saved figure7_embedding_distance")


# =============================================================================
# Figure 8: Experimental Validation (Held-out Isolates)
# =============================================================================


def load_experimental_validation_data():
    """Load experimental validation data from the single held-out TSV file.

    VALIDATION_DATA_FILE is the sole source of truth: it contains taxonomy
    (phylum, class, order, family, genus, closest relative) and measurement
    values for every held-out isolate in wide format (one row per isolate,
    one column per metabolite).  'ND' cells (not determined) are treated as
    missing.

    Returns DataFrame with columns:
        isolate, genus, family, order, species_label, metabolite, value, media
    Returns None if the file is not available.
    """
    if not VALIDATION_DATA_FILE.exists():
        logger.warning(f"Validation data file not found: {VALIDATION_DATA_FILE}")
        return None

    raw = pd.read_csv(VALIDATION_DATA_FILE, sep='\t')

    # Normalise column names and string values (TSV cells can have trailing spaces)
    raw.rename(columns={raw.columns[0]: 'isolate'}, inplace=True)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.apply(lambda col: col.str.strip() if pd.api.types.is_string_dtype(col) else col)


    # Drop the time-point column — not needed for predictions
    raw.drop(columns=[c for c in raw.columns if 'time point' in c.lower()], inplace=True)

    # Taxonomy columns (present in the file)
    tax_cols = ['phylum', 'class', 'order', 'family', 'genus',
                'closest relative in ncbi 16s database']
    # Normalise column names to lower-case for robust matching
    raw.columns = [c.lower() for c in raw.columns]
    tax_cols = [c for c in tax_cols if c in raw.columns]

    # Derive a clean display label from the closest-relative field
    def _species_label(cr, genus):
        if pd.notna(cr) and str(cr).strip():
            parts = str(cr).strip().split()
            return ' '.join(parts[:2])
        return f"{genus} Sp."

    raw['species_label'] = raw.apply(
        lambda r: _species_label(
            r.get('closest relative in ncbi 16s database', None),
            r.get('genus', '')
        ), axis=1
    )

    # Metabolite columns = everything that is not a taxonomy or id column
    id_cols = ['isolate', 'species_label'] + tax_cols
    met_cols = [c for c in raw.columns if c not in id_cols]

    melted = raw.melt(id_vars=id_cols, value_vars=met_cols,
                      var_name='metabolite', value_name='value')

    # 'ND' (not determined) → NaN; coerce to float
    melted['value'] = pd.to_numeric(melted['value'].replace('ND', pd.NA), errors='coerce')
    melted.dropna(subset=['value'], inplace=True)

    melted['media'] = 'nldm'

    logger.info(f"Loaded validation data: {len(melted)} measurements, "
                f"{melted['isolate'].nunique()} isolates, "
                f"{melted['genus'].nunique()} genera")

    return melted


def compute_genus_predictions_any_model(model, training_data):
    """Compute genus-, family-, and order-level mean predictions for any model.

    Returns a dict with keys 'genus', 'family', 'order', each mapping to a
    DataFrame with columns: <level>, metabolite, mean_score, n_isolates.
    n_isolates is the number of training isolates that contributed to the mean,
    used downstream to detect when a genus/family has no training coverage and a
    coarser taxonomic fallback is required.
    """
    idx_to_microbe = training_data.idx_to_microbe
    idx_to_metabolite = training_data.idx_to_metabolite

    all_scores = []
    for microbe_idx in range(training_data.n_microbes):
        microbe_id = idx_to_microbe[microbe_idx]
        scores = model.predict_scores(microbe_idx)
        taxonomy = training_data.microbe_taxonomy.get(microbe_id)
        genus  = taxonomy.genus  if taxonomy else None
        family = taxonomy.family if taxonomy and hasattr(taxonomy, 'family') else None
        order  = taxonomy.order  if taxonomy and hasattr(taxonomy, 'order')  else None
        for met_idx, score in enumerate(scores):
            all_scores.append({
                'microbe_id': microbe_id,
                'metabolite': idx_to_metabolite[met_idx],
                'score': float(score),
                'genus': genus,
                'family': family,
                'order': order,
            })

    df = pd.DataFrame(all_scores)

    def _aggregate(df, level):
        grp = df.dropna(subset=[level]).groupby([level, 'metabolite'])
        agg = grp['score'].agg(mean_score='mean', n_isolates='count').reset_index()
        agg.rename(columns={level: level}, inplace=True)
        return agg

    return {
        'genus':  _aggregate(df, 'genus'),
        'family': _aggregate(df, 'family'),
        'order':  _aggregate(df, 'order'),
    }


def find_optimal_threshold(training_data, model, method='top_k'):
    """Find optimal classification threshold for a model.
    
    Args:
        training_data: TrainingData object
        model: Trained model
        method: 'top_k' or 'prevalence'
    
    Returns:
        threshold value or k (for top-k)
    """
    if method == 'top_k':
        # Use consumption rate from training data
        total_observations = training_data.n_microbes * training_data.n_metabolites
        total_consumed = sum(len(pos) for pos in training_data.positives.values())
        consumption_rate = total_consumed / total_observations
        return consumption_rate
    
    elif method == 'prevalence':
        # Find threshold where predicted positive rate matches training prevalence
        all_scores = []
        for microbe_idx in range(training_data.n_microbes):
            scores = model.predict_scores(microbe_idx)
            all_scores.extend(scores)
        
        all_scores = np.array(all_scores)
        total_observations = training_data.n_microbes * training_data.n_metabolites
        total_consumed = sum(len(pos) for pos in training_data.positives.values())
        consumption_rate = total_consumed / total_observations
        
        threshold = np.percentile(all_scores, (1 - consumption_rate) * 100)
        return float(threshold)
    
    else:
        return 0.0


def compute_precision_recall(experimental_df, predictions_df, threshold=0, method='threshold'):
    """Compute precision and recall for experimental validation.
    
    Args:
        experimental_df: DataFrame with columns [genus, metabolite, value]
        predictions_df: DataFrame with columns [genus, metabolite, mean_score]
        threshold: Score threshold for 'threshold' method, or consumption rate for 'top_k'
        method: 'threshold' or 'top_k'
    
    Returns:
        DataFrame with precision/recall per genus
    """
    # Map metabolites
    from loaders.compound_mapper import map_compound_names_batch
    
    unique_metabolites = experimental_df['metabolite'].unique()
    compound_mapping = map_compound_names_batch(unique_metabolites)
    
    experimental_df = experimental_df.copy()
    experimental_df['mapped_metabolite'] = experimental_df['metabolite'].apply(
        lambda x: compound_mapping[x].representative_name if x in compound_mapping else None
    )
    
    # Merge experimental data with predictions
    merged = pd.merge(
        experimental_df,
        predictions_df,
        left_on=['genus', 'mapped_metabolite'],
        right_on=['genus', 'metabolite'],
        how='left'
    )
    merged = merged[pd.notna(merged['mean_score'])]
    
    if len(merged) == 0:
        logger.warning("No matching metabolites found between experimental data and model")
        return pd.DataFrame()
    
    # Compute precision/recall per genus
    results = []
    for genus in merged['genus'].unique():
        genus_data = merged[merged['genus'] == genus]
        
        # True labels: consumed if value <= -2 (matches build threshold)
        true_consumed = (genus_data['value'] <= -2).astype(int)
        
        # Predicted labels
        if method == 'top_k':
            # Predict top k% as consumed
            k = int(len(genus_data) * threshold)
            k = max(1, k)  # At least 1
            top_k_idx = np.argsort(-genus_data['mean_score'].values)[:k]
            predicted_consumed = np.zeros(len(genus_data), dtype=int)
            predicted_consumed[top_k_idx] = 1
        else:
            # Use threshold
            predicted_consumed = (genus_data['mean_score'] > threshold).astype(int)
        
        if predicted_consumed.sum() == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = precision_score(true_consumed, predicted_consumed, zero_division=0)
            recall = recall_score(true_consumed, predicted_consumed, zero_division=0)
        
        results.append({
            'genus': genus,
            'precision': precision,
            'recall': recall,
            'n_metabolites': len(genus_data),
            'n_consumed': true_consumed.sum(),
            'n_predicted': predicted_consumed.sum()
        })
    
    return pd.DataFrame(results)


def figure_8_experimental_validation_single_model(
    model,
    training_data,
    model_name: str,
    output_dir: Path,
    experimental_df: pd.DataFrame = None,
    k: int = 30
):
    """Figure 8: Experimental validation histograms for a single model.

    Shows distribution of model scores for consumed vs neutral metabolites
    in held-out isolates, broken down by genus. Per-genus ROC-AUC is annotated
    inside each subplot.

    Args:
        model: Trained model
        training_data: TrainingData object
        model_name: Name of model for figure title
        output_dir: Output directory
        experimental_df: Optional experimental data. If None, will try to load.
        k: Unused (kept for API compatibility)
    """
    # Load experimental data if not provided
    if experimental_df is None:
        experimental_df = load_experimental_validation_data()
        if experimental_df is None:
            logger.warning(f"Cannot generate Figure 8 for {model_name}: no experimental data")
            return None
    
    # Compute predictions at genus, family, and order levels
    logger.info(f"Computing multi-level predictions for {model_name}")
    predictions = compute_genus_predictions_any_model(model, training_data)

    # Map experimental metabolite names to canonical representative names
    from loaders.compound_mapper import map_compound_names_batch

    experimental_df = experimental_df.copy()
    compound_mapping = map_compound_names_batch(experimental_df['metabolite'].unique())
    experimental_df['mapped_metabolite'] = experimental_df['metabolite'].apply(
        lambda x: compound_mapping[x].representative_name if x in compound_mapping else None
    )
    experimental_df = experimental_df[experimental_df['media'] == 'nldm']

    # species_label is already computed by load_experimental_validation_data()
    isolate_display = (
        experimental_df[['isolate', 'species_label']]
        .drop_duplicates()
        .set_index('isolate')['species_label']
        .to_dict()
    )

    # Merge per isolate with taxonomic fallback: genus → family → order
    merged_parts = []
    isolate_meta = {}  # isolate -> {tax_level, n_training}

    for isolate, iso_df in experimental_df.groupby('isolate'):
        genus  = iso_df['genus'].iloc[0]
        family = iso_df['family'].iloc[0]
        order  = iso_df['order'].iloc[0]

        merged_iso = None
        for level, key in [('genus', genus), ('family', family), ('order', order)]:
            if pd.isna(key):
                continue
            pred = predictions[level]
            level_rows = pred[pred[level] == key]
            if len(level_rows) == 0:
                continue
            n_training = int(level_rows['n_isolates'].iloc[0])
            iso_merged = pd.merge(
                iso_df,
                level_rows[['metabolite', 'mean_score', 'n_isolates']],
                left_on='mapped_metabolite',
                right_on='metabolite',
                how='left',
                suffixes=('_exp', ''),
            )
            iso_merged['tax_level'] = level
            iso_merged['n_training'] = n_training
            iso_merged['isolate_label'] = isolate
            merged_iso = iso_merged[pd.notna(iso_merged['mean_score'])]
            isolate_meta[isolate] = {
                'tax_level': level, 'n_training': n_training,
                'genus': genus, 'display': isolate_display.get(isolate, genus),
            }
            logger.info(f"  {isolate}: using {level}-level predictions "
                        f"(n_training={n_training})")
            break

        if merged_iso is None or len(merged_iso) == 0:
            logger.warning(f"  {isolate}: no predictions found at any taxonomic level — skipped")
            continue
        merged_parts.append(merged_iso)

    if not merged_parts:
        logger.warning(f"No matching data for {model_name}")
        return None

    merged = pd.concat(merged_parts, ignore_index=True)

    # One panel per isolate, sorted by genus then isolate id
    isolates = sorted(merged['isolate_label'].unique(),
                      key=lambda x: (isolate_meta.get(x, {}).get('genus', ''), x))
    n_panels = len(isolates)
    ncols = 4
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(figsize=(ncols * 4.5, nrows * 4.5), ncols=ncols, nrows=nrows,
                             sharex=True)
    axes = axes.flatten() if n_panels > 1 else [axes]

    # Dynamic bin edges across all isolates
    score_min = merged['mean_score'].min()
    score_max = merged['mean_score'].max()
    padding = max((score_max - score_min) * 0.05, 0.1)
    edges = np.linspace(score_min - padding, score_max + padding, 25)

    from sklearn.metrics import roc_auc_score

    all_auc = []
    legend_handles = None

    for i, isolate in enumerate(isolates):
        ax = axes[i]
        subset = merged[merged['isolate_label'] == isolate]
        meta = isolate_meta.get(isolate, {})
        n_training = meta.get('n_training', 0)
        display_name = meta.get('display', isolate)
        tax_level = meta.get('tax_level', 'genus')

        consumed_idx = subset['value'] <= -2
        neutral_idx  = (subset['value'] > -2) & (subset['value'] < 2)
        consumed_scores = subset[consumed_idx]['mean_score']
        neutral_scores  = subset[neutral_idx]['mean_score']

        ax.hist(consumed_scores, bins=edges, alpha=0.7,
                label='Consumed (≤ −2)', density=False, color='#1f77b4')
        ax.hist(neutral_scores, bins=edges, alpha=0.7,
                label='Neutral (−2 to 2)', density=False, color='#ff7f0e')
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        cn_subset = subset[consumed_idx | neutral_idx]
        true_consumed = (cn_subset['value'] <= -2).astype(int)
        n_consumed = consumed_idx.sum()
        n_total    = neutral_idx.sum() + n_consumed

        if n_consumed > 0 and n_consumed < n_total:
            try:
                auc = roc_auc_score(true_consumed, cn_subset['mean_score'].values)
            except Exception:
                auc = float('nan')
        else:
            auc = float('nan')

        all_auc.append(auc)
        logger.info(f"  {isolate} ({display_name}): n_total={n_total}, "
                    f"n_consumed={n_consumed}, AUC={auc:.3f}, level={tax_level}")

        # Title: italicised species name + training-isolate count
        title = f"{display_name} ({n_training})"
        ax.set_title(title, fontsize=18, style='italic')
        # Annotate fallback level if not genus
        if tax_level != 'genus':
            ax.set_title(title + f"\n[{tax_level}-level]", fontsize=16, style='italic')
        ax.tick_params(axis='both', labelsize=20)
        if not np.isnan(auc):
            ax.text(0.04, 0.97, f"AUC={auc:.2f}",
                    transform=ax.transAxes, fontsize=19,
                    ha='left', va='top')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Remove extra axes
    for i in range(n_panels, len(axes)):
        axes[i].remove()

    # Single-line figure-level legend above the subplots
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(legend_labels), frameon=False,
                   fontsize=20, title="experimental observation",
                   title_fontsize=20)

    # Shared axis labels — positioned inside tight_layout rect
    fig.text(0.5, 0.01, f'Mean {model_name} Score (Genus-Level)', ha='center', fontsize=22)
    fig.text(0.01, 0.5, 'Number of Metabolites', va='center', rotation='vertical', fontsize=22)

    # Reserve left/bottom margins for shared labels; top margin for legend
    plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.88])
    
    # Save figure
    safe_name = model_name.lower().replace('-', '_').replace(' ', '_')
    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f"figure8_experimental_validation_{safe_name}.{ext}",
                   dpi=600, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved figure8_experimental_validation_{safe_name}")
    valid_auc = [a for a in all_auc if not np.isnan(a)]
    logger.info(f"  Mean AUC across isolates: {np.mean(valid_auc):.3f}" if valid_auc else "  No valid AUC scores")

    return {
        'model': model_name,
        'mean_auc': np.mean(valid_auc) if valid_auc else float('nan'),
        'isolates': isolates,
        'per_isolate_auc': dict(zip(isolates, all_auc)),
        'isolate_meta': isolate_meta,
    }


def figure_8_experimental_validation_all_models(
    output_dir: Path,
    trained_model_dir: Path = None,
    embedding_dim: int = 32,
    k: int = 30,
    assertion_type: AssertionType = AssertionType.CONSUMED,
):
    """Generate Figure 8 for Ordinal-BPR + AUC summary table.

    Uses per-genus ROC-AUC as the evaluation metric.

    Args:
        output_dir: Output directory for figures and tables
        trained_model_dir: Directory with trained models (if available)
        embedding_dim: Embedding dimension to use for training
        k: Unused (kept for API compatibility)
        assertion_type: Whether to use consumed or secreted data
    """
    # Load experimental data
    experimental_df = load_experimental_validation_data()
    if experimental_df is None:
        logger.warning("Cannot generate Figure 8: no experimental data available")
        return

    # Load training data
    logger.info(f"Loading {assertion_type.name.lower()} training data for Figure 8")
    df = load_exometabolite()
    training_data = prepare_training_data(
        df,
        preference_type=assertion_type,
        include_secreted=False,
    )
    
    # Only validate with Ordinal-BPR
    model_configs = {
        "Ordinal-BPR": {"model_type": "ordinal_bpr", "kwargs": {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01, "hard_neg_ratio": 0.5, "lambda_ord": 0.1}},
    }
    
    all_results = []
    all_genus_results = []
    
    for model_name, config in model_configs.items():
        logger.info(f"Processing {model_name} model")
        
        try:
            # Train model
            artifacts = train_model(
                training_data=training_data,
                model_type=config["model_type"],
                embedding_dim=embedding_dim,
                verbose=False,
                **config["kwargs"]
            )
            model = artifacts.model
            
            # Generate figure
            result = figure_8_experimental_validation_single_model(
                model=model,
                training_data=training_data,
                model_name=model_name,
                output_dir=output_dir,
                experimental_df=experimental_df,
                k=k
            )
            
            if result:
                all_results.append(result)
                
                # Note: Detailed per-genus metrics already computed in figure generation
                # Store summary for later if needed
                pass
        
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            continue
    
    # Create summary table
    if all_results:
        # Per-isolate AUC rows
        rows = []
        for result in all_results:
            for isolate_id, auc in result['per_isolate_auc'].items():
                meta = result.get('isolate_meta', {}).get(isolate_id, {})
                rows.append({
                    'model': result['model'],
                    'isolate': isolate_id,
                    'genus': meta.get('genus', ''),
                    'tax_level': meta.get('tax_level', 'genus'),
                    'auc': auc,
                })
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / 'figure8_auc_per_isolate.csv', index=False)

        for result in all_results:
            logger.info(f"  {result['model']}: mean AUC = {result['mean_auc']:.3f}")


def export_taxonomic_score_matrices(
    artifacts: ModelArtifacts,
    output_dir: Path,
):
    """Export three TSV tables with taxonomic multiindex (rows) and metabolites (columns).
    
    Creates three separate TSV files:
    - raw_scores: Matrix of raw prediction scores
    - distinctive_scores: Matrix of distinctive scores (relative to global background)
    - ranks: Matrix of metabolite ranks per organism
    
    Each row represents a single organism with multiindex: kingdom, class, order, family, genus, isolate
    Each column represents a metabolite.
    
    Args:
        artifacts: ModelArtifacts object containing model and training data
        output_dir: Output directory (will be created if it doesn't exist)
    """
    model = artifacts.model
    training_data = artifacts.training_data
    
    n_microbes = training_data.n_microbes
    n_metabolites = training_data.n_metabolites
    idx_to_metabolite = training_data.idx_to_metabolite
    idx_to_microbe = training_data.idx_to_microbe
    microbe_taxonomy = training_data.microbe_taxonomy
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting taxonomic score matrices to {output_dir}")
    
    # Build metabolite column headers (in order)
    metabolite_names = [idx_to_metabolite[i] for i in range(n_metabolites)]
    
    # Accumulate data for all three matrices
    raw_score_rows = []
    distinctive_score_rows = []
    rank_rows = []
    
    index_rows = []  # For multiindex (kingdom, class, order, family, genus, isolate)
    
    # Compute global background for distinctive scores
    global_embeddings = model.microbe_embeddings.mean(axis=0)
    metabolite_embeddings = model.metabolite_embeddings
    global_scores = metabolite_embeddings @ global_embeddings
    
    # Process each organism
    for microbe_idx in range(n_microbes):
        microbe_id = idx_to_microbe[microbe_idx]
        
        # Get taxonomy
        taxonomy = microbe_taxonomy.get(microbe_id)
        if taxonomy is None:
            logger.warning(f"No taxonomy found for {microbe_id}, skipping")
            continue
        
        # Extract taxonomy levels (use kingdom instead of phylum)
        kingdom = taxonomy.kingdom or "Unknown"
        class_ = taxonomy.class_ or "Unknown"
        order = taxonomy.order or "Unknown"
        family = taxonomy.family or "Unknown"
        genus = taxonomy.genus or "Unknown"
        isolate = microbe_id  # Use the organism ID as isolate name
        
        # Get raw scores
        raw_scores = model.predict_scores(microbe_idx)
        
        # Get distinctive scores (relative to global background)
        # distinctive = organism_specific_score
        microbe_embedding = model.microbe_embeddings[microbe_idx]
        embedding_relative = microbe_embedding - global_embeddings
        distinctive_scores = metabolite_embeddings @ embedding_relative
        
        # Get ranks (1 = best, n = worst)
        # argsort gives indices in ascending order; negate scores for descending
        # Then argsort again to get ranks
        rank_indices = np.argsort(-raw_scores)  # Indices sorted by score (best to worst)
        ranks = np.empty_like(rank_indices)
        ranks[rank_indices] = np.arange(1, len(rank_indices) + 1)  # Assign ranks 1, 2, 3, ...
        
        # Store taxonomy info
        index_rows.append((kingdom, class_, order, family, genus, isolate))
        
        # Store scores and ranks
        raw_score_rows.append(raw_scores)
        distinctive_score_rows.append(distinctive_scores)
        rank_rows.append(ranks)
    
    if len(raw_score_rows) == 0:
        logger.error("No valid organisms found for export")
        return
    
    logger.info(f"Exporting {len(raw_score_rows)} organisms")
    
    # Convert to numpy arrays
    raw_scores_array = np.array(raw_score_rows, dtype=np.float32)
    distinctive_scores_array = np.array(distinctive_score_rows, dtype=np.float32)
    ranks_array = np.array(rank_rows, dtype=np.int32)
    
    # Create multiindex
    multiindex = pd.MultiIndex.from_tuples(
        index_rows,
        names=["kingdom", "class", "order", "family", "genus", "isolate"]
    )
    
    # Create DataFrames
    df_raw_scores = pd.DataFrame(
        raw_scores_array,
        index=multiindex,
        columns=metabolite_names,
        dtype=np.float32
    )
    
    df_distinctive_scores = pd.DataFrame(
        distinctive_scores_array,
        index=multiindex,
        columns=metabolite_names,
        dtype=np.float32
    )
    
    df_ranks = pd.DataFrame(
        ranks_array,
        index=multiindex,
        columns=metabolite_names,
        dtype=np.int32
    )
    
    # Save as TSVs
    raw_scores_path = output_dir / "taxonomic_raw_scores.tsv"
    distinctive_scores_path = output_dir / "taxonomic_distinctive_scores.tsv"
    ranks_path = output_dir / "taxonomic_ranks.tsv"
    
    df_raw_scores.to_csv(raw_scores_path, sep="\t")
    df_distinctive_scores.to_csv(distinctive_scores_path, sep="\t")
    df_ranks.to_csv(ranks_path, sep="\t")
    
    logger.info(f"Saved raw scores to {raw_scores_path}")
    logger.info(f"Saved distinctive scores to {distinctive_scores_path}")
    logger.info(f"Saved ranks to {ranks_path}")
    
    # Print summary statistics
    logger.info(f"Raw scores - min: {df_raw_scores.values.min():.3f}, max: {df_raw_scores.values.max():.3f}, mean: {df_raw_scores.values.mean():.3f}")
    logger.info(f"Distinctive scores - min: {df_distinctive_scores.values.min():.3f}, max: {df_distinctive_scores.values.max():.3f}, mean: {df_distinctive_scores.values.mean():.3f}")
    logger.info(f"Ranks - min: {df_ranks.values.min()}, max: {df_ranks.values.max()}, mean: {df_ranks.values.mean():.1f}")


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
        description="Generate publication figures from trained models"
    )
    parser.add_argument(
        "--assertion-type",
        choices=["consumed", "secreted", "both"],
        default="consumed",
        help="Assertion type to generate figures for: 'consumed', 'secreted', or 'both' (default: consumed)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Path to Ordinal-BPR model directory for figures 2, 3, 6, 7, and TSV export. "
            "Defaults to <models-dir>/ordinal_bpr_<assertion-type>."
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("cache/exometabolite.parquet"),
        help="Path to exometabolite parquet file (used for Figure 3 taxonomy ordering)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("trained_models"),
        help="Path to parent directory containing all trained models",
    )
    parser.add_argument(
        "--figure1-dims",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64, 128],
        help="Embedding dimensions for Figure 1 sweep",
    )
    parser.add_argument(
        "--figure1-repeats",
        type=int,
        default=5,
        help="Number of leave-one-out repeats for Figure 1",
    )
    parser.add_argument(
        "--figure1-recompute",
        action="store_true",
        help="Force recomputation of Figure 1 benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manuscript_figures_nat_microbiology_first_submission"),
        help="Output directory for publication figures",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension (used for some calculations)",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        default=[1, 2, 3, 6, 7, 8],
        help="Which figures to generate (1, 2, 3, 6, 7, 8 are implemented)",
    )
    parser.add_argument(
        "--figure8-k",
        type=int,
        default=30,
        help="Number of top predictions for precision@k and recall@k in Figure 8 (default: 30)",
    )
    parser.add_argument(
        "--no-export-tsv",
        action="store_false",
        dest="export_tsv",
        help="Skip exporting taxonomic score matrices (raw scores, distinctive scores, ranks) as TSVs",
    )
    parser.set_defaults(export_tsv=True)

    args = parser.parse_args()

    # Determine which assertion types to process
    if args.assertion_type == "both":
        assertion_types = ["consumed", "secreted"]
    else:
        assertion_types = [args.assertion_type]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for atype in assertion_types:
        at_enum = AssertionType.CONSUMED if atype == "consumed" else AssertionType.SECRETED

        # When generating both types, write into subdirectories to avoid collisions
        if len(assertion_types) > 1:
            type_output_dir = args.output_dir / atype
        else:
            type_output_dir = args.output_dir
        type_output_dir.mkdir(parents=True, exist_ok=True)

        # Derive the BPR model directory: explicit override or auto-detect from models-dir
        model_dir = args.model_dir or (args.models_dir / f"ordinal_bpr_{atype}")

        logger.info(f"=== Assertion type: {atype} | output: {type_output_dir} ===")

        # Figure 1: Model comparison (benchmark across all 6 model types)
        if 1 in args.figures:
            try:
                logger.info("Generating Figure 1: Model comparison")
                figure_1_model_comparison(
                    type_output_dir,
                    figure1_dims=args.figure1_dims,
                    figure1_repeats=args.figure1_repeats,
                    figure1_recompute=args.figure1_recompute,
                    assertion_type=at_enum,
                )
            except Exception as e:
                logger.error(f"Failed to generate Figure 1: {e}")

        # For figures 2/3/6/7 and TSV export, load the BPR model
        if any(f in args.figures for f in [2, 3, 6, 7]) or args.export_tsv:
            logger.info(f"Loading model from {model_dir}")
            try:
                artifacts = load_model_artifacts(model_dir)
                model = artifacts.model
                training_data = artifacts.training_data
                logger.info(
                    f"Loaded {type(model).__name__} with "
                    f"{training_data.n_microbes} microbes and "
                    f"{training_data.n_metabolites} metabolites"
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                continue

            # Export taxonomic score matrices as TSVs
            if args.export_tsv:
                try:
                    logger.info("Exporting taxonomic score matrices to TSV")
                    export_taxonomic_score_matrices(artifacts, type_output_dir)
                except Exception as e:
                    logger.error(f"Failed to export taxonomic score matrices: {e}")

            # Figure 3: Genus x metabolite biclustering heatmap
            if 3 in args.figures:
                try:
                    logger.info("Generating Figure 3: Distinctive-score biclustering")
                    if not isinstance(model, (BPRModel, OrdinalBPRModel)):
                        logger.warning("Figure 3 expects a BPR/Ordinal-BPR model; using loaded embeddings anyway")

                    if args.data.exists():
                        df = pd.read_parquet(args.data)
                    else:
                        logger.info("Data file not found at --data, loading exometabolite cache via loader")
                        df = load_exometabolite()

                    profiles = compute_genus_profiles_from_bpr(model, training_data)
                    categories = compute_metabolite_categories_from_profiles(profiles)
                    figure_3_heatmap_seaborn(
                        df,
                        profiles,
                        categories,
                        type_output_dir,
                        min_isolates=1,
                        exclude_universal=True,
                        max_metabolites=60,
                        assertion_type=at_enum,
                        score_col="distinctive_score",
                        figure_name="figure3_heatmap_biclustering_distinctive",
                    )
                    figure_3_heatmap_seaborn(
                        df,
                        profiles,
                        categories,
                        type_output_dir,
                        min_isolates=1,
                        exclude_universal=True,
                        max_metabolites=60,
                        assertion_type=at_enum,
                        score_col="raw_score",
                        figure_name="figure3_heatmap_biclustering_raw",
                    )
                    figure_3_heatmap_seaborn(
                        df,
                        profiles,
                        categories,
                        type_output_dir,
                        min_isolates=1,
                        exclude_universal=False,
                        max_metabolites=60,
                        assertion_type=at_enum,
                        score_col="distinctive_score",
                        compound_selection="most_measured",
                        figure_name="figure3_heatmap_biclustering_most-measured-distinctive",
                    )
                    figure_3_heatmap_seaborn(
                        df,
                        profiles,
                        categories,
                        type_output_dir,
                        min_isolates=1,
                        exclude_universal=False,
                        max_metabolites=60,
                        assertion_type=at_enum,
                        score_col="raw_score",
                        compound_selection="most_measured",
                        figure_name="figure3_heatmap_biclustering_most-measured-raw",
                    )
                except Exception as e:
                    logger.error(f"Failed to generate Figure 3: {e}")

            # Figure 2: t-SNE of microbe embeddings
            if 2 in args.figures:
                try:
                    logger.info("Generating Figure 2: t-SNE embeddings")
                    figure_2_tsne(model, training_data, type_output_dir, embedding_dim=args.embedding_dim)
                except Exception as e:
                    logger.error(f"Failed to generate Figure 2: {e}")

            # Figure 6: Rank clustermap
            if 6 in args.figures:
                try:
                    logger.info("Generating Figure 6: Rank clustermap")
                    if "df" not in locals():
                        df = load_exometabolite()
                    figure_6_rank_clustermap(model, training_data, type_output_dir, df=df, embedding_dim=args.embedding_dim)
                except Exception as e:
                    logger.error(f"Failed to generate Figure 6: {e}")

            # Figure 7: Embedding distance by taxonomy
            if 7 in args.figures:
                try:
                    logger.info("Generating Figure 7: Embedding distance by taxonomy")
                    figure_7_embedding_distance(model, training_data, type_output_dir)
                except Exception as e:
                    logger.error(f"Failed to generate Figure 7: {e}")

        # Figure 8: Experimental validation (all 5 model types)
        if 8 in args.figures:
            try:
                logger.info(f"Generating Figure 8: Experimental validation (Ordinal-BPR) using precision@{args.figure8_k}")
                figure_8_experimental_validation_all_models(
                    output_dir=type_output_dir,
                    trained_model_dir=model_dir,
                    embedding_dim=args.embedding_dim,
                    k=args.figure8_k,
                    assertion_type=at_enum,
                )
            except Exception as e:
                logger.error(f"Failed to generate Figure 8: {e}")

    logger.info("Figure generation complete")
