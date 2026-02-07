# benchmark.py
"""Benchmarking suite for metabolite preference models.

Provides three categories of benchmarks:

1. Core Model Quality: Standard leave-one-out AUC comparing BPR, WR-MF, SVD
2. Taxonomic Aggregation Quality: How well do aggregated embeddings predict?
3. Unified Taxonomic Holdout: Hold out at each taxonomic level, retrain, predict
   from the next level up. BPR only.

Usage:
    python benchmark.py --all --output-dir figures/
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import (
    BaseEmbeddingModel,
    BPRModel,
    WRMFModel,
    SVDModel,
    prepare_training_data,
    compute_taxonomy_embeddings,
)
from taxon_metabolite_profiles import compute_global_direction, project_out
logger = logging.getLogger(__name__)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class PredictionMetrics:
    """Metrics for evaluating preference predictions."""
    mean_percentile: float  # Mean percentile rank of true positives (lower = better)
    hits_at_10: float       # Fraction of top-10 predictions that are true positives
    hits_at_20: float       # Fraction of top-20 predictions that are true positives
    gross_misses: int       # True positives predicted in bottom 25%
    gross_miss_rate: float  # gross_misses / n_positives
    n_positives: int        # Number of true positives evaluated
    auc: float              # Standard AUC for ranking


def compute_metrics(
    predicted_scores: np.ndarray,
    true_positives: set[int],
    exclude_from_ranking: set[int] = None,
) -> PredictionMetrics:
    """Compute prediction metrics for a single microbe.

    Args:
        predicted_scores: Scores for all metabolites (n_metabolites,)
        true_positives: Set of positive metabolite indices
        exclude_from_ranking: Metabolites to exclude (e.g., training positives)

    Returns:
        PredictionMetrics dataclass
    """
    n_metabolites = len(predicted_scores)

    if exclude_from_ranking is None:
        exclude_from_ranking = set()

    if len(true_positives) == 0:
        return PredictionMetrics(
            mean_percentile=0.5,
            hits_at_10=0.0,
            hits_at_20=0.0,
            gross_misses=0,
            gross_miss_rate=0.0,
            n_positives=0,
            auc=0.5,
        )

    # Compute ranks (0 = highest score)
    mask = np.ones(n_metabolites, dtype=bool)
    for idx in exclude_from_ranking:
        mask[idx] = False

    valid_scores = predicted_scores.copy()
    valid_scores[~mask] = -np.inf

    ranks = np.argsort(np.argsort(-valid_scores))
    n_valid = mask.sum()

    # Percentile ranks for true positives
    percentiles = []
    for pos_idx in true_positives:
        if pos_idx in exclude_from_ranking:
            continue
        percentile = ranks[pos_idx] / n_valid
        percentiles.append(percentile)

    if len(percentiles) == 0:
        return PredictionMetrics(
            mean_percentile=0.5,
            hits_at_10=0.0,
            hits_at_20=0.0,
            gross_misses=0,
            gross_miss_rate=0.0,
            n_positives=0,
            auc=0.5,
        )

    mean_percentile = np.mean(percentiles)

    # Hits@K
    top_10 = set(np.argsort(-valid_scores)[:10])
    top_20 = set(np.argsort(-valid_scores)[:20])

    eval_positives = true_positives - exclude_from_ranking
    hits_at_10 = len(top_10 & eval_positives) / min(10, len(eval_positives))
    hits_at_20 = len(top_20 & eval_positives) / min(20, len(eval_positives))

    # Gross misses: true positives in bottom 25%
    gross_misses = sum(1 for p in percentiles if p > 0.75)
    gross_miss_rate = gross_misses / len(percentiles)

    # AUC: fraction of negatives ranked below each positive
    negative_scores = [
        predicted_scores[i] for i in range(n_metabolites)
        if i not in true_positives and i not in exclude_from_ranking
    ]

    if len(negative_scores) == 0:
        auc = 0.5
    else:
        auc_scores = []
        for pos_idx in true_positives:
            if pos_idx in exclude_from_ranking:
                continue
            pos_score = predicted_scores[pos_idx]
            n_below = sum(1 for ns in negative_scores if ns < pos_score)
            n_ties = sum(1 for ns in negative_scores if ns == pos_score)
            auc_scores.append((n_below + 0.5 * n_ties) / len(negative_scores))
        auc = np.mean(auc_scores) if auc_scores else 0.5

    return PredictionMetrics(
        mean_percentile=mean_percentile,
        hits_at_10=hits_at_10,
        hits_at_20=hits_at_20,
        gross_misses=gross_misses,
        gross_miss_rate=gross_miss_rate,
        n_positives=len(percentiles),
        auc=auc,
    )


def aggregate_metrics(metrics_list: list[PredictionMetrics]) -> dict[str, float]:
    """Aggregate metrics across multiple microbes (mean of means)."""
    if not metrics_list:
        return {
            "mean_percentile": 0.5,
            "hits_at_10": 0.0,
            "hits_at_20": 0.0,
            "gross_miss_rate": 0.0,
            "auc": 0.5,
            "n_microbes": 0,
        }

    valid = [m for m in metrics_list if m.n_positives > 0]

    if not valid:
        return {
            "mean_percentile": 0.5,
            "hits_at_10": 0.0,
            "hits_at_20": 0.0,
            "gross_miss_rate": 0.0,
            "auc": 0.5,
            "n_microbes": 0,
        }

    return {
        "mean_percentile": np.mean([m.mean_percentile for m in valid]),
        "hits_at_10": np.mean([m.hits_at_10 for m in valid]),
        "hits_at_20": np.mean([m.hits_at_20 for m in valid]),
        "gross_miss_rate": np.mean([m.gross_miss_rate for m in valid]),
        "auc": np.mean([m.auc for m in valid]),
        "n_microbes": len(valid),
    }


# =============================================================================
# Data Splitting
# =============================================================================

def leave_one_out_split(
    positives: dict[int, set[int]],
    seed: int = 42,
) -> tuple[dict[int, set[int]], list[tuple[int, int]]]:
    """Leave-one-out split for standard BPR evaluation.

    For each microbe with ≥2 positives, hold out one random positive.

    Returns:
        (train_positives, test_pairs) where test_pairs is list of (microbe_idx, metabolite_idx)
    """
    rng = np.random.default_rng(seed)

    train_positives = {}
    test_pairs = []

    for m_idx, pos_set in positives.items():
        if len(pos_set) < 2:
            train_positives[m_idx] = pos_set.copy()
            continue

        pos_list = list(pos_set)
        test_item = rng.choice(pos_list)

        train_positives[m_idx] = pos_set - {test_item}
        test_pairs.append((m_idx, test_item))

    return train_positives, test_pairs


# =============================================================================
# Benchmark 1: Core Model Quality (Figure 6 style)
# =============================================================================

def benchmark_core_models(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    dimensions: list[int] = None,
    n_repeats: int = 5,
    models_to_test: list[str] = None,
) -> pd.DataFrame:
    """Benchmark BPR vs WR-MF vs SVD using leave-one-out AUC.

    This replicates Figure 6 from the BPR paper.
    """
    if dimensions is None:
        dimensions = [4, 8, 16, 32, 64, 128]

    if models_to_test is None:
        models_to_test = ["BPR", "WR-MF", "SVD"]

    MODEL_CLASSES = {
        "BPR": BPRModel,
        "WR-MF": WRMFModel,
        "SVD": SVDModel,
    }

    data = prepare_training_data(df, preference_type)
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]

    n_pairs = sum(len(v) for v in positives.values())
    logger.info(f"Core benchmark ({preference_type}): {n_microbes} microbes, "
                f"{n_metabolites} metabolites, {n_pairs} positive pairs")

    results = []

    for repeat in range(n_repeats):
        logger.info(f"  Repeat {repeat + 1}/{n_repeats}")

        train_pos, test_pairs = leave_one_out_split(positives, seed=repeat)
        logger.info(f"    Test pairs: {len(test_pairs)}")

        for dim in dimensions:
            for model_name in models_to_test:
                model_class = MODEL_CLASSES[model_name]

                try:
                    model = model_class(embedding_dim=dim)
                    model.fit(train_pos, n_microbes, n_metabolites, verbose=False)
                    pred_matrix = model.predict_matrix()

                    metrics_list = []
                    for m_idx, test_item in test_pairs:
                        train_pos_for_m = train_pos.get(m_idx, set())
                        metrics = compute_metrics(
                            pred_matrix[m_idx],
                            true_positives={test_item},
                            exclude_from_ranking=train_pos_for_m,
                        )
                        metrics_list.append(metrics)

                    agg = aggregate_metrics(metrics_list)

                    results.append({
                        "method": model_name,
                        "dim": dim,
                        "repeat": repeat,
                        **agg,
                    })

                    logger.info(f"    {model_name} dim={dim}: AUC={agg['auc']:.3f}, "
                                f"MeanPct={agg['mean_percentile']:.3f}")

                except Exception as e:
                    logger.warning(f"    {model_name} dim={dim} failed: {e}")
                    continue

    return pd.DataFrame(results)


# =============================================================================
# Benchmark 2: Taxonomic Aggregation Quality (no retraining)
# =============================================================================

def benchmark_taxonomic_degradation(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
    models_to_test: list[str] = None,
    levels: list[str] = None,
) -> pd.DataFrame:
    """Evaluate how prediction quality degrades with taxonomic aggregation.

    Trains once on all data, then for each isolate predicts using aggregated
    embeddings at each taxonomic level (excluding the focal isolate from
    the aggregation). No retraining.
    """
    if models_to_test is None:
        models_to_test = ["BPR", "WR-MF", "SVD"]

    if levels is None:
        levels = ["isolate", "genus", "family", "order", "class"]

    MODEL_CLASSES = {
        "BPR": BPRModel,
        "WR-MF": WRMFModel,
        "SVD": SVDModel,
    }

    data = prepare_training_data(df, preference_type)
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    microbe_to_idx = data["microbe_to_idx"]
    microbe_taxonomy = data["microbe_taxonomy"]
    idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}

    logger.info(f"Taxonomic degradation benchmark ({preference_type})")

    results = []

    for model_name in models_to_test:
        logger.info(f"  Training {model_name}...")

        model_class = MODEL_CLASSES[model_name]
        model = model_class(embedding_dim=embedding_dim)
        model.fit(positives, n_microbes, n_metabolites, verbose=False)

        microbe_embeddings = model.microbe_embeddings
        metabolite_embeddings = model.metabolite_embeddings

        tax_embeddings = compute_taxonomy_embeddings(
            microbe_embeddings,
            microbe_to_idx,
            df,
            levels=["genus", "family", "order", "class"],
        )

        for level in levels:
            logger.info(f"    Evaluating at {level} level...")

            for m_idx, pos_set in positives.items():
                if len(pos_set) == 0:
                    continue

                org_id = idx_to_microbe[m_idx]
                taxonomy = microbe_taxonomy.get(org_id, {})

                if level == "isolate":
                    embedding = microbe_embeddings[m_idx]
                else:
                    taxon_name = taxonomy.get(level)
                    if not taxon_name or pd.isna(taxon_name):
                        continue

                    other_indices = [
                        microbe_to_idx[other_org]
                        for other_org, other_tax in microbe_taxonomy.items()
                        if other_tax.get(level) == taxon_name
                        and other_org != org_id
                        and other_org in microbe_to_idx
                    ]

                    if len(other_indices) == 0:
                        continue

                    embedding = microbe_embeddings[other_indices].mean(axis=0)

                scores = embedding @ metabolite_embeddings.T
                metrics = compute_metrics(scores, pos_set)

                results.append({
                    "method": model_name,
                    "level": level,
                    "microbe_idx": m_idx,
                    "organism_id": org_id,
                    "embedding_dim": embedding_dim,
                    "mean_percentile": metrics.mean_percentile,
                    "hits_at_10": metrics.hits_at_10,
                    "hits_at_20": metrics.hits_at_20,
                    "gross_miss_rate": metrics.gross_miss_rate,
                    "auc": metrics.auc,
                    "n_positives": metrics.n_positives,
                })

            level_df = pd.DataFrame([r for r in results if r["method"] == model_name and r["level"] == level])
            if len(level_df) > 0:
                logger.info(f"      n={len(level_df)}, MeanPct={level_df['mean_percentile'].mean():.3f}, "
                            f"AUC={level_df['auc'].mean():.3f}")

    return pd.DataFrame(results)


# =============================================================================
# Benchmark 3: Unified Taxonomic Holdout (retrain each time)
# =============================================================================

TAXONOMY_HIERARCHY = ["class", "order", "family", "genus"]
PREDICT_FROM = {
    "genus": "family",
    "family": "order",
    "order": "class",
}


def _get_holdout_groups(
    microbe_taxonomy: dict[str, dict[str, str]],
    microbe_to_idx: dict[str, int],
    holdout_level: str,
    min_group_size: int = 1,
) -> dict[str, list[tuple[str, int]]]:
    """Group microbes by a taxonomic level for holdout.

    Args:
        microbe_taxonomy: {organism_id: {level: taxon_name}}
        microbe_to_idx: {organism_id: int}
        holdout_level: "genus", "family", "order", or "isolate"
        min_group_size: Minimum isolates per group to include

    Returns:
        {group_name: [(organism_id, microbe_idx), ...]}
    """
    groups = defaultdict(list)

    for org_id, idx in microbe_to_idx.items():
        tax = microbe_taxonomy.get(org_id, {})

        if holdout_level == "isolate":
            group_name = org_id
        else:
            group_name = tax.get(holdout_level)
            if not group_name or pd.isna(group_name):
                continue

        groups[group_name].append((org_id, idx))

    return {g: members for g, members in groups.items() if len(members) >= min_group_size}


def _get_prediction_embedding(
    held_out_group_name: str,
    holdout_level: str,
    microbe_taxonomy: dict[str, dict[str, str]],
    microbe_to_idx: dict[str, int],
    microbe_embeddings: np.ndarray,
    held_out_indices: set[int],
) -> np.ndarray | None:
    """Compute the embedding used to predict for a held-out group.

    For isolate holdout: average embedding of other isolates in the same genus.
    For genus holdout: average embedding of other genera in the same family.
    For family holdout: average embedding of other families in the same order.
    For order holdout: average embedding of other orders in the same class.

    Returns:
        Embedding vector, or None if no relatives are available.
    """
    if holdout_level == "isolate":
        # Find the genus of this isolate, average other isolates in that genus
        predict_level = "genus"
        org_id = held_out_group_name
        tax = microbe_taxonomy.get(org_id, {})
        target_taxon = tax.get(predict_level)
    else:
        predict_level = PREDICT_FROM.get(holdout_level)
        if predict_level is None:
            return None
        # Get the predict_level taxon from any member of the held-out group
        sample_org = None
        for other_org, other_tax in microbe_taxonomy.items():
            if other_tax.get(holdout_level) == held_out_group_name:
                sample_org = other_org
                break
        if sample_org is None:
            return None
        tax = microbe_taxonomy.get(sample_org, {})
        target_taxon = tax.get(predict_level)

    if not target_taxon or pd.isna(target_taxon):
        return None

    # Collect embeddings from organisms in target_taxon but NOT in held-out set
    relative_indices = [
        microbe_to_idx[other_org]
        for other_org, other_tax in microbe_taxonomy.items()
        if other_tax.get(predict_level) == target_taxon
        and other_org in microbe_to_idx
        and microbe_to_idx[other_org] not in held_out_indices
    ]

    if len(relative_indices) == 0:
        return None

    return microbe_embeddings[relative_indices].mean(axis=0)


def benchmark_unified_holdout(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
    holdout_levels: list[str] = None,
) -> pd.DataFrame:
    """Unified taxonomic holdout benchmark.

    For each holdout level (order, family, genus, isolate):
      - For each group at that level, hold out all its isolates from training
      - Retrain BPR on remaining data
      - Predict for each held-out isolate using the embedding from the
        next taxonomic level up (averaged from non-held-out relatives)
      - Record per-isolate metrics

    This produces one large results table with columns:
      holdout_level, held_out_identity, predict_from_level, predict_from_identity,
      organism_id, microbe_idx, + all metric columns

    Args:
        df: Exometabolite dataframe
        preference_type: "consumed" or "secreted"
        embedding_dim: Embedding dimension for BPR
        holdout_levels: Which levels to hold out. Default: ["order", "family", "genus", "isolate"]

    Returns:
        DataFrame with per-isolate results across all holdout levels
    """
    if holdout_levels is None:
        holdout_levels = ["order", "family", "genus", "isolate"]

    data = prepare_training_data(df, preference_type)
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    microbe_to_idx = data["microbe_to_idx"]
    microbe_taxonomy = data["microbe_taxonomy"]
    idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}

    n_pairs = sum(len(v) for v in positives.values())
    logger.info(f"Unified holdout benchmark ({preference_type}): "
                f"{n_microbes} microbes, {n_metabolites} metabolites, {n_pairs} pairs")

    results = []

    for holdout_level in holdout_levels:
        predict_from_level = (
            "genus" if holdout_level == "isolate"
            else PREDICT_FROM.get(holdout_level)
        )

        if predict_from_level is None:
            logger.warning(f"  No prediction level defined for holdout_level={holdout_level}, skipping")
            continue

        # Get groups to hold out
        min_size = 1 if holdout_level == "isolate" else 1
        groups = _get_holdout_groups(microbe_taxonomy, microbe_to_idx, holdout_level, min_group_size=min_size)

        logger.info(f"  Holdout level: {holdout_level} → predict from: {predict_from_level}")
        logger.info(f"    {len(groups)} groups to evaluate")

        n_skipped = 0
        n_evaluated = 0

        for group_name, members in groups.items():
            held_out_indices = set(idx for _, idx in members)

            # Build training positives excluding held-out group
            train_positives = {
                m_idx: pos for m_idx, pos in positives.items()
                if m_idx not in held_out_indices
            }

            if len(train_positives) == 0:
                n_skipped += 1
                continue

            # Train BPR on remaining data
            model = BPRModel(embedding_dim=embedding_dim)
            model.fit(train_positives, n_microbes, n_metabolites, verbose=False)

            microbe_embeddings = model.microbe_embeddings
            metabolite_embeddings = model.metabolite_embeddings
            # Compute global direction from training embeddings
            global_dir = compute_global_direction(microbe_embeddings)
            # Get prediction embedding from next level up
            pred_embedding = _get_prediction_embedding(
                held_out_group_name=group_name,
                holdout_level=holdout_level,
                microbe_taxonomy=microbe_taxonomy,
                microbe_to_idx=microbe_to_idx,
                microbe_embeddings=microbe_embeddings,
                held_out_indices=held_out_indices,
            )

            if pred_embedding is None:
                n_skipped += 1
                continue

            # Determine what the predict_from identity is
            if holdout_level == "isolate":
                predict_from_identity = microbe_taxonomy.get(group_name, {}).get("genus", "unknown")
            else:
                sample_org = members[0][0]
                predict_from_identity = microbe_taxonomy.get(sample_org, {}).get(predict_from_level, "unknown")



            # Raw scores (as before)
            raw_scores = pred_embedding @ metabolite_embeddings.T

            # Distinctive scores: remove global popularity from prediction embedding
            distinctive_embedding = project_out(pred_embedding, global_dir)
            distinctive_scores = distinctive_embedding @ metabolite_embeddings.T

            for org_id, m_idx in members:
                pos_set = positives.get(m_idx, set())
                if len(pos_set) == 0:
                    continue

                raw_metrics = compute_metrics(raw_scores, pos_set)
                distinctive_metrics = compute_metrics(distinctive_scores, pos_set)

                base_row = {
                    "holdout_level": holdout_level,
                    "held_out_identity": group_name,
                    "predict_from_level": predict_from_level,
                    "predict_from_identity": predict_from_identity,
                    "organism_id": org_id,
                    "microbe_idx": m_idx,
                    "embedding_dim": embedding_dim,
                    "n_positives": raw_metrics.n_positives,
                }

                results.append({
                    **base_row,
                    "score_type": "raw",
                    "mean_percentile": raw_metrics.mean_percentile,
                    "hits_at_10": raw_metrics.hits_at_10,
                    "hits_at_20": raw_metrics.hits_at_20,
                    "gross_miss_rate": raw_metrics.gross_miss_rate,
                    "auc": raw_metrics.auc,
                })

                results.append({
                    **base_row,
                    "score_type": "distinctive",
                    "mean_percentile": distinctive_metrics.mean_percentile,
                    "hits_at_10": distinctive_metrics.hits_at_10,
                    "hits_at_20": distinctive_metrics.hits_at_20,
                    "gross_miss_rate": distinctive_metrics.gross_miss_rate,
                    "auc": distinctive_metrics.auc,
                })

                n_evaluated += 1

        logger.info(f"    Evaluated {n_evaluated} isolates, skipped {n_skipped} groups (no relatives)")

        # Log summary for this holdout level
        level_rows = [r for r in results if r["holdout_level"] == holdout_level]
        if level_rows:
            level_df = pd.DataFrame(level_rows)
            logger.info(f"    Summary: MeanPct={level_df['mean_percentile'].mean():.3f}, "
                        f"AUC={level_df['auc'].mean():.3f}, "
                        f"n_isolates={len(level_df)}")

    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

def plot_core_benchmark(results_df: pd.DataFrame, output_dir: Path, preference_type: str):
    """Plot Figure 6-style benchmark results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    style = {
        "BPR": {"color": "#d62728", "marker": "s"},
        "WR-MF": {"color": "#1f77b4", "marker": "o"},
        "SVD": {"color": "#2ca02c", "marker": "^"},
    }

    ax = axes[0]
    for method in ["BPR", "WR-MF", "SVD"]:
        method_df = results_df[results_df["method"] == method]
        if len(method_df) == 0:
            continue
        summary = method_df.groupby("dim")["auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["dim"], summary["mean"], yerr=summary["std"],
            label=method, marker=style[method]["marker"],
            color=style[method]["color"], capsize=4, linewidth=2,
        )
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("AUC")
    ax.set_title(f"Leave-One-Out AUC ({preference_type})")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method in ["BPR", "WR-MF", "SVD"]:
        method_df = results_df[results_df["method"] == method]
        if len(method_df) == 0:
            continue
        summary = method_df.groupby("dim")["mean_percentile"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["dim"], summary["mean"], yerr=summary["std"],
            label=method, marker=style[method]["marker"],
            color=style[method]["color"], capsize=4, linewidth=2,
        )
    ax.axhline(0.5, color="gray", linestyle="--", label="Random")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Mean percentile of positives")
    ax.set_title(f"Mean Percentile (lower = better)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"core_benchmark_{preference_type}.{ext}", dpi=150)
    plt.close()
    logger.info(f"Saved core_benchmark_{preference_type}.png/pdf")


def plot_taxonomic_degradation(results_df: pd.DataFrame, output_dir: Path, preference_type: str):
    """Plot how prediction quality degrades with taxonomic aggregation (no retraining)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    level_order = ["isolate", "genus", "family", "order", "class"]
    style = {
        "BPR": {"color": "#d62728", "marker": "s"},
        "WR-MF": {"color": "#1f77b4", "marker": "o"},
        "SVD": {"color": "#2ca02c", "marker": "^"},
    }

    ax = axes[0]
    for method in ["BPR", "WR-MF", "SVD"]:
        method_df = results_df[results_df["method"] == method]
        if len(method_df) == 0:
            continue
        summary = method_df.groupby("level")["mean_percentile"].agg(["mean", "std"])
        summary = summary.reindex(level_order).dropna()
        x_pos = [level_order.index(l) for l in summary.index]
        ax.errorbar(
            x_pos, summary["mean"], yerr=summary["std"],
            label=method, marker=style[method]["marker"],
            color=style[method]["color"], capsize=4, linewidth=2,
        )
    ax.axhline(0.5, color="gray", linestyle="--", label="Random")
    ax.set_xticks(range(len(level_order)))
    ax.set_xticklabels(level_order)
    ax.set_xlabel("Embedding source")
    ax.set_ylabel("Mean percentile (lower = better)")
    ax.set_title(f"Prediction Quality vs Taxonomic Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for method in ["BPR", "WR-MF", "SVD"]:
        method_df = results_df[results_df["method"] == method]
        if len(method_df) == 0:
            continue
        summary = method_df.groupby("level")["auc"].agg(["mean", "std"])
        summary = summary.reindex(level_order).dropna()
        x_pos = [level_order.index(l) for l in summary.index]
        ax.errorbar(
            x_pos, summary["mean"], yerr=summary["std"],
            label=method, marker=style[method]["marker"],
            color=style[method]["color"], capsize=4, linewidth=2,
        )
    ax.axhline(0.5, color="gray", linestyle="--", label="Random")
    ax.set_xticks(range(len(level_order)))
    ax.set_xticklabels(level_order)
    ax.set_xlabel("Embedding source")
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC vs Taxonomic Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"taxonomic_degradation_{preference_type}.{ext}", dpi=150)
    plt.close()
    logger.info(f"Saved taxonomic_degradation_{preference_type}.png/pdf")

def plot_unified_holdout(results_df: pd.DataFrame, output_dir: Path, preference_type: str):
    """Plot unified taxonomic holdout results with raw and distinctive scores."""

    has_score_type = "score_type" in results_df.columns

    if not has_score_type:
        # Legacy format without score_type column
        _plot_unified_holdout_legacy(results_df, output_dir, preference_type)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    level_order = ["order", "family", "genus", "isolate"]
    present_levels = [l for l in level_order if l in results_df["holdout_level"].values]

    colors = {"raw": "#377eb8", "distinctive": "#e41a1c"}

    for ax, metric, ylabel, title in [
        (axes[0], "mean_percentile", "Mean percentile (lower = better)", "Prediction Quality by Holdout Level"),
        (axes[1], "auc", "AUC", "AUC by Holdout Level"),
    ]:
        positions = []
        data_boxes = []
        box_colors = []
        tick_positions = []
        tick_labels = []

        for i, level in enumerate(present_levels):
            center = i * 3
            for j, score_type in enumerate(["raw", "distinctive"]):
                vals = results_df[
                    (results_df["holdout_level"] == level) &
                    (results_df["score_type"] == score_type)
                ][metric].values

                if len(vals) == 0:
                    continue

                pos = center + j - 0.5
                positions.append(pos)
                data_boxes.append(vals)
                box_colors.append(colors[score_type])

            tick_positions.append(center)
            tick_labels.append(f"Hold out {level}")

        if data_boxes:
            bp = ax.boxplot(data_boxes, positions=positions, widths=0.8,
                            patch_artist=True)
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=colors["raw"], alpha=0.6, label="Raw"),
                Patch(facecolor=colors["distinctive"], alpha=0.6, label="Distinctive"),
            ],
            loc="best",
        )

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"unified_holdout_{preference_type}.{ext}", dpi=150)
    plt.close()
    logger.info(f"Saved unified_holdout_{preference_type}.png/pdf")


def plot_unified_holdout_summary_line(results_df: pd.DataFrame, output_dir: Path, preference_type: str):
    """Plot line/point summary of unified holdout: median + IQR by holdout level."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    level_order = ["order", "family", "genus", "isolate"]
    present_levels = [l for l in level_order if l in results_df["holdout_level"].values]

    color = "#d62728"

    for ax, metric, ylabel, title in [
        (axes[0], "mean_percentile", "Mean percentile (lower = better)", "Prediction Quality vs Holdout Level"),
        (axes[1], "auc", "AUC", "AUC vs Holdout Level"),
    ]:
        medians = []
        q25s = []
        q75s = []

        for level in present_levels:
            vals = results_df[results_df["holdout_level"] == level][metric].values
            medians.append(np.median(vals))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))

        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        x = np.arange(len(present_levels))
        ax.plot(x, medians, marker="s", color=color, linewidth=2, markersize=8, zorder=3)
        ax.fill_between(x, q25s, q75s, color=color, alpha=0.15, zorder=1)
        ax.errorbar(x, medians, yerr=[medians - q25s, q75s - medians],
                    color=color, capsize=6, linewidth=0, elinewidth=1.5, zorder=2)

        ax.axhline(0.5, color="gray", linestyle="--", label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(present_levels)
        ax.set_xlabel("Holdout level")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Annotate n per level
        for i, level in enumerate(present_levels):
            n = len(results_df[results_df["holdout_level"] == level])
            ax.annotate(f"n={n}", (x[i], q75s[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"unified_holdout_line_{preference_type}.{ext}", dpi=150)
    plt.close()
    logger.info(f"Saved unified_holdout_line_{preference_type}.png/pdf")


def plot_summary_table(
    core_results: pd.DataFrame,
    tax_results: pd.DataFrame,
    holdout_results: pd.DataFrame,
    output_dir: Path,
    preference_type: str,
):
    """Generate summary table as figure."""

    summary_data = []

    for method in ["BPR", "WR-MF", "SVD"]:
        row = {"Method": method}

        # Core benchmark (best dimension)
        core_method = core_results[core_results["method"] == method]
        if len(core_method) > 0:
            best_dim = core_method.groupby("dim")["auc"].mean().idxmax()
            best_core = core_method[core_method["dim"] == best_dim]
            row["Core AUC"] = f"{best_core['auc'].mean():.3f} ± {best_core['auc'].std():.3f}"
            row["Best Dim"] = int(best_dim)

        # Taxonomic degradation at genus level
        tax_method = tax_results[(tax_results["method"] == method) & (tax_results["level"] == "genus")]
        if len(tax_method) > 0:
            row["Genus AUC"] = f"{tax_method['auc'].mean():.3f}"
            row["Genus MeanPct"] = f"{tax_method['mean_percentile'].mean():.3f}"

        summary_data.append(row)

    # Add unified holdout summary (BPR only)
    if len(holdout_results) > 0:
        for holdout_level in ["isolate", "genus", "family", "order"]:
            level_df = holdout_results[holdout_results["holdout_level"] == holdout_level]
            if len(level_df) > 0:
                # Add to BPR row
                for row in summary_data:
                    if row["Method"] == "BPR":
                        row[f"Holdout-{holdout_level} AUC"] = f"{level_df['auc'].mean():.3f}"

    summary_df = pd.DataFrame(summary_data)

    summary_df.to_csv(output_dir / f"summary_{preference_type}.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for j, col in enumerate(summary_df.columns):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    plt.title(f"Benchmark Summary ({preference_type})", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"summary_{preference_type}.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary_{preference_type}.csv/png/pdf")

    return summary_df


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_benchmarks(
    df: pd.DataFrame,
    output_dir: Path,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
    quick: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run all benchmarks and generate plots.

    Args:
        df: Exometabolite dataframe
        output_dir: Directory for output files
        preference_type: "consumed" or "secreted"
        embedding_dim: Embedding dimension for taxonomic benchmarks
        quick: If True, run reduced benchmarks for testing

    Returns:
        Dict with benchmark result DataFrames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Benchmark 1: Core model comparison
    logger.info("=" * 60)
    logger.info("Benchmark 1: Core Model Comparison")
    logger.info("=" * 60)

    dimensions = [8, 32] if quick else [4, 8, 16, 32, 64, 128]
    n_repeats = 2 if quick else 5

    core_results = benchmark_core_models(
        df,
        preference_type=preference_type,
        dimensions=dimensions,
        n_repeats=n_repeats,
    )
    results["core"] = core_results
    core_results.to_csv(output_dir / f"core_results_{preference_type}.csv", index=False)
    plot_core_benchmark(core_results, output_dir, preference_type)

    # Benchmark 2: Taxonomic degradation (no retraining)
    logger.info("=" * 60)
    logger.info("Benchmark 2: Taxonomic Degradation (no retraining)")
    logger.info("=" * 60)

    tax_results = benchmark_taxonomic_degradation(
        df,
        preference_type=preference_type,
        embedding_dim=embedding_dim,
    )
    results["taxonomic"] = tax_results
    tax_results.to_csv(output_dir / f"taxonomic_results_{preference_type}.csv", index=False)
    plot_taxonomic_degradation(tax_results, output_dir, preference_type)

    # Benchmark 3: Unified taxonomic holdout (retrain each time)
    logger.info("=" * 60)
    logger.info("Benchmark 3: Unified Taxonomic Holdout")
    logger.info("=" * 60)

    holdout_levels = ["genus", "isolate"] if quick else ["order", "family", "genus", "isolate"]

    holdout_results = benchmark_unified_holdout(
        df,
        preference_type=preference_type,
        embedding_dim=embedding_dim,
        holdout_levels=holdout_levels,
    )
    results["unified_holdout"] = holdout_results
    holdout_results.to_csv(output_dir / f"unified_holdout_results_{preference_type}.csv", index=False)
    plot_unified_holdout(holdout_results, output_dir, preference_type)
    plot_unified_holdout_summary_line(holdout_results, output_dir, preference_type)

    # Summary table
    logger.info("=" * 60)
    logger.info("Generating Summary")
    logger.info("=" * 60)

    summary_df = plot_summary_table(
        core_results, tax_results, holdout_results, output_dir, preference_type
    )
    results["summary"] = summary_df

    logger.info(f"All benchmarks complete. Results saved to {output_dir}")

    return results


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

    parser = argparse.ArgumentParser(description="Run metabolite preference benchmarks")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("cache/exometabolite.parquet"),
        help="Path to exometabolite parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--preference-type",
        choices=["consumed", "secreted"],
        default="consumed",
        help="Which preference type to benchmark",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for taxonomic benchmarks",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced benchmarks for testing",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Only run core model benchmark",
    )
    parser.add_argument(
        "--taxonomic-only",
        action="store_true",
        help="Only run taxonomic degradation benchmark",
    )
    parser.add_argument(
        "--holdout-only",
        action="store_true",
        help="Only run unified taxonomic holdout benchmark",
    )

    args = parser.parse_args()

    logger.info(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.core_only:
        dimensions = [8, 32] if args.quick else [4, 8, 16, 32, 64, 128]
        n_repeats = 2 if args.quick else 5
        results = benchmark_core_models(
            df,
            preference_type=args.preference_type,
            dimensions=dimensions,
            n_repeats=n_repeats,
        )
        results.to_csv(args.output_dir / f"core_results_{args.preference_type}.csv", index=False)
        plot_core_benchmark(results, args.output_dir, args.preference_type)

    elif args.taxonomic_only:
        results = benchmark_taxonomic_degradation(
            df,
            preference_type=args.preference_type,
            embedding_dim=args.embedding_dim,
        )
        results.to_csv(args.output_dir / f"taxonomic_results_{args.preference_type}.csv", index=False)
        plot_taxonomic_degradation(results, args.output_dir, args.preference_type)

    elif args.holdout_only:
        holdout_levels = ["genus", "isolate"] if args.quick else ["order", "family", "genus", "isolate"]
        results = benchmark_unified_holdout(
            df,
            preference_type=args.preference_type,
            embedding_dim=args.embedding_dim,
            holdout_levels=holdout_levels,
        )
        results.to_csv(args.output_dir / f"unified_holdout_results_{args.preference_type}.csv", index=False)
        plot_unified_holdout(results, args.output_dir, args.preference_type)
        plot_unified_holdout_summary_line(results, args.output_dir, args.preference_type)

    else:
        run_all_benchmarks(
            df,
            args.output_dir,
            preference_type=args.preference_type,
            embedding_dim=args.embedding_dim,
            quick=args.quick,
        )
