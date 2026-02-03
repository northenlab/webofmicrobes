"""
benchmarking.py

Benchmark comparison of BPR, WR-MF, and SVD for metabolite preference prediction.
Also includes validation analyses for the implicit feedback assumption.

Usage:
    python benchmarking.py
    
Or from build.py:
    from benchmarking import run_benchmark, run_assertion_validation
    run_benchmark(df, output_dir=figures_dir)
    run_assertion_validation(df, model_wrapper, output_dir=figures_dir)
"""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


# ============================================
# Data preparation
# ============================================

def prepare_data(df: pd.DataFrame, preference_type: str = "consumed") -> dict:
    """Build mappings and positive sets from exometabolite data."""
    df = df[df["chebi_id"].notna()].copy()
    
    microbes = df["organism_id"].unique()
    metabolites = df["chebi_id"].unique()
    
    microbe_to_idx = {m: i for i, m in enumerate(microbes)}
    metabolite_to_idx = {int(m): i for i, m in enumerate(metabolites)}
    
    n_microbes = len(microbes)
    n_metabolites = len(metabolites)
    
    # Build positive set per microbe
    positives = defaultdict(set)
    for _, row in df.iterrows():
        if row["assertion"] == preference_type:
            m_idx = microbe_to_idx[row["organism_id"]]
            c_idx = metabolite_to_idx[int(row["chebi_id"])]
            positives[m_idx].add(c_idx)
    
    return {
        "n_microbes": n_microbes,
        "n_metabolites": n_metabolites,
        "positives": dict(positives),
        "microbe_to_idx": microbe_to_idx,
        "metabolite_to_idx": metabolite_to_idx,
    }


def leave_one_out_split(positives: dict, seed: int = 42) -> tuple[dict, list]:
    """
    Leave-one-out split as in BPR paper.
    
    For each microbe with ≥2 positives, hold out one random positive for testing.
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


def compute_auc(
    predictions: np.ndarray, 
    train_positives: dict, 
    test_pairs: list, 
    n_metabolites: int
) -> float:
    """Compute AUC for ranking evaluation (as in BPR paper Eq. 2)."""
    auc_scores = []
    
    for m_idx, test_item in test_pairs:
        scores = predictions[m_idx]
        test_score = scores[test_item]
        
        train_pos = train_positives.get(m_idx, set())
        negative_indices = [
            j for j in range(n_metabolites) 
            if j not in train_pos and j != test_item
        ]
        
        if len(negative_indices) == 0:
            continue
        
        neg_scores = scores[negative_indices]
        n_below = np.sum(neg_scores < test_score)
        n_ties = np.sum(neg_scores == test_score)
        
        auc = (n_below + 0.5 * n_ties) / len(negative_indices)
        auc_scores.append(auc)
    
    return np.mean(auc_scores) if auc_scores else 0.5


# ============================================
# Model implementations
# ============================================

class BPRModelBenchmark(nn.Module):
    """Minimal BPR model for benchmarking."""
    
    def __init__(self, n_microbes: int, n_metabolites: int, embedding_dim: int):
        super().__init__()
        self.microbe_emb = nn.Embedding(n_microbes, embedding_dim)
        self.metabolite_emb = nn.Embedding(n_metabolites, embedding_dim)
        nn.init.normal_(self.microbe_emb.weight, std=0.01)
        nn.init.normal_(self.metabolite_emb.weight, std=0.01)
    
    def forward(self, m_idx, pos_idx, neg_idx):
        m = self.microbe_emb(m_idx)
        pos = self.metabolite_emb(pos_idx)
        neg = self.metabolite_emb(neg_idx)
        return (m * pos).sum(1), (m * neg).sum(1)
    
    def predict_matrix(self) -> np.ndarray:
        with torch.no_grad():
            return (self.microbe_emb.weight @ self.metabolite_emb.weight.T).numpy()


def train_bpr(
    train_positives: dict, 
    n_microbes: int, 
    n_metabolites: int, 
    embedding_dim: int,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 0.01,
) -> np.ndarray:
    """Train BPR and return prediction matrix."""
    model = BPRModelBenchmark(n_microbes, n_metabolites, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    microbe_list = [m for m, pos in train_positives.items() if len(pos) > 0]
    
    for _ in range(n_epochs):
        m_batch, pos_batch, neg_batch = [], [], []
        
        for _ in range(batch_size):
            m_idx = np.random.choice(microbe_list)
            pos_idx = np.random.choice(list(train_positives[m_idx]))
            neg_idx = np.random.randint(n_metabolites)
            while neg_idx in train_positives[m_idx]:
                neg_idx = np.random.randint(n_metabolites)
            
            m_batch.append(m_idx)
            pos_batch.append(pos_idx)
            neg_batch.append(neg_idx)
        
        pos_scores, neg_scores = model(
            torch.tensor(m_batch),
            torch.tensor(pos_batch),
            torch.tensor(neg_batch),
        )
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model.predict_matrix()


def train_wrmf(
    train_positives: dict,
    n_microbes: int,
    n_metabolites: int,
    embedding_dim: int,
    iterations: int = 50,
    regularization: float = 0.1,
) -> np.ndarray | None:
    """Train WR-MF (ALS) and return prediction matrix."""
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        logger.warning("implicit library not installed - pip install implicit")
        return None
    
    rows, cols, data = [], [], []
    for m_idx, pos_set in train_positives.items():
        for c_idx in pos_set:
            rows.append(m_idx)
            cols.append(c_idx)
            data.append(1.0)
    
    sparse_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_microbes, n_metabolites),
    )
    
    model = AlternatingLeastSquares(
        factors=embedding_dim,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
    )
    
    model.fit(sparse_matrix, show_progress=False)
    
    return model.user_factors @ model.item_factors.T


def train_svd(
    train_positives: dict,
    n_microbes: int,
    n_metabolites: int,
    embedding_dim: int,
) -> np.ndarray:
    """Train truncated SVD and return prediction matrix."""
    M = np.zeros((n_microbes, n_metabolites))
    for m_idx, pos_set in train_positives.items():
        for c_idx in pos_set:
            M[m_idx, c_idx] = 1.0
    
    n_components = min(embedding_dim, min(M.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    M_reduced = svd.fit_transform(M)
    
    return M_reduced @ svd.components_


def compute_most_popular(
    train_positives: dict,
    n_microbes: int,
    n_metabolites: int,
) -> np.ndarray:
    """Baseline: rank by global popularity (non-personalized)."""
    popularity = np.zeros(n_metabolites)
    for pos_set in train_positives.values():
        for c_idx in pos_set:
            popularity[c_idx] += 1
    
    return np.tile(popularity, (n_microbes, 1))


# ============================================
# Main benchmark
# ============================================

def run_benchmark(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    dimensions: list[int] | None = None,
    n_repeats: int = 5,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run benchmark comparing BPR, WR-MF, SVD, and most-popular baseline.
    """
    if dimensions is None:
        dimensions = [2, 4, 8, 16, 32, 64, 128]
    
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = prepare_data(df, preference_type)
    n_microbes = data["n_microbes"]
    n_metabolites = data["n_metabolites"]
    positives = data["positives"]
    
    n_pairs = sum(len(v) for v in positives.values())
    n_microbes_with_pos = sum(1 for v in positives.values() if len(v) > 0)
    avg_pos = n_pairs / n_microbes_with_pos if n_microbes_with_pos > 0 else 0
    
    logger.info(f"Benchmark ({preference_type}): {n_microbes} microbes, "
                f"{n_metabolites} metabolites, {n_pairs} positive pairs")
    logger.info(f"  Microbes with positives: {n_microbes_with_pos}, "
                f"avg positives/microbe: {avg_pos:.1f}")
    
    results = []
    
    for repeat in range(n_repeats):
        logger.info(f"  Repeat {repeat + 1}/{n_repeats}")
        
        train_pos, test_pairs = leave_one_out_split(positives, seed=repeat)
        logger.info(f"    Test pairs: {len(test_pairs)}")
        
        # Most-popular baseline (dimension-independent)
        pred_pop = compute_most_popular(train_pos, n_microbes, n_metabolites)
        auc_pop = compute_auc(pred_pop, train_pos, test_pairs, n_metabolites)
        
        for dim in dimensions:
            results.append({
                "method": "Most Popular",
                "dim": dim,
                "repeat": repeat,
                "auc": auc_pop,
            })
            
            # BPR
            pred_bpr = train_bpr(train_pos, n_microbes, n_metabolites, dim)
            auc_bpr = compute_auc(pred_bpr, train_pos, test_pairs, n_metabolites)
            results.append({
                "method": "BPR-MF",
                "dim": dim,
                "repeat": repeat,
                "auc": auc_bpr,
            })
            
            # WR-MF
            pred_wrmf = train_wrmf(train_pos, n_microbes, n_metabolites, dim)
            if pred_wrmf is not None:
                auc_wrmf = compute_auc(pred_wrmf, train_pos, test_pairs, n_metabolites)
                results.append({
                    "method": "WR-MF",
                    "dim": dim,
                    "repeat": repeat,
                    "auc": auc_wrmf,
                })
            else:
                auc_wrmf = np.nan
            
            # SVD
            pred_svd = train_svd(train_pos, n_microbes, n_metabolites, dim)
            auc_svd = compute_auc(pred_svd, train_pos, test_pairs, n_metabolites)
            results.append({
                "method": "SVD-MF",
                "dim": dim,
                "repeat": repeat,
                "auc": auc_svd,
            })
            
            logger.info(f"    dim={dim}: BPR={auc_bpr:.3f}, WR-MF={auc_wrmf:.3f}, SVD={auc_svd:.3f}")
    
    results_df = pd.DataFrame(results)
    
    csv_path = output_dir / f"benchmark_{preference_type}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    plot_benchmark(results_df, preference_type, output_dir)
    
    return results_df


def plot_benchmark(results_df: pd.DataFrame, preference_type: str, output_dir: Path):
    """Generate Figure 6-style plot: AUC vs dimensions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    style = {
        "BPR-MF": {"color": "#d62728", "marker": "s"},
        "WR-MF": {"color": "#1f77b4", "marker": "o"},
        "SVD-MF": {"color": "#2ca02c", "marker": "^"},
        "Most Popular": {"color": "#7f7f7f", "marker": "+", "linestyle": "--"},
    }
    
    for method in ["BPR-MF", "WR-MF", "SVD-MF", "Most Popular"]:
        method_df = results_df[results_df["method"] == method]
        if len(method_df) == 0:
            continue
        
        summary = method_df.groupby("dim")["auc"].agg(["mean", "std"]).reset_index()
        
        ax.errorbar(
            summary["dim"],
            summary["mean"],
            yerr=summary["std"],
            label=method,
            marker=style[method]["marker"],
            color=style[method]["color"],
            linestyle=style[method].get("linestyle", "-"),
            markersize=8,
            capsize=4,
            linewidth=2,
        )
    
    ax.set_xlabel("Number of dimensions", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"Metabolite Preference Prediction: {preference_type.title()}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xscale("log", base=2)
    
    dims = sorted(results_df["dim"].unique())
    ax.set_xticks(dims)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    
    ymin = max(0.4, results_df["auc"].min() - 0.05)
    ymax = min(1.0, results_df["auc"].max() + 0.02)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    for ext in ["png", "pdf"]:
        path = output_dir / f"benchmark_{preference_type}.{ext}"
        fig.savefig(path, dpi=150)
    
    logger.info(f"Saved figure to {output_dir}/benchmark_{preference_type}.png/pdf")
    plt.close()


# ============================================
# Assertion validation analyses
# ============================================

def analyze_assertion_overlap(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Analyze co-occurrence of consumption and secretion for the same microbe-metabolite pairs.
    
    This addresses the question: if a microbe can secrete a metabolite, 
    can it also consume it (and vice versa)?
    """
    df = df[df["chebi_id"].notna()].copy()
    
    # Get unique microbe-metabolite pairs by assertion type
    pairs_by_assertion = {}
    for assertion in df["assertion"].unique():
        pairs = set(
            zip(df[df["assertion"] == assertion]["organism_id"],
                df[df["assertion"] == assertion]["chebi_id"].astype(int))
        )
        pairs_by_assertion[assertion] = pairs
        logger.info(f"  {assertion}: {len(pairs)} unique pairs")
    
    consumed = pairs_by_assertion.get("consumed", set())
    secreted = pairs_by_assertion.get("secreted", set())
    unchanged = pairs_by_assertion.get("unchanged", set())
    
    # Compute overlaps
    both_consumed_secreted = consumed & secreted
    consumed_and_unchanged = consumed & unchanged
    secreted_and_unchanged = secreted & unchanged
    
    results = {
        "consumed_only": len(consumed - secreted - unchanged),
        "secreted_only": len(secreted - consumed - unchanged),
        "unchanged_only": len(unchanged - consumed - secreted),
        "consumed_and_secreted": len(both_consumed_secreted),
        "consumed_and_unchanged": len(consumed_and_unchanged),
        "secreted_and_unchanged": len(secreted_and_unchanged),
        "total_consumed": len(consumed),
        "total_secreted": len(secreted),
        "total_unchanged": len(unchanged),
    }
    
    logger.info(f"\nAssertion Overlap Analysis:")
    logger.info(f"  Consumed only: {results['consumed_only']}")
    logger.info(f"  Secreted only: {results['secreted_only']}")
    logger.info(f"  Both consumed AND secreted: {results['consumed_and_secreted']}")
    
    if len(consumed) > 0:
        overlap_pct = 100 * results["consumed_and_secreted"] / len(consumed)
        logger.info(f"  Overlap as % of consumed: {overlap_pct:.1f}%")
        results["overlap_pct_of_consumed"] = overlap_pct
    
    if len(secreted) > 0:
        overlap_pct = 100 * results["consumed_and_secreted"] / len(secreted)
        logger.info(f"  Overlap as % of secreted: {overlap_pct:.1f}%")
        results["overlap_pct_of_secreted"] = overlap_pct
    
    # Venn diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        from matplotlib_venn import venn3
        venn3(
            [consumed, secreted, unchanged],
            set_labels=("Consumed", "Secreted", "Unchanged"),
            ax=ax,
        )
        ax.set_title("Microbe-Metabolite Pair Overlap by Assertion Type")
    except ImportError:
        # Fallback: bar chart
        labels = ["Consumed\nonly", "Secreted\nonly", "Both", "Unchanged\nonly"]
        values = [
            results["consumed_only"],
            results["secreted_only"],
            results["consumed_and_secreted"],
            results["unchanged_only"],
        ]
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#95a5a6"]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Number of microbe-metabolite pairs")
        ax.set_title("Microbe-Metabolite Pair Overlap by Assertion Type")
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, str(v), ha="center", fontsize=10)
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"assertion_overlap.{ext}", dpi=150)
    plt.close()
    
    logger.info(f"Saved assertion_overlap figure to {output_dir}")
    
    return results


def cross_prediction_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    embedding_dim: int = 32,
) -> pd.DataFrame:
    """
    Train model on 'consumed', test if it predicts 'secreted' better than chance.
    
    If consumption and secretion reflect similar underlying capability,
    the consumption model should rank secreted metabolites higher than unchanged.
    """
    df = df[df["chebi_id"].notna()].copy()
    
    # Train consumption model
    logger.info("Training consumption model for cross-prediction analysis...")
    data = prepare_data(df, "consumed")
    
    pred_matrix = train_bpr(
        data["positives"],
        data["n_microbes"],
        data["n_metabolites"],
        embedding_dim,
        n_epochs=1000,
    )
    
    microbe_to_idx = data["microbe_to_idx"]
    metabolite_to_idx = data["metabolite_to_idx"]
    n_metabolites = data["n_metabolites"]
    
    # Score all observed pairs by assertion type
    results = {"chebi_id": [], "organism_id": [], "predicted_rank_pct": [], "assertion": []}
    
    for assertion in ["consumed", "secreted", "unchanged"]:
        subset = df[df["assertion"] == assertion].drop_duplicates(["organism_id", "chebi_id"])
        
        for _, row in subset.iterrows():
            org_id = row["organism_id"]
            chebi_id = int(row["chebi_id"])
            
            if org_id not in microbe_to_idx:
                continue
            if chebi_id not in metabolite_to_idx:
                continue
            
            m_idx = microbe_to_idx[org_id]
            c_idx = metabolite_to_idx[chebi_id]
            
            scores = pred_matrix[m_idx]
            rank = (scores > scores[c_idx]).sum()
            percentile = rank / n_metabolites
            
            results["chebi_id"].append(chebi_id)
            results["organism_id"].append(org_id)
            results["predicted_rank_pct"].append(percentile)
            results["assertion"].append(assertion)
    
    results_df = pd.DataFrame(results)
    
    # Statistical tests
    logger.info("\nCross-Prediction Analysis Results:")
    logger.info("(Lower percentile = model ranks it higher, 0.5 = random)")
    
    summary = results_df.groupby("assertion")["predicted_rank_pct"].agg(["mean", "median", "std", "count"])
    logger.info(f"\n{summary.round(3)}")
    
    # Mann-Whitney U tests
    consumed_ranks = results_df[results_df["assertion"] == "consumed"]["predicted_rank_pct"]
    secreted_ranks = results_df[results_df["assertion"] == "secreted"]["predicted_rank_pct"]
    unchanged_ranks = results_df[results_df["assertion"] == "unchanged"]["predicted_rank_pct"]
    
    if len(secreted_ranks) > 0 and len(unchanged_ranks) > 0:
        stat, p = mannwhitneyu(secreted_ranks, unchanged_ranks, alternative="less")
        logger.info(f"\nSecreted vs Unchanged (Mann-Whitney U, one-sided):")
        logger.info(f"  U={stat:.0f}, p={p:.2e}")
        logger.info(f"  Secreted ranked higher than unchanged: {p < 0.05}")
    
    if len(consumed_ranks) > 0 and len(secreted_ranks) > 0:
        stat, p = mannwhitneyu(consumed_ranks, secreted_ranks, alternative="less")
        logger.info(f"\nConsumed vs Secreted (Mann-Whitney U, one-sided):")
        logger.info(f"  U={stat:.0f}, p={p:.2e}")
    
    # Plot: distribution of predicted ranks by assertion type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    colors = {"consumed": "#2ecc71", "secreted": "#3498db", "unchanged": "#95a5a6"}
    
    for assertion in ["consumed", "secreted", "unchanged"]:
        subset = results_df[results_df["assertion"] == assertion]
        if len(subset) == 0:
            continue
        ax.hist(
            subset["predicted_rank_pct"],
            bins=30,
            alpha=0.5,
            label=f"{assertion} (n={len(subset)})",
            color=colors.get(assertion, "gray"),
            density=True,
        )
    
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Random expectation")
    ax.set_xlabel("Predicted rank percentile (0 = top ranked)")
    ax.set_ylabel("Density")
    ax.set_title("Consumption Model Predictions\nby Actual Assertion Type")
    ax.legend()
    
    # Box plot
    ax = axes[1]
    assertion_order = ["consumed", "secreted", "unchanged"]
    plot_data = [
        results_df[results_df["assertion"] == a]["predicted_rank_pct"].values
        for a in assertion_order
        if len(results_df[results_df["assertion"] == a]) > 0
    ]
    plot_labels = [
        a for a in assertion_order
        if len(results_df[results_df["assertion"] == a]) > 0
    ]
    
    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    for patch, label in zip(bp["boxes"], plot_labels):
        patch.set_facecolor(colors.get(label, "gray"))
        patch.set_alpha(0.6)
    
    ax.axhline(0.5, color="red", linestyle="--", linewidth=2, label="Random")
    ax.set_ylabel("Predicted rank percentile")
    ax.set_title("Consumption Model Predictions\n(box plot)")
    ax.legend()
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"cross_prediction_analysis.{ext}", dpi=150)
    plt.close()
    
    # Save detailed results
    results_df.to_csv(output_dir / "cross_prediction_results.csv", index=False)
    
    logger.info(f"\nSaved cross_prediction figures and CSV to {output_dir}")
    
    return results_df


def metabolite_contingency_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    For each metabolite, count how many microbes show each assertion type.
    Tests if metabolites that are often consumed are also often secreted.
    """
    df = df[df["chebi_id"].notna()].copy()
    
    # Count unique microbes per (metabolite, assertion)
    contingency = df.groupby(["chebi_id", "assertion"])["organism_id"].nunique().unstack(fill_value=0)
    
    # Add metabolite names
    name_map = df.drop_duplicates("chebi_id").set_index("chebi_id")["chebi_name"].to_dict()
    contingency["chebi_name"] = contingency.index.map(name_map)
    
    logger.info("\nMetabolite Contingency Analysis:")
    logger.info(f"  Metabolites with consumption data: {(contingency.get('consumed', 0) > 0).sum()}")
    logger.info(f"  Metabolites with secretion data: {(contingency.get('secreted', 0) > 0).sum()}")
    
    # Correlation between consumed and secreted counts
    if "consumed" in contingency.columns and "secreted" in contingency.columns:
        r, p = spearmanr(contingency["consumed"], contingency["secreted"])
        logger.info(f"\nSpearman correlation (consumed vs secreted counts per metabolite):")
        logger.info(f"  r = {r:.3f}, p = {p:.2e}")
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(7, 6))
        
        ax.scatter(
            contingency["consumed"],
            contingency["secreted"],
            alpha=0.5,
            s=30,
        )
        
        # Label top metabolites
        top_consumed = contingency.nlargest(5, "consumed")
        for idx, row in top_consumed.iterrows():
            ax.annotate(
                row.get("chebi_name", str(idx))[:15],
                (row["consumed"], row["secreted"]),
                fontsize=8,
                alpha=0.7,
            )
        
        ax.set_xlabel("# microbes consuming")
        ax.set_ylabel("# microbes secreting")
        ax.set_title(f"Metabolite Usage Correlation\nSpearman r={r:.3f}, p={p:.2e}")
        
        # Add regression line
        z = np.polyfit(contingency["consumed"], contingency["secreted"], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(0, contingency["consumed"].max(), 100)
        ax.plot(x_line, p_line(x_line), "r--", alpha=0.5)
        
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(output_dir / f"metabolite_contingency.{ext}", dpi=150)
        plt.close()
    
    # Save contingency table
    contingency.to_csv(output_dir / "metabolite_contingency.csv")
    logger.info(f"Saved metabolite_contingency.csv to {output_dir}")
    
    return contingency


def run_assertion_validation(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    embedding_dim: int = 32,
) -> dict:
    """
    Run all assertion validation analyses.
    
    These analyses justify treating consumption as implicit positive feedback
    without explicitly modeling secretion as negative.
    """
    if output_dir is None:
        output_dir = Path("figures")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ASSERTION VALIDATION ANALYSES")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. Overlap analysis
    logger.info("\n1. Assertion Overlap Analysis")
    logger.info("-" * 40)
    results["overlap"] = analyze_assertion_overlap(df, output_dir)
    
    # 2. Metabolite contingency
    logger.info("\n2. Metabolite Contingency Analysis")
    logger.info("-" * 40)
    results["contingency"] = metabolite_contingency_analysis(df, output_dir)
    
    # 3. Cross-prediction
    logger.info("\n3. Cross-Prediction Analysis")
    logger.info("-" * 40)
    results["cross_prediction"] = cross_prediction_analysis(df, output_dir, embedding_dim)
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    
    return results


# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark metabolite preference models")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmark")
    parser.add_argument("--validate", action="store_true", help="Run assertion validation")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    
    if not any([args.benchmark, args.validate, args.all]):
        args.all = True
    
    from config import CACHE_FILES
    
    logger.info("Loading exometabolite data...")
    df = pd.read_parquet(CACHE_FILES["exometabolite"])
    
    if args.all or args.validate:
        run_assertion_validation(df, args.output_dir)
    
    if args.all or args.benchmark:
        for pref_type in ["consumed", "secreted"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"BENCHMARK: {pref_type}")
            logger.info(f"{'='*60}")
            
            results = run_benchmark(
                df,
                preference_type=pref_type,
                dimensions=args.dims,
                n_repeats=args.repeats,
                output_dir=args.output_dir,
            )
            
            summary = results.groupby(["method", "dim"])["auc"].mean().unstack()
            print(f"\nAUC Summary ({pref_type}):")
            print(summary.round(3).to_string())