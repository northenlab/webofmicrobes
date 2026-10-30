"""taxonomic_group_analysis.py — Phylogenetic conservation of metabolite preferences.

Answers: "How conserved are metabolite preferences as a function of phylogenetic distance?"

Produces two parallel sets of 4-panel figures for the Nature Microbiology manuscript:

  EMPIRICAL (output_dir/empirical/)  — raw exometabolite measurements
    Metric: pairwise Spearman ρ between isolate assertion-value profiles
    (pairwise complete observations on shared tested compounds)

  EMBEDDING (output_dir/embedding/)  — ordinal BPR model latent space
    Metric: pairwise cosine similarity between microbe embedding vectors

Each set contains:
  A: Distance-decay curve (mean similarity vs. taxonomic level, violin + 95% CI)
  B: Per-class consistency boxplots (within-genus similarity grouped by class)
  C: Per-genus Genus Consistency Score forest plot
  D: Mini-heatmaps of most vs. least conserved genera
     (assertion values for empirical; predicted BPR scores for embedding)

Statistics (all non-parametric):
  - Kruskal-Wallis omnibus test across distance classes
  - Pairwise Mann-Whitney U post-hoc with Bonferroni correction (15 contrasts)
  - Bootstrap 95% CI for mean similarity per group (n_bootstrap resamples)
  - PERMANOVA (genus label permutation, pseudo-F statistic)

Usage:
    cd wom/
    python scripts/taxonomic_group_analysis.py
    python scripts/taxonomic_group_analysis.py --output-dir figures/taxonomic_conservation
    python scripts/taxonomic_group_analysis.py --model-dir trained_models/ordinal_bpr_consumed
    python scripts/taxonomic_group_analysis.py --n-bootstrap 5000 --n-permutations 999 --verbose
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial.distance import cosine as cosine_dist
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import load_model_artifacts
from loaders.loaders import load_exometabolite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy class colors (shared with make_figure3_heatmap.py)
# ---------------------------------------------------------------------------

CLASS_COLORS: dict[str, str] = {
    # Sasha Trubetskoy 20-distinct-color palette — chosen to minimise confusion
    # between any pair of classes.  Each hue/lightness combination is unique.
    "Actinomycetes":       "#e6194b",  # vivid red
    "Bacilli":             "#f58231",  # orange
    "Alphaproteobacteria": "#3cb44b",  # green
    "Betaproteobacteria":  "#4363d8",  # blue
    "Gammaproteobacteria": "#911eb4",  # purple
    "Flavobacteriia":      "#42d4f4",  # cyan
    "Sphingobacteriia":    "#f032e6",  # magenta
    "Cytophagia":          "#bfef45",  # lime
    "Cyanophyceae":        "#469990",  # teal
    "Actinobacteria":      "#9a6324",  # brown
    "Bacteroidia":         "#800000",  # maroon
    "Chitinophagia":       "#ffe119",  # yellow
    "Clostridia":          "#aaffc3",  # mint
    "Coriobacteriia":      "#dcbeff",  # lavender
    "Deltaproteobacteria": "#000075",  # navy
    "Erysipelotrichia":    "#ffd8b1",  # apricot
    "Fusobacteriia":       "#fabed4",  # pink
    "Negativicutes":       "#808000",  # olive
    "Tissierellia":        "#fffac8",  # beige
    "Verrucomicrobiae":    "#a9a9a9",  # grey
}
DEFAULT_CLASS_COLOR = "#aaaaaa"

DISTANCE_LABELS: dict[int, str] = {
    1: "Same\ngenus",
    2: "Same\nfamily",
    3: "Same\norder",
    4: "Same\nclass",
    5: "Same\nphylum",
    6: "Different\nphylum",
}

TAX_LEVELS = ["genus", "family", "order", "class", "phylum"]


def get_class_color(class_name: str) -> str:
    for key, color in CLASS_COLORS.items():
        if key.lower() in str(class_name).lower():
            return color
    return DEFAULT_CLASS_COLOR


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(
    min_isolates_per_genus: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load exometabolite data and build wide isolate × compound matrix.

    Returns:
        df:       Raw exometabolite DataFrame
        wide:     Wide matrix (organism_id index, compound columns, assertion_value)
        tax_map:  Per-isolate taxonomy (organism_id index, TAX_LEVELS columns)
    """
    logger.info("Loading exometabolite data...")
    df = load_exometabolite()
    logger.info(
        f"  {len(df):,} rows, {df['organism_id'].nunique()} isolates, "
        f"{df['representative_compound'].nunique()} compounds"
    )

    wide = df.pivot_table(
        index="organism_id",
        columns="representative_compound",
        values="assertion_value",
        aggfunc="mean",
    )
    logger.info(f"  Wide matrix: {wide.shape[0]} isolates × {wide.shape[1]} compounds")

    tax_cols = ["organism_id"] + TAX_LEVELS
    # Keep only columns that exist
    existing = [c for c in tax_cols if c in df.columns]
    tax_map = (
        df[existing]
        .dropna(subset=["genus"])
        .drop_duplicates("organism_id")
        .set_index("organism_id")
    )

    return df, wide, tax_map


# ---------------------------------------------------------------------------
# Taxonomic distance
# ---------------------------------------------------------------------------

def taxonomic_distance(row1: pd.Series, row2: pd.Series) -> int:
    """Return integer taxonomic distance (1 = same genus, …, 6 = different phylum).

    Checks from most to least specific. Returns the most specific shared level
    as distance (genus=1, family=2, …, phylum=5). Returns 6 if nothing is shared.
    """
    for dist, level in enumerate(TAX_LEVELS, start=1):
        v1 = row1.get(level)
        v2 = row2.get(level)
        if pd.notna(v1) and pd.notna(v2) and str(v1) == str(v2):
            return dist
    return 6


# ---------------------------------------------------------------------------
# Pairwise Spearman ρ computation
# ---------------------------------------------------------------------------

def compute_all_pairs(
    wide: pd.DataFrame,
    tax_map: pd.DataFrame,
    min_shared_compounds: int = 5,
) -> pd.DataFrame:
    """Compute Spearman ρ for all isolate pairs using shared tested compounds.

    Returns DataFrame with columns:
        org1, org2, rho, n_shared, distance, genus1, genus2, class1, class2
    """
    valid_orgs = wide.index.intersection(tax_map.index)
    wide = wide.loc[valid_orgs]
    tax = tax_map.loc[valid_orgs]

    orgs = wide.index.tolist()
    n = len(orgs)
    n_pairs_total = n * (n - 1) // 2
    logger.info(
        f"Computing pairwise Spearman ρ for {n} isolates "
        f"({n_pairs_total:,} pairs)..."
    )

    wide_vals = wide.values  # numpy for speed

    records = []
    skipped = 0
    for i, j in combinations(range(n), 2):
        v1 = wide_vals[i]
        v2 = wide_vals[j]
        mask = ~(np.isnan(v1) | np.isnan(v2))
        n_shared = int(mask.sum())
        if n_shared < min_shared_compounds:
            skipped += 1
            continue
        rho, _ = spearmanr(v1[mask], v2[mask])
        if not np.isfinite(rho):
            skipped += 1
            continue
        dist = taxonomic_distance(tax.iloc[i], tax.iloc[j])
        records.append({
            "org1": orgs[i],
            "org2": orgs[j],
            "rho": float(rho),
            "n_shared": n_shared,
            "distance": dist,
            "genus1": tax.iloc[i].get("genus"),
            "genus2": tax.iloc[j].get("genus"),
            "class1": tax.iloc[i].get("class"),
            "class2": tax.iloc[j].get("class"),
        })

    pairs = pd.DataFrame(records)
    dist_counts = pairs["distance"].value_counts().sort_index().to_dict()
    logger.info(
        f"  {len(pairs):,} valid pairs ({skipped:,} skipped, <{min_shared_compounds} shared compounds)"
    )
    logger.info(f"  Pairs by distance: {dist_counts}")
    return pairs


# ---------------------------------------------------------------------------
# BPR embedding loading and pairwise computation
# ---------------------------------------------------------------------------

def load_bpr_embeddings(
    model_dir: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame, np.ndarray, list[str]]:
    """Load trained BPR model and return embedding matrix + metadata.

    Returns:
        emb_matrix:   (n_isolates, embedding_dim) microbe embedding vectors
        org_ids:      list of organism_id strings (row order of emb_matrix)
        tax_map:      DataFrame indexed by organism_id with TAX_LEVELS columns
        met_matrix:   (n_compounds, embedding_dim) metabolite embedding vectors
        met_names:    list of compound names (row order of met_matrix)
    """
    logger.info(f"Loading BPR model from {model_dir}...")
    artifacts = load_model_artifacts(model_dir)
    td = artifacts.training_data

    emb_matrix = artifacts.model.microbe_embeddings   # (n_isolates, dim)
    met_matrix  = artifacts.model.metabolite_embeddings  # (n_compounds, dim)

    # Build ordered org_ids list matching emb_matrix row order
    org_ids = [td.idx_to_microbe[i] for i in range(td.n_microbes)]

    # Build metabolite name list
    met_names = [td.idx_to_metabolite[i] for i in range(td.n_metabolites)]

    # Build taxonomy DataFrame from Taxonomy objects
    # Taxonomy.to_dict() returns {"class": ..., "genus": ..., etc.}  (class, not class_)
    tax_rows = []
    for org_id in org_ids:
        tax = td.microbe_taxonomy.get(org_id)
        if tax is not None:
            row = tax.to_dict()
            row["organism_id"] = org_id
            tax_rows.append(row)
    tax_map = pd.DataFrame(tax_rows).set_index("organism_id")
    # Keep only the standard taxonomy levels that exist
    for col in TAX_LEVELS:
        if col not in tax_map.columns:
            tax_map[col] = None

    logger.info(
        f"  Loaded {len(org_ids)} isolates, {len(met_names)} compounds, "
        f"embedding dim = {emb_matrix.shape[1]}"
    )
    return emb_matrix, org_ids, tax_map, met_matrix, met_names


def compute_embedding_pairs(
    emb_matrix: np.ndarray,
    org_ids: list[str],
    tax_map: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise cosine similarity between BPR microbe embedding vectors.

    Returns DataFrame with same schema as compute_all_pairs():
        org1, org2, rho (= cosine similarity), n_shared (= embedding_dim),
        distance, genus1, genus2, class1, class2
    """
    n = len(org_ids)
    n_pairs_total = n * (n - 1) // 2
    logger.info(
        f"Computing pairwise cosine similarity for {n} isolates "
        f"({n_pairs_total:,} pairs)..."
    )

    # Normalise rows for faster cosine similarity via dot product
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_norm = emb_matrix / norms  # (n, dim)

    # Restrict to isolates that have taxonomy
    valid_orgs = [o for o in org_ids if o in tax_map.index]
    valid_idx  = [org_ids.index(o) for o in valid_orgs]
    emb_sub    = emb_norm[valid_idx]
    tax        = tax_map.loc[valid_orgs]

    n_sub = len(valid_orgs)
    embedding_dim = emb_matrix.shape[1]

    records = []
    for i, j in combinations(range(n_sub), 2):
        cos_sim = float(emb_sub[i] @ emb_sub[j])  # dot product of L2-normalised = cosine sim
        dist = taxonomic_distance(tax.iloc[i], tax.iloc[j])
        records.append({
            "org1": valid_orgs[i],
            "org2": valid_orgs[j],
            "rho": cos_sim,
            "n_shared": embedding_dim,
            "distance": dist,
            "genus1": tax.iloc[i].get("genus"),
            "genus2": tax.iloc[j].get("genus"),
            "class1": tax.iloc[i].get("class"),
            "class2": tax.iloc[j].get("class"),
        })

    pairs = pd.DataFrame(records)
    dist_counts = pairs["distance"].value_counts().sort_index().to_dict()
    logger.info(f"  {len(pairs):,} embedding pairs by distance: {dist_counts}")
    return pairs


# ---------------------------------------------------------------------------
# Statistical analyses
# ---------------------------------------------------------------------------

def _pseudo_f(dists: np.ndarray, labels: np.ndarray) -> float:
    """PERMANOVA pseudo-F statistic.

    Adapted from make_figures_for_manuscript.py:1119.
    """
    n = len(labels)
    unique = np.unique(labels)
    a = len(unique)
    if a < 2 or n <= a:
        return np.nan
    ss_total = (dists ** 2).sum() / (2 * n)
    ss_within = 0.0
    for g in unique:
        mask = labels == g
        n_g = int(mask.sum())
        if n_g > 1:
            group_dists = dists[np.ix_(mask, mask)]
            ss_within += (group_dists ** 2).sum() / (2 * n_g)
    ss_between = ss_total - ss_within
    return (ss_between / (a - 1)) / (ss_within / (n - a))


def distance_decay_analysis(
    pairs: pd.DataFrame,
    n_bootstrap: int = 5000,
    n_permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Full statistical analysis of distance-decay pattern.

    Returns dict with keys:
        summary:       DataFrame with per-distance stats + bootstrap CI
        kw_stat:       Kruskal-Wallis H statistic
        kw_p:          Kruskal-Wallis p-value
        posthoc:       DataFrame with all 15 pairwise Mann-Whitney comparisons
        permanova_f:   PERMANOVA pseudo-F (genus grouping)
        permanova_p:   PERMANOVA p-value
        permanova_r2:  PERMANOVA R² effect size
        groups:        Dict {distance: array of ρ values}
    """
    rng = np.random.default_rng(seed)

    distances = sorted(pairs["distance"].unique())
    groups = {d: pairs.loc[pairs["distance"] == d, "rho"].values for d in distances}

    # --- Kruskal-Wallis omnibus test ---
    kw_stat, kw_p = kruskal(*groups.values())
    logger.info(f"Kruskal-Wallis: H = {kw_stat:.2f}, p = {kw_p:.2e}")

    # --- Bootstrap CI for mean ρ per distance ---
    summary_rows = []
    for d in distances:
        rhos = groups[d]
        boot_means = np.array([
            rng.choice(rhos, size=len(rhos), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        summary_rows.append({
            "distance": d,
            "label": DISTANCE_LABELS.get(d, str(d)),
            "n_pairs": len(rhos),
            "mean_rho": float(rhos.mean()),
            "median_rho": float(np.median(rhos)),
            "sd_rho": float(rhos.std()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        })
    summary = pd.DataFrame(summary_rows).sort_values("distance").reset_index(drop=True)

    # --- Pairwise Mann-Whitney U post-hoc (Bonferroni) ---
    posthoc_rows = []
    for d1, d2 in combinations(distances, 2):
        stat, p = mannwhitneyu(groups[d1], groups[d2], alternative="two-sided")
        posthoc_rows.append({"dist1": d1, "dist2": d2, "U": stat, "p_uncorrected": p})
    posthoc = pd.DataFrame(posthoc_rows)
    n_tests = len(posthoc)
    posthoc["p_bonferroni"] = np.minimum(posthoc["p_uncorrected"] * n_tests, 1.0)
    posthoc["significant"] = posthoc["p_bonferroni"] < 0.05
    n_sig = posthoc["significant"].sum()
    logger.info(
        f"  {n_sig}/{n_tests} pairwise contrasts significant "
        f"(Bonferroni α = {0.05/n_tests:.4f})"
    )

    # --- PERMANOVA: genus label permutation on pair-wise distance matrix ---
    # Build a matrix of isolates with both valid genus and enough data
    # Use the within-genus pairs to define the label set; PERMANOVA runs on
    # a per-isolate basis using the mean assertion-value vector.
    permanova_f = permanova_p = permanova_r2 = np.nan
    permanova_n = permanova_n_groups = 0
    try:
        logger.info(f"  Running PERMANOVA ({n_permutations} permutations)...")
        # Get per-isolate aggregate rows (mean across replicates if any)
        # from the pairs DataFrame we already have all the organism IDs
        all_orgs = list(set(pairs["org1"].tolist() + pairs["org2"].tolist()))
        genus_of = {
            **dict(zip(pairs["org1"], pairs["genus1"])),
            **dict(zip(pairs["org2"], pairs["genus2"])),
        }
        genus_labels = np.array([genus_of.get(o, "Unknown") for o in all_orgs])
        # Only use organisms with known genus and in genus with ≥2 members
        from collections import Counter
        genus_counts = Counter(genus_labels)
        valid_mask = np.array([
            genus_labels[k] != "Unknown" and
            genus_counts[genus_labels[k]] >= 2
            for k in range(len(all_orgs))
        ])
        if valid_mask.sum() >= 4:
            # Build pairwise distance matrix from pairs DataFrame
            # Use 1 - rho as distance between each pair of isolates
            org_idx = {o: k for k, o in enumerate(all_orgs)}
            n_orgs = len(all_orgs)
            dist_mat = np.zeros((n_orgs, n_orgs))
            count_mat = np.zeros((n_orgs, n_orgs))
            for _, row in pairs.iterrows():
                i, j = org_idx[row["org1"]], org_idx[row["org2"]]
                # Spearman ρ in [-1, 1]; use 1 - rho as distance
                dist_mat[i, j] += 1.0 - row["rho"]
                dist_mat[j, i] += 1.0 - row["rho"]
                count_mat[i, j] += 1
                count_mat[j, i] += 1
            # Average in case of duplicate pairs (shouldn't happen)
            with np.errstate(invalid="ignore"):
                dist_mat = np.where(count_mat > 0, dist_mat / count_mat, 0.0)

            sub_dists = dist_mat[np.ix_(valid_mask, valid_mask)]
            sub_labels = genus_labels[valid_mask]

            observed_f = _pseudo_f(sub_dists, sub_labels)
            if np.isfinite(observed_f):
                perm_rng = np.random.default_rng(seed + 1)
                perm_fs = np.array([
                    _pseudo_f(sub_dists, perm_rng.permutation(sub_labels))
                    for _ in range(n_permutations)
                ])
                permanova_p = (np.sum(perm_fs >= observed_f) + 1) / (n_permutations + 1)

                # R² = SS_between / SS_total
                n_sub = len(sub_labels)
                ss_total = (sub_dists ** 2).sum() / (2 * n_sub)
                ss_within = 0.0
                for g in np.unique(sub_labels):
                    mask = sub_labels == g
                    n_g = int(mask.sum())
                    if n_g > 1:
                        gd = sub_dists[np.ix_(mask, mask)]
                        ss_within += (gd ** 2).sum() / (2 * n_g)
                permanova_r2 = 1.0 - ss_within / ss_total
                permanova_f = observed_f
                permanova_n = int(n_sub)
                permanova_n_groups = int(len(np.unique(sub_labels)))

                logger.info(
                    f"  PERMANOVA: pseudo-F = {permanova_f:.2f}, "
                    f"R² = {permanova_r2:.3f}, p = {permanova_p:.4f}"
                )
    except Exception as exc:
        logger.warning(f"PERMANOVA failed: {exc}")

    return {
        "summary": summary,
        "kw_stat": float(kw_stat),
        "kw_p": float(kw_p),
        "posthoc": posthoc,
        "permanova_f": permanova_f,
        "permanova_p": permanova_p,
        "permanova_r2": permanova_r2,
        "permanova_n": permanova_n,  # pseudo-F d.f. = (n_groups - 1, n - n_groups)
        "permanova_n_groups": permanova_n_groups,
        "groups": groups,
    }


def per_genus_analysis(
    pairs: pd.DataFrame,
    tax_map: pd.DataFrame,
    min_isolates_per_genus: int = 3,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute Genus Consistency Score (GCS = mean within-genus pairwise ρ) per genus.

    Returns DataFrame sorted by GCS descending with columns:
        genus, n_isolates, n_pairs, gcs, ci_low, ci_high, class, phylum
    """
    rng = np.random.default_rng(seed)

    within = pairs[pairs["distance"] == 1].copy()

    genus_isolate_counts = (
        tax_map["genus"]
        .value_counts()
        .rename("n_isolates")
    )
    valid_genera = genus_isolate_counts[
        genus_isolate_counts >= min_isolates_per_genus
    ].index.tolist()

    logger.info(
        f"Per-genus: {len(valid_genera)} genera with ≥{min_isolates_per_genus} isolates"
    )

    genus_to_class = (
        tax_map.groupby("genus")["class"]
        .agg(lambda x: str(x.dropna().mode().iloc[0]) if x.notna().any() else "Unknown")
    )
    genus_to_phylum = (
        tax_map.groupby("genus")["phylum"]
        .agg(lambda x: str(x.dropna().mode().iloc[0]) if x.notna().any() else "Unknown")
        if "phylum" in tax_map.columns else pd.Series(dtype=str)
    )

    rows = []
    for genus in valid_genera:
        genus_pairs = within[
            (within["genus1"] == genus) & (within["genus2"] == genus)
        ]
        if len(genus_pairs) < 1:
            continue
        rhos = genus_pairs["rho"].values

        if len(rhos) >= 2:
            boot_means = np.array([
                rng.choice(rhos, size=len(rhos), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci_low = ci_high = float(rhos[0])

        rows.append({
            "genus": genus,
            "n_isolates": int(genus_isolate_counts.get(genus, 0)),
            "n_pairs": len(rhos),
            "gcs": float(rhos.mean()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "class": genus_to_class.get(genus, "Unknown"),
            "phylum": genus_to_phylum.get(genus, "Unknown") if len(genus_to_phylum) > 0 else "Unknown",
        })

    genus_df = (
        pd.DataFrame(rows)
        .sort_values("gcs", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        f"  Top 5: {genus_df.head(5)['genus'].tolist()}\n"
        f"  Bottom 5: {genus_df.tail(5)['genus'].tolist()}"
    )
    return genus_df


def per_class_analysis(
    pairs: pd.DataFrame,
    tax_map: pd.DataFrame,
    min_pairs_per_class: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-genus ρ grouped by class.

    Returns (class_summary DataFrame, within_genus_with_class DataFrame).
    """
    within = pairs[pairs["distance"] == 1].copy()

    genus_to_class = (
        tax_map.groupby("genus")["class"]
        .agg(lambda x: str(x.dropna().mode().iloc[0]) if x.notna().any() else "Unknown")
    )
    within["class"] = within["genus1"].map(genus_to_class)
    within = within.dropna(subset=["class"])
    within = within[within["class"] != "Unknown"]

    class_counts = within["class"].value_counts()
    valid_classes = class_counts[class_counts >= min_pairs_per_class].index.tolist()
    within = within[within["class"].isin(valid_classes)].copy()

    class_groups = [grp["rho"].values for _, grp in within.groupby("class")]
    if len(class_groups) >= 2:
        kw_stat, kw_p = kruskal(*class_groups)
        logger.info(f"Per-class KW: H = {kw_stat:.2f}, p = {kw_p:.2e}")
    else:
        kw_stat, kw_p = np.nan, np.nan

    class_summary = (
        within.groupby("class")["rho"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"count": "n_pairs"})
        .sort_values("median", ascending=False)
        .reset_index()
    )
    class_summary["kw_stat"] = kw_stat
    class_summary["kw_p"] = kw_p

    return class_summary, within


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def figure_distance_decay(
    groups: dict,
    summary: pd.DataFrame,
    posthoc: pd.DataFrame,
    kw_stat: float,
    kw_p: float,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
    metric_label: str = "Spearman ρ (metabolite profile similarity)",
) -> Path:
    """Panel A: Violin plots of pairwise similarity by taxonomic distance, mean ± 95% CI."""
    distances = sorted(groups.keys())
    x_pos = np.arange(len(distances))

    palette = sns.color_palette("RdYlBu_r", len(distances))

    fig, ax = plt.subplots(figsize=(9, 5))

    # Violin plots
    parts = ax.violinplot(
        [groups[d] for d in distances],
        positions=x_pos,
        showmedians=False,
        showextrema=False,
        widths=0.72,
    )
    for pc, color in zip(parts["bodies"], palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
        pc.set_edgecolor("none")

    # Mean ± 95% CI overlay
    for k, d in enumerate(distances):
        row = summary[summary["distance"] == d].iloc[0]
        ax.errorbar(
            [k], [row["mean_rho"]],
            yerr=[[row["mean_rho"] - row["ci_low"]], [row["ci_high"] - row["mean_rho"]]],
            fmt="o", color=palette[k], markersize=8, capsize=5, capthick=2,
            markeredgecolor="white", markeredgewidth=1.5, lw=2, zorder=5,
        )

    # Reference line
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Significance stars for adjacent comparisons
    adjacent_sig = posthoc[
        (posthoc["dist2"] - posthoc["dist1"] == 1) & posthoc["significant"]
    ]
    y_max = summary["ci_high"].max()
    step = 0.04
    for _, row in adjacent_sig.iterrows():
        k1 = distances.index(int(row["dist1"]))
        k2 = distances.index(int(row["dist2"]))
        mid = (k1 + k2) / 2
        p_b = row["p_bonferroni"]
        stars = "***" if p_b < 0.001 else ("**" if p_b < 0.01 else "*")
        y_ann = y_max + step
        ax.annotate(
            stars,
            xy=(mid, y_ann),
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )
        ax.plot([k1, k2], [y_ann - 0.005, y_ann - 0.005], color="black", lw=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{DISTANCE_LABELS.get(d, d)}\n(n={len(groups[d]):,})" for d in distances],
        fontsize=9.5,
    )
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_xlabel("Taxonomic relationship", fontsize=12)
    p_str = f"{kw_p:.1e}" if kw_p < 0.001 else f"{kw_p:.4f}"
    ax.set_title(
        f"Metabolite preference similarity decays with phylogenetic distance\n"
        f"Kruskal–Wallis H = {kw_stat:.1f},  p = {p_str},  "
        f"n = {sum(len(v) for v in groups.values()):,} isolate pairs",
        fontsize=10.5,
    )
    sns.despine(ax=ax)
    fig.tight_layout()

    stem = output_dir / "figure_A_distance_decay"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def figure_class_boxplots(
    within_genus: pd.DataFrame,
    class_summary: pd.DataFrame,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
    xlabel: str = "Within-genus Spearman ρ",
) -> Path:
    """Panel B: Horizontal boxplots of within-genus similarity per class, sorted by median."""
    classes = class_summary["class"].tolist()
    colors = [get_class_color(c) for c in classes]
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.48 * n_classes + 1.5)))

    data_by_class = [
        within_genus[within_genus["class"] == c]["rho"].values for c in classes
    ]

    bp = ax.boxplot(
        data_by_class,
        vert=False,
        patch_artist=True,
        medianprops={"color": "black", "lw": 2},
        boxprops={"lw": 1.5},
        whiskerprops={"lw": 1},
        capprops={"lw": 1},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4, "lw": 0},
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Annotate with n_pairs on the right margin
    xlim_right = ax.get_xlim()[1]
    for k, (cls, n_pairs) in enumerate(
        zip(class_summary["class"], class_summary["n_pairs"]), start=1
    ):
        ax.text(
            xlim_right + 0.01 * (xlim_right - ax.get_xlim()[0]),
            k, f"n={int(n_pairs)}",
            va="center", ha="left", fontsize=8, color="#555555",
        )

    ax.set_yticks(range(1, n_classes + 1))
    ax.set_yticklabels(classes, fontsize=9, style="italic")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

    kw_p = class_summary["kw_p"].iloc[0]
    kw_stat = class_summary["kw_stat"].iloc[0]
    p_str = f"{kw_p:.1e}" if kw_p < 0.001 else f"{kw_p:.4f}"
    ax.set_title(
        f"Within-genus consistency by taxonomic class\n"
        f"Kruskal–Wallis H = {kw_stat:.1f},  p = {p_str}",
        fontsize=10,
    )
    sns.despine(ax=ax)
    fig.tight_layout()

    stem = output_dir / "figure_B_class_boxplots"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def figure_genus_forest(
    genus_df: pd.DataFrame,
    output_dir: Path,
    n_show: int = 10,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
    xlabel: str = "Genus Consistency Score (mean within-genus Spearman ρ)",
) -> Path:
    """Panel C: Forest plot of GCS (mean within-genus similarity) per genus."""
    top = genus_df.head(n_show).copy()
    bottom = genus_df.tail(n_show).copy()
    separator = pd.DataFrame([{
        "genus": "···", "gcs": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "n_isolates": 0, "class": "",
    }])
    plot_df = pd.concat([top, separator, bottom], ignore_index=True)

    n_rows = len(plot_df)
    fig, ax = plt.subplots(figsize=(7.5, max(5, 0.40 * n_rows + 2)))

    y_pos = np.arange(n_rows)[::-1]  # highest GCS at top

    for k, (_, row) in enumerate(plot_df.iterrows()):
        y = y_pos[k]
        if row["genus"] == "···":
            ax.text(0, y, "···", ha="center", va="center", fontsize=13, color="gray")
            continue
        color = get_class_color(str(row["class"]))
        # CI bar
        if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            ax.plot(
                [row["ci_low"], row["ci_high"]], [y, y],
                "-", color=color, lw=2.5, alpha=0.7, solid_capstyle="round",
            )
        # Point estimate
        ax.plot(
            row["gcs"], y, "o", color=color, markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5,
        )

    # Genus labels (italic) and n= annotations
    ax.set_yticks(y_pos)
    genus_labels = []
    for _, row in plot_df.iterrows():
        if row["genus"] == "···":
            genus_labels.append("")
        else:
            genus_labels.append(f"$\\it{{{row['genus'].replace(' ', '\\ ')}}}$")
    ax.set_yticklabels(genus_labels, fontsize=9)

    # n= annotation at right of each bar
    x_right = genus_df["ci_high"].max() + 0.02
    for k, (_, row) in enumerate(plot_df.iterrows()):
        y = y_pos[k]
        if row["genus"] == "···" or row["n_isolates"] == 0:
            continue
        ax.text(
            x_right, y,
            f"n={int(row['n_isolates'])}",
            va="center", ha="left", fontsize=8, color="#555555",
        )

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(
        f"Metabolic conservatism varies widely across genera\n"
        f"(top {n_show} most conserved  ·  bottom {n_show} least conserved)",
        fontsize=10.5,
    )

    # Class legend
    seen_classes = [
        row["class"] for _, row in plot_df.iterrows()
        if row["genus"] != "···" and row["class"]
    ]
    unique_classes = list(dict.fromkeys(seen_classes))  # preserve order, deduplicate
    handles = [
        mpatches.Patch(color=get_class_color(c), label=c, alpha=0.75)
        for c in unique_classes
    ]
    if handles:
        ax.legend(
            handles=handles, fontsize=8, frameon=False,
            loc="lower right", title="Class", title_fontsize=8,
        )

    sns.despine(ax=ax)
    fig.tight_layout()

    stem = output_dir / "figure_C_genus_forest"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def figure_contrasting_heatmaps(
    wide: pd.DataFrame,
    tax_map: pd.DataFrame,
    genus_df: pd.DataFrame,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
) -> Path | None:
    """Panel D: Mini assertion-value heatmaps for the most vs. least consistent genus."""

    def get_genus_matrix(genus_name: str) -> pd.DataFrame | None:
        isolates = tax_map[tax_map["genus"] == genus_name].index.tolist()
        isolates = [o for o in isolates if o in wide.index]
        if len(isolates) < 3:
            return None
        mat = wide.loc[isolates]
        # Keep compounds tested in ≥2 isolates in this genus
        mat = mat.loc[:, mat.notna().sum() >= 2]
        return mat if mat.shape[1] >= 3 else None

    best_genus = best_mat = worst_genus = worst_mat = None
    for genus in genus_df.head(10)["genus"]:
        mat = get_genus_matrix(genus)
        if mat is not None:
            best_genus, best_mat = genus, mat
            break
    for genus in genus_df.tail(10)["genus"]:
        mat = get_genus_matrix(genus)
        if mat is not None and genus != best_genus:
            worst_genus, worst_mat = genus, mat
            break

    if best_genus is None or worst_genus is None:
        logger.warning("Could not find suitable contrasting genera for Panel D")
        return None

    best_gcs = genus_df.loc[genus_df["genus"] == best_genus, "gcs"].iloc[0]
    worst_gcs = genus_df.loc[genus_df["genus"] == worst_genus, "gcs"].iloc[0]

    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    cmap = "RdBu_r"
    max_compounds = 30

    def prepare_matrix(mat: pd.DataFrame, max_cols: int) -> pd.DataFrame:
        # Select top compounds by number tested, fill NaN with 0 for display
        n_tested = mat.notna().sum()
        top_cols = n_tested.sort_values(ascending=False).head(max_cols).index
        mat = mat[top_cols].copy()
        # Cluster isolates if possible
        if len(mat) >= 3 and mat.shape[1] >= 2:
            from scipy.cluster.hierarchy import leaves_list, linkage
            link = linkage(mat.fillna(0).values, method="ward")
            order = leaves_list(link)
            mat = mat.iloc[order]
        return mat

    best_plot = prepare_matrix(best_mat, max_compounds)
    worst_plot = prepare_matrix(worst_mat, max_compounds)

    max_rows = max(len(best_plot), len(worst_plot))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, max_rows * 0.35 + 2)))

    for ax, mat, genus, gcs in [
        (axes[0], best_plot, best_genus, best_gcs),
        (axes[1], worst_plot, worst_genus, worst_gcs),
    ]:
        im = ax.imshow(
            mat.fillna(0).values,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(mat)))
        # Shorten isolate IDs for readability
        ax.set_yticklabels(
            [str(o)[:20] for o in mat.index],
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_xlabel(f"{mat.shape[1]} shared compounds", fontsize=9)
        label = "most conserved" if genus == best_genus else "least conserved"
        ax.set_title(
            f"$\\it{{{genus}}}$  ({label})\n"
            f"GCS = {gcs:.2f}  (n={len(mat)} isolates)",
            fontsize=10, fontweight="bold",
        )

    cbar = fig.colorbar(
        im, ax=axes, orientation="vertical",
        label="Assertion value\n(–5 consumed → +5 secreted)",
        fraction=0.025, pad=0.04,
    )
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Most vs. least conserved genera: per-isolate metabolite profiles",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)

    stem = output_dir / "figure_D_contrasting_heatmaps"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def figure_contrasting_heatmaps_embedding(
    emb_matrix: np.ndarray,
    org_ids: list[str],
    met_matrix: np.ndarray,
    met_names: list[str],
    tax_map: pd.DataFrame,
    genus_df: pd.DataFrame,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
) -> Path | None:
    """Panel D (embedding): Predicted BPR score heatmaps for most vs. least conserved genus.

    Shows score_matrix = emb_isolates @ met_matrix.T for each isolate in the genus.
    Directly analogous to figure_contrasting_heatmaps() but using model predictions.
    """
    org_to_idx = {o: k for k, o in enumerate(org_ids)}
    n_met = len(met_names)

    def get_genus_score_matrix(genus_name: str) -> tuple[np.ndarray, list[str]] | None:
        isolates = tax_map[tax_map["genus"] == genus_name].index.tolist()
        isolates = [o for o in isolates if o in org_to_idx]
        if len(isolates) < 3:
            return None
        indices = [org_to_idx[o] for o in isolates]
        scores = emb_matrix[indices] @ met_matrix.T  # (n_iso, n_met)
        return scores, isolates

    best_genus = best_data = worst_genus = worst_data = None
    for genus in genus_df.head(10)["genus"]:
        result = get_genus_score_matrix(genus)
        if result is not None:
            best_genus, best_data = genus, result
            break
    for genus in genus_df.tail(10)["genus"]:
        result = get_genus_score_matrix(genus)
        if result is not None and genus != best_genus:
            worst_genus, worst_data = genus, result
            break

    if best_genus is None or worst_genus is None:
        logger.warning("Could not find suitable contrasting genera for embedding Panel D")
        return None

    best_scores, best_isolates = best_data
    worst_scores, worst_isolates = worst_data
    best_gcs = genus_df.loc[genus_df["genus"] == best_genus, "gcs"].iloc[0]
    worst_gcs = genus_df.loc[genus_df["genus"] == worst_genus, "gcs"].iloc[0]

    # Select top 30 compounds by score variance across isolates (most discriminating)
    max_compounds = 30

    def select_and_cluster(scores: np.ndarray, isolates: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
        var = scores.var(axis=0)
        top_idx = np.argsort(var)[::-1][:max_compounds]
        sub = scores[:, top_idx]
        sub_names = [met_names[k] for k in top_idx]
        # Cluster isolates by score profile
        if len(isolates) >= 3:
            from scipy.cluster.hierarchy import leaves_list, linkage
            link = linkage(sub, method="ward")
            order = leaves_list(link)
            sub = sub[order]
            isolates = [isolates[k] for k in order]
        return sub, isolates, sub_names

    best_plot, best_iso_ord, best_met_names = select_and_cluster(best_scores, best_isolates)
    worst_plot, worst_iso_ord, worst_met_names = select_and_cluster(worst_scores, worst_isolates)

    # Shared colour scale centred at 0
    all_vals = np.concatenate([best_plot.ravel(), worst_plot.ravel()])
    abs_max = np.percentile(np.abs(all_vals), 99)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    max_rows = max(len(best_plot), len(worst_plot))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, max_rows * 0.45 + 2)))

    for ax, scores, iso_names, mnames, genus, gcs in [
        (axes[0], best_plot, best_iso_ord, best_met_names, best_genus, best_gcs),
        (axes[1], worst_plot, worst_iso_ord, worst_met_names, worst_genus, worst_gcs),
    ]:
        im = ax.imshow(
            scores,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(iso_names)))
        ax.set_yticklabels([str(o)[:20] for o in iso_names], fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel(f"{scores.shape[1]} compounds (ranked by variance)", fontsize=9)
        label = "most conserved" if genus == best_genus else "least conserved"
        ax.set_title(
            f"$\\it{{{genus}}}$  ({label})\n"
            f"GCS = {gcs:.2f}  (n={len(iso_names)} isolates)",
            fontsize=10, fontweight="bold",
        )

    cbar = fig.colorbar(
        im, ax=axes, orientation="vertical",
        label="BPR predicted score\n(negative = consumed, positive = secreted)",
        fraction=0.025, pad=0.04,
    )
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Most vs. least conserved genera: per-isolate BPR predicted score profiles",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.88)

    stem = output_dir / "figure_D_contrasting_heatmaps"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def assemble_4panel_figure(
    png_a: Path,
    png_b: Path,
    png_c: Path,
    png_d: Path | None,
    output_dir: Path,
    dpi_png: int = 300,
    dpi_pdf: int = 600,
) -> None:
    """Read saved panel PNGs and assemble a combined 4-panel figure."""
    panels = [("A", png_a), ("B", png_b), ("C", png_c), ("D", png_d)]
    panels = [(label, p) for label, p in panels if p is not None and p.exists()]

    if len(panels) < 2:
        logger.warning("Fewer than 2 panels available; skipping combined figure")
        return

    n_cols = 2
    n_rows = (len(panels) + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for k, (label, png_path) in enumerate(panels):
        img = mpimg.imread(str(png_path))
        axes[k].imshow(img)
        axes[k].axis("off")
        axes[k].text(
            -0.02, 1.02, label,
            transform=axes[k].transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right",
        )

    fig.tight_layout(pad=0.5)
    stem = output_dir / "figure_combined_4panel"
    fig.savefig(str(stem) + ".png", dpi=dpi_png, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", dpi=dpi_pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined 4-panel figure")


# ---------------------------------------------------------------------------
# Statistics output
# ---------------------------------------------------------------------------

def save_statistics(
    summary: pd.DataFrame,
    posthoc: pd.DataFrame,
    kw_stat: float,
    kw_p: float,
    permanova_f: float,
    permanova_p: float,
    permanova_r2: float,
    genus_df: pd.DataFrame,
    class_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all statistics to CSV files."""
    # Embed KW and PERMANOVA into distance-decay summary
    s = summary.copy()
    s["kw_stat"] = kw_stat
    s["kw_p"] = kw_p
    s["permanova_f"] = permanova_f
    s["permanova_p"] = permanova_p
    s["permanova_r2"] = permanova_r2
    s.to_csv(output_dir / "stats_distance_decay.csv", index=False)

    posthoc.to_csv(output_dir / "stats_pairwise_posthoc.csv", index=False)
    genus_df.to_csv(output_dir / "stats_genus_consistency.csv", index=False)
    class_summary.to_csv(output_dir / "stats_class_consistency.csv", index=False)

    logger.info(f"Statistics saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_analysis(
    pairs: pd.DataFrame,
    tax_map: pd.DataFrame,
    output_dir: Path,
    n_bootstrap: int,
    n_permutations: int,
    n_show_genera: int,
    seed: int,
    dpi_png: int,
    dpi_pdf: int,
    metric_label: str,
    xlabel_b: str,
    xlabel_c: str,
    panel_d_fn,        # callable that produces Panel D and returns a Path | None
) -> None:
    """Run the full analysis pipeline (statistics + figures) for one data source.

    Used by main() for both the empirical and embedding branches.
    panel_d_fn() is called with output_dir, dpi_png, dpi_pdf as keyword args.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(output_dir / "pairs_all.csv", index=False)

    logger.info("  Distance-decay analysis...")
    decay = distance_decay_analysis(
        pairs, n_bootstrap=n_bootstrap, n_permutations=n_permutations, seed=seed,
    )

    logger.info("  Per-genus analysis...")
    genus_df = per_genus_analysis(
        pairs, tax_map,
        n_bootstrap=min(n_bootstrap, 1000),
        seed=seed,
    )
    logger.info(f"  Top 5: {genus_df.head(5)['genus'].tolist()}")
    logger.info(f"  Bottom 5: {genus_df.tail(5)['genus'].tolist()}")

    logger.info("  Per-class analysis...")
    class_summary, within_genus = per_class_analysis(pairs, tax_map)

    save_statistics(
        decay["summary"], decay["posthoc"],
        decay["kw_stat"], decay["kw_p"],
        decay["permanova_f"], decay["permanova_p"], decay["permanova_r2"],
        genus_df, class_summary,
        output_dir,
    )

    logger.info("  Generating figures...")
    fig_kw = {"output_dir": output_dir, "dpi_png": dpi_png, "dpi_pdf": dpi_pdf}

    png_a = figure_distance_decay(
        decay["groups"], decay["summary"], decay["posthoc"],
        decay["kw_stat"], decay["kw_p"],
        metric_label=metric_label, **fig_kw,
    )
    png_b = figure_class_boxplots(within_genus, class_summary, xlabel=xlabel_b, **fig_kw)
    png_c = figure_genus_forest(genus_df, n_show=n_show_genera, xlabel=xlabel_c, **fig_kw)
    png_d = panel_d_fn(**fig_kw)

    assemble_4panel_figure(png_a, png_b, png_c, png_d, output_dir, dpi_png, dpi_pdf)
    logger.info(f"  Done → {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phylogenetic conservation of metabolite preferences (Nature Microbiology)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("figures/taxonomic_conservation"),
        help="Root output directory; empirical/ and embedding/ subdirs are created inside",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("trained_models/ordinal_bpr_consumed"),
        help="Path to trained BPR model directory for embedding analysis (default: ordinal_bpr_consumed)",
    )
    parser.add_argument(
        "--min-isolates-per-genus", type=int, default=3,
        help="Minimum isolates per genus for genus-level analysis (default: 3)",
    )
    parser.add_argument(
        "--min-shared-compounds", type=int, default=5,
        help="Minimum shared compounds for a valid empirical isolate pair (default: 5)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=5000,
        help="Bootstrap resamples for 95%% CI (default: 5000)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=999,
        help="Permutations for PERMANOVA (default: 999)",
    )
    parser.add_argument(
        "--n-show-genera", type=int, default=10,
        help="Top and bottom N genera shown in forest plot (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dpi-png", type=int, default=300)
    parser.add_argument("--dpi-pdf", type=int, default=600)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    shared_kw = dict(
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        n_show_genera=args.n_show_genera,
        seed=args.seed,
        dpi_png=args.dpi_png,
        dpi_pdf=args.dpi_pdf,
    )

    # -----------------------------------------------------------------------
    # EMPIRICAL ANALYSIS — pairwise Spearman ρ on assertion-value profiles
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EMPIRICAL ANALYSIS (raw exometabolite measurements)")
    logger.info("=" * 60)

    df, wide, tax_map_emp = load_and_preprocess(
        min_isolates_per_genus=args.min_isolates_per_genus
    )
    pairs_emp = compute_all_pairs(
        wide, tax_map_emp, min_shared_compounds=args.min_shared_compounds
    )
    if len(pairs_emp) == 0:
        logger.error("No valid empirical pairs. Lower --min-shared-compounds.")
        sys.exit(1)

    emp_dir = args.output_dir / "empirical"

    def _panel_d_empirical(**kw):
        return figure_contrasting_heatmaps(wide, tax_map_emp, genus_df_emp, **kw)

    # We need genus_df_emp before calling _run_analysis; compute it here so the
    # closure captures it.  _run_analysis recomputes it internally — that's fine.
    _decay_tmp = distance_decay_analysis(
        pairs_emp, n_bootstrap=min(args.n_bootstrap, 100),
        n_permutations=min(args.n_permutations, 9), seed=args.seed,
    )
    genus_df_emp = per_genus_analysis(
        pairs_emp, tax_map_emp,
        n_bootstrap=min(args.n_bootstrap, 100), seed=args.seed,
    )

    _run_analysis(
        pairs=pairs_emp,
        tax_map=tax_map_emp,
        output_dir=emp_dir,
        metric_label="Spearman ρ (empirical metabolite profiles)",
        xlabel_b="Within-genus Spearman ρ",
        xlabel_c="Genus Consistency Score (mean within-genus Spearman ρ)",
        panel_d_fn=_panel_d_empirical,
        **shared_kw,
    )

    # -----------------------------------------------------------------------
    # EMBEDDING ANALYSIS — pairwise cosine similarity in BPR latent space
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("EMBEDDING ANALYSIS (ordinal BPR model latent space)")
    logger.info("=" * 60)

    emb_matrix, org_ids, tax_map_emb, met_matrix, met_names = load_bpr_embeddings(
        args.model_dir
    )
    pairs_emb = compute_embedding_pairs(emb_matrix, org_ids, tax_map_emb)
    if len(pairs_emb) == 0:
        logger.error("No valid embedding pairs. Check model directory.")
        sys.exit(1)

    emb_dir = args.output_dir / "embedding"

    # Pre-compute genus_df_emb for the Panel D closure
    genus_df_emb = per_genus_analysis(
        pairs_emb, tax_map_emb,
        n_bootstrap=min(args.n_bootstrap, 100), seed=args.seed,
    )

    def _panel_d_embedding(**kw):
        return figure_contrasting_heatmaps_embedding(
            emb_matrix, org_ids, met_matrix, met_names,
            tax_map_emb, genus_df_emb, **kw,
        )

    _run_analysis(
        pairs=pairs_emb,
        tax_map=tax_map_emb,
        output_dir=emb_dir,
        metric_label="Cosine similarity (BPR embedding vectors)",
        xlabel_b="Within-genus cosine similarity",
        xlabel_c="Genus Consistency Score (mean within-genus cosine similarity)",
        panel_d_fn=_panel_d_embedding,
        **shared_kw,
    )

    logger.info("=" * 60)
    logger.info(f"All outputs in {args.output_dir}/")
    logger.info(f"  Empirical figures → {emp_dir}/")
    logger.info(f"  Embedding figures → {emb_dir}/")


if __name__ == "__main__":
    main()
