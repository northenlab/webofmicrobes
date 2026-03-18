"""
Compare BPR model predictions trained with vs without gut microbes.

Analyses:
1. Dataset summary (microbe/metabolite overlap)
2. Metabolite presence/absence across conditions
3. Rank entropy per metabolite (discrimination power)
4. Metabolite universality curves (collapse detection)
5. Effective differentiation rank between microbe pairs
6. Rank profile correlation for shared microbes

Output: figures saved to wom/manuscript_figures/comparison/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "..", "manuscript_figures")
OUT_DIR = os.path.join(BASE, "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = {
    "with":    os.path.join(BASE, "trained_with_gut_microbes"),
    "without": os.path.join(BASE, "trained_without_gut_microbes"),
}
MODES = ["consumed", "secreted"]

TAXONOMY_COLS = ["kingdom", "class", "order", "family", "genus", "isolate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(condition: str, mode: str) -> pd.DataFrame:
    path = os.path.join(CONDITIONS[condition], mode, "taxonomic_ranks.tsv")
    df = pd.read_csv(path, sep="\t", index_col=None)
    df = df.set_index("isolate")
    return df


def rank_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the numeric rank columns (drop taxonomy)."""
    meta = [c for c in TAXONOMY_COLS if c in df.columns]
    return df.drop(columns=meta)


def metabolite_entropy(ranks: pd.DataFrame) -> pd.Series:
    """
    For each metabolite (column), compute the entropy of its rank distribution
    across microbes.  Higher entropy → more discriminative.
    """
    # Bin ranks into 20 equal-width bins
    bins = np.linspace(0.5, ranks.values.max() + 0.5, 21)
    entropies = {}
    for met in ranks.columns:
        counts, _ = np.histogram(ranks[met].values, bins=bins)
        counts = counts + 1e-9          # Laplace smoothing
        entropies[met] = entropy(counts / counts.sum())
    return pd.Series(entropies)


def universality_matrix(ranks: pd.DataFrame, ks: np.ndarray) -> pd.DataFrame:
    """
    For each K in `ks`, compute the fraction of microbes that rank each
    metabolite in their top-K.
    Returns DataFrame shape (n_metabolites, len(ks)).
    """
    results = {}
    n_microbes = len(ranks)
    for k in ks:
        in_topk = (ranks <= k).mean(axis=0)   # fraction of microbes with rank ≤ k
        results[k] = in_topk
    return pd.DataFrame(results, index=ranks.columns)


def effective_diff_rank(ranks: pd.DataFrame, n_pairs: int = 500, seed: int = 42) -> np.ndarray:
    """
    For a random sample of microbe pairs, find the minimum K such that
    their top-K metabolite sets are not identical (i.e. first rank at which
    they differ).  Returns array of per-pair effective differentiation ranks.
    """
    rng = np.random.default_rng(seed)
    microbes = ranks.index.tolist()
    n_microbes = len(microbes)
    if n_microbes < 2:
        return np.array([])
    max_pairs = n_microbes * (n_microbes - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    # Sample unique ordered pairs (i < j)
    chosen = rng.choice(max_pairs, size=n_pairs, replace=False)
    # Convert flat indices to (i, j) pairs where i < j
    # For flat index k, i and j satisfy: k = i*(n-1) - i*(i-1)//2 + (j-i-1)
    # Simpler: generate via triangular indexing
    ii, jj = np.tril_indices(n_microbes, k=-1)
    chosen_i = ii[chosen]
    chosen_j = jj[chosen]
    pairs = np.stack([chosen_i, chosen_j], axis=1)

    rank_arr = ranks.values   # shape (n_microbes, n_metabolites)
    diff_ranks = []
    for i, j in pairs:
        r_i = rank_arr[i]
        r_j = rank_arr[j]
        # For each K from 1 to max, check if top-K sets differ
        order_i = np.argsort(r_i)
        order_j = np.argsort(r_j)
        max_k = ranks.shape[1]
        found = max_k
        for k in range(1, max_k + 1):
            set_i = set(order_i[:k])
            set_j = set(order_j[:k])
            if set_i != set_j:
                found = k
                break
        diff_ranks.append(found)
    return np.array(diff_ranks)


def rank_profile_correlation(ranks_with: pd.DataFrame,
                              ranks_without: pd.DataFrame) -> pd.Series:
    """
    For isolates present in BOTH conditions, compute the Spearman rank
    correlation of their full metabolite rank vectors between conditions.
    Only metabolites shared between both datasets are used.
    """
    shared_isolates = ranks_with.index.intersection(ranks_without.index)
    shared_mets = ranks_with.columns.intersection(ranks_without.columns)
    if len(shared_isolates) == 0 or len(shared_mets) == 0:
        return pd.Series(dtype=float)

    corrs = {}
    for iso in shared_isolates:
        r_w = ranks_with.loc[iso, shared_mets].values.astype(float)
        r_wo = ranks_without.loc[iso, shared_mets].values.astype(float)
        rho, _ = spearmanr(r_w, r_wo)
        corrs[iso] = rho
    return pd.Series(corrs)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {"with": "#E07B39", "without": "#3B82C4"}
LABELS = {"with": "With gut microbes", "without": "Without gut microbes"}


def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Analysis 1 — Dataset summary
# ---------------------------------------------------------------------------

def analysis_summary(data: dict) -> None:
    """Print and save a simple summary table."""
    print("\n=== Analysis 1: Dataset Summary ===")
    rows = []
    for mode in MODES:
        for cond in ("with", "without"):
            df = data[cond][mode]
            rm = rank_matrix(df)
            rows.append({
                "Condition": LABELS[cond],
                "Mode": mode,
                "Microbes": len(rm),
                "Metabolites": rm.shape[1],
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # Metabolite overlap
    for mode in MODES:
        m_with = set(rank_matrix(data["with"][mode]).columns)
        m_wo   = set(rank_matrix(data["without"][mode]).columns)
        print(f"\n  [{mode}] metabolites only in 'with':    {len(m_with - m_wo)}")
        print(f"  [{mode}] metabolites only in 'without': {len(m_wo - m_with)}")
        print(f"  [{mode}] shared metabolites:             {len(m_with & m_wo)}")

        # Isolate overlap
        i_with = set(data["with"][mode].index)
        i_wo   = set(data["without"][mode].index)
        print(f"  [{mode}] isolates only in 'with':    {len(i_with - i_wo)}")
        print(f"  [{mode}] isolates only in 'without': {len(i_wo - i_with)}")
        print(f"  [{mode}] shared isolates:             {len(i_with & i_wo)}")


# ---------------------------------------------------------------------------
# Analysis 2 — Rank entropy per metabolite
# ---------------------------------------------------------------------------

def analysis_entropy(data: dict) -> None:
    print("\n=== Analysis 2: Rank Entropy per Metabolite ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Rank entropy per metabolite\n"
                 "(higher = more discriminative across microbes)", fontsize=13)

    for ax, mode in zip(axes, MODES):
        ents = {}
        for cond in ("with", "without"):
            rm = rank_matrix(data[cond][mode])
            ents[cond] = metabolite_entropy(rm)

        # Only shared metabolites for a fair comparison
        shared = ents["with"].index.intersection(ents["without"].index)
        for cond in ("with", "without"):
            ax.hist(
                ents[cond].loc[shared],
                bins=40, alpha=0.6,
                color=COLORS[cond], label=LABELS[cond],
                density=True,
            )
        # Summary stats
        for cond in ("with", "without"):
            med = ents[cond].loc[shared].median()
            ax.axvline(med, color=COLORS[cond], linestyle="--", linewidth=1.5)

        ax.set_title(mode.capitalize())
        ax.set_xlabel("Shannon entropy of rank distribution")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    save(fig, "analysis2_rank_entropy.png")

    # Also save the per-metabolite entropy table
    for mode in MODES:
        rows = []
        for cond in ("with", "without"):
            rm = rank_matrix(data[cond][mode])
            e = metabolite_entropy(rm).rename(cond)
            rows.append(e)
        tbl = pd.concat(rows, axis=1)
        tbl["entropy_diff (without - with)"] = tbl["without"] - tbl["with"]
        tbl = tbl.sort_values("entropy_diff (without - with)", ascending=False)
        out = os.path.join(OUT_DIR, f"analysis2_entropy_{mode}.tsv")
        tbl.to_csv(out, sep="\t")
        print(f"  Saved entropy table: {out}")
        print(f"  [{mode}] Top 10 metabolites gaining discrimination WITHOUT gut data:")
        print(tbl.head(10)[["with", "without", "entropy_diff (without - with)"]].to_string())
        print(f"  [{mode}] Top 10 metabolites losing discrimination WITH gut data:")
        print(tbl.tail(10)[["with", "without", "entropy_diff (without - with)"]].to_string())


# ---------------------------------------------------------------------------
# Analysis 3 — Metabolite universality curves
# ---------------------------------------------------------------------------

def analysis_universality(data: dict) -> None:
    """
    For each condition, sweep K from 1 to 50.
    At each K, count how many metabolites are 'universal'
    (ranked in top-K by ≥ X% of microbes).
    Threshold X values: 25%, 50%, 75%.
    """
    print("\n=== Analysis 3: Metabolite Universality Curves ===")
    ks = np.arange(1, 101)
    thresholds = [0.25, 0.50, 0.75]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "How many distinct metabolites appear in the top-K predictions for ≥X% of microbes?\n"
        "(collapse signal: curve rises steeply — many metabolites are universally top-ranked)",
        fontsize=11,
    )

    for ax, mode in zip(axes, MODES):
        for cond in ("with", "without"):
            rm = rank_matrix(data[cond][mode])
            univ = universality_matrix(rm, ks)   # shape (n_mets, len(ks))
            for thresh in thresholds:
                n_universal = (univ >= thresh).sum(axis=0)   # Series indexed by k
                ax.plot(
                    ks, n_universal.values,
                    color=COLORS[cond],
                    linestyle={0.25: ":", 0.50: "--", 0.75: "-"}[thresh],
                    linewidth=1.8,
                    label=f"{LABELS[cond]} ≥{int(thresh*100)}% of microbes",
                )

        ax.set_title(mode.capitalize())
        ax.set_xlabel("K  (each microbe contributes its top-K metabolites)")
        ax.set_ylabel("# distinct metabolites crossing the popularity threshold\n"
                      "(out of all metabolites measured)")
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlim(1, ks[-1])

    save(fig, "analysis3_universality_curves.png")

    # Supplementary: at K=10, list the universal metabolites in each condition
    for mode in MODES:
        print(f"\n  [{mode}] Metabolites in top-10 for ≥50% of microbes:")
        for cond in ("with", "without"):
            rm = rank_matrix(data[cond][mode])
            frac = (rm <= 10).mean(axis=0)
            universal = frac[frac >= 0.50].sort_values(ascending=False)
            print(f"    {LABELS[cond]} ({len(universal)} metabolites):")
            for met, f in universal.items():
                print(f"      {met}: {f:.1%}")


# ---------------------------------------------------------------------------
# Analysis 4 — Effective differentiation rank
# ---------------------------------------------------------------------------

def analysis_diff_rank(data: dict) -> None:
    """
    For random microbe pairs, find the minimum K where their top-K sets differ.
    A collapsed model → large minimum K (everyone looks the same until deep).
    """
    print("\n=== Analysis 4: Effective Differentiation Rank ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Effective differentiation rank between microbe pairs\n"
        "(collapsed model → must go deeper before profiles diverge)",
        fontsize=12,
    )

    for ax, mode in zip(axes, MODES):
        all_diffs = {}
        for cond in ("with", "without"):
            rm = rank_matrix(data[cond][mode])
            print(f"  Computing diff ranks for [{mode}] {cond}... ", end="", flush=True)
            diffs = effective_diff_rank(rm, n_pairs=1000)
            all_diffs[cond] = diffs
            print(f"done (median={np.median(diffs):.0f}, mean={np.mean(diffs):.1f})")

        max_k = max(d.max() for d in all_diffs.values() if len(d) > 0)
        bins = np.arange(1, min(max_k + 2, 60))
        for cond in ("with", "without"):
            d = all_diffs[cond]
            ax.hist(
                np.clip(d, 1, bins[-1]),
                bins=bins, alpha=0.6,
                color=COLORS[cond], label=LABELS[cond],
                density=True,
            )
            ax.axvline(np.median(d), color=COLORS[cond], linestyle="--", linewidth=1.5,
                       label=f"median={np.median(d):.0f}")

        ax.set_title(mode.capitalize())
        ax.set_xlabel("Minimum K where top-K sets differ")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    save(fig, "analysis4_effective_diff_rank.png")


# ---------------------------------------------------------------------------
# Analysis 5 — Rank profile correlation for shared isolates
# ---------------------------------------------------------------------------

def analysis_shared_isolate_correlation(data: dict) -> None:
    """
    For isolates present in BOTH conditions, measure how much their
    predicted rank profile changed due to adding gut data.
    Low correlation → gut data substantially changed the prediction.
    """
    print("\n=== Analysis 5: Shared Isolate Rank Profile Correlation ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Spearman correlation of rank profiles (with vs without gut data)\n"
        "for isolates present in both conditions\n"
        "(low ρ → gut data substantially reshuffled predictions)",
        fontsize=11,
    )

    for ax, mode in zip(axes, MODES):
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])
        corrs = rank_profile_correlation(rm_with, rm_wo)

        if len(corrs) == 0:
            ax.text(0.5, 0.5, "No shared isolates", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mode.capitalize())
            continue

        print(f"  [{mode}] {len(corrs)} shared isolates")
        print(f"    mean ρ = {corrs.mean():.3f}  median ρ = {corrs.median():.3f}")
        print(f"    min  ρ = {corrs.min():.3f}  max ρ = {corrs.max():.3f}")
        print(f"    isolates with ρ < 0.5: {(corrs < 0.5).sum()}")

        ax.hist(corrs.values, bins=30, color="#6B4FA0", alpha=0.8, edgecolor="white")
        ax.axvline(corrs.median(), color="black", linestyle="--", linewidth=1.5,
                   label=f"median ρ={corrs.median():.2f}")
        ax.set_title(mode.capitalize())
        ax.set_xlabel("Spearman ρ (rank profile: with vs without)")
        ax.set_ylabel("# isolates")
        ax.legend(fontsize=10)

        # Save per-isolate table
        tax_cols_with = [c for c in TAXONOMY_COLS if c in data["with"][mode].columns]
        meta = data["with"][mode].loc[corrs.index, tax_cols_with]
        tbl = corrs.rename("spearman_rho").to_frame().join(meta)
        out = os.path.join(OUT_DIR, f"analysis5_shared_isolate_corr_{mode}.tsv")
        tbl.sort_values("spearman_rho").to_csv(out, sep="\t")
        print(f"  Saved: {out}")

    save(fig, "analysis5_shared_isolate_correlation.png")


# ---------------------------------------------------------------------------
# Analysis 6 — Top-N rank shift for shared isolates
# ---------------------------------------------------------------------------

def analysis_rank_shift(data: dict) -> None:
    """
    For each shared isolate, look at how its top-N metabolites change between
    conditions.  Report Jaccard similarity at K=10,20,30,50 and plot the
    distribution.  Also identify metabolites that consistently move up or down
    in rank when gut data is added.
    """
    print(f"\n=== Analysis 6: Top-N Rank Shift for Shared Isolates ===")
    ks = [10, 20, 30, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Jaccard similarity of top-K metabolite sets (with vs without gut data)\n"
        "for shared isolates  (lower = more reshuffling by gut data)",
        fontsize=11,
    )

    for ax, mode in zip(axes, MODES):
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])
        shared_iso  = rm_with.index.intersection(rm_wo.index)
        shared_mets = rm_with.columns.intersection(rm_wo.columns)

        if len(shared_iso) == 0:
            ax.text(0.5, 0.5, "No shared isolates", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mode.capitalize())
            continue

        rw  = rm_with.loc[shared_iso, shared_mets]
        rwo = rm_wo.loc[shared_iso, shared_mets]

        jaccards = {k: [] for k in ks}
        for iso in shared_iso:
            for k in ks:
                tw  = set(rw.loc[iso].nsmallest(k).index)
                two = set(rwo.loc[iso].nsmallest(k).index)
                j = len(tw & two) / len(tw | two) if (tw | two) else 1.0
                jaccards[k].append(j)

        positions = np.arange(len(ks))
        ax.boxplot(
            [jaccards[k] for k in ks],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="#3B82C4", alpha=0.7),
            medianprops=dict(color="black", linewidth=2),
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"top-{k}" for k in ks])
        ax.set_ylim(0, 1.05)
        ax.set_title(mode.capitalize())
        ax.set_xlabel("Top-K threshold")
        ax.set_ylabel("Jaccard similarity")
        for k in ks:
            med = np.median(jaccards[k])
            print(f"  [{mode}] top-{k} Jaccard median={med:.3f}  mean={np.mean(jaccards[k]):.3f}")

        # Metabolite rank shift: mean rank with gut minus mean rank without gut
        # Positive = metabolite moved DOWN (worse rank) when gut added
        mean_rank_with = rw.mean(axis=0)
        mean_rank_wo   = rwo.mean(axis=0)
        shift = (mean_rank_with - mean_rank_wo).sort_values()
        out = os.path.join(OUT_DIR, f"analysis6_rank_shift_{mode}.tsv")
        shift.rename("mean_rank_with_minus_without").to_frame().to_csv(out, sep="\t")
        print(f"  [{mode}] Top 15 metabolites ranked HIGHER (↑) when gut data added:")
        for met, s in shift.head(15).items():
            print(f"    {met}: {s:+.1f}")
        print(f"  [{mode}] Top 15 metabolites ranked LOWER (↓) when gut data added:")
        for met, s in shift.tail(15).items():
            print(f"    {met}: {s:+.1f}")

    save(fig, "analysis6_rank_shift_jaccard.png")


# ---------------------------------------------------------------------------
# Analysis 6b — Reconcile high Spearman ρ with large rank shifts
# ---------------------------------------------------------------------------

def analysis_rank_change_distribution(data: dict) -> None:
    """
    High Spearman ρ and large rank shifts seem contradictory.
    This analysis reveals why they're compatible by showing:
      (a) the full distribution of per-cell absolute rank changes
          (every metabolite × shared isolate pair)
      (b) per-metabolite mean absolute change sorted, to show
          whether big shifts are a few outliers or broadly spread
      (c) the cumulative fraction of cells with |Δrank| ≤ X

    Key insight: Spearman measures *relative ordering*, not absolute
    displacement.  Most cells can change by small amounts (preserving ρ)
    while a handful of outlier metabolites change by a lot.
    """
    print("\n=== Analysis 6b: Rank-change distribution (reconciling ρ vs shift) ===")
    print("  NOTE: 'with' has 391 metabolites, 'without' has 257.")
    print("  Adding 134 new metabolites mechanically inflates absolute ranks.")
    print("  A metabolite at rank r in 'without' is expected at r*(391/257)≈r*1.52")
    print("  in 'with' even if relative ordering is PERFECTLY preserved.")
    print("  Spearman ρ is scale-invariant; absolute rank shifts are partially artifactual.")
    print("  This analysis shows both raw |Δrank| and scale-corrected |Δpercentile|.\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Reconciling high Spearman ρ with large rank shifts\n"
        "Top: raw |Δrank|   Bottom: |Δpercentile rank| (corrects for different pool sizes)\n"
        "'with'=391 metabolites, 'without'=257 — scale difference inflates raw Δrank",
        fontsize=10,
    )

    for col, mode in enumerate(MODES):
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])
        shared_iso  = rm_with.index.intersection(rm_wo.index)
        shared_mets = rm_with.columns.intersection(rm_wo.columns)

        rw  = rm_with.loc[shared_iso, shared_mets]
        rwo = rm_wo.loc[shared_iso, shared_mets]

        n_with = rm_with.shape[1]   # 391
        n_wo   = rm_wo.shape[1]     # 257

        # Absolute rank change (raw — confounded by pool size difference)
        abs_delta_raw = (rw - rwo).abs()
        flat_raw = abs_delta_raw.values.ravel()

        # Percentile rank change (scale-corrected: rank / total metabolites)
        pct_with = rw  / n_with   # 0..1
        pct_wo   = rwo / n_wo     # 0..1
        abs_delta_pct = (pct_with - pct_wo).abs()
        flat_pct = abs_delta_pct.values.ravel()

        # Expected raw Δrank if ordering were perfectly preserved (pure scale artifact)
        expected_delta = rwo * (n_with / n_wo - 1)   # r * (391/257 - 1) ≈ 0.52r
        expected_flat  = expected_delta.values.ravel()

        print(f"  [{mode}] Pool sizes: with={n_with}, without={n_wo}, ratio={n_with/n_wo:.2f}")
        print(f"    Raw |Δrank|: median={np.median(flat_raw):.1f}  mean={np.mean(flat_raw):.1f}")
        print(f"    Expected |Δrank| from scale alone: median={np.median(expected_flat):.1f}  mean={np.mean(expected_flat):.1f}")
        print(f"    Scale-corrected |Δpercentile|: median={np.median(flat_pct):.3f}  mean={np.mean(flat_pct):.3f}")
        print(f"    (|Δpercentile|>0.10 = >10 pct-point shift: {(flat_pct>0.10).mean()*100:.1f}% of pairs)")

        # --- top panel: raw |Δrank| with expected-from-scale overlay ---
        ax_raw = axes[0, col]
        bins_raw = np.linspace(0, flat_raw.max(), 60)
        ax_raw.hist(flat_raw, bins=bins_raw, color="#6B4FA0", alpha=0.7,
                    density=True, label="Observed |Δrank|")
        ax_raw.hist(expected_flat, bins=bins_raw, color="#AAAAAA", alpha=0.5,
                    density=True, label="Expected from scale alone\n(if ordering perfectly preserved)")
        ax_raw.axvline(np.median(flat_raw), color="#6B4FA0", linestyle="--",
                       label=f"Observed median={np.median(flat_raw):.0f}")
        ax_raw.axvline(np.median(expected_flat), color="gray", linestyle=":",
                       label=f"Expected median={np.median(expected_flat):.0f}")
        ax_raw.set_title(f"{mode.capitalize()}: raw |Δrank|")
        ax_raw.set_xlabel("|rank_with − rank_without|  per (metabolite × isolate)")
        ax_raw.set_ylabel("Density")
        ax_raw.legend(fontsize=7)

        # --- bottom panel: scale-corrected |Δpercentile| ---
        ax_pct = axes[1, col]
        ax_pct.hist(flat_pct, bins=50, color="#3B82C4", alpha=0.8, edgecolor="white",
                    density=True)
        ax_pct.axvline(np.median(flat_pct), color="black", linestyle="--",
                       label=f"median={np.median(flat_pct):.3f}")
        ax_pct.axvline(0.10, color="red", linestyle=":", linewidth=1.2,
                       label="10 pct-point threshold")
        frac_large = (flat_pct > 0.10).mean() * 100
        ax_pct.set_title(
            f"{mode.capitalize()}: scale-corrected |Δpercentile rank|\n"
            f"{frac_large:.0f}% of pairs shift >10 percentile points — this is real biological signal"
        )
        ax_pct.set_xlabel("|percentile_with − percentile_without|  (0=no change, 1=max shift)")
        ax_pct.set_ylabel("Density")
        ax_pct.legend(fontsize=9)

    plt.tight_layout()
    save(fig, "analysis6b_rank_change_distribution.png")


# ---------------------------------------------------------------------------
# Analysis 7 — Gut-only vs aerobic-only metabolite profile comparison
# ---------------------------------------------------------------------------

def analysis_gut_vs_aerobic(data: dict) -> None:
    """
    Partition the 'with' dataset into gut isolates (present only in 'with')
    and aerobic isolates (present in 'without').  Compare the average rank
    profiles of these two groups to see if the model differentiates them.
    If the model is collapsed, the mean rank profiles will be nearly identical.
    """
    print("\n=== Analysis 7: Gut-only vs Aerobic isolate average rank profiles ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Mean metabolite rank: gut-only isolates vs aerobic isolates\n"
        "(if model is collapsed → profiles overlap completely)",
        fontsize=12,
    )

    for ax, mode in zip(axes, MODES):
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])

        aerobic_iso = rm_with.index.intersection(rm_wo.index)
        gut_iso     = rm_with.index.difference(rm_wo.index)

        if len(gut_iso) == 0:
            ax.text(0.5, 0.5, "No gut-only isolates found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mode.capitalize())
            continue

        print(f"  [{mode}] gut-only isolates: {len(gut_iso)}, aerobic isolates: {len(aerobic_iso)}")

        # Only shared metabolites for comparison
        shared_mets = rm_with.columns  # all metabolites in 'with'
        mean_gut     = rm_with.loc[gut_iso,     shared_mets].mean(axis=0)
        mean_aerobic = rm_with.loc[aerobic_iso, shared_mets].mean(axis=0)

        # Sort by mean aerobic rank for readability (top 60 metabolites)
        top_mets = mean_aerobic.sort_values().head(60).index
        x = np.arange(len(top_mets))

        ax.scatter(x, mean_aerobic.loc[top_mets].values, color="#3B82C4",
                   s=15, alpha=0.8, label=f"Aerobic (n={len(aerobic_iso)})")
        ax.scatter(x, mean_gut.loc[top_mets].values, color="#E07B39",
                   s=15, alpha=0.8, label=f"Gut-only (n={len(gut_iso)})")
        ax.set_xticks(x)
        ax.set_xticklabels(top_mets, rotation=90, fontsize=5)
        ax.set_title(mode.capitalize())
        ax.set_xlabel("Metabolite (sorted by mean aerobic rank, top 60)")
        ax.set_ylabel("Mean rank (lower = higher confidence)")
        ax.legend(fontsize=9)

        # Compute correlation between the two mean profiles
        rho, pval = spearmanr(mean_gut.loc[top_mets], mean_aerobic.loc[top_mets])
        print(f"    Spearman ρ of mean profiles (top 60 mets): {rho:.3f} (p={pval:.2e})")

        # Save full table
        tbl = pd.DataFrame({
            "mean_rank_gut": mean_gut,
            "mean_rank_aerobic": mean_aerobic,
            "diff (gut - aerobic)": mean_gut - mean_aerobic,
        }).sort_values("diff (gut - aerobic)")
        out = os.path.join(OUT_DIR, f"analysis7_gut_vs_aerobic_{mode}.tsv")
        tbl.to_csv(out, sep="\t")
        print(f"  Saved: {out}")

    save(fig, "analysis7_gut_vs_aerobic.png")


# ---------------------------------------------------------------------------
# Analysis 8 — Phantom metabolite audit (C18-only metabolites for aerobes)
# ---------------------------------------------------------------------------

def analysis_phantom_metabolites(data: dict) -> None:
    """
    The 134 metabolites present only in the 'with' dataset are C18-only —
    measured exclusively from gut microbes, never from aerobic Berkeley Lab
    microbes.  There is ZERO training signal linking aerobic microbes to
    these metabolites.

    A coherent model should rank C18-only metabolites near the bottom for
    aerobic microbes.  If gut embeddings bleed into aerobic embeddings, these
    metabolites creep up — a hallucination the model has no business making.

    Null expectation: if ranks were random over 391 metabolites, C18-only
    metabolites would average around rank 196 (the midpoint).
    """
    print("\n=== Analysis 8: Phantom metabolite audit (C18-only for aerobic microbes) ===")
    print("  C18-only metabolites were NEVER measured for aerobic microbes.")
    print("  Any high ranking for an aerobe is a model hallucination.\n")

    for mode in MODES:
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])

        c18_only    = rm_with.columns.difference(rm_wo.columns)   # 134 phantom mets
        aerobic_iso = rm_with.index.intersection(rm_wo.index)    # 225 aerobes
        n_total_mets = rm_with.shape[1]                          # 391

        print(f"  [{mode}] C18-only metabolites: {len(c18_only)}")
        print(f"  [{mode}] Aerobic isolates checked: {len(aerobic_iso)}")
        print(f"  [{mode}] Total metabolites in 'with' model: {n_total_mets}")
        print(f"  Null expectation for C18-only ranks (random): ~{n_total_mets/2:.0f}")

        # Ranks of C18-only metabolites for aerobic isolates
        phantom_ranks = rm_with.loc[aerobic_iso, c18_only]   # (225, 134)
        flat = phantom_ranks.values.ravel()

        print(f"\n  [{mode}] C18-only rank stats across aerobic isolates:")
        print(f"    median rank = {np.median(flat):.1f}  (null={n_total_mets/2:.0f})")
        print(f"    mean rank   = {np.mean(flat):.1f}")
        print(f"    % in top-20  = {(flat <= 20).mean()*100:.1f}%  "
              f"(null={20/n_total_mets*100:.1f}%)")
        print(f"    % in top-50  = {(flat <= 50).mean()*100:.1f}%  "
              f"(null={50/n_total_mets*100:.1f}%)")
        print(f"    % in top-100 = {(flat <= 100).mean()*100:.1f}%  "
              f"(null={100/n_total_mets*100:.1f}%)")

        # Per-metabolite: how often does each C18 metabolite crack top-K for aerobes?
        phantom_topk = {}
        for k in [20, 50, 100]:
            phantom_topk[k] = (phantom_ranks <= k).mean(axis=0)  # fraction of aerobes

        # Worst offenders: C18 metabolites most commonly in top-50 for aerobes
        worst = phantom_topk[50].sort_values(ascending=False)
        print(f"\n  [{mode}] Top-20 worst offenders "
              f"(C18-only mets most often in aerobic top-50):")
        for met, frac in worst.head(20).items():
            med_rank = phantom_ranks[met].median()
            print(f"    {met}: in top-50 for {frac*100:.1f}% of aerobes  "
                  f"(median rank={med_rank:.0f})")

        # How many aerobic microbes have ≥1 C18-only metabolite in their top-20?
        n_polluted = {k: (phantom_ranks <= k).any(axis=1).sum() for k in [20, 50, 100]}
        for k, n in n_polluted.items():
            print(f"  [{mode}] Aerobes with ≥1 C18 metabolite in top-{k}: "
                  f"{n}/{len(aerobic_iso)} ({n/len(aerobic_iso)*100:.1f}%)")

        # Save per-metabolite table
        tbl = pd.DataFrame({
            "median_rank_in_aerobes": phantom_ranks.median(axis=0),
            "mean_rank_in_aerobes":   phantom_ranks.mean(axis=0),
            "frac_aerobes_top20":     phantom_topk[20],
            "frac_aerobes_top50":     phantom_topk[50],
            "frac_aerobes_top100":    phantom_topk[100],
        }).sort_values("median_rank_in_aerobes")
        out = os.path.join(OUT_DIR, f"analysis8_phantom_mets_{mode}.tsv")
        tbl.to_csv(out, sep="\t")
        print(f"  Saved: {out}\n")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Phantom metabolite audit: where do C18-only metabolites rank for aerobic microbes?\n"
        "C18 metabolites were NEVER measured for aerobes — any high ranking is a hallucination.\n"
        "Dashed line = null expectation if model were uninformed (random rank)",
        fontsize=10,
    )

    for col, mode in enumerate(MODES):
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])
        c18_only    = rm_with.columns.difference(rm_wo.columns)
        aerobic_iso = rm_with.index.intersection(rm_wo.index)
        n_total     = rm_with.shape[1]

        phantom_ranks = rm_with.loc[aerobic_iso, c18_only]
        flat = phantom_ranks.values.ravel()

        # top: rank distribution histogram
        ax_hist = axes[0, col]
        bins = np.linspace(1, n_total, 50)
        ax_hist.hist(flat, bins=bins, color="#E07B39", alpha=0.8,
                     edgecolor="white", density=True, label="Observed C18 ranks")
        # Null: uniform over 1..n_total
        null_density = 1.0 / n_total
        ax_hist.axhline(null_density, color="black", linestyle="--", linewidth=1.5,
                        label=f"Null (uniform) = {null_density:.4f}")
        ax_hist.axvline(np.median(flat), color="#E07B39", linestyle=":",
                        label=f"Observed median = {np.median(flat):.0f}")
        ax_hist.axvline(n_total / 2, color="black", linestyle=":",
                        label=f"Null median = {n_total/2:.0f}")
        ax_hist.set_title(f"{mode.capitalize()}: rank distribution of C18-only mets\n"
                          f"for aerobic isolates (n={len(aerobic_iso)} isolates × "
                          f"{len(c18_only)} C18 mets)")
        ax_hist.set_xlabel("Rank in 'with' model  (1=top prediction, "
                           f"{n_total}=bottom)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)

        # bottom: per-metabolite median rank, sorted
        ax_sorted = axes[1, col]
        per_met = phantom_ranks.median(axis=0).sort_values()
        colors = ["#E07B39" if v < n_total * 0.25 else
                  "#F5A623" if v < n_total * 0.50 else
                  "#3B82C4" for v in per_met.values]
        ax_sorted.bar(np.arange(len(per_met)), per_met.values,
                      color=colors, width=1.0, edgecolor="none")
        ax_sorted.axhline(n_total / 2, color="black", linestyle="--",
                          linewidth=1.2, label=f"Null median = {n_total/2:.0f}")
        ax_sorted.axhline(n_total * 0.25, color="red", linestyle=":",
                          linewidth=1.0, label=f"Top 25% threshold = {n_total*0.25:.0f}")
        n_top_quarter = (per_met < n_total * 0.25).sum()
        ax_sorted.set_title(
            f"{mode.capitalize()}: median rank of each C18 metabolite across aerobes\n"
            f"Red/orange = ranked in top 25% of predictions  "
            f"({n_top_quarter}/{len(per_met)} C18 mets are hallucinated into top quartile)"
        )
        ax_sorted.set_xlabel("C18-only metabolites (sorted by median rank in aerobes)")
        ax_sorted.set_ylabel(f"Median rank  (1=top, {n_total}=bottom)")
        ax_sorted.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "analysis8_phantom_metabolites.png")


# ---------------------------------------------------------------------------
# Analysis 9 — Profile drift toward gut centroid
# ---------------------------------------------------------------------------

def analysis_gut_centroid_drift(data: dict) -> None:
    """
    Does adding gut data pull aerobic microbe predictions toward the gut
    microbe profile?

    Method:
      1. Compute the gut centroid = mean rank profile of gut-only isolates
         over the 257 shared HILIC metabolites (in the 'with' model).
      2. For each of the 225 aerobic isolates, compute Spearman ρ between
         its HILIC rank profile and the gut centroid, under both conditions:
           ρ_without: aerobic profile from 'without' model  vs gut centroid
           ρ_with:    aerobic profile from 'with' model     vs gut centroid
      3. If ρ_with > ρ_without for most isolates → gut data is dragging
         aerobic predictions toward gut biology.

    A scatter plot of ρ_with vs ρ_without (one dot per aerobic isolate)
    tells the story at a glance: points above the diagonal = pulled toward gut.
    """
    print("\n=== Analysis 9: Profile drift toward gut centroid ===")
    print("  Asking: does adding gut data make aerobic microbes LOOK MORE LIKE gut microbes?\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        "Profile drift toward gut centroid\n"
        "Each point = one aerobic isolate.  "
        "X axis = similarity to gut centroid WITHOUT gut training data.\n"
        "Y axis = similarity to gut centroid WITH gut training data.  "
        "Points above diagonal → gut data pulled that aerobe toward gut biology.",
        fontsize=10,
    )

    for ax, mode in zip(axes, MODES):
        rm_with = rank_matrix(data["with"][mode])
        rm_wo   = rank_matrix(data["without"][mode])

        hilic_mets  = rm_with.columns.intersection(rm_wo.columns)  # 257 shared
        aerobic_iso = rm_with.index.intersection(rm_wo.index)      # 225 aerobes
        gut_iso     = rm_with.index.difference(rm_wo.index)        # 504 gut-only

        # Gut centroid over HILIC metabolites (from 'with' model)
        gut_centroid = rm_with.loc[gut_iso, hilic_mets].mean(axis=0).values

        rho_without = {}
        rho_with    = {}
        for iso in aerobic_iso:
            profile_wo   = rm_wo.loc[iso, hilic_mets].values.astype(float)
            profile_with = rm_with.loc[iso, hilic_mets].values.astype(float)
            rho_without[iso], _ = spearmanr(profile_wo,   gut_centroid)
            rho_with[iso],    _ = spearmanr(profile_with, gut_centroid)

        rho_wo_arr  = np.array(list(rho_without.values()))
        rho_w_arr   = np.array(list(rho_with.values()))
        drift = rho_w_arr - rho_wo_arr
        n_drifted = (drift > 0).sum()

        print(f"  [{mode}] Aerobic isolates pulled TOWARD gut centroid "
              f"(ρ_with > ρ_without): {n_drifted}/{len(aerobic_iso)} "
              f"({n_drifted/len(aerobic_iso)*100:.1f}%)")
        print(f"    Mean drift (ρ_with − ρ_without): {drift.mean():+.3f}")
        print(f"    Median drift:                    {np.median(drift):+.3f}")
        print(f"    Max drift toward gut:            {drift.max():+.3f}")
        print(f"    Max drift away from gut:         {drift.min():+.3f}")

        # Scatter plot
        lim_min = min(rho_wo_arr.min(), rho_w_arr.min()) - 0.05
        lim_max = max(rho_wo_arr.max(), rho_w_arr.max()) + 0.05
        pulled_toward = drift > 0.05
        ax.scatter(rho_wo_arr[~pulled_toward], rho_w_arr[~pulled_toward],
                   s=18, alpha=0.5, color="#3B82C4", label="Stable (|drift| ≤ 0.05)")
        ax.scatter(rho_wo_arr[pulled_toward], rho_w_arr[pulled_toward],
                   s=25, alpha=0.8, color="#E07B39",
                   label=f"Pulled toward gut (drift >0.05, n={pulled_toward.sum()})")
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="black", linestyle="--", linewidth=1, label="No change")
        ax.plot([lim_min, lim_max], [lim_min + 0.05, lim_max + 0.05],
                color="#E07B39", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal")
        ax.set_xlabel("ρ (aerobic profile vs gut centroid)  WITHOUT gut data")
        ax.set_ylabel("ρ (aerobic profile vs gut centroid)  WITH gut data")
        ax.set_title(
            f"{mode.capitalize()}\n"
            f"{n_drifted}/{len(aerobic_iso)} aerobes pulled toward gut centroid  "
            f"(mean drift={drift.mean():+.3f})"
        )
        ax.legend(fontsize=8)

        # Save per-isolate table with taxonomy
        tax_cols = [c for c in TAXONOMY_COLS if c in data["with"][mode].columns]
        meta = data["with"][mode].loc[aerobic_iso, tax_cols]
        tbl = pd.DataFrame({
            "rho_without_vs_gut": rho_without,
            "rho_with_vs_gut":    rho_with,
            "drift":              {iso: rho_with[iso] - rho_without[iso]
                                   for iso in aerobic_iso},
        }).join(meta).sort_values("drift", ascending=False)
        out = os.path.join(OUT_DIR, f"analysis9_gut_centroid_drift_{mode}.tsv")
        tbl.to_csv(out, sep="\t")
        print(f"  Saved: {out}")

        print(f"\n  [{mode}] Top 10 aerobes most pulled toward gut biology:")
        for iso, row in tbl.head(10).iterrows():
            print(f"    {iso} ({row.get('genus','?')}): "
                  f"ρ_without={row['rho_without_vs_gut']:.3f} → "
                  f"ρ_with={row['rho_with_vs_gut']:.3f}  "
                  f"(drift={row['drift']:+.3f})")
        print(f"\n  [{mode}] Top 10 aerobes most resistant to gut drift:")
        for iso, row in tbl.tail(10).iterrows():
            print(f"    {iso} ({row.get('genus','?')}): "
                  f"ρ_without={row['rho_without_vs_gut']:.3f} → "
                  f"ρ_with={row['rho_with_vs_gut']:.3f}  "
                  f"(drift={row['drift']:+.3f})")

    save(fig, "analysis9_gut_centroid_drift.png")


# ---------------------------------------------------------------------------
# Analysis 10 — Biological plausibility of consumed metabolite rank changes
# ---------------------------------------------------------------------------

# Biochemical categories for HILIC metabolites
MET_CATEGORIES = {
    "Nucleosides":        ["adenosine", "guanosine", "uridine", "inosine", "cytidine",
                           "thymidine", "2'-deoxyadenosine", "2'-deoxyguanosine",
                           "2'-deoxyuridine", "2'-deoxycytidine", "5-methyluridine",
                           "1-methyladenosine", "S-methyl-5'-thioadenosine",
                           "5'-deoxyadenosine", "xanthosine"],
    "Purine & pyrimidine\nbases": ["hypoxanthine", "xanthine", "adenine", "guanine",
                           "cytosine", "uracil", "thymine", "5-methylcytosine"],
    "Nucleotides":        ["5'-AMP", "5'-GMP", "5'-UMP", "5'-CDP",
                           "2'-deoxyadenosine 5'-monophosphate",
                           "2'-deoxyguanosine 5'-monophosphate",
                           "cytidine 5'-monophosphate", "3'-CMP", "α-NAD+",
                           "β-nicotinamide D-ribonucleotide",
                           "3',5'-cyclic AMP", "3',5'-cyclic GMP",
                           "2',3'-cyclophospho-AMP", "2',3'-cyclophospho-CMP",
                           "ADP-D-ribose"],
    "Vitamins &\ncofactors":      ["nicotinamide", "nicotinate", "riboflavin", "biotin",
                           "thiamine", "pyridoxine", "pyridoxamine", "4-pyridoxate",
                           "lumichrome", "dethiobiotin", "N-methylnicotinate",
                           "isonicotinate", "6-hydroxynicotinate"],
    "Amino acids":        ["glutamate", "glycine", "L-serine", "L-valine", "L-lysine",
                           "arginine", "methionine", "L-tyrosine", "L-asparagine",
                           "L-citrulline", "L-ornithine", "L-pipecolate", "L-ascorbate",
                           "β-alanine", "L-ectoine", "D-alanine", "D-aspartate",
                           "D-glutamine", "D-histidine", "D-leucine", "D-proline",
                           "D-phenylalanine", "D-tryptophan", "D-threonine",
                           "D-isoleucine", "L-homocysteine", "L-kynurenine"],
    "TCA & organic\nacids":       ["citrate", "succinate", "malate", "pyruvate", "lactate",
                           "cis-aconitate", "2-oxoglutarate", "D-threo-isocitrate",
                           "itaconate", "mesaconate", "malonate", "formate",
                           "glycerate", "fumarate", "tartrate", "2-oxopentanoate",
                           "3-hydroxybutanoate", "4-hydroxybutanoate", "citramalate"],
    "Aromatics":          ["benzoate", "4-hydroxybenzoate", "3-hydroxybenzoate",
                           "2-phenylacetate", "4-coumarate", "vanillate", "salicylate",
                           "anthranilate", "3,4-dihydroxybenzoate", "2,3-dihydroxybenzoate",
                           "mandelate", "quinaldate", "kynurenate", "cinnamate",
                           "2,5-dihydroxybenzoate", "4-hydroxyphenylacetate"],
    "Indoles":            ["tryptamine", "serotonin", "(indol-3-yl)acetate",
                           "indole-3-propanoate", "indole-3-pyruvate",
                           "indole-3-acetaldehyde", "indole-3-carbaldehyde",
                           "indole-3-acetamide", "(indol-3-yl)butanoate",
                           "5-hydroxy-L-tryptophan"],
    "Sugars & polyols":   ["D-glucose", "sucrose", "D-maltose", "D-maltotriose",
                           "D-xylose", "D-sorbitol", "inositol", "ribitol",
                           "raffinose", "α,α-trehalose", "D-fucose", "D-allose",
                           "D-glucuronate", "2-deoxy-D-glucose"],
    "Polyamines":         ["putrescine", "cadaverine", "N-acetylputrescine",
                           "N,N-dimethylglycine", "sarcosine"],
}

# Canonical aerobic strains for case studies (isolate_id: display_label)
CASE_STUDY_STRAINS = {
    "Pp":                    "Pseudomonas putida (Pp)",
    "BW25113":               "Escherichia coli K-12 (BW25113)",
    "iso8_BS":               "Bacillus sp. (iso8_BS)",
    "FW305-C-30-S16":        "Caulobacter sp. (FW305-C-30-S16)",
    "Sphingomonas":          "Sphingomonas sp.",
    "Bb":                    "Brevundimonas sp. (Bb)",
}

# Nucleosides/nucleotides with known biology for the spotlight
NUCLEOSIDE_SPOTLIGHT = [
    "nicotinamide", "adenosine", "uridine", "guanosine", "inosine",
    "cytidine", "thymidine", "hypoxanthine", "S-methyl-5'-thioadenosine",
    "2'-deoxyuridine",
]


def analysis_biological_plausibility(data: dict) -> None:
    """
    Three-part biological interpretation of consumed rank changes:

    A) Nucleoside rank collapse:
       Boxplots of rank distribution across aerobic microbes in both conditions
       for key nucleosides.  Collapse = all aerobes predicted equally.

    B) Biochemical category rank shift:
       Lollipop chart showing mean percentile rank shift per metabolite class.
       Tells a biologist which chemistries are systematically elevated/suppressed.

    C) Case studies for canonical aerobes:
       Side-by-side top-15 consumed metabolites for well-characterised strains.
       C18-only metabolites flagged as hallucinations.
    """
    mode = "consumed"   # user-requested focus
    rm_with = rank_matrix(data["with"][mode])
    rm_wo   = rank_matrix(data["without"][mode])

    hilic_mets  = rm_with.columns.intersection(rm_wo.columns)   # 257 shared
    c18_only    = rm_with.columns.difference(rm_wo.columns)      # 134 phantom
    aerobic_iso = rm_with.index.intersection(rm_wo.index)        # 225 aerobes

    rw  = rm_with.loc[aerobic_iso, hilic_mets]
    rwo = rm_wo.loc[aerobic_iso, hilic_mets]

    n_with = rm_with.shape[1]   # 391  (for percentile in 'with')
    n_wo   = rm_wo.shape[1]     # 257  (for percentile in 'without')

    # -----------------------------------------------------------------------
    # Panel A — Nucleoside rank collapse
    # -----------------------------------------------------------------------
    print("\n=== Analysis 10A: Nucleoside rank collapse (consumed, aerobic isolates) ===")

    spotlight = [m for m in NUCLEOSIDE_SPOTLIGHT if m in hilic_mets]
    fig_a, ax_a = plt.subplots(figsize=(14, 6))
    fig_a.suptitle(
        "Rank distribution of nucleosides/vitamins across aerobic microbes\n"
        "Narrow box = model treats all aerobes identically (collapse).\n"
        "Wide box = model discriminates between aerobes (desirable).",
        fontsize=11,
    )

    n = len(spotlight)
    positions_wo   = np.arange(n) * 3.0
    positions_with = np.arange(n) * 3.0 + 1.0

    # Convert to percentile rank within each model's metabolite space
    pct_wo   = rwo[spotlight] / n_wo
    pct_with = rw[spotlight]  / n_with

    ax_a.boxplot(
        [pct_wo[m].values for m in spotlight],
        positions=positions_wo, widths=0.8, patch_artist=True,
        boxprops=dict(facecolor=COLORS["without"], alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    ax_a.boxplot(
        [pct_with[m].values for m in spotlight],
        positions=positions_with, widths=0.8, patch_artist=True,
        boxprops=dict(facecolor=COLORS["with"], alpha=0.7),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )

    ax_a.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                 label="Median (random)")
    ax_a.set_xticks(positions_wo + 0.5)
    ax_a.set_xticklabels(spotlight, rotation=35, ha="right", fontsize=9)
    ax_a.set_ylabel("Percentile rank  (0 = top prediction, 1 = bottom)")
    ax_a.set_ylim(0, 1.05)

    from matplotlib.patches import Patch
    ax_a.legend(handles=[
        Patch(facecolor=COLORS["without"], alpha=0.7, label=LABELS["without"]),
        Patch(facecolor=COLORS["with"],    alpha=0.7, label=LABELS["with"]),
    ], fontsize=9)

    # Print IQR as proxy for discrimination
    print(f"  {'Metabolite':<35} {'IQR_without':>12} {'IQR_with':>10} {'Median_without':>15} {'Median_with':>12}")
    for m in spotlight:
        iqr_wo   = np.percentile(pct_wo[m], 75) - np.percentile(pct_wo[m], 25)
        iqr_with = np.percentile(pct_with[m], 75) - np.percentile(pct_with[m], 25)
        print(f"  {m:<35} {iqr_wo:>12.3f} {iqr_with:>10.3f} "
              f"{pct_wo[m].median():>15.3f} {pct_with[m].median():>12.3f}")

    save(fig_a, "analysis10a_nucleoside_collapse.png")

    # -----------------------------------------------------------------------
    # Panel B — Biochemical category rank shift
    # -----------------------------------------------------------------------
    print("\n=== Analysis 10B: Biochemical category rank shift ===")

    cat_results = []
    for cat, mets in MET_CATEGORIES.items():
        present = [m for m in mets if m in hilic_mets]
        if len(present) < 2:
            continue
        # Mean percentile rank across aerobic isolates for this category
        mean_pct_wo   = (rwo[present] / n_wo).values.ravel().mean()
        mean_pct_with = (rw[present]  / n_with).values.ravel().mean()
        shift = mean_pct_with - mean_pct_wo   # positive = moved DOWN (worse), negative = moved UP
        cat_results.append({
            "category": cat,
            "n_metabolites": len(present),
            "mean_pct_without": mean_pct_wo,
            "mean_pct_with": mean_pct_with,
            "shift": shift,
        })
        print(f"  {cat.replace(chr(10),' '):<30} n={len(present):>3}  "
              f"without={mean_pct_wo:.3f}  with={mean_pct_with:.3f}  "
              f"shift={shift:+.3f}  "
              f"({'↑ better' if shift < 0 else '↓ worse'})")

    cat_df = pd.DataFrame(cat_results).sort_values("shift")
    out = os.path.join(OUT_DIR, "analysis10b_category_shift.tsv")
    cat_df.to_csv(out, sep="\t", index=False)
    print(f"  Saved: {out}")

    fig_b, ax_b = plt.subplots(figsize=(10, 7))
    fig_b.suptitle(
        "Effect of gut data on consumed metabolite predictions by biochemical class\n"
        "(aerobic microbes only, HILIC shared metabolites)\n"
        "↑ Blue = gut data improves average rank (moves up)   "
        "↓ Red = gut data suppresses rank (moves down)",
        fontsize=10,
    )

    cats    = [r["category"] for _, r in cat_df.iterrows()]
    shifts  = cat_df["shift"].values
    n_mets  = cat_df["n_metabolites"].values
    colors_b = ["#3B82C4" if s < 0 else "#E07B39" for s in shifts]

    y = np.arange(len(cats))
    ax_b.barh(y, shifts, color=colors_b, alpha=0.8, edgecolor="white", height=0.6)
    ax_b.axvline(0, color="black", linewidth=1)
    for i, (s, n_m) in enumerate(zip(shifts, n_mets)):
        x_pos = s + (0.002 if s >= 0 else -0.002)
        ha = "left" if s >= 0 else "right"
        ax_b.text(x_pos, i, f"n={n_m}", va="center", ha=ha, fontsize=8)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(cats, fontsize=10)
    ax_b.set_xlabel("Δ mean percentile rank (with − without)\n"
                    "Negative = gut data pushes this class HIGHER in predictions\n"
                    "Positive = gut data pushes this class LOWER (suppressed)")
    ax_b.set_title("Consumed metabolite class rank shift in aerobic microbes")

    save(fig_b, "analysis10b_category_shift.png")

    # -----------------------------------------------------------------------
    # Panel C — Rank-flow bump charts, one figure per strain
    # -----------------------------------------------------------------------
    print("\n=== Analysis 10C: Rank-flow case studies (one figure per strain) ===")

    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    present_strains = {iso: label for iso, label in CASE_STUDY_STRAINS.items()
                       if iso in aerobic_iso}
    top_k   = 50          # metabolites shown per side
    off_y   = top_k + 2   # y-position for off-chart endpoints

    # x coordinates for the two columns and the curve endpoints
    X_LEFT_LABEL  = 0.34   # right edge of left labels
    X_LEFT_RANK   = 0.005
    X_RIGHT_LABEL = 0.66   # left edge of right labels
    X_RIGHT_RANK  = 0.995
    X_LINE_LEFT   = 0.36   # where lines begin on left
    X_LINE_RIGHT  = 0.64   # where lines end on right

    for iso, label in present_strains.items():
        ranks_wo   = rm_wo.loc[iso]       # ranks out of 257
        ranks_with = rm_with.loc[iso]     # ranks out of 391

        top_wo_names   = list(ranks_wo.nsmallest(top_k).index)
        top_with_names = list(ranks_with.nsmallest(top_k).index)

        # Union, preserving first-seen order (without side first)
        seen = {}
        for m in top_wo_names + top_with_names:
            if m not in seen:
                seen[m] = True
        all_mets = list(seen.keys())

        # Print table
        print(f"\n  {label}")
        print(f"  {'Rank':<6} {'WITHOUT gut data':<45} {'WITH gut data'}")
        for r in range(top_k):
            wo_m   = top_wo_names[r]   if r < len(top_wo_names)   else ""
            with_m = top_with_names[r] if r < len(top_with_names) else ""
            flag    = " [C18!]" if with_m in c18_only else ""
            entered = " ← entered top-50" if (with_m and with_m not in top_wo_names) else ""
            print(f"  {r+1:<6} {wo_m:<45} {with_m}{flag}{entered}")

        # Build figure
        fig_h = max(14, top_k * 0.30 + 3)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        ax.set_xlim(0, 1)
        ax.set_ylim(off_y + 1, 0)   # rank 1 at top, rank top_k at bottom
        ax.axis("off")

        # Alternating row stripes (only within top_k visible rows)
        for r in range(1, top_k + 1):
            if r % 2 == 0:
                ax.axhspan(r - 0.5, r + 0.5, color="#F5F5F5",
                           xmin=0, xmax=1, zorder=0)

        # Column header box shading
        ax.axvspan(0, X_LINE_LEFT,  ymin=0, ymax=1, color=COLORS["without"],
                   alpha=0.06, zorder=0)
        ax.axvspan(X_LINE_RIGHT, 1, ymin=0, ymax=1, color=COLORS["with"],
                   alpha=0.06, zorder=0)

        # Column headers
        ax.text(X_LEFT_LABEL / 2,  -0.9, "Without gut data",
                fontsize=11, fontweight="bold", ha="center",
                color=COLORS["without"])
        ax.text((X_RIGHT_LABEL + 1) / 2, -0.9, "With gut data",
                fontsize=11, fontweight="bold", ha="center",
                color=COLORS["with"])
        ax.text(0.5, -1.8, label,
                fontsize=13, fontweight="bold", ha="center", color="black")
        ax.text(0.5, -0.9,
                "Lines: gray=stable  blue=moved up  orange=moved down  "
                "red=C18 hallucination (never measured)",
                fontsize=7.5, ha="center", color="#555555", style="italic")

        # Draw lines then labels (lines behind labels)
        for met in all_mets:
            rank_wo   = int(ranks_wo[met])   if met in ranks_wo.index   else 9999
            rank_with = int(ranks_with[met])

            in_left  = rank_wo   <= top_k
            in_right = rank_with <= top_k
            if not in_left and not in_right:
                continue

            y_left  = rank_wo   if in_left  else off_y
            y_right = rank_with if in_right else off_y

            is_c18 = met in c18_only
            delta  = rank_with - rank_wo if met in ranks_wo.index else None

            if is_c18:
                color, lw, alpha = "#CC3300", 2.2, 0.85
            elif delta is not None and delta < -10:
                color, lw, alpha = "#3B82C4", 1.8, 0.75   # moved up
            elif delta is not None and delta > 10:
                color, lw, alpha = "#E07B39", 1.8, 0.75   # moved down
            else:
                color, lw, alpha = "#BBBBBB", 1.0, 0.55   # stable

            # Dashed if going off-chart on either side
            style = "--" if (not in_left or not in_right) else "-"

            verts = [(X_LINE_LEFT, y_left), (0.5, y_left),
                     (0.5, y_right), (X_LINE_RIGHT, y_right)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            ax.add_patch(PathPatch(Path(verts, codes),
                                   facecolor="none", edgecolor=color,
                                   linewidth=lw, alpha=alpha,
                                   linestyle=style))

        # Draw labels (on top of lines)
        for met in all_mets:
            rank_wo   = int(ranks_wo[met])   if met in ranks_wo.index else 9999
            rank_with = int(ranks_with[met])
            in_left   = rank_wo   <= top_k
            in_right  = rank_with <= top_k
            is_c18    = met in c18_only

            # Left label
            if in_left:
                ax.text(X_LEFT_RANK, rank_wo, f"{rank_wo}",
                        fontsize=6.5, ha="left", va="center", color="#AAAAAA")
                ax.text(X_LEFT_LABEL, rank_wo, met,
                        fontsize=7.5, ha="right", va="center",
                        color="#CC3300" if is_c18 else "black",
                        fontweight="bold" if is_c18 else "normal")

            # Right label
            if in_right:
                entered = (not in_left) and (not is_c18)
                text_col = ("#CC3300" if is_c18 else
                            "#1A7A1A" if entered else "black")
                fw = "bold" if (is_c18 or entered) else "normal"
                suffix = "  [C18]" if is_c18 else ("  ←" if entered else "")
                ax.text(X_RIGHT_RANK, rank_with, f"{rank_with}",
                        fontsize=6.5, ha="right", va="center", color="#AAAAAA")
                ax.text(X_RIGHT_LABEL, rank_with, met + suffix,
                        fontsize=7.5, ha="left", va="center",
                        color=text_col, fontweight=fw)

        # "off-chart" label at bottom
        ax.text(0.5, off_y + 0.5, "▼ off chart",
                fontsize=7, ha="center", va="top", color="#AAAAAA", style="italic")
        ax.axhline(top_k + 0.5, color="#DDDDDD", linewidth=0.7, xmin=0, xmax=1)

        iso_clean = iso.replace("/", "_").replace(" ", "_")
        save(fig, f"analysis10c_{iso_clean}_rankflow.png")


# ---------------------------------------------------------------------------
# Analysis 11 — Enzyme presence vs BPR rank correlation
# ---------------------------------------------------------------------------

COMPOUND_SUPPORT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "processed_data",
    "taxon_function_compound_support.parquet",
)


def load_enzyme_support() -> dict:
    """
    Returns {genus: set_of_model_compounds_with_enzyme_support}.
    Uses taxon_function_compound_support.parquet filtered to genus level.
    """
    df = pd.read_parquet(COMPOUND_SUPPORT_PATH)
    genus_df = df[df["level"] == "genus"][["taxon", "representative_compound",
                                           "n_proteins_support"]].copy()
    support = {}
    for genus, grp in genus_df.groupby("taxon"):
        support[genus] = dict(zip(grp["representative_compound"],
                                  grp["n_proteins_support"]))
    return support


def enzyme_auc_for_isolate(ranks: pd.Series, supported: dict) -> float | None:
    """
    AUC for one isolate: are enzyme-supported compounds ranked higher?

    ranks   : Series (index=compound name, value=rank, lower=better)
    supported: {compound: n_proteins_support} for this isolate's genus

    Returns None if there are no positives or no negatives.
    """
    from sklearn.metrics import roc_auc_score
    mets = ranks.index
    y_true  = np.array([1 if m in supported else 0 for m in mets])
    scores  = -ranks.values.astype(float)   # lower rank → higher score

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return roc_auc_score(y_true, scores)


def analysis_enzyme_bpr_correlation(data: dict) -> None:
    """
    External validation completely orthogonal to the training signal.

    Training signal : observed metabolite consumption (mass spec)
    Validation signal: UniProt enzyme annotations — which metabolites does
                       each genus have enzymatic machinery to process?

    If BPR is learning biochemistry:
      → enzyme-supported compounds should rank higher (lower rank number)
      → per-microbe AUC >> 0.5

    If BPR is only recovering training noise or collapsing to the mean:
      → AUC ≈ 0.5  (no concordance with enzyme presence)

    Additional question: does adding gut data improve or hurt concordance?
    """
    print("\n=== Analysis 11: Enzyme presence vs BPR rank correlation ===")
    print("  Validation is ORTHOGONAL to training — UniProt annotations were")
    print("  never seen by the model.  AUC > 0.5 = model learns biochemistry.\n")

    support = load_enzyme_support()
    n_genera_with_support = len(support)
    all_supported_mets = set(m for s in support.values() for m in s)
    print(f"  Genera with enzyme support data: {n_genera_with_support}")
    print(f"  Unique compounds with any enzyme support: {len(all_supported_mets)}")

    mode = "consumed"   # primary focus; enzyme annotations are directionally agnostic

    results = {}   # {condition: DataFrame(isolate, genus, auc, n_supported)}
    for cond in ("with", "without"):
        df_full = data[cond][mode]          # includes taxonomy columns
        rm      = rank_matrix(df_full)      # rank columns only

        rows = []
        for iso in rm.index:
            genus = df_full.loc[iso, "genus"] if "genus" in df_full.columns else None
            if genus is None or genus not in support:
                continue
            genus_support = support[genus]
            # Only use compounds present in this condition's model
            local_support = {m: v for m, v in genus_support.items()
                             if m in rm.columns}
            if len(local_support) < 3:
                continue
            auc = enzyme_auc_for_isolate(rm.loc[iso], local_support)
            if auc is None:
                continue
            rows.append({
                "isolate": iso,
                "genus":   genus,
                "auc":     auc,
                "n_supported": len(local_support),
                "condition": cond,
            })
        results[cond] = pd.DataFrame(rows)
        n = len(rows)
        auc_vals = results[cond]["auc"].values
        print(f"  [{cond}] isolates with enzyme data: {n}")
        print(f"    median AUC = {np.median(auc_vals):.3f}")
        print(f"    mean AUC   = {np.mean(auc_vals):.3f}")
        print(f"    % AUC > 0.5: {(auc_vals > 0.5).mean()*100:.1f}%")
        print(f"    % AUC > 0.6: {(auc_vals > 0.6).mean()*100:.1f}%")
        print(f"    % AUC > 0.7: {(auc_vals > 0.7).mean()*100:.1f}%")

    # -------------------------------------------------------------------
    # Figure A — Per-isolate AUC distribution
    # -------------------------------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=(9, 5))
    fig_a.suptitle(
        "BPR rank concordance with UniProt enzyme presence\n"
        "AUC per isolate: are metabolites the genus has enzymes for ranked higher?\n"
        "AUC = 0.5 (dashed) → random.  AUC > 0.5 → model learned biochemistry.",
        fontsize=10,
    )
    bins = np.linspace(0, 1, 41)
    for cond in ("without", "with"):
        if cond not in results:
            continue
        auc_vals = results[cond]["auc"].values
        ax_a.hist(auc_vals, bins=bins, alpha=0.6, color=COLORS[cond],
                  label=f"{LABELS[cond]}  (median={np.median(auc_vals):.3f}, n={len(auc_vals)})",
                  density=True)
        ax_a.axvline(np.median(auc_vals), color=COLORS[cond],
                     linestyle="--", linewidth=1.5)
    ax_a.axvline(0.5, color="black", linestyle=":", linewidth=1.2, label="Random (AUC=0.5)")
    ax_a.set_xlabel("AUC (enzyme-supported compounds ranked higher than unsupported)")
    ax_a.set_ylabel("Density")
    ax_a.legend(fontsize=9)
    save(fig_a, "analysis11a_enzyme_auc_distribution.png")

    # -------------------------------------------------------------------
    # Figure B — Per-genus median AUC (both conditions, sorted by 'without')
    # -------------------------------------------------------------------
    for cond in ("with", "without"):
        if cond not in results or results[cond].empty:
            continue
        tbl = (results[cond]
               .groupby("genus")["auc"]
               .agg(median_auc="median", n_isolates="count")
               .reset_index()
               .query("n_isolates >= 2"))
        out = os.path.join(OUT_DIR, f"analysis11b_genus_auc_{cond}.tsv")
        tbl.sort_values("median_auc", ascending=False).to_csv(out, sep="\t", index=False)
        print(f"\n  [{cond}] Top 15 genera by enzyme concordance (AUC):")
        for _, row in tbl.nlargest(15, "median_auc").iterrows():
            print(f"    {row['genus']:<30} median AUC={row['median_auc']:.3f}  (n={row['n_isolates']})")
        print(f"  [{cond}] Bottom 10 genera (lowest concordance):")
        for _, row in tbl.nsmallest(10, "median_auc").iterrows():
            print(f"    {row['genus']:<30} median AUC={row['median_auc']:.3f}  (n={row['n_isolates']})")

    # Combined per-genus plot (shared genera, both conditions side by side)
    if "with" in results and "without" in results:
        g_wo = (results["without"].groupby("genus")["auc"]
                .agg(median_auc="median", n="count")
                .query("n >= 2"))
        g_w  = (results["with"].groupby("genus")["auc"]
                .agg(median_auc="median", n="count")
                .query("n >= 2"))
        shared_genera = g_wo.index.intersection(g_w.index)
        g_merged = pd.DataFrame({
            "without": g_wo.loc[shared_genera, "median_auc"],
            "with":    g_w.loc[shared_genera, "median_auc"],
        }).sort_values("without", ascending=False)

        n_show = min(40, len(g_merged))
        g_plot = g_merged.head(n_show)
        y = np.arange(len(g_plot))
        fig_b, ax_b = plt.subplots(figsize=(9, max(6, n_show * 0.28)))
        ax_b.barh(y - 0.18, g_plot["without"], height=0.34,
                  color=COLORS["without"], alpha=0.8, label=LABELS["without"])
        ax_b.barh(y + 0.18, g_plot["with"],    height=0.34,
                  color=COLORS["with"],    alpha=0.8, label=LABELS["with"])
        ax_b.axvline(0.5, color="black", linestyle=":", linewidth=1.0)
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(g_plot.index, fontsize=8)
        ax_b.set_xlabel("Median AUC (enzyme-supported compounds ranked higher)")
        ax_b.set_title(
            f"Per-genus BPR–enzyme concordance (top {n_show} genera by 'without' AUC)\n"
            "Sorted by 'without' AUC.  Bar left of dashed line = below chance."
        )
        ax_b.legend(fontsize=9)
        ax_b.set_xlim(0, 1)
        save(fig_b, "analysis11b_genus_auc.png")

    # -------------------------------------------------------------------
    # Figure C — Dose-response: n_proteins_support vs mean BPR percentile rank
    # -------------------------------------------------------------------
    print("\n  [Dose-response] n_proteins_support bins vs mean BPR percentile rank:")
    fig_c, axes_c = plt.subplots(1, 2, figsize=(12, 5))
    fig_c.suptitle(
        "Dose-response: more enzymatic evidence → better BPR rank?\n"
        "Each point = one (isolate × compound) pair.  "
        "Downward slope = model learned biochemistry.",
        fontsize=10,
    )

    support_bins = [1, 5, 20, 100, 500, 10_000]
    bin_labels   = ["1", "2–5", "6–20", "21–100", "101–500", ">500"]

    for ax_c, cond in zip(axes_c, ("without", "with")):
        df_full = data[cond][mode]
        rm      = rank_matrix(df_full)
        n_mets  = rm.shape[1]

        rows_c = []
        for iso in rm.index:
            genus = df_full.loc[iso, "genus"] if "genus" in df_full.columns else None
            if genus is None or genus not in support:
                continue
            for met, n_prot in support[genus].items():
                if met not in rm.columns:
                    continue
                pct_rank = rm.loc[iso, met] / n_mets   # 0..1, lower=better
                rows_c.append({"n_proteins_support": n_prot, "pct_rank": pct_rank})

        df_c = pd.DataFrame(rows_c)
        df_c["bin"] = pd.cut(df_c["n_proteins_support"],
                             bins=[0] + support_bins,
                             labels=bin_labels[:len(support_bins)])
        bin_means = df_c.groupby("bin", observed=True)["pct_rank"].agg(
            mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)), n="count"
        ).reset_index()

        print(f"  [{cond}]")
        for _, row in bin_means.iterrows():
            print(f"    n_proteins_support={row['bin']:>6}: "
                  f"mean percentile rank = {row['mean']:.3f} ± {row['sem']:.3f}  "
                  f"(n={row['n']:,} pairs)")

        x = np.arange(len(bin_means))
        ax_c.bar(x, bin_means["mean"], yerr=bin_means["sem"],
                 color=COLORS[cond], alpha=0.8, edgecolor="white",
                 capsize=4)
        ax_c.axhline(0.5, color="black", linestyle=":", linewidth=1,
                     label="Random (0.5)")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(bin_means["bin"], fontsize=9)
        ax_c.set_xlabel("# UniProt proteins supporting this enzyme in this genus")
        ax_c.set_ylabel("Mean percentile rank  (lower = predicted more confidently)")
        ax_c.set_ylim(0, 0.75)
        ax_c.set_title(f"{LABELS[cond]}")
        ax_c.legend(fontsize=9)

    save(fig_c, "analysis11c_enzyme_dose_response.png")

    # -------------------------------------------------------------------
    # Figure D — Per-compound AUC: which metabolites are best predicted
    #            by enzyme presence?
    # -------------------------------------------------------------------
    print("\n  [Per-compound AUC] Which metabolites are best/worst enzyme-concordant?")
    fig_d, axes_d = plt.subplots(1, 2, figsize=(14, 7))
    fig_d.suptitle(
        "Per-compound enzyme concordance AUC ('without' model, aerobic isolates)\n"
        "For each compound: AUC of enzyme-support vs BPR rank across all isolates\n"
        "that have enzymatic data for their genus.",
        fontsize=10,
    )

    for ax_d, cond in zip(axes_d, ("without", "with")):
        from sklearn.metrics import roc_auc_score as _auc
        df_full = data[cond][mode]
        rm      = rank_matrix(df_full)
        n_mets  = rm.shape[1]

        # For each compound, gather (has_enzyme_support_for_genus, rank)
        met_data = {met: {"y": [], "score": []} for met in rm.columns
                    if met in all_supported_mets}

        for iso in rm.index:
            genus = df_full.loc[iso, "genus"] if "genus" in df_full.columns else None
            if genus is None or genus not in support:
                continue
            genus_support = support[genus]
            for met in rm.columns:
                if met not in met_data:
                    continue
                met_data[met]["y"].append(1 if met in genus_support else 0)
                met_data[met]["score"].append(-rm.loc[iso, met])

        met_aucs = {}
        for met, d in met_data.items():
            if sum(d["y"]) < 5 or sum(1 - v for v in d["y"]) < 5:
                continue
            try:
                met_aucs[met] = _auc(d["y"], d["score"])
            except Exception:
                pass

        if not met_aucs:
            ax_d.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_d.transAxes)
            continue

        met_auc_s = pd.Series(met_aucs).sort_values(ascending=False)
        n_show    = min(40, len(met_auc_s))
        top_mets  = met_auc_s.head(n_show // 2)
        bot_mets  = met_auc_s.tail(n_show // 2)
        display   = pd.concat([top_mets, bot_mets])

        colors_d = ["#3B82C4" if v >= 0.5 else "#E07B39" for v in display.values]
        y_d = np.arange(len(display))
        ax_d.barh(y_d, display.values, color=colors_d, alpha=0.8, edgecolor="white")
        ax_d.axvline(0.5, color="black", linestyle=":", linewidth=1)
        ax_d.set_yticks(y_d)
        ax_d.set_yticklabels(display.index, fontsize=7.5)
        ax_d.set_xlabel("AUC (enzyme support predicts BPR rank for this compound)")
        ax_d.set_title(f"{LABELS[cond]}\nTop {n_show//2} and bottom {n_show//2} compounds")
        ax_d.set_xlim(0, 1)

        print(f"\n  [{cond}] Top 15 compounds with best enzyme concordance:")
        for met, auc in met_auc_s.head(15).items():
            print(f"    {met:<45} AUC={auc:.3f}")
        print(f"  [{cond}] Bottom 15 compounds (worst enzyme concordance):")
        for met, auc in met_auc_s.tail(15).items():
            print(f"    {met:<45} AUC={auc:.3f}")

        out = os.path.join(OUT_DIR, f"analysis11d_compound_auc_{cond}.tsv")
        met_auc_s.rename("auc").to_frame().to_csv(out, sep="\t")

    save(fig_d, "analysis11d_compound_auc.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    data = {
        cond: {mode: load(cond, mode) for mode in MODES}
        for cond in ("with", "without")
    }
    print("Done.\n")

    analysis_summary(data)
    analysis_entropy(data)
    analysis_universality(data)
    analysis_diff_rank(data)
    analysis_shared_isolate_correlation(data)
    analysis_rank_shift(data)
    analysis_rank_change_distribution(data)
    analysis_gut_vs_aerobic(data)
    analysis_phantom_metabolites(data)
    analysis_gut_centroid_drift(data)
    analysis_biological_plausibility(data)
    analysis_enzyme_bpr_correlation(data)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
