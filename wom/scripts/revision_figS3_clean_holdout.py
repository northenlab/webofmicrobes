# revision_figS3_clean_holdout.py
"""Regenerate Fig. S3 from the clean hold-out Ordinal-BPR models (mSystems revision).

Same layout as the submitted figure
(manuscript_figures_nat_microbiology_first_submission/figure8_experimental_validation_ordinal_bpr.png):
one panel per held-out isolate, histograms of reference-level mean Ordinal-BPR
scores for consumed (<= -2) vs neutral (-2 < v < 2) metabolites, secreted (>= 2)
excluded. Models are trained with every validation isolate that appears in the
training data (GW456-R20, FW507-4G11) removed.

  Histograms   seed-0 clean model
  Annotation   ROC-AUC mean ± SD over the 20 clean seeds; checked against
               clean_holdout_roc_auc_mean / _sd in table_S_validation_12_isolates.csv
  Title        genus, isolate ID, (number of reference training isolates, rank)

Evaluation helpers come from revision_validation_12_isolates.py.

Usage:
    python scripts/revision_figS3_clean_holdout.py
"""

import sys
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.types import AssertionType
from loaders.loaders import prepare_training_data, load_exometabolite
from scripts.revision_validation_12_isolates import (
    load_validation_with_mapping,
    find_validation_isolates_in_training,
    evaluate_model,
    train_seeded,
    corrected_labels,
)

logger = logging.getLogger(__name__)

N_SEEDS = 20           # matches the clean_holdout columns of table_S_validation_12_isolates.csv
HISTOGRAM_SEED = 0
AUC_FORMAT = "{:.4f}"  # same precision as table_S_validation_12_isolates.csv
CONSUMED_COLOR = "#1f77b4"  # same colours as the submitted figure
NEUTRAL_COLOR = "#ff7f0e"


def clean_holdout_results(exp: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Train the clean hold-out seeds.

    Returns (per-seed per-isolate metrics, seed-0 merged score rows, removed isolate IDs).
    """
    in_training = find_validation_isolates_in_training(exp, df)
    df_clean = df[~df["organism_id"].astype(str).str.strip().isin(in_training)]
    training_clean = prepare_training_data(df_clean, preference_type=AssertionType.CONSUMED,
                                           include_secreted=False)
    logger.info(f"Removed from training: {sorted(in_training)}; "
                f"{training_clean.n_microbes} microbes remain")

    per_seed, histogram_rows = [], None
    for seed in range(N_SEEDS):
        model = train_seeded(training_clean, seed)
        per_iso, merged = evaluate_model(model, training_clean, exp)
        per_iso.insert(0, "seed", seed)
        per_seed.append(per_iso)
        if seed == HISTOGRAM_SEED:
            histogram_rows = merged
        logger.info(f"  clean seed {seed}: mean ROC-AUC = {per_iso['roc_auc'].mean():.3f}")
    return pd.concat(per_seed, ignore_index=True), histogram_rows, sorted(in_training)


def summarize_and_check(per_seed: pd.DataFrame, table_csv: Path) -> pd.DataFrame:
    """Per-isolate panel metadata and ROC-AUC mean/SD, verified against the S-table."""
    grp = per_seed.groupby("isolate_id")
    panels = grp[["reference_rank_used", "reference_taxon", "n_source_training_isolates",
                  "n_consumed", "n_neutral"]].first()
    panels["roc_auc_mean"] = grp["roc_auc"].mean()
    panels["roc_auc_sd"] = grp["roc_auc"].std(ddof=1)

    table = pd.read_csv(table_csv).set_index("isolate_id")
    checks = pd.DataFrame({
        "panel_mean": panels["roc_auc_mean"],
        "table_mean": table["clean_holdout_roc_auc_mean"],
        "panel_sd": panels["roc_auc_sd"],
        "table_sd": table["clean_holdout_roc_auc_sd"],
        "panel_n_ref": panels["n_source_training_isolates"],
        "table_n_ref": table["clean_holdout_n_source_training_isolates"],
        "panel_rank": panels["reference_rank_used"],
        "table_rank": table["clean_holdout_reference_rank_used"],
    })
    # Panels print AUC_FORMAT, the table's own precision; rounding to fewer decimals
    # would double-round values sitting on a boundary (e.g. SD 0.017492 vs table 0.0175).
    mismatched = (
        (checks["panel_mean"].map(AUC_FORMAT.format) != checks["table_mean"].map(AUC_FORMAT.format))
        | (checks["panel_sd"].map(AUC_FORMAT.format) != checks["table_sd"].map(AUC_FORMAT.format))
        | (checks["panel_n_ref"] != checks["table_n_ref"])
        | (checks["panel_rank"] != checks["table_rank"])
    )
    if mismatched.any():
        raise ValueError(f"Panel values do not match {table_csv.name}:\n{checks[mismatched]}")
    logger.info(f"All {len(checks)} panel ROC-AUC mean/SD values, reference counts and ranks "
                f"match {table_csv.name}")
    return panels


def plot_figure(histogram_rows: pd.DataFrame, panels: pd.DataFrame, exp: pd.DataFrame,
                output_dir: Path):
    genus_of = exp.groupby("isolate")["genus"].first()
    isolates = sorted(panels.index, key=lambda x: (genus_of[x], x))

    scores = histogram_rows["mean_score"]
    lo = min(-5.0, np.floor(scores.min()))
    hi = max(15.0, np.ceil(scores.max()))
    edges = np.linspace(-5, 15, 25) if (lo, hi) == (-5.0, 15.0) else np.arange(lo, hi + 0.8, 0.8)

    ncols = 4
    nrows = (len(isolates) + ncols - 1) // ncols
    fig, axes = plt.subplots(figsize=(ncols * 5.2, nrows * 4.6), ncols=ncols, nrows=nrows,
                             sharex=True)
    axes = axes.flatten()

    for ax, isolate in zip(axes, isolates):
        sub = histogram_rows[histogram_rows["isolate"] == isolate]
        labels = corrected_labels(sub["value"])
        consumed = sub.loc[labels == 1, "mean_score"]
        neutral = sub.loc[labels == 0, "mean_score"]

        for vals, color, label in [(consumed, CONSUMED_COLOR, "Consumed (≤ −2)"),
                                   (neutral, NEUTRAL_COLOR, "Neutral (−2 < v < 2)")]:
            ax.hist(vals, bins=edges, alpha=0.55, color=color, label=label)
            # Outline keeps each distribution's shape readable where the fills overlap
            ax.hist(vals, bins=edges, histtype="step", linewidth=1.6, color=color)

        meta = panels.loc[isolate]
        genus = genus_of[isolate]
        ax.set_title(f"$\\mathit{{{genus}}}$ {isolate} "
                     f"({int(meta['n_source_training_isolates'])}, {meta['reference_rank_used']})",
                     fontsize=12.5)
        annotation = (f"ROC-AUC = {AUC_FORMAT.format(meta['roc_auc_mean'])} "
                      f"± {AUC_FORMAT.format(meta['roc_auc_sd'])}")
        if int(meta["n_neutral"]) == 1:
            annotation += "\n1 neutral compound"
        ax.text(0.04, 0.97, annotation, transform=ax.transAxes, fontsize=14,
                ha="left", va="top")

        # Headroom so the annotation sits above the tallest bar
        max_count = max(np.histogram(consumed, bins=edges)[0].max(),
                        np.histogram(neutral, bins=edges)[0].max())
        ax.set_ylim(0, max_count * (1.55 if "\n" in annotation else 1.35))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax.set_xlim(edges[0], edges[-1])
        ax.set_xticks(np.arange(-5, edges[-1] + 1, 5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(2.5))
        ax.tick_params(axis="both", labelsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(isolates):]:
        ax.remove()

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=c, alpha=0.7)
               for c in (CONSUMED_COLOR, NEUTRAL_COLOR)]
    fig.legend(handles, ["Consumed (≤ −2)", "Neutral (−2 < v < 2)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=18, title="experimental observation (secreted, ≥ 2, excluded)",
               title_fontsize=18)
    fig.text(0.5, 0.905,
             f"Models trained without any validation isolate. Histograms: seed-{HISTOGRAM_SEED} model; "
             f"ROC-AUC: mean ± SD over {N_SEEDS} seeds. Title: (reference training isolates, rank)",
             ha="center", va="center", fontsize=14)
    fig.text(0.5, 0.01, "Mean Ordinal-BPR score (at reference taxonomic level)",
             ha="center", fontsize=20)
    fig.text(0.01, 0.5, "Number of metabolites", va="center", rotation="vertical", fontsize=20)

    plt.tight_layout(rect=[0.03, 0.04, 1.0, 0.88])
    for ext in ["png", "pdf"]:
        path = output_dir / f"figS3_validation_clean_holdout.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def main(output_dir: Path, table_csv: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    exp = load_validation_with_mapping()
    df = load_exometabolite()

    per_seed, histogram_rows, removed = clean_holdout_results(exp, df)
    panels = summarize_and_check(per_seed, table_csv)
    with pd.option_context("display.width", 200):
        logger.info("\n" + panels.to_string(float_format="%.3f"))
    plot_figure(histogram_rows, panels, exp, output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Regenerate Fig. S3 from the clean hold-out Ordinal-BPR models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WOM_DIR / "manuscript_figures" / "msystems_revision",
        help="Output directory",
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        default=WOM_DIR / "manuscript_figures" / "msystems_revision"
        / "table_S_validation_12_isolates.csv",
        help="Per-isolate table whose clean_holdout columns the panels must match",
    )
    args = parser.parse_args()

    main(args.output_dir, args.table_csv)
