# revision_validation_12_isolates.py
"""Per-isolate validation table for the 12 held-out NLDM isolates (mSystems revision).

Re-evaluates the submitted Fig. S3, which was produced by
figure_8_experimental_validation_single_model() in
make_figures_for_manuscript_nature_microbiology_first_submission.py from
exometabolite_data/validation/held-out-validation-data.tsv. The validation
loader and the taxon-mean score aggregation are imported from that script so the
inputs are identical; the existing figure scripts and figure files are not
modified.

Fig. S3 trained a fresh, unseeded Ordinal-BPR model at run time and never saved
it, so that exact model cannot be reloaded. This script evaluates:

  saved     trained_models/ordinal_bpr_consumed (32-dim, same training data
            and hyper-parameters as the Fig. S3 run)
  retrain   N seeded retrains on the same training data (run-to-run spread)
  clean     N seeded retrains with every validation isolate that is present in
            the training data removed (true hold-out)

Labels:
  corrected  consumed: value <= -2; neutral: -2 < value < 2; secreted (>= 2) excluded
             (these are also the labels Fig. S3 used)
  legacy     consumed: value < -2; neutral: everything else, including secreted
             (labels of the older genus-level precision@k figure)

Scores are taxon-mean scores at genus level, falling back to family and then
order only when the genus (or family) has no training isolates.

Usage:
    python scripts/revision_validation_12_isolates.py
    python scripts/revision_validation_12_isolates.py --n-seeds 5
"""

import sys
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from core.io import load_model_artifacts
from core.training import train_model
from core.types import AssertionType
from loaders.loaders import prepare_training_data, load_exometabolite
from loaders.compound_mapper import map_compound_names_batch
from scripts.make_figures_for_manuscript_nature_microbiology_first_submission import (
    load_experimental_validation_data,
    compute_genus_predictions_any_model,
)

logger = logging.getLogger(__name__)

# Hyper-parameters used by Fig. S3 (figure_8_experimental_validation_all_models)
# and by train_all_models.py for trained_models/ordinal_bpr_consumed.
ORDINAL_BPR_KWARGS = {
    "n_epochs": 1000,
    "batch_size": 256,
    "learning_rate": 0.01,
    "hard_neg_ratio": 0.5,
    "lambda_ord": 0.1,
}
EMBEDDING_DIM = 32
TAX_LEVELS = ("genus", "family", "order")


# =============================================================================
# Labels
# =============================================================================


def corrected_labels(values: pd.Series) -> pd.Series:
    """1 = consumed (<= -2), 0 = neutral (-2 < v < 2), NaN = secreted (>= 2, excluded)."""
    labels = pd.Series(np.nan, index=values.index)
    labels[values <= -2] = 1
    labels[(values > -2) & (values < 2)] = 0
    return labels


def legacy_labels(values: pd.Series) -> pd.Series:
    """1 = consumed (< -2), 0 = everything else, including secreted."""
    return (values < -2).astype(float)


def score_metrics(labels: pd.Series, scores: pd.Series) -> dict:
    """ROC-AUC, average precision, and prevalence over rows with a non-NaN label."""
    keep = labels.notna()
    y = labels[keep].astype(int).values
    s = scores[keep].values
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        roc, pr = float("nan"), float("nan")
    else:
        roc = float(roc_auc_score(y, s))
        pr = float(average_precision_score(y, s))
    prevalence = n_pos / len(y) if len(y) else float("nan")
    return {"roc_auc": roc, "pr_auc": pr, "prevalence": prevalence,
            "n_pos": n_pos, "n_neg": n_neg}


# =============================================================================
# Data
# =============================================================================


def load_validation_with_mapping() -> pd.DataFrame:
    """Load held-out measurements (Fig. S3 loader) and add representative compound names."""
    exp = load_experimental_validation_data()
    if exp is None:
        raise FileNotFoundError("held-out-validation-data.tsv not found")
    mapping = map_compound_names_batch(exp["metabolite"].unique())
    exp = exp.copy()
    exp["mapped_metabolite"] = exp["metabolite"].apply(
        lambda x: mapping[x].representative_name if x in mapping else None
    )
    return exp[exp["media"] == "nldm"]


def find_validation_isolates_in_training(exp: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Return {isolate_id: [source files]} for validation isolates present in the training table."""
    ids = set(exp["isolate"].unique())
    hits = df[df["organism_id"].astype(str).str.strip().isin(ids)]
    return {
        iso: sorted(grp["source_file"].unique())
        for iso, grp in hits.groupby(hits["organism_id"].astype(str).str.strip())
    }


# =============================================================================
# Evaluation
# =============================================================================


def merge_with_taxon_predictions(exp: pd.DataFrame, predictions: dict) -> tuple[pd.DataFrame, dict]:
    """Attach taxon-mean scores to each isolate using genus -> family -> order fallback.

    Mirrors the fallback in figure_8_experimental_validation_single_model(): the
    first level whose taxon has any training isolates is used.

    Returns (merged rows with mean_score, {isolate: reference metadata}).
    """
    merged_parts = []
    reference = {}
    for isolate, iso_df in exp.groupby("isolate"):
        for level in TAX_LEVELS:
            key = iso_df[level].iloc[0]
            if pd.isna(key):
                continue
            pred = predictions[level]
            level_rows = pred[pred[level] == key]
            if len(level_rows) == 0:
                continue
            iso_merged = pd.merge(
                iso_df,
                level_rows[["metabolite", "mean_score"]],
                left_on="mapped_metabolite",
                right_on="metabolite",
                how="left",
                suffixes=("_exp", ""),
            )
            reference[isolate] = {
                "reference_rank_used": level,
                "reference_taxon": key,
                "n_source_training_isolates": int(level_rows["n_isolates"].iloc[0]),
            }
            merged_parts.append(iso_merged[pd.notna(iso_merged["mean_score"])])
            break
        else:
            logger.warning(f"  {isolate}: no training isolates at genus, family or order level")
            reference[isolate] = {
                "reference_rank_used": None,
                "reference_taxon": None,
                "n_source_training_isolates": 0,
            }
    merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()
    return merged, reference


def evaluate_model(model, training_data, exp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-isolate counts and metrics under corrected and legacy labels.

    Returns (per-isolate DataFrame, merged rows used for scoring).
    """
    predictions = compute_genus_predictions_any_model(model, training_data)
    merged, reference = merge_with_taxon_predictions(exp, predictions)

    rows = []
    for isolate, iso_exp in exp.groupby("isolate"):
        sub = merged[merged["isolate"] == isolate] if len(merged) else iso_exp.iloc[0:0]
        values = sub["value"]
        corrected = score_metrics(corrected_labels(values), sub["mean_score"])
        legacy = score_metrics(legacy_labels(values), sub["mean_score"])
        rows.append({
            "isolate_id": isolate,
            **reference[isolate],
            "n_measured": int(iso_exp["value"].notna().sum()),
            "n_mapped_to_model": int(len(sub)),
            "n_consumed": corrected["n_pos"],
            "n_neutral": corrected["n_neg"],
            "n_secreted_excluded": int((values >= 2).sum()),
            "roc_auc": corrected["roc_auc"],
            "pr_auc": corrected["pr_auc"],
            "prevalence": corrected["prevalence"],
            "legacy_n_consumed": legacy["n_pos"],
            "legacy_n_neutral": legacy["n_neg"],
            "legacy_roc_auc": legacy["roc_auc"],
            "legacy_pr_auc": legacy["pr_auc"],
            "legacy_prevalence": legacy["prevalence"],
        })
    return pd.DataFrame(rows), merged


def train_seeded(training_data, seed: int):
    """Train Ordinal-BPR with the Fig. S3 hyper-parameters and a fixed seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    artifacts = train_model(
        training_data=training_data,
        model_type="ordinal_bpr",
        embedding_dim=EMBEDDING_DIM,
        verbose=False,
        **ORDINAL_BPR_KWARGS,
    )
    return artifacts.model


def run_seeds(training_data, exp: pd.DataFrame, seeds: list[int], label: str) -> pd.DataFrame:
    """Evaluate one retrained model per seed; returns long-format per-isolate results."""
    results = []
    for seed in seeds:
        model = train_seeded(training_data, seed)
        per_iso, _ = evaluate_model(model, training_data, exp)
        per_iso.insert(0, "seed", seed)
        per_iso.insert(0, "model_variant", label)
        results.append(per_iso)
        logger.info(f"  {label} seed {seed}: mean ROC-AUC = {per_iso['roc_auc'].mean():.3f}")
    return pd.concat(results, ignore_index=True)


# =============================================================================
# Summaries
# =============================================================================


def summarize_auc(aucs: pd.Series) -> dict:
    aucs = aucs.dropna()
    return {
        "n_isolates": int(len(aucs)),
        "mean_roc_auc": float(aucs.mean()),
        "median_roc_auc": float(aucs.median()),
        "n_roc_auc_ge_0.8": int((aucs >= 0.8).sum()),
        "min_roc_auc": float(aucs.min()),
    }


def build_summary(table: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean/median ROC-AUC per model variant, for all isolates and for true hold-outs only."""
    subsets = {
        "all_12": table["isolate_id"],
        "excluding_in_training": table.loc[~table["in_training_data"], "isolate_id"],
    }
    rows = []
    single_runs = {
        "fig_s3_submitted_run": "fig_s3_submitted_roc_auc",
        "saved_model_corrected_labels": "roc_auc",
        "saved_model_legacy_labels": "legacy_roc_auc",
    }
    for variant, col in single_runs.items():
        for subset, ids in subsets.items():
            vals = table.set_index("isolate_id").loc[ids, col]
            rows.append({"model_variant": variant, "subset": subset, "n_seeds": 1,
                         **summarize_auc(vals)})

    for variant, grp in per_seed.groupby("model_variant"):
        for subset, ids in subsets.items():
            per_run = [summarize_auc(g.set_index("isolate_id").loc[ids, "roc_auc"])
                       for _, g in grp.groupby("seed")]
            per_run = pd.DataFrame(per_run)
            rows.append({
                "model_variant": f"{variant}_corrected_labels",
                "subset": subset,
                "n_seeds": len(per_run),
                "n_isolates": int(per_run["n_isolates"].iloc[0]),
                "mean_roc_auc": float(per_run["mean_roc_auc"].mean()),
                "median_roc_auc": float(per_run["median_roc_auc"].mean()),
                "n_roc_auc_ge_0.8": float(per_run["n_roc_auc_ge_0.8"].mean()),
                "min_roc_auc": float(per_run["min_roc_auc"].mean()),
                "mean_roc_auc_sd_across_seeds": float(per_run["mean_roc_auc"].std(ddof=1)),
                "mean_roc_auc_min_across_seeds": float(per_run["mean_roc_auc"].min()),
                "mean_roc_auc_max_across_seeds": float(per_run["mean_roc_auc"].max()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main(model_dir: Path, output_dir: Path, submitted_auc_csv: Path, n_seeds: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = load_validation_with_mapping()
    logger.info(f"Validation data: {exp['isolate'].nunique()} isolates, {len(exp)} measurements")

    df = load_exometabolite()
    in_training = find_validation_isolates_in_training(exp, df)
    for iso, sources in in_training.items():
        logger.warning(f"  {iso} is present in the training data ({', '.join(sources)})")

    # Saved final model
    artifacts = load_model_artifacts(model_dir)
    logger.info(f"Loaded {type(artifacts.model).__name__} ({artifacts.model.embedding_dim}-dim) "
                f"from {model_dir}")
    table, _ = evaluate_model(artifacts.model, artifacts.training_data, exp)

    taxonomy = exp.groupby("isolate")[["species_label", "genus", "family", "order"]].first()
    table = table.join(taxonomy, on="isolate_id")
    table["in_training_data"] = table["isolate_id"].isin(in_training)
    table["training_source_files"] = table["isolate_id"].map(
        lambda x: ";".join(in_training.get(x, [])))

    if submitted_auc_csv.exists():
        submitted = pd.read_csv(submitted_auc_csv).set_index("isolate")["auc"]
        table["fig_s3_submitted_roc_auc"] = table["isolate_id"].map(submitted)
    else:
        logger.warning(f"Submitted per-isolate AUC file not found: {submitted_auc_csv}")
        table["fig_s3_submitted_roc_auc"] = np.nan

    # Seeded retrains: same training data, and with in-training validation isolates removed
    seeds = list(range(n_seeds))
    training_full = prepare_training_data(df, preference_type=AssertionType.CONSUMED,
                                          include_secreted=False)
    df_clean = df[~df["organism_id"].astype(str).str.strip().isin(in_training)]
    training_clean = prepare_training_data(df_clean, preference_type=AssertionType.CONSUMED,
                                           include_secreted=False)
    logger.info(f"Retraining {n_seeds} seeds on full data ({training_full.n_microbes} microbes) "
                f"and clean data ({training_clean.n_microbes} microbes)")
    per_seed = pd.concat([
        run_seeds(training_full, exp, seeds, "retrain"),
        run_seeds(training_clean, exp, seeds, "clean_holdout"),
    ], ignore_index=True)

    for variant in ("retrain", "clean_holdout"):
        grp = per_seed[per_seed["model_variant"] == variant].groupby("isolate_id")
        table[f"{variant}_roc_auc_mean"] = table["isolate_id"].map(grp["roc_auc"].mean())
        table[f"{variant}_roc_auc_sd"] = table["isolate_id"].map(grp["roc_auc"].std(ddof=1))
        table[f"{variant}_pr_auc_mean"] = table["isolate_id"].map(grp["pr_auc"].mean())
    clean_ref = per_seed[per_seed["model_variant"] == "clean_holdout"].groupby("isolate_id").first()
    table["clean_holdout_reference_rank_used"] = table["isolate_id"].map(clean_ref["reference_rank_used"])
    table["clean_holdout_n_source_training_isolates"] = table["isolate_id"].map(
        clean_ref["n_source_training_isolates"])

    table["delta_roc_auc_corrected_minus_legacy"] = table["roc_auc"] - table["legacy_roc_auc"]
    table["delta_roc_auc_saved_minus_submitted"] = table["roc_auc"] - table["fig_s3_submitted_roc_auc"]

    columns = [
        "isolate_id", "species_label", "genus", "family", "order",
        "in_training_data", "training_source_files",
        "reference_rank_used", "reference_taxon", "n_source_training_isolates",
        "n_measured", "n_mapped_to_model",
        "n_consumed", "n_neutral", "n_secreted_excluded",
        "roc_auc", "pr_auc", "prevalence",
        "legacy_n_consumed", "legacy_n_neutral", "legacy_roc_auc", "legacy_pr_auc",
        "legacy_prevalence",
        "fig_s3_submitted_roc_auc",
        "retrain_roc_auc_mean", "retrain_roc_auc_sd", "retrain_pr_auc_mean",
        "clean_holdout_reference_rank_used", "clean_holdout_n_source_training_isolates",
        "clean_holdout_roc_auc_mean", "clean_holdout_roc_auc_sd", "clean_holdout_pr_auc_mean",
        "delta_roc_auc_corrected_minus_legacy", "delta_roc_auc_saved_minus_submitted",
    ]
    table = table[columns].sort_values(["genus", "isolate_id"]).reset_index(drop=True)
    table.to_csv(output_dir / "table_S_validation_12_isolates.csv", index=False,
                 float_format="%.4f")
    per_seed.to_csv(output_dir / "validation_12_isolates_per_seed.csv", index=False,
                    float_format="%.4f")

    summary = build_summary(table, per_seed)
    summary.to_csv(output_dir / "validation_12_isolates_summary.csv", index=False,
                   float_format="%.4f")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        logger.info("\n" + table[["isolate_id", "genus", "reference_rank_used",
                                  "n_source_training_isolates", "in_training_data",
                                  "fig_s3_submitted_roc_auc", "roc_auc", "legacy_roc_auc",
                                  "retrain_roc_auc_mean", "clean_holdout_roc_auc_mean"]]
                    .to_string(index=False, float_format="%.3f"))
        logger.info("\n" + summary.to_string(index=False, float_format="%.3f"))
    logger.info(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Per-isolate validation table for the 12 held-out NLDM isolates"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=WOM_DIR / "trained_models" / "ordinal_bpr_consumed",
        help="Saved final Ordinal-BPR model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WOM_DIR / "manuscript_figures" / "msystems_revision",
        help="Output directory",
    )
    parser.add_argument(
        "--submitted-auc-csv",
        type=Path,
        default=WOM_DIR / "manuscript_figures_nat_microbiology_first_submission"
        / "figure8_auc_per_isolate.csv",
        help="Per-isolate AUCs written by the submitted Fig. S3 run (read only)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of seeded retrains per training set (default: 20)",
    )
    args = parser.parse_args()

    main(args.model_dir, args.output_dir, args.submitted_auc_csv, args.n_seeds)
