# revision_fig1b_benchmark.py
"""Corrected Fig. 1b benchmark (mSystems revision, Reviewer 3 major comment 1).

The submitted Fig. 1b came from run_figure1_oos_benchmark() in
make_figures_for_manuscript_nature_microbiology_first_submission.py (cached as
figure1_core_results_consumed_4models_ordinal.csv). That benchmark has two
problems:

  2a  The per-repeat TrainingData is rebuilt without known_negatives, so the
      "Ordinal-BPR" model received an empty known-negative set and trained as
      plain BPR (no hard negatives, no ordinal term).
  2b  compute_loo_auc() ranks the withheld consumed compound against every
      compound that is not a training positive (neutral + secreted + untested),
      not against measured-neutral compounds as the Results state.

This script:
  * re-runs the submitted code unmodified (imported) and records the
    known_negatives that reach OrdinalBPRModel.fit()
  * shows that, on the submitted split, Ordinal-BPR is bit-identical with the
    ordinal modification switched off (hard_neg_ratio=0, lambda_ord=0)
  * Protocol A: leave-one-out as in the paper, known_negatives passed through,
    AUC against three negative pools (neutral / untested / mixed)
  * Protocol B: strict hold-out of 20% of each microbe's consumed and neutral
    pairs, evaluated on the withheld pairs only
  * Ordinal-BPR component ablations (lambda_ord=0, hard_neg_ratio=0) at dim 32

Pair labels (from the per-row assertion, which is consumed <= -2, neutral
-2 < v < 2, secreted >= 2): consumed if any row is consumed (the training
positives), secreted if any row is secreted and none consumed, neutral if all
rows are neutral, untested if never measured. Secreted is never a negative.

Usage:
    python scripts/revision_fig1b_benchmark.py
    python scripts/revision_fig1b_benchmark.py --dims 4 32 --repeats 2
"""

import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

from core.models import OrdinalBPRModel
from core.training import train_model
from core.types import AssertionType, TrainingData
from loaders.loaders import prepare_training_data, load_exometabolite
import scripts.make_figures_for_manuscript_nature_microbiology_first_submission as submitted

logger = logging.getLogger(__name__)

# Hyper-parameters of the submitted run (run_figure1_oos_benchmark); plain BPR as in
# the 6-model benchmark of make_figures_for_manuscript_ordinal.py.
MODEL_TYPES = {
    "Ordinal-BPR": "ordinal_bpr",
    "BPR": "bpr",
    "WR-MF": "wrmf",
    "SVD": "svd",
    "LMF": "lmf",
}
MODEL_KWARGS = {
    "ordinal_bpr": {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01,
                    "hard_neg_ratio": 0.5, "lambda_ord": 0.1},
    "bpr": {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01},
    "wrmf": {"iterations": 50, "regularization": 0.1},
    "svd": {},
    "lmf": {"iterations": 30, "learning_rate": 1.0, "regularization": 0.1},
}
# Submitted Fig. 1b colours; plain BPR added in a colour that passes CVD checks
# against them, with a dashed line and per-model markers as secondary encoding.
MODEL_STYLES = {
    "Ordinal-BPR": {"color": "#e377c2", "marker": "*", "linestyle": "-"},
    "BPR": {"color": "#bcbd22", "marker": "s", "linestyle": "--"},
    "WR-MF": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "SVD": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
    "LMF": {"color": "#8c564b", "marker": "P", "linestyle": "-"},
}
# Ordinal-BPR component ablations, run at ABLATION_DIM only (not plotted). The ordinal
# pairs are only formed in the hard-negative branch, so hard_neg_ratio=0 disables both;
# soft negatives still exclude known negatives.
ABLATIONS = {
    "Ordinal-BPR [lambda_ord=0]": {"lambda_ord": 0.0},
    "Ordinal-BPR [hard_neg_ratio=0]": {"hard_neg_ratio": 0.0},
}
ABLATION_DIM = 32
HOLDOUT_FRACTION = 0.2
LABELS = ("consumed", "neutral", "secreted", "untested")
RESULT_COLUMNS = ["model", "dim", "repeat", "protocol", "pool", "roc_auc", "roc_auc_pooled",
                  "pr_auc_pooled", "prevalence", "n_microbes", "n_pairs", "n_pos", "n_neg",
                  "known_negatives_passed", "mean_score_withheld", "mean_score_neutral",
                  "mean_score_untested"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_seed(repeat: int, dim: int) -> int:
    """Shared by all models at a given repeat and dimension, so models are paired."""
    return 10_000 * repeat + dim


def n_known_negatives(known_negatives: dict) -> int:
    return int(sum(len(v) for v in known_negatives.values()))


# =============================================================================
# Data
# =============================================================================


def build_label_matrix(df: pd.DataFrame, training_data: TrainingData) -> np.ndarray:
    """(n_microbes, n_metabolites) array of label codes, indexed like training_data."""
    labels = np.full((training_data.n_microbes, training_data.n_metabolites),
                     LABELS.index("untested"), dtype=np.int8)
    assertions = df.groupby(["organism_id", "representative_compound"])["assertion"].agg(set)
    for (org, compound), values in assertions.items():
        if "consumed" in values:
            label = "consumed"
        elif "secreted" in values:
            label = "secreted"
        else:
            label = "neutral"
        labels[training_data.microbe_to_idx[org], training_data.metabolite_to_idx[compound]] = \
            LABELS.index(label)

    consumed = labels == LABELS.index("consumed")
    positives_matrix = np.zeros_like(consumed)
    for m, items in training_data.positives.items():
        positives_matrix[m, list(items)] = True
    assert np.array_equal(consumed, positives_matrix), "consumed labels must equal training positives"
    return labels


def split_training_data(training_data: TrainingData, positives: dict,
                        known_negatives: dict) -> TrainingData:
    """Copy of training_data with the given positives and known_negatives."""
    return TrainingData(
        n_microbes=training_data.n_microbes,
        n_metabolites=training_data.n_metabolites,
        microbe_to_idx=training_data.microbe_to_idx,
        metabolite_to_idx=training_data.metabolite_to_idx,
        idx_to_microbe=training_data.idx_to_microbe,
        idx_to_metabolite=training_data.idx_to_metabolite,
        positives=positives,
        known_negatives=known_negatives,
        microbe_taxonomy=training_data.microbe_taxonomy,
        preference_type=training_data.preference_type,
    )


def n_withheld(n: int) -> int:
    """20% (rounded half up), at least 1, and only if at least one pair stays in training."""
    if n < 2:
        return 0
    return max(1, int(np.floor(HOLDOUT_FRACTION * n + 0.5)))


def holdout_split(training_data: TrainingData, labels: np.ndarray, seed: int):
    """Protocol B split: withhold consumed and neutral pairs per microbe.

    Returns (train positives, train known_negatives, withheld (microbe, compound) array).
    """
    rng = np.random.default_rng(seed)
    neutral_code = LABELS.index("neutral")
    train_pos, train_kn, withheld = {}, {}, []
    for m in range(training_data.n_microbes):
        consumed = sorted(training_data.positives.get(m, set()))
        neutral = np.flatnonzero(labels[m] == neutral_code).tolist()
        held_c = set(rng.choice(consumed, n_withheld(len(consumed)), replace=False).tolist()) \
            if n_withheld(len(consumed)) else set()
        held_n = set(rng.choice(neutral, n_withheld(len(neutral)), replace=False).tolist()) \
            if n_withheld(len(neutral)) else set()
        if consumed:
            train_pos[m] = set(consumed) - held_c
        kn = training_data.known_negatives.get(m, set()) - held_n
        if kn:
            train_kn[m] = kn
        withheld.extend((m, c) for c in sorted(held_c | held_n))
    return train_pos, train_kn, np.array(withheld, dtype=int)


# =============================================================================
# Models and metrics
# =============================================================================


def train(training_data: TrainingData, model_name: str, dim: int, seed: int, **overrides):
    seed_everything(seed)
    if model_name in ABLATIONS:
        overrides = {**ABLATIONS[model_name], **overrides}
        model_name = "Ordinal-BPR"
    model_type = MODEL_TYPES[model_name]
    kwargs = {**MODEL_KWARGS[model_type], **overrides}
    artifacts = train_model(training_data=training_data, model_type=model_type,
                            embedding_dim=dim, verbose=False, **kwargs)
    return artifacts.model


def score_matrix(model) -> np.ndarray:
    return model.microbe_embeddings @ model.metabolite_embeddings.T


def pair_auc(pos_score: float, neg_scores: np.ndarray) -> float:
    """Same statistic as compute_loo_auc: P(neg < pos) + 0.5 P(tie)."""
    return float(((neg_scores < pos_score).sum() + 0.5 * (neg_scores == pos_score).sum())
                 / len(neg_scores))


def evaluate_loo(scores: np.ndarray, test_pairs: list, train_pos: dict,
                 labels: np.ndarray) -> dict:
    """Protocol A: per-pair AUC of each withheld consumed compound against three pools.

    Also returns, per pool row, the mean (over microbes) score of the withheld compound
    and of the microbe's neutral and untested compounds.
    """
    pools = {"neutral": [], "untested": [], "mixed": []}
    n_neg = {k: 0 for k in pools}
    tier_means = []
    all_items = np.arange(scores.shape[1])
    for m, test_item in test_pairs:
        row = scores[m]
        candidates = {
            "neutral": np.flatnonzero(labels[m] == LABELS.index("neutral")),
            "untested": np.flatnonzero(labels[m] == LABELS.index("untested")),
            "mixed": np.setdiff1d(all_items, list(train_pos.get(m, set())) + [test_item]),
        }
        for pool, idx in candidates.items():
            if len(idx):
                pools[pool].append(pair_auc(row[test_item], row[idx]))
                n_neg[pool] += len(idx)
        tier_means.append([row[test_item],
                           row[candidates["neutral"]].mean() if len(candidates["neutral"]) else np.nan,
                           row[candidates["untested"]].mean() if len(candidates["untested"]) else np.nan])
    withheld_mean, neutral_mean, untested_mean = np.nanmean(np.array(tier_means), axis=0)
    return {
        pool: {"roc_auc": float(np.mean(aucs)) if aucs else np.nan,
               "n_microbes": len(aucs), "n_pos": len(aucs), "n_neg": n_neg[pool],
               "mean_score_withheld": float(withheld_mean),
               "mean_score_neutral": float(neutral_mean),
               "mean_score_untested": float(untested_mean)}
        for pool, aucs in pools.items()
    }


def evaluate_holdout(scores: np.ndarray, withheld: np.ndarray, labels: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Protocol B metrics over withheld consumed (1) vs withheld neutral (0) pairs."""
    m_idx, c_idx = withheld[:, 0], withheld[:, 1]
    y = (labels[m_idx, c_idx] == LABELS.index("consumed")).astype(int)
    s = scores[m_idx, c_idx]
    per_microbe = []
    for m in np.unique(m_idx):
        sel = m_idx == m
        if 0 < y[sel].sum() < sel.sum():
            per_microbe.append(roc_auc_score(y[sel], s[sel]))
    metrics = {
        "roc_auc": float(np.mean(per_microbe)),
        "roc_auc_pooled": float(roc_auc_score(y, s)),
        "pr_auc_pooled": float(average_precision_score(y, s)),
        "prevalence": float(y.mean()),
        "n_microbes": len(per_microbe),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }
    return metrics, y, s


# =============================================================================
# Submitted code: re-run and bug checks
# =============================================================================


def rerun_submitted_code(dims: list[int], repeats: int) -> pd.DataFrame:
    """Run the submitted run_figure1_oos_benchmark() unmodified, recording the
    known_negatives that reach OrdinalBPRModel.fit()."""
    recorded = []
    original_fit = OrdinalBPRModel.fit

    def recording_fit(self, *args, known_negatives=None, **kwargs):
        recorded.append(n_known_negatives(known_negatives or {}))
        return original_fit(self, *args, known_negatives=known_negatives, **kwargs)

    OrdinalBPRModel.fit = recording_fit
    try:
        seed_everything(0)  # the submitted code sets no global seed
        rows = submitted.run_figure1_oos_benchmark(dims, repeats, AssertionType.CONSUMED)
    finally:
        OrdinalBPRModel.fit = original_fit

    ordinal_rows = rows["method"] == "Ordinal-BPR"
    assert ordinal_rows.sum() == len(recorded)
    rows["known_negatives_passed"] = np.nan
    rows.loc[ordinal_rows, "known_negatives_passed"] = recorded
    return pd.DataFrame({
        "model": rows["method"], "dim": rows["dim"], "repeat": rows["repeat"],
        "protocol": "submitted_code", "pool": "mixed", "roc_auc": rows["auc"],
        "n_microbes": rows["n_microbes"], "n_pos": rows["n_microbes"],
        "known_negatives_passed": rows["known_negatives_passed"],
    })


def ordinal_inertness_check(training_data: TrainingData, labels: np.ndarray, dim: int = 32) -> dict:
    """On the submitted split (no known_negatives), Ordinal-BPR with and without its
    ordinal settings must give identical embeddings if the modification is inert."""
    train_pos, test_pairs = submitted.leave_one_out_split(training_data.positives, seed=0)
    # Same constructor call as run_figure1_oos_benchmark() in the submitted script
    as_submitted = TrainingData(
        n_microbes=training_data.n_microbes,
        n_metabolites=training_data.n_metabolites,
        microbe_to_idx=training_data.microbe_to_idx,
        metabolite_to_idx=training_data.metabolite_to_idx,
        positives=train_pos,
        microbe_taxonomy=training_data.microbe_taxonomy,
        preference_type=training_data.preference_type,
    )
    seed = training_seed(0, dim)
    default = train(as_submitted, "Ordinal-BPR", dim, seed)
    switched_off = train(as_submitted, "Ordinal-BPR", dim, seed, hard_neg_ratio=0.0, lambda_ord=0.0)
    with_kn = train(split_training_data(training_data, train_pos, training_data.known_negatives),
                    "Ordinal-BPR", dim, seed)
    auc = {name: evaluate_loo(score_matrix(model), test_pairs, train_pos, labels)["mixed"]["roc_auc"]
           for name, model in [("default", default), ("switched_off", switched_off),
                               ("with_known_negatives", with_kn)]}
    return {
        "dim": dim,
        "repeat": 0,
        "seed": seed,
        "known_negatives_in_submitted_split": n_known_negatives(as_submitted.known_negatives),
        "known_negatives_in_full_training_data": n_known_negatives(training_data.known_negatives),
        "embeddings_identical_default_vs_switched_off": bool(
            np.array_equal(default.microbe_embeddings, switched_off.microbe_embeddings)
            and np.array_equal(default.metabolite_embeddings, switched_off.metabolite_embeddings)),
        "embeddings_identical_default_vs_with_known_negatives": bool(
            np.array_equal(default.microbe_embeddings, with_kn.microbe_embeddings)),
        "mixed_pool_auc_default": auc["default"],
        "mixed_pool_auc_switched_off": auc["switched_off"],
        "mixed_pool_auc_with_known_negatives": auc["with_known_negatives"],
    }


# =============================================================================
# Benchmark
# =============================================================================


def run_benchmark(training_data: TrainingData, labels: np.ndarray, dims: list[int],
                  repeats: int, model_names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Protocols A and B. By default runs the five main models at every dim plus the
    ablations at ABLATION_DIM; pass model_names to run exactly those at every dim."""
    rows, heldout = [], []
    idx_to_microbe = np.array([m for m, _ in sorted(training_data.microbe_to_idx.items(), key=lambda x: x[1])])
    idx_to_compound = np.array([c for c, _ in sorted(training_data.metabolite_to_idx.items(), key=lambda x: x[1])])

    for repeat in range(repeats):
        # Protocol A: the submitted leave-one-out split, known_negatives passed through
        loo_pos, test_pairs = submitted.leave_one_out_split(training_data.positives, seed=repeat)
        loo_data = split_training_data(training_data, loo_pos, training_data.known_negatives)
        # Protocol B: withheld consumed and neutral pairs are untested for every model
        hb_pos, hb_kn, withheld = holdout_split(training_data, labels, seed=repeat)
        hb_data = split_training_data(training_data, hb_pos, hb_kn)

        for dim in dims:
            seed = training_seed(repeat, dim)
            if model_names is None:
                names = list(MODEL_TYPES) + (list(ABLATIONS) if dim == ABLATION_DIM else [])
            else:
                names = model_names
            for model_name in names:
                is_ordinal = model_name.startswith("Ordinal-BPR")
                t0 = time.time()
                scores = score_matrix(train(loo_data, model_name, dim, seed))
                for pool, metrics in evaluate_loo(scores, test_pairs, loo_pos, labels).items():
                    rows.append({"model": model_name, "dim": dim, "repeat": repeat,
                                 "protocol": "A_loo", "pool": pool, **metrics,
                                 "known_negatives_passed":
                                     n_known_negatives(loo_data.known_negatives) if is_ordinal else np.nan})

                scores = score_matrix(train(hb_data, model_name, dim, seed))
                metrics, y, s = evaluate_holdout(scores, withheld, labels)
                rows.append({"model": model_name, "dim": dim, "repeat": repeat,
                             "protocol": "B_holdout", "pool": "neutral", **metrics,
                             "known_negatives_passed":
                                 n_known_negatives(hb_data.known_negatives) if is_ordinal else np.nan})
                heldout.append(pd.DataFrame({
                    "model": model_name, "dim": dim, "repeat": repeat,
                    "microbe": idx_to_microbe[withheld[:, 0]],
                    "compound": idx_to_compound[withheld[:, 1]],
                    "label": np.where(y == 1, "consumed", "neutral"),
                    "score": s.astype(np.float32),
                }))
                logger.info(f"  repeat {repeat} dim {dim:>3} {model_name:<11} "
                            f"A1={rows[-4]['roc_auc']:.3f} B={metrics['roc_auc']:.3f} "
                            f"({time.time() - t0:.1f}s)")

    heldout = pd.concat(heldout, ignore_index=True)
    for col in ["model", "microbe", "compound", "label"]:
        heldout[col] = heldout[col].astype("category")
    return pd.DataFrame(rows), heldout


# =============================================================================
# Figure
# =============================================================================


def plot_revised_figure(results: pd.DataFrame, output_dir: Path):
    panels = [
        ("A_loo", "neutral", "roc_auc", "a  Leave-one-out vs measured neutral", "AUC", (0.5, 1.0)),
        ("B_holdout", "neutral", "roc_auc", "b  Strict hold-out, per microbe", "ROC-AUC", (0.5, 1.0)),
        ("B_holdout", "neutral", "pr_auc_pooled", "c  Strict hold-out, pooled", "PR-AUC", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (protocol, pool, metric, title, ylabel, ylim) in zip(axes, panels):
        sub = results[(results["protocol"] == protocol) & (results["pool"] == pool)]
        stats = sub.groupby(["model", "dim"])[metric].agg(["mean", "std"]).reset_index()
        for model_name, style in MODEL_STYLES.items():
            m = stats[stats["model"] == model_name].sort_values("dim")
            ax.errorbar(m["dim"], m["mean"], yerr=m["std"], label=model_name,
                        color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                        capsize=8, linewidth=4, markersize=10)
        if metric == "pr_auc_pooled":
            prevalence = sub["prevalence"].mean()
            ax.axhline(prevalence, color="#555555", linestyle=":", linewidth=2.5)
            ax.text(sub["dim"].min(), prevalence + 0.02, f"prevalence = {prevalence:.2f}",
                    fontsize=16, color="#333333", va="bottom")

        dims = sorted(stats["dim"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([fr"$2^{{{int(np.log2(d))}}}$" for d in dims])
        ax.set_xlabel("Embedding dimension", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=18, loc="left")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels),
               frameon=False, fontsize=18)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig1b_revised.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'fig1b_revised.png'}")


# =============================================================================
# Main
# =============================================================================


def main(output_dir: Path, dims: list[int], repeats: int):
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_exometabolite()
    training_data = prepare_training_data(df, preference_type=AssertionType.CONSUMED,
                                          include_secreted=False)
    labels = build_label_matrix(df, training_data)
    counts = {label: int((labels == i).sum()) for i, label in enumerate(LABELS)}
    logger.info(f"{training_data.n_microbes} microbes x {training_data.n_metabolites} metabolites; "
                f"pair labels: {counts}")

    logger.info("Re-running the submitted Fig. 1b code (unmodified)")
    t0 = time.time()
    submitted_rows = rerun_submitted_code(dims, repeats)
    submitted_secs = time.time() - t0

    logger.info("Checking whether the ordinal modification is active on the submitted split")
    checks = ordinal_inertness_check(training_data, labels)
    checks["known_negatives_passed_in_submitted_code_rerun"] = sorted(
        submitted_rows["known_negatives_passed"].dropna().astype(int).unique().tolist())
    checks["pair_label_counts"] = counts
    logger.info(json.dumps(checks, indent=2))

    logger.info("Running Protocols A and B")
    t0 = time.time()
    results, heldout = run_benchmark(training_data, labels, dims, repeats)
    benchmark_secs = time.time() - t0

    results = pd.concat([submitted_rows, results], ignore_index=True)
    results["n_pairs"] = results["n_pos"] + results["n_neg"]
    results[RESULT_COLUMNS].to_csv(output_dir / "fig1b_benchmark_results.csv", index=False,
                                   float_format="%.6f")
    heldout.to_parquet(output_dir / "fig1b_heldout_scores.parquet", index=False)
    plot_revised_figure(results, output_dir)

    checks["runtime_seconds"] = {"submitted_code_rerun": round(submitted_secs, 1),
                                 "protocols_A_B": round(benchmark_secs, 1),
                                 "total": round(time.time() - start, 1)}
    with open(output_dir / "fig1b_bug_checks.json", "w") as f:
        json.dump(checks, f, indent=2)
    logger.info(f"Done in {time.time() - start:.0f}s; outputs in {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Corrected Fig. 1b benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WOM_DIR / "manuscript_figures" / "msystems_revision",
        help="Output directory",
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    main(args.output_dir, args.dims, args.repeats)
