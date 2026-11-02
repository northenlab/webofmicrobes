# revision_r1_learning_curve.py
"""Learning curve for BPR (Reviewer 1, comment 6; mSystems revision).

Uses the Fig. 1b hold-out (Protocol B of revision_fig1b_benchmark.py: 20% of every
isolate's consumed and neutral pairs withheld, five repeats) and trains BPR (32
dimensions, lambda_ord = 0, the settings of Fig. 1b) on nested random subsets of the
remaining measured pairs (consumed and neutral), so the held-out pairs are fixed while
the amount of training data varies. For each subset it reports per-isolate ROC-AUC and
PR-AUC on the withheld pairs, and calibration after cross-fitted Platt scaling.
The full subset must reproduce the BPR dim-32 hold-out rows of fig1b_benchmark_results.csv.

BPR draws one batch of 256 (isolate, consumed, negative) triples per epoch, so the
number of gradient steps does not depend on the number of training pairs.

Usage:
    python scripts/revision_r1_learning_curve.py --output-dir <dir> --fig1b-results <csv>
"""

import sys
import time
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.types import AssertionType
from loaders.loaders import prepare_training_data, load_exometabolite
import scripts.revision_fig1b_benchmark as fig1b
from scripts.revision_r1_metrics import per_isolate, crossfit_probabilities, calibration_stats, baseline_scores

logger = logging.getLogger(__name__)

MODEL = "Ordinal-BPR [lambda_ord=0]"  # "BPR" in the revised manuscript
DIM = 32
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
BPR_COLOR = "#e377c2"  # Fig. 1b colour of BPR


def subsample(train_pos: dict, train_kn: dict, fraction: float, seed: int) -> tuple[dict, dict]:
    """Nested random subset: the first `fraction` of one permutation of all training pairs."""
    pairs = [(m, c, 1) for m, cs in train_pos.items() for c in sorted(cs)] + \
            [(m, c, 0) for m, cs in train_kn.items() for c in sorted(cs)]
    pairs.sort()
    order = np.random.default_rng(seed).permutation(len(pairs))
    keep = [pairs[i] for i in order[:int(round(fraction * len(pairs)))]]
    pos, kn = {}, {}
    for m, c, y in keep:
        (pos if y else kn).setdefault(m, set()).add(c)
    return pos, kn


def run(repeats: int) -> pd.DataFrame:
    df = load_exometabolite()
    td = prepare_training_data(df, preference_type=AssertionType.CONSUMED, include_secreted=False)
    labels = fig1b.build_label_matrix(df, td, rule="methods")
    genus_of = {i: td.microbe_taxonomy[m].genus if isinstance(td.microbe_taxonomy[m].genus, str) else None
                for m, i in td.microbe_to_idx.items()}
    n_matrix = td.n_microbes * td.n_metabolites
    rows = []
    for repeat in range(repeats):
        hb_pos, hb_kn, withheld = fig1b.holdout_split(td, labels, seed=repeat)
        n_train_all = sum(map(len, hb_pos.values())) + sum(map(len, hb_kn.values()))
        for fraction in FRACTIONS:
            t0 = time.time()
            if fraction == 1.0:
                pos, kn = hb_pos, hb_kn  # exactly the Fig. 1b training data
            else:
                pos, kn = subsample(hb_pos, hb_kn, fraction, seed=1000 + repeat)
            data = fig1b.split_training_data(td, pos, kn)
            scores = fig1b.score_matrix(fig1b.train(data, MODEL, DIM, fig1b.training_seed(repeat, DIM)))
            metrics, y, s = fig1b.evaluate_holdout(scores, withheld, labels)
            iso = per_isolate(y, s, withheld[:, 0])
            p = crossfit_probabilities(y, s, withheld[:, 0], seed=repeat)
            cal = calibration_stats(y, p)
            n_pairs = sum(map(len, pos.values())) + sum(map(len, kn.values()))
            base = {}
            for name, bs in baseline_scores(pos, kn, td.n_microbes, td.n_metabolites, genus_of).items():
                b = per_isolate(y, bs[withheld[:, 0], withheld[:, 1]], withheld[:, 0])
                base[f"{name}_roc_auc"], base[f"{name}_pr_auc"] = float(b.roc_auc.mean()), float(b.pr_auc.mean())
            rows.append({
                "repeat": repeat, "fraction": fraction, "n_train_pairs": n_pairs,
                "n_train_consumed": sum(map(len, pos.values())), "n_train_neutral": sum(map(len, kn.values())),
                "pct_of_training_pairs": 100 * n_pairs / n_train_all,
                "pct_of_matrix": 100 * n_pairs / n_matrix,
                "n_isolates_with_training_consumed": sum(1 for v in pos.values() if v),
                "n_isolates_evaluated": len(iso),
                "roc_auc": float(iso.roc_auc.mean()), "pr_auc": float(iso.pr_auc.mean()),
                "prevalence": float(iso.prevalence.mean()),
                "roc_auc_fig1b_definition": metrics["roc_auc"], "pr_auc_pooled": metrics["pr_auc_pooled"],
                "pooled_prevalence": metrics["prevalence"],
                **{f"cal_{k}": v for k, v in cal.items() if k not in ("prevalence", "n_pairs")},
                "brier_prevalence_only": float(y.mean() * (1 - y.mean())),
                **base,
            })
            logger.info(f"repeat {repeat} fraction {fraction:.2f} pairs {n_pairs:>6} "
                        f"ROC-AUC {rows[-1]['roc_auc']:.3f} PR-AUC {rows[-1]['pr_auc']:.3f} "
                        f"Brier {cal['brier']:.3f} | popularity {base['popularity_roc_auc']:.3f} "
                        f"genus {base['genus_roc_auc']:.3f} ({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


def check_full_fraction(results: pd.DataFrame, fig1b_csv: Path):
    """At fraction 1 the ROC-AUC must equal the BPR dim-32 hold-out rows of Fig. 1b."""
    ref = pd.read_csv(fig1b_csv)
    ref = ref[(ref.model == MODEL) & (ref.dim == DIM) & (ref.protocol == "B_holdout")].set_index("repeat")
    full = results[results.fraction == 1.0].set_index("repeat")
    diff = (full.roc_auc_fig1b_definition - ref.loc[full.index, "roc_auc"]).abs().max()
    if diff > 5e-7 + 1e-9:
        raise ValueError(f"fraction 1.0 does not reproduce {fig1b_csv.name} (max |diff| {diff:.2e})")
    logger.info(f"fraction 1.0 reproduces {fig1b_csv.name} (max |diff| {diff:.1e})")


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["n_train_pairs", "pct_of_training_pairs", "pct_of_matrix", "n_isolates_with_training_consumed",
            "roc_auc", "pr_auc", "prevalence", "cal_brier", "brier_prevalence_only", "cal_ols_slope",
            "cal_ols_intercept", "cal_ece", "popularity_roc_auc", "popularity_pr_auc", "genus_roc_auc",
            "genus_pr_auc"]
    g = results.groupby("fraction")[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return out.reset_index()


def plot(summary: pd.DataFrame, out_stem: Path):
    x = summary.pct_of_matrix_mean
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    baselines = [("popularity", "popularity (no learning)", "-."), ("genus", "genus prevalence (no learning)", ":")]
    panels = [("roc_auc", "ROC-AUC (mean per isolate)", baselines, None),
              ("pr_auc", "PR-AUC (mean per isolate)", baselines, ("prevalence", "random ranking (prevalence)")),
              ("cal_brier", "Brier score, calibrated (lower is better)", [],
               ("brier_prevalence_only", "prevalence-only prediction"))]
    for ax, letter, (col, ylabel, base, ref) in zip(axes, "abc", panels):
        ax.errorbar(x, summary[f"{col}_mean"], yerr=summary[f"{col}_sd"], color=BPR_COLOR, marker="*",
                    markersize=11, linewidth=2, capsize=3, label="BPR")
        for name, label, ls in base:
            ax.errorbar(x, summary[f"{name}_{col}_mean"], yerr=summary[f"{name}_{col}_sd"], color="#4d4d4d",
                        linestyle=ls, linewidth=1.6, capsize=2, label=label)
        if ref:
            ax.plot(x, summary[f"{ref[0]}_mean"], color="#9a9a9a", linestyle="--", linewidth=1.5, label=ref[1])
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 40])
        ax.set_xticklabels(["2", "5", "10", "20", "40"])
        ax.set_xlabel("Training pairs (% of the 433 × 311 matrix)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
    handles, labels = {}, {}
    for ax in axes:  # one legend for all panels, in first-seen order
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", ncol=5, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 1.07))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(output_dir: Path, fig1b_csv: Path, repeats: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = run(repeats)
    check_full_fraction(results, fig1b_csv)
    results.to_csv(output_dir / "learning_curve.csv", index=False, float_format="%.6f")
    summary = summarise(results)
    summary.to_csv(output_dir / "learning_curve_summary.csv", index=False, float_format="%.6f")
    plot(summary, output_dir / "figS8_learning_curve")
    logger.info(f"Saved {output_dir} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="BPR learning curve on the Fig. 1b hold-out")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fig1b-results", type=Path, required=True,
                        help="fig1b_benchmark_results.csv of the same labelling rule (check at fraction 1)")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    main(args.output_dir, args.fig1b_results, args.repeats)
