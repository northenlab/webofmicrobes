# revision_fig1b_main.py
"""Main-text Fig. 1b for the revision: strict hold-out per-microbe ROC-AUC vs dimension.

Plots Protocol B (20% of each microbe's consumed and neutral pairs withheld;
ROC-AUC per microbe, averaged over microbes) as mean ± SD over 5 repeats for
the adopted model, labelled "BPR" in the revised manuscript (code path:
Ordinal-BPR with lambda_ord = 0), and WR-MF, SVD and LMF, the models of the
submitted figure. Values come from revision_fig1b_benchmark.py run with the
revised-Methods labelling (lambda0/fig1b/fig1b_benchmark_results.csv); nothing
is retrained here. Styled as the submitted figure1_benchmark.png.

Usage:
    python scripts/revision_fig1b_main.py --output-dir <dir>
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

logger = logging.getLogger(__name__)

REVISION_DIR = WOM_DIR / "manuscript_figures" / "msystems_revision"
# (model name in the results CSV, legend label, style); legend in the submitted order
MODELS = [
    ("Ordinal-BPR [lambda_ord=0]", "BPR", {"color": "#e377c2", "linestyle": "-"}),
    ("WR-MF", "WR-MF", {"color": "#1f77b4", "linestyle": "-"}),
    ("SVD", "SVD", {"color": "#2ca02c", "linestyle": "-"}),
    ("LMF", "LMF", {"color": "#8c564b", "linestyle": "-"}),
]


def load_strict_holdout(results_csv: Path) -> pd.DataFrame:
    results = pd.read_csv(results_csv)
    sub = results[(results["protocol"] == "B_holdout") & (results["pool"] == "neutral")
                  & results["model"].isin([m for m, _, _ in MODELS])]
    stats = (sub.groupby(["model", "dim"])["roc_auc"]
             .agg(mean="mean", sd="std", n_repeats="count").reset_index())
    if (stats["n_repeats"] != 5).any():
        raise ValueError(f"Expected 5 repeats per model and dim:\n{stats}")
    return stats


def plot(stats: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for model, label, style in MODELS:
        m = stats[stats["model"] == model].sort_values("dim")
        ax.errorbar(m["dim"], m["mean"], yerr=m["sd"], label=label, marker="o",
                    capsize=8, linewidth=4, markersize=8, **style)

    dims = sorted(stats["dim"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([fr"$2^{{{int(np.log2(d))}}}$" for d in dims])
    ax.set_xlabel("Embedding dimension", fontsize=20)
    ax.set_ylabel("ROC-AUC", fontsize=20)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=16, loc="lower left")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig1b_main.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'fig1b_main.png'}")


def main(results_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = load_strict_holdout(results_csv)
    labels = {m: label for m, label, _ in MODELS}
    order = {m: i for i, (m, _, _) in enumerate(MODELS)}
    data = stats.assign(order=stats["model"].map(order)).sort_values(["order", "dim"])
    data = data.assign(model=data["model"].map(labels))[["model", "dim", "mean", "sd"]]
    data.to_csv(output_dir / "fig1b_main_data.csv", index=False, float_format="%.4f")
    logger.info("\n" + data.to_string(index=False, float_format="%.3f"))
    plot(stats, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Revision Fig. 1b (strict hold-out)")
    parser.add_argument("--results-csv", type=Path,
                        default=REVISION_DIR / "lambda0" / "fig1b" / "fig1b_benchmark_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.results_csv, args.output_dir)
