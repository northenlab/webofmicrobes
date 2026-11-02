# revision_fig1b_lambda0_sweep.py
"""Dimension sweep for Ordinal-BPR without the ordinal term (mSystems revision).

Runs "Ordinal-BPR [lambda_ord=0]" (hard_neg_ratio 0.5, all other settings as
Ordinal-BPR) at every dimension under Protocol A (neutral / untested / mixed
pools) and Protocol B, with the seeds and splits of revision_fig1b_benchmark.py.
At dim 32 the rows must equal the ablation rows already in
fig1b_benchmark_results.csv; the script checks this before writing.

Usage:
    python scripts/revision_fig1b_lambda0_sweep.py
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

from core.types import AssertionType
from loaders.loaders import prepare_training_data, load_exometabolite
import scripts.revision_fig1b_benchmark as fig1b

logger = logging.getLogger(__name__)

MODEL = "Ordinal-BPR [lambda_ord=0]"


def check_against_existing(sweep: pd.DataFrame, existing_csv: Path):
    """Rows at ABLATION_DIM must reproduce the ablation rows of the main benchmark."""
    existing = pd.read_csv(existing_csv)
    existing = existing[(existing["model"] == MODEL) & (existing["dim"] == fig1b.ABLATION_DIM)
                        & existing["repeat"].isin(sweep["repeat"].unique())]
    if existing.empty:
        raise ValueError(f"No {MODEL} rows at dim {fig1b.ABLATION_DIM} in {existing_csv.name}")
    keys = ["model", "dim", "repeat", "protocol", "pool"]
    merged = sweep.merge(existing, on=keys, suffixes=("", "_existing"), validate="one_to_one")
    if len(merged) != len(existing):
        raise ValueError(f"Expected {len(existing)} matching rows, found {len(merged)}")
    numeric = [c for c in fig1b.RESULT_COLUMNS if c not in keys]
    nan_mismatch = [c for c in numeric if (merged[c].isna() != merged[f"{c}_existing"].isna()).any()]
    if nan_mismatch:
        raise ValueError(f"Missing-value pattern differs from {existing_csv.name}: {nan_mismatch}")
    # The existing CSV is written with 6 decimals, so unrounded values agree to half a unit
    diffs = {c: float(np.nanmax(np.abs(merged[c] - merged[f"{c}_existing"])))
             if merged[c].notna().any() else 0.0 for c in numeric}
    worst = max(diffs.values())
    if worst > 5e-7 + 1e-9:
        raise ValueError(f"Sweep does not reproduce {existing_csv.name} at dim "
                         f"{fig1b.ABLATION_DIM}: {diffs}")
    logger.info(f"Dim {fig1b.ABLATION_DIM} rows reproduce {existing_csv.name} "
                f"({len(merged)} rows, max |diff| {worst:.1e})")


def main(output_dir: Path, dims: list[int], repeats: int):
    start = time.time()
    df = load_exometabolite()
    training_data = prepare_training_data(df, preference_type=AssertionType.CONSUMED,
                                          include_secreted=False)
    # same labelling as fig1b_benchmark_results.csv, which the dim-32 check compares against
    labels = fig1b.build_label_matrix(df, training_data, rule="secreted_over_neutral")

    results, _ = fig1b.run_benchmark(training_data, labels, dims, repeats, model_names=[MODEL])
    results["n_pairs"] = results["n_pos"] + results["n_neg"]
    results = results[fig1b.RESULT_COLUMNS]

    existing_csv = output_dir / "fig1b_benchmark_results.csv"
    if fig1b.ABLATION_DIM in dims:
        check_against_existing(results, existing_csv)

    out = output_dir / "fig1b_lambda0_sweep.csv"
    results.to_csv(out, index=False, float_format="%.6f")
    logger.info(f"Saved {out} ({len(results)} rows) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=f"Dimension sweep for {MODEL}")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WOM_DIR / "manuscript_figures" / "msystems_revision",
        help="Directory holding fig1b_benchmark_results.csv; the sweep CSV is written here",
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    main(args.output_dir, args.dims, args.repeats)
