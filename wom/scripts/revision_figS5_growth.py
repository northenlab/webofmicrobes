# revision_figS5_growth.py
"""Supplementary Fig. 5: growth of Acinetobacter guillouiae GW821-FHT01E09 on single carbon sources.

Re-plots Markus de Raad's succinic acid experiment (Growth_Data_Acinetobacter_Succinic_Acid.xlsx,
three independent experiments with three replicate wells per condition; notebook
20261102MdR_Acinetobacter_growth_curve_succinic_acid.ipynb) in the style of his notebook, with
two changes: the time axis ends at 21 h, when the shortest experiment (Exp 2, last read 21.24 h)
stopped, so no experiment is extrapolated (the notebook interpolated all runs to 36 h, which held
Exp 2 at its last value); and the band is the sample SD (n - 1) across the three experiments.
Replicate wells are averaged within each experiment and interpolated onto a common grid.

Usage:
    python scripts/revision_figS5_growth.py --data <xlsx> --output-dir <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHEETS = ["Exp1_AMD1706", "Exp2_AMD1706", "Exp3_AMD1706"]
CONDITIONS = ["NLDM", "Succinic", "Xylose", "Arginine", "N-Ac-Glu", "H2O"]
LABELS = {"NLDM": "NLDM", "Succinic": "Succinic Acid", "Xylose": "Xylose", "Arginine": "Arginine",
          "N-Ac-Glu": "N-Acetyl-Glutamate", "H2O": "Water (Negative Control)"}
COLORS = {"NLDM": "#0072B2", "Succinic": "#E69F00", "Xylose": "#009E73", "Arginine": "#CC79A7",
          "N-Ac-Glu": "#56B4E9", "H2O": "#999999"}  # the notebook's colours
N_GRID = 150


def load(data: Path) -> tuple[dict, list]:
    """Raw sheets (for the data table) and per-experiment well means on a common grid."""
    sheets = {s: pd.read_excel(data, sheet_name=s).dropna(subset=["Time"]) for s in SHEETS}
    start = max(df.Time.min() for df in sheets.values())
    end = min(df.Time.max() for df in sheets.values())
    end = np.floor(end)  # whole hours, within every experiment
    grid = np.linspace(start, end, N_GRID)
    means = {c: [] for c in CONDITIONS}
    for df in sheets.values():
        for c in CONDITIONS:
            cols = [col for col in df.columns if str(col).split(".")[0].lower() == c.lower()]
            assert len(cols) == 3, (c, cols)
            means[c].append(np.interp(grid, df.Time.to_numpy(float), df[cols].mean(axis=1).to_numpy(float)))
    return sheets, grid, {c: np.vstack(v) for c, v in means.items()}


def plot(grid, means, out_stem: Path):
    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"],
                         "axes.linewidth": 0.8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for c in CONDITIONS:
        m, sd = means[c].mean(0), means[c].std(0, ddof=1)
        ax.plot(grid, m, label=LABELS[c], color=COLORS[c], linewidth=1.2)
        ax.fill_between(grid, m - sd, m + sd, color=COLORS[c], alpha=0.15, linewidth=0)
    ax.set_xlabel("Time (h)", fontsize=8)
    ax.set_ylabel("Optical density (OD600)", fontsize=8)
    ax.set_xlim(0, grid[-1])
    ax.set_xticks(np.arange(0, grid[-1] + 0.1, 3))
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Growth condition", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7,
              title_fontsize=7, handlelength=1.2, labelspacing=0.4)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def summary(grid, means) -> pd.DataFrame:
    rows = []
    for c in CONDITIONS:
        m, sd = means[c].mean(0), means[c].std(0, ddof=1)
        rows.append({"condition": LABELS[c], "od_end_mean": m[-1], "od_end_sd": sd[-1], "od_max_mean": m.max(),
                     "time_of_max_h": grid[m.argmax()], "end_h": grid[-1]})
    return pd.DataFrame(rows)


def main(data: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets, grid, means = load(data)
    plot(grid, means, output_dir / "figS5_growth")
    s = summary(grid, means)
    s.to_csv(output_dir / "figS5_summary.csv", index=False, float_format="%.4f")
    print(s.round(3).to_string())
    for name, df in sheets.items():
        print(name, f"{df.Time.min():.2f}-{df.Time.max():.2f} h, {len(df)} reads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.output_dir)
