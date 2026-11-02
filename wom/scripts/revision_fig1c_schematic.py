# revision_fig1c_schematic.py
"""Figure 1c schematic: what BPR embeds and how scores are formed (Reviewer 1, comment 4).

Drawn in the visual language of Figure 1a (same toy matrix, cell colours and
labels): the isolate x metabolite assertion matrix is the only input; BPR learns one
32-value embedding per isolate and per metabolite from sampled (isolate, depleted,
negative) triples, half of the negatives measured-neutral and half unobserved (untested or secreted); the score
of every isolate-metabolite pair is the dot product of the two embeddings, which ranks
all metabolites for each isolate, including unobserved ones. Values are illustrative.

Usage:
    python scripts/revision_fig1c_schematic.py --output-dir <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

DEPLETED, UNOBSERVED, NEUTRAL, INK = "#1e8aae", "#b3b3b3", "#ffffff", "#000000"
STATE_COLOR = {"T": DEPLETED, "G": UNOBSERVED, "W": NEUTRAL}
# Figure 1a input matrix: rows Met 1-7, columns Org 1-7 (T depleted, W neutral, G unobserved)
MATRIX = ["GGGWGGT", "GGGTGGW", "GWGGWWT", "GWTGGGG", "GGGGWGG", "TTGGGGW", "GGGGWTG"]
EMB = LinearSegmentedColormap.from_list("emb", ["#5e3c99", "#f2f2f2", "#e66101"])  # diverging, grey midpoint
WIDTH, HEIGHT = 20.48, 7.4


def cell_grid(ax, x0, y_top, values, size, colors=None, cmap=None, row_labels=None, col_labels=None,
              label_size=15):
    n_rows, n_cols = len(values), len(values[0])
    for r in range(n_rows):
        for c in range(n_cols):
            face = colors[values[r][c]] if colors else cmap((values[r][c] + 1) / 2)
            ax.add_patch(Rectangle((x0 + c * size, y_top - (r + 1) * size), size, size,
                                   facecolor=face, edgecolor=INK, linewidth=2))
    if row_labels:
        for r, lab in enumerate(row_labels):
            ax.text(x0 - 0.12, y_top - (r + 0.5) * size, lab, ha="right", va="center", fontsize=label_size,
                    fontweight="bold")
    if col_labels:
        for c, lab in enumerate(col_labels):
            ax.text(x0 + (c + 0.5) * size, y_top + 0.12, lab, ha="center", va="bottom", fontsize=label_size,
                    fontweight="bold", rotation=0)


def arrow(ax, x0, x1, y, label=None, label_y=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=30, linewidth=3, color=INK))
    if label:
        ax.text((x0 + x1) / 2, label_y if label_y is not None else y + 0.25, label, ha="center", va="bottom",
                fontsize=16, fontweight="bold")


def draw(out_stem: Path):
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")
    ax.text(0.3, HEIGHT - 0.5, "c", fontsize=50, va="center")
    title_y, sub_y, top = HEIGHT - 0.5, HEIGHT - 1.1, HEIGHT - 2.35

    # 1. input matrix (Figure 1a)
    size, x0 = 0.5, 1.75
    ax.text(x0 + 3.5 * size, title_y, "Input: assertion matrix", ha="center", va="center", fontsize=26,
            fontweight="bold")
    ax.text(x0 + 3.5 * size, sub_y, "433 isolates × 311 metabolites;\nno other features", ha="center",
            va="center", fontsize=17, linespacing=1.15)
    cell_grid(ax, x0, top, MATRIX, size, colors=STATE_COLOR, row_labels=[f"Met {i}" for i in range(1, 8)],
              col_labels=[f"Org\n{i}" for i in range(1, 8)], label_size=15)
    legend_y = top - 7 * size - 0.45
    for k, (state, lab) in enumerate([("T", "depleted\n(positive)"), ("W", "neutral\n(measured\nnegative)"),
                                      ("G", "unobserved\n(untested or\nsecreted)")]):
        cx = x0 + 0.25 + k * 1.5
        ax.add_patch(Rectangle((cx - 0.2, legend_y - 0.2), 0.4, 0.4, facecolor=STATE_COLOR[state],
                               edgecolor=INK, linewidth=2))
        ax.text(cx, legend_y - 0.35, lab, ha="center", va="top", fontsize=15, fontweight="bold",
                linespacing=1.1)

    # 2. training
    arrow(ax, 5.75, 8.35, top - 0.9, "BPR training", label_y=top - 0.7)
    ax.add_patch(FancyBboxPatch((5.75, top - 4.75), 2.6, 3.3, boxstyle="round,pad=0.12", facecolor="#f5f5f5",
                                edgecolor="#9a9a9a", linewidth=1.5))
    ax.text(7.05, top - 1.6, "sampled triples (i, j, k)", ha="center", va="top", fontsize=16, fontweight="bold")
    ax.text(5.95, top - 2.15, "i  isolate\nj  metabolite depleted\n    by i\nk  neutral (50%) or\n    unobserved (50%)",
            ha="left", va="top", fontsize=15.5, linespacing=1.3)
    ax.text(7.05, top - 4.3, r"push $\mathbf{u}_i\cdot\mathbf{v}_j > \mathbf{u}_i\cdot\mathbf{v}_k$", ha="center",
            va="center", fontsize=18)

    # 3. embeddings
    rng = np.random.default_rng(7)
    esize = 0.46
    dims = ["d1", "d2", "d3", "…", "d32"]
    for k, (title, rows, ex0) in enumerate([("Isolate\nembeddings", [f"Org {i}" for i in range(1, 8)], 9.65),
                                            ("Metabolite\nembeddings", [f"Met {i}" for i in range(1, 8)], 12.75)]):
        vals = rng.uniform(-1, 1, (7, 5))
        ax.text(ex0 + 2.5 * esize, title_y - 0.1, title, ha="center", va="center", fontsize=24, fontweight="bold",
                linespacing=1.0)
        ax.text(ex0 + 2.5 * esize, sub_y - 0.3, ("u" if k == 0 else "v") + " (32 values each)",
                ha="center", va="center", fontsize=17, style="italic")
        cell_grid(ax, ex0, top, vals, esize, cmap=EMB, row_labels=rows, col_labels=dims, label_size=15)
        for r in range(7):  # the "…" column
            ax.add_patch(Rectangle((ex0 + 3 * esize, top - (r + 1) * esize), esize, esize, facecolor="white",
                                   edgecolor="white", linewidth=0))
            ax.text(ex0 + 3.5 * esize, top - (r + 0.5) * esize, "…", ha="center", va="center", fontsize=16)
    cb_x, cb_y = 11.2, top - 7 * esize - 0.75
    for i, v in enumerate(np.linspace(-1, 1, 9)):
        ax.add_patch(Rectangle((cb_x + i * 0.34, cb_y), 0.34, 0.32, facecolor=EMB((v + 1) / 2), edgecolor=INK,
                               linewidth=1))
    ax.text(cb_x - 0.12, cb_y + 0.16, "−", ha="right", va="center", fontsize=22, fontweight="bold")
    ax.text(cb_x + 9 * 0.34 + 0.12, cb_y + 0.16, "+", ha="left", va="center", fontsize=22, fontweight="bold")
    ax.text(cb_x + 4.5 * 0.34, cb_y - 0.12, "learned values (illustrative)", ha="center", va="top", fontsize=15)

    # 4. scores and ranking for one isolate
    arrow(ax, 15.3, 16.45, top - 0.9, None)
    ax.text(15.88, top - 0.7, "score", ha="center", va="bottom", fontsize=19, fontweight="bold")
    ax.text(15.88, top - 1.2, r"$\mathbf{u}_i\cdot\mathbf{v}_j$", ha="center", va="top", fontsize=21)
    ax.text(18.55, title_y, "Every pair scored", ha="center", va="center", fontsize=26, fontweight="bold")
    ax.text(18.55, sub_y, "ranking for Org 4\n(illustrative)", ha="center", va="center", fontsize=17,
            linespacing=1.15)
    org4 = {f"Met {r + 1}": MATRIX[r][3] for r in range(7)}
    scores = {"Met 2": 3.1, "Met 6": 2.2, "Met 3": 1.4, "Met 7": 0.6, "Met 5": -0.4, "Met 4": -1.1, "Met 1": -1.9}
    bx, bscale, bh, step = 18.45, 0.55, 0.38, 0.5
    for r, (met, sc) in enumerate(scores.items()):
        y = top - (r + 0.5) * step
        ax.text(bx - 1.35, y, met, ha="right", va="center", fontsize=15, fontweight="bold")
        ax.add_patch(Rectangle((bx, y - bh / 2), sc * bscale, bh, facecolor=STATE_COLOR[org4[met]], edgecolor=INK,
                               linewidth=1.8))
    ax.plot([bx, bx], [top - 7 * step - 0.05, top + 0.05], color=INK, linewidth=1.5)
    ax.text(bx, top - 7 * step - 0.2, "score", ha="center", va="top", fontsize=15)
    ax.text(18.55, top - 7 * step - 0.75, "unobserved metabolites (grey)\nare ranked with measured ones",
            ha="center", va="top", fontsize=15, linespacing=1.15)

    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(args.output_dir / "fig1c_schematic")
