# Figure 3 — Taxonomic hold-out cross-validation (Ordinal-BPR vs. naive baseline)

Code and results behind manuscript Figure 3 (`Figure3_holdout_CV.pdf/png`):
Ordinal-BPR generalization to organisms withheld from training, compared against
a learning-free genus/parent-mean baseline, at four hold-out levels of
increasing phylogenetic distance (isolate, genus, family, order).

## Files

| File | What it is |
|---|---|
| `holdout_cv.py` | Runs the hierarchical taxonomic hold-out CV. For each group at each rank, retrains a fresh Ordinal-BPR (dim 32, 400 epochs) on the remainder and predicts each held-out isolate from the mean embedding of its nearest remaining relatives. Writes `holdout_cv_results.csv`. Slow (retrains one model per held-out group; expect ~30–40 min). |
| `baseline_all_levels.py` | Naive baseline mirroring the same peer selection: score = consumption prevalence among peers; compounds untested in the peer set can't be scored (that's panel c's coverage). Writes `baseline_all_levels_results.csv`. Fast. |
| `make_holdout_figure.py` | Builds the 3-panel figure from the two CSVs. Writes `Figure3_holdout_CV.pdf/png` into this directory. |
| `holdout_cv_results.csv` | Per-isolate BPR hold-out AUCs (the run used for the manuscript figure). |
| `baseline_all_levels_results.csv` | Per-isolate baseline AUCs + coverage (ditto). |

## Reproduce the figure only (uses committed CSVs, seconds)

```bash
pip install -r wom/requirements.txt   # needs numpy, pandas, matplotlib, scipy, torch
python wom/scripts/figure3_holdout_cv/make_holdout_figure.py
```

## Rerun the full analysis

```bash
python wom/scripts/figure3_holdout_cv/holdout_cv.py           # ~30-40 min
python wom/scripts/figure3_holdout_cv/baseline_all_levels.py  # fast
python wom/scripts/figure3_holdout_cv/make_holdout_figure.py
```

Data are loaded via `wom.loaders.prepare_training_data(include_secreted=False)`
(311 metabolites; consumed vs. neutral assertions). Seed is fixed (`SEED = 0`)
in `holdout_cv.py`.
