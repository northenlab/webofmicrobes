# Ordinal-BPR without the ordinal term: dimension sweep, and Ordinal-BPR dependency inventory

Branch `msystems-revision`. This follows `fig1b_REPORT.md`.

| File | Contents |
|---|---|
| `fig1b_lambda0_sweep.csv` | "Ordinal-BPR [lambda_ord=0]" at dims 4–128 × 5 repeats, Protocols A (neutral / untested / mixed) and B. 120 rows, same columns as `fig1b_benchmark_results.csv` |
| `wom/scripts/revision_fig1b_lambda0_sweep.py` | Companion script. It reuses `run_benchmark()` from `revision_fig1b_benchmark.py` |

`run_benchmark()` gained an optional `model_names` argument. Its default output is byte-identical to before; this was checked by re-running
the dims 4 and 32 configuration and comparing the CSVs.

## Part 1. Sweep results

**Setup**
- **Model:** `hard_neg_ratio` 0.5, `lambda_ord` 0. All other settings match Ordinal-BPR (1000 epochs, batch 256, lr 0.01).
  Known negatives are passed through: 59,749 in Protocol A and 47,835 in Protocol B.
- **Seeds and splits:** same as the main benchmark, i.e. `training_seed(repeat, dim)` = 10,000 × repeat + dim, with the leave-one-out and
  hold-out splits seeded by repeat. Every comparison below is therefore paired.
- **Reproduction check:** at dim 32, the sweep reproduces the 20 ablation rows already in `fig1b_benchmark_results.csv` (max |diff| 4.9 × 10⁻⁷, i.e. CSV rounding).
  The script runs this check before writing.
- **Runtime:** 316 s on CPU.

### "Ordinal-BPR [lambda_ord=0]", mean ± SD over 5 repeats

| dim | A1 neutral | A2 untested | A3 mixed | B per-microbe ROC-AUC | B pooled ROC-AUC | B pooled PR-AUC |
|---|---|---|---|---|---|---|
| 4 | 0.861 ± 0.008 | 0.974 ± 0.005 | 0.920 ± 0.006 | 0.849 ± 0.007 | 0.872 ± 0.007 | 0.541 ± 0.024 |
| 8 | 0.879 ± 0.007 | 0.977 ± 0.003 | 0.924 ± 0.005 | 0.859 ± 0.006 | 0.885 ± 0.006 | 0.592 ± 0.020 |
| 16 | 0.893 ± 0.006 | 0.977 ± 0.002 | 0.928 ± 0.004 | **0.865 ± 0.006** | 0.898 ± 0.002 | 0.646 ± 0.008 |
| 32 | **0.895 ± 0.009** | 0.974 ± 0.003 | 0.924 ± 0.005 | **0.865 ± 0.009** | 0.901 ± 0.004 | 0.658 ± 0.012 |
| 64 | 0.892 ± 0.008 | 0.975 ± 0.004 | 0.922 ± 0.005 | 0.856 ± 0.005 | 0.895 ± 0.002 | 0.641 ± 0.009 |
| 128 | 0.876 ± 0.005 | 0.968 ± 0.004 | 0.910 ± 0.006 | 0.835 ± 0.007 | 0.874 ± 0.005 | 0.611 ± 0.011 |

The prevalence baseline for B pooled PR-AUC is 0.158.

### Comparison on the two consumed-vs-neutral metrics

Per-dimension means for the variant, default Ordinal-BPR and plain BPR:

| dim | A1: λ=0 | A1: Ordinal-BPR | A1: BPR | B: λ=0 | B: Ordinal-BPR | B: BPR |
|---|---|---|---|---|---|---|
| 4 | **0.861** | 0.846 | 0.852 | **0.849** | **0.849** | 0.840 |
| 8 | **0.879** | 0.855 | 0.865 | **0.859** | 0.852 | 0.846 |
| 16 | **0.893** | 0.844 | 0.867 | **0.865** | 0.854 | 0.842 |
| 32 | **0.895** | 0.792 | 0.856 | **0.865** | 0.827 | 0.824 |
| 64 | **0.892** | 0.678 | 0.833 | **0.856** | 0.755 | 0.804 |
| 128 | **0.876** | 0.536 | 0.790 | **0.835** | 0.666 | 0.770 |

Paired differences, each given as mean (number of λ=0 wins out of 5):

| dim | λ=0 − BPR, A1 | λ=0 − BPR, B | λ=0 − BPR, pooled PR | λ=0 − Ordinal-BPR, A1 | λ=0 − Ordinal-BPR, B | λ=0 − Ordinal-BPR, pooled PR |
|---|---|---|---|---|---|---|
| 4 | +0.009 (4) | +0.009 (5) | +0.035 (4) | +0.014 (5) | −0.000 (3) | −0.093 (0) |
| 8 | +0.014 (4) | +0.013 (5) | +0.034 (5) | +0.023 (5) | +0.007 (3) | −0.068 (0) |
| 16 | +0.026 (5) | +0.024 (5) | +0.047 (5) | +0.049 (5) | +0.011 (4) | −0.054 (0) |
| 32 | +0.040 (5) | +0.041 (5) | +0.073 (5) | +0.104 (5) | +0.039 (5) | −0.034 (0) |
| 64 | +0.059 (5) | +0.051 (5) | +0.100 (5) | +0.215 (5) | +0.101 (5) | +0.003 (3) |
| 128 | +0.085 (5) | +0.065 (5) | +0.135 (5) | +0.339 (5) | +0.169 (5) | +0.042 (5) |

### Best model at each dimension, among the five main models plus λ=0

| Metric | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| A1 neutral | λ=0 | λ=0 | λ=0 | λ=0 | λ=0 | λ=0 |
| B per-microbe ROC-AUC | Ordinal-BPR ≈ λ=0 | λ=0 | λ=0 | λ=0 | λ=0 | λ=0 |
| B pooled ROC-AUC | WR-MF | WR-MF | WR-MF | Ordinal-BPR (0.902 vs 0.901) | λ=0 | λ=0 |
| B pooled PR-AUC | WR-MF | WR-MF | WR-MF | Ordinal-BPR | λ=0 | λ=0 |
| A3 mixed | BPR | BPR | BPR | BPR | BPR | λ=0 |
| A2 untested | Ordinal-BPR | Ordinal-BPR | Ordinal-BPR | Ordinal-BPR | Ordinal-BPR | Ordinal-BPR |

### Stability across dimensions

Range of the per-dimension means over dims 4–128 (lower is more stable):

| Model | A1 | A2 | A3 | B per-microbe | B pooled ROC | B pooled PR |
|---|---|---|---|---|---|---|
| **Ordinal-BPR [λ=0]** | **0.035** | **0.010** | **0.018** | **0.030** | **0.029** | **0.117** |
| Ordinal-BPR | 0.319 | **0.010** | 0.230 | 0.188 | 0.081 | 0.131 |
| BPR | 0.077 | 0.035 | 0.037 | 0.076 | 0.087 | 0.123 |
| WR-MF | 0.182 | 0.216 | 0.192 | 0.219 | 0.262 | 0.551 |
| SVD | 0.247 | 0.303 | 0.271 | 0.273 | 0.357 | 0.603 |
| LMF | 0.127 | 0.191 | 0.174 | 0.131 | 0.218 | 0.324 |

### Interpretation

- **Within-microbe ranking of consumed vs measured-neutral compounds:**
  - **On A1, the leave-one-out comparison the Results describe,** removing the ordinal term makes this the best model at every dimension and the most stable. It beats plain BPR in 28/30 paired runs. At dim 32 it scores 0.895 ± 0.009, against 0.856 for plain BPR and 0.792 for default Ordinal-BPR.
  - **On B, the strict hold-out,** it beats plain BPR in 30/30 runs and is best at dims 8–128. At dims 4–16 it is within about 0.01 of default Ordinal-BPR; at dims ≥ 32 it is far ahead. B peaks at 0.865 at dims 16 and 32.
- **Pooled metrics, which compare scores across microbes:**
  - The ordinal term helps here at low dimensions. Default Ordinal-BPR has higher pooled PR-AUC at dims 4–32 (0/5 wins for λ=0 at each), and WR-MF/SVD are best overall at dims 4–16 (pooled PR-AUC ≈ 0.80–0.83).
  - λ=0 still beats plain BPR on pooled PR-AUC at every dimension.
- **The paper's qualitative Fig. 1b claims hold for this variant.** "Best consumed-vs-neutral ranking and stable across all dimensions" is true of λ=0, not of the ordinal model as specified. Adopting it would mean:
  - The Methods would describe BPR with hard negatives sampled from measured-neutral compounds, and drop the ordinal term.
  - The model name would need reconsidering.
- **Selection caveat:** λ was chosen after seeing these benchmark results, so the λ=0 numbers carry some selection optimism.
  - Only λ_ord ∈ {0, 0.1} was compared, with `hard_neg_ratio` fixed at 0.5.
  - An independent check would be to re-run the 12-isolate validation (Fig. S3) and the taxonomic hold-out (Fig. 3) with λ=0. Their costs are in Part 2.

## Part 2. Everything that uses Ordinal-BPR, with regeneration cost

**This inventory comes from reading the code. Nothing in Part 2 was run.**
- **Measured runtimes** come from runs earlier in this revision on the same CPU. A single Ordinal-BPR fit (dim 32, 1000 epochs, full data) takes 6 s.
- **Estimated runtimes** are marked "est." and are based on the number of model fits and the bootstrap/permutation counts in the code.
- **Figure numbers** follow the mSystems submission as given (Fig. 2, Fig. 3, Fig. S1–S3, SI Table 2). The repo copy of the manuscript uses older numbering, so the mapping is by caption content.

### A. Deployed artifacts

| Artifact | How Ordinal-BPR enters | Regenerate with | Runtime |
|---|---|---|---|
| `wom/trained_models/ordinal_bpr_consumed/` (git-tracked; last committed `36022a4`, 2026-03-25) | The saved model. Trained by `scripts/train_all_models.py` (config at line 37: dim 32, 1000 epochs, `hard_neg_ratio` 0.5, `lambda_ord` 0.1). Its training data match the current parquet | `train_all_models.py` (trains all 12 models), or a single `train_model(...)` call | 6 s per Ordinal-BPR fit (measured). The full 12-model script is est. 1–3 min |
| `wom/trained_models/ordinal_bpr_secreted/` | Same algorithm and settings, trained on secreted data. Used for secretion queries | Same | est. ~6 s |
| MCP server (`mcp/wom_mcp_server.py`; docker service `wom-mcp`, image built with `COPY . /app`) | `_get_wom_engine()` builds `QueryEngine(trained_models/latest, model_type="ordinal_bpr", interaction_type=…)`, which resolves to `trained_models/ordinal_bpr_{consumed,secreted}`. All tools default to `model_type="ordinal_bpr"` | Retrain, commit the model files, rebuild the image, restart the container. Engines are cached per process, so a restart is required | Image build est. 10–20 min (conda/mamba + pip); restart < 1 min |
| Website (Streamlit, docker service `wom-streamlit`: `app_homepage.py`, `pages/mcp_agent_client.py`) | No model files. `agents/ollama_agent.TestAgent` calls `http://wom-mcp:8000/mcp` | Nothing; it picks up the new model once `wom-mcp` is redeployed | 0 |
| REST API `scripts/wom_api.py` and CLI `scripts/wom.py` | Default `--model-dir trained_models/latest`. The `latest` symlink is not present in this checkout; `train_all_models.py` suggests `ln -sf ordinal_bpr_consumed trained_models/latest` | Restart after retraining | < 1 min |

**Configuration notes:**
- **`docker-compose.yml` sets `MODEL_DIR: trained_models/bpr_consumed` on `wom-mcp`.** The server never reads that variable, so it is dead configuration. It should be removed or corrected so it doesn't suggest the server runs plain BPR.
- **`generate_system_prompt.py` builds its examples from `trained_models/bpr_consumed`.** The deployed agent loads `prompts/wom_agent_system_prompt_mcp_version.md` instead.
- **No benchmark claims in agent-facing text:** neither that prompt nor the MCP tool text cites the Fig. 1b AUC or a stability claim.

### B. Manuscript figures and tables

| Item | Code (entry point) | Ordinal-BPR source | Runtime to regenerate |
|---|---|---|---|
| **Fig. 1b** (benchmark) | `scripts/revision_fig1b_benchmark.py`, plus `revision_fig1b_lambda0_sweep.py` | Trains its own models | 18.5 min (measured) + 5.3 min sweep (measured) |
| **Fig. 2** (phylogenetic conservation: panel a BPR cosine similarity vs taxonomic distance; panel b GCS vs BPR embedding diversity) | `scripts/make_figure_biology.py`. Writes `figure_biology_main`, `figure_gcs_ranking`, `figure_biology_panel_A_stats.txt` and `stats_gcs_emb_diversity.csv` to `manuscript_figures_nat_microbiology_first_submission/` by default | Saved model (`--model-dir`, default `trained_models/ordinal_bpr_consumed`) | est. 5–15 min (>90,000 isolate pairs, 1000 bootstraps, 999 permutations). The empirical (Spearman/GCS) parts do not change |
| **Fig. 3** (taxonomic hold-out CV, Ordinal-BPR vs naive baseline) | `scripts/figure3_holdout_cv/holdout_cv.py` → `baseline_all_levels.py` → `make_holdout_figure.py` | Retrains `OrdinalBPRModel` directly for every held-out group (dim 32, **400 epochs**, known negatives passed, `lambda_ord` default 0.1, `SEED = 0`) | ~30–40 min (README) + seconds |
| Fig. 3, alternate version (`molecular_system_biology_fig3.py`, untracked; results cached in `molecular_system_biology_fig3_results.csv`) | Same design (400 epochs) | Retrains | ~30–40 min (docstring); rendering from cache is fast |
| **Fig. 4** (agent predictions: *P. fluorescens* temporal order, Prompt 1; gut genera, Prompt 5) and the SI table `Supplementary Tables - GeminiRankingPseudomonas.tsv` (`bpr rank` column) + `benchmarking/output/plot_test1_boxplots.py` | Fig. 4 plotting code is not in this repo. The BPR values came from the deployed agent/model | Deployed model | Re-querying BPR ranks via `QueryEngine` takes seconds and re-plotting takes seconds. Re-running agent narratives takes minutes per prompt and LLM output is not deterministic |
| **SI agent transcripts** (Prompts 1–4/5: amino-acid and xylose competition, growth enhancement), which quote BPR scores (e.g. *Caulobacter* 3.17 ± 3.30) | Deployed agent → MCP | Deployed model | Minutes per prompt via the agent; LLM output not deterministic |
| Fig. 5 (*A. guillouiae* growth on an agent-selected carbon source) | Experimental; the recommendation came from the deployed agent at the time | Historical | Not regenerated; the experiment stands. Re-asking Prompt 4 could give a different recommendation |
| **Fig. S1** (genus × metabolite raw-score heatmap, 60 most-measured metabolites, Ward clustering) | `make_figures_for_manuscript_nature_microbiology_first_submission.py --figures 3` (`figure_3_heatmap_seaborn` → `figure3_heatmap_biclustering_{raw,distinctive,most-measured-raw,most-measured-distinctive}`). An alternate is `make_figure3_heatmap.py` | Saved model | est. 1–3 min |
| **Fig. S2** (rank clustermap: consumed metabolites concentrate at top ranks) | Same script, `--figures 6` (`figure6_rank_clustermap`) | Saved model | est. 1–2 min |
| Extended Data Fig. 3 (embedding distance by taxonomic level), if retained | Same script, `--figures 7` | Saved model | est. < 1 min |
| t-SNE of embeddings (`figure2_tsne`, older numbering), if retained | Same script, `--figures 2` | Saved model | est. < 1 min |
| `taxonomic_{raw_scores,distinctive_scores,ranks}.tsv` | Same script; exported on every run unless `--no-export-tsv` is passed (`export_tsv` defaults to True) | Saved model | est. < 1 min |
| **Fig. S3 + SI Table 2** (12 held-out isolates) | `scripts/revision_validation_12_isolates.py` → `table_S_validation_12_isolates.csv`; `scripts/revision_figS3_clean_holdout.py` → `figS3_validation_clean_holdout.{png,pdf}` | The saved model (`roc_auc` columns) plus 20 + 20 seeded retrains. The figure uses 20 clean retrains. Settings are hard-coded in `ORDINAL_BPR_KWARGS` (`lambda_ord` 0.1) | ≈ 4 min + ≈ 2.5 min (measured) |
| LOIO validation (`figure_loio_validation`, `loio_auc_results.csv`, in the first-submission folder, 2026-05-05), if used in the SI | `scripts/loio_validation.py` (untracked) | Retrains Ordinal-BPR (1000 epochs, `TRAIN_KWARGS`) once per isolate (433), plus one full model; resumes from a cache | est. 45–75 min (433 × 6–10 s) with `--force` |
| Submitted Fig. S3 (original) | First-submission script `--figures 8` | Retrains an unseeded Ordinal-BPR | Superseded by the revision scripts above |

**Exploratory scripts with no manuscript figure identified:**
- `scripts/taxonomic_group_analysis.py` and `scripts/diffusion_map_metabolic.py` load the saved model and write under `wom/figures/`.
- `scripts/phylogenetic_conservation_analysis.py` uses empirical data only.

**Not dependent on the current Ordinal-BPR:**
- `compare_with_without_gut_microbes.py` reads pre-Ordinal models in `manuscript_figures/trained_with(out)_gut_microbes/`.
- `benchmarking/prompt_tester.py` queries Gemini without WoM tools.
- `make_figures_for_manuscript.py` and `make_figures_for_manuscript_ordinal.py` are superseded pipelines. The latter loads `ordinal_bpr_consumed` for its 2026-03-18 `manuscript_figures_ordinal/` outputs.
- The unit tests in `wom/tests/` train throwaway models.

### C. If the λ=0 variant is adopted

**Code changes:**

| Where | Current setting | Change needed |
|---|---|---|
| `train_all_models.py:37` | `lambda_ord` 0.1 | Set it to 0, retrain, and commit the model files. Then rebuild and redeploy `wom-mcp` |
| `figure3_holdout_cv/holdout_cv.py` and `molecular_system_biology_fig3.py` | Use the `fit()` default (0.1) | Pass `lambda_ord=0` |
| `loio_validation.py` | `TRAIN_KWARGS` | Set `lambda_ord` to 0 |
| `revision_validation_12_isolates.py` and `revision_figS3_clean_holdout.py` | `ORDINAL_BPR_KWARGS` | Set `lambda_ord` to 0 |
| Saved-model consumers (Fig. 2, S1, S2, the TSVs, MCP) | Read the model files | Need only the retrained model files |

**Naming:**
- **Model directory and MCP name:** the directory `ordinal_bpr_*` and the MCP `model_type="ordinal_bpr"` would then hold a model with no ordinal term. Decide whether to rename them (this affects the MCP tool docs and `QueryEngine` path resolution) or keep the names and document the setting.

**Compute:**
- Retraining the two deployed models takes about 15 s.
- Fig. 2, S1, S2 and the TSVs take an est. 10–20 min.
- Fig. 3 takes 30–40 min, or 60–80 min for both versions.
- Fig. S3 + SI Table 2 take about 7 min.
- LOIO, if used, takes an est. 45–75 min.
- The MCP image rebuild takes an est. 10–20 min.
- **Total compute: roughly 1.5–3 h**, dominated by Fig. 3 and LOIO.
- **Additional manual work:** the Fig. 4 BPR values and the SI agent transcripts need re-querying, and the transcripts' LLM output is not deterministic.

**Overwrite risk:** the first-submission figure script writes into `manuscript_figures_nat_microbiology_first_submission/` by default. Its default `--figures` list
also includes Fig. 1 (from cache) and Fig. 8, which retrains a model. Run it with an explicit `--figures 2 3 6 7` and a new `--output-dir`
so the submitted figures are not overwritten. `make_figure_biology.py` has the same default output directory.

## Reproduce

```bash
cd wom
.venv/bin/python scripts/revision_fig1b_lambda0_sweep.py   # ~5.3 min; requires fig1b_benchmark_results.csv for the dim-32 check
```
