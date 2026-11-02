#!/usr/bin/env bash
# revision_lambda0_pipeline.sh
# Regenerate every model-derived figure/table of the mSystems revision with the
# adopted model (Ordinal-BPR, lambda_ord = 0). Run from wom/:
#     bash scripts/revision_lambda0_pipeline.sh
#
# All outputs go under manuscript_figures/msystems_revision/lambda0/; every step
# passes an explicit --output-dir (and --figures where the script has defaults
# that would write elsewhere), so submitted outputs are never touched.
# Resumable: a finished step leaves lambda0/.pipeline/<step>.done and is skipped
# on rerun; Fig. 3 also caches per hold-out group. Delete a .done file to rerun
# that step. Step logs go to lambda0/logs/, runtimes to lambda0/pipeline_runtime.tsv.
#
# Prerequisite: trained_models/ordinal_bpr_{consumed,secreted} retrained with
#     python scripts/train_all_models.py --models ordinal_bpr --seed 0
set -euo pipefail

PY=.venv/bin/python
OUT=manuscript_figures/msystems_revision/lambda0
NEW_MODEL=trained_models/ordinal_bpr_consumed
OLD_MODEL=trained_models/archive_submitted_lambda0.1/ordinal_bpr_consumed

mkdir -p "$OUT/.pipeline" "$OUT/logs"
[ -f "$OUT/pipeline_runtime.tsv" ] || printf "step\tseconds\tfinished\n" > "$OUT/pipeline_runtime.tsv"

grep -q '"lambda_ord": 0.0' "$NEW_MODEL/training_config.json" || {
    echo "ERROR: $NEW_MODEL is not the lambda_ord = 0 model" >&2; exit 1; }

step() {
    local name=$1; shift
    if [ -f "$OUT/.pipeline/$name.done" ]; then
        echo "[skip] $name (done)"
        return
    fi
    echo "[run ] $name"
    local start=$(date +%s)
    "$@" > "$OUT/logs/$name.log" 2>&1
    local secs=$(( $(date +%s) - start ))
    touch "$OUT/.pipeline/$name.done"
    printf "%s\t%s\t%s\n" "$name" "$secs" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT/pipeline_runtime.tsv"
    echo "[done] $name (${secs}s)"
}

# Fig. 1b benchmark with the revised-Methods pair labelling (BPR = lambda_ord 0, the
# submitted comparison models, plain BPR and the submitted Ordinal-BPR), then the
# main-text panel (strict hold-out)
step fig1b_benchmark $PY scripts/revision_fig1b_benchmark.py --label-rule methods --skip-submitted-checks \
                    --models "Ordinal-BPR [lambda_ord=0]" WR-MF SVD LMF BPR Ordinal-BPR \
                    --output-dir "$OUT/fig1b"
step fig1b      $PY scripts/revision_fig1b_main.py --output-dir "$OUT/fig1b"

# BPR scores / ranks quoted in the SI (no LLM prompts re-run)
step si_scores  $PY scripts/revision_lambda0_si_scores.py --output-dir "$OUT/si_tables"

# Fig. 2 and Supplementary Fig. 1 (GCS ranking), new model; and the same code on
# the archived submitted model, for old values that were never saved to file
step fig2       $PY scripts/make_figure_biology.py --model-dir "$NEW_MODEL" --output-dir "$OUT/fig2"
step fig2_ref   $PY scripts/make_figure_biology.py --model-dir "$OLD_MODEL" \
                    --output-dir "$OUT/reference_lambda0.1/fig2"

# Supplementary Fig. 2 (genus x metabolite heatmaps) and the other saved-model SI
# figures (t-SNE, rank clustermap, embedding distance), plus taxonomic_*.tsv
step si_figures $PY scripts/make_figures_for_manuscript_nature_microbiology_first_submission.py \
                    --figures 2 3 6 7 --model-dir "$NEW_MODEL" --output-dir "$OUT/si_figures"

# Supplementary Fig. 2 as submitted (portrait genus x metabolite heatmap, make_figure3_heatmap.py)
step si_fig2_heatmap $PY scripts/make_figure3_heatmap.py --model-dir "$NEW_MODEL" \
                    --output-dir "$OUT/si_figures" --figure-name figS2_genus_heatmap

# Supplementary Fig. 3 + SI Table 2: 12 held-out isolates, clean hold-out, 20 seeds
step figS3_table  $PY scripts/revision_validation_12_isolates.py --lambda-ord 0 \
                      --model-dir "$NEW_MODEL" --output-dir "$OUT/figS3"
step figS3_figure $PY scripts/revision_figS3_clean_holdout.py --lambda-ord 0 \
                      --output-dir "$OUT/figS3" --table-csv "$OUT/figS3/table_S_validation_12_isolates.csv"
# same analysis with the archived submitted model (lambda_ord = 0.1) and the same
# clean hold-out, for old values computed by identical code
step figS3_table_ref $PY scripts/revision_validation_12_isolates.py --lambda-ord 0.1 \
                      --model-dir "$OLD_MODEL" --output-dir "$OUT/reference_lambda0.1/figS3"

# Fig. 3: taxonomic hold-out CV (script that produced the submitted Fig. 3)
step fig3       $PY scripts/molecular_system_biology_fig3.py --lambda-ord 0 \
                    --results-csv "$OUT/fig3/fig3_holdout_results.csv" --fig-stem "$OUT/fig3/fig3_holdout"

# Reviewer 1: same hierarchical hold-out with PR-AUC (average precision) columns added;
# the ROC-AUC columns must reproduce fig3/ (seeded), checked by revision_r1_pr_calibration.py
step fig3_pr    $PY scripts/molecular_system_biology_fig3.py --lambda-ord 0 \
                    --results-csv "$OUT/r1/fig3_pr/fig3_holdout_results.csv" --fig-stem "$OUT/r1/fig3_pr/fig3_holdout"

# Reviewer 1, comment 6: learning curve on the Fig. 1b hold-out (fraction 1 reproduces Fig. 1b)
step r1_learning_curve $PY scripts/revision_r1_learning_curve.py --output-dir "$OUT/r1/learning_curve" \
                    --fig1b-results "$OUT/fig1b/fig1b_benchmark_results.csv"

# Reviewer 1, comment 4: Figure 1c schematic (what is embedded and how scores are formed)
step r1_fig1c   $PY scripts/revision_fig1c_schematic.py --output-dir "$OUT/r1/fig1c"

# Reviewer 1, comments 2, 3 and 5: PR-AUC, learning-free baselines, calibration
# (Supplementary Figs. 6 and 7); needs fig1b and fig3_pr
step r1_pr_calibration $PY scripts/revision_r1_pr_calibration.py --lambda0-dir "$OUT"

echo "Pipeline complete; runtimes in $OUT/pipeline_runtime.tsv"
