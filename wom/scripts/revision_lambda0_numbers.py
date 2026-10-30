# revision_lambda0_numbers.py
"""Old -> new values for every model-derived number quoted in the mSystems manuscript.

Reads the outputs of revision_lambda0_pipeline.sh (lambda_ord = 0 model) and the
corresponding submitted / lambda_ord = 0.1 outputs, and writes the tables of
lambda0/NUMBERS_FOR_MANUSCRIPT.md. "Quoted" is the value printed in the
manuscript; "old" is recomputed from the submitted model or results with the same
code as "new", so the two columns differ only by the model.

Usage:
    python scripts/revision_lambda0_numbers.py
"""

import ast
import sys
import json
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

REV = WOM_DIR / "manuscript_figures" / "msystems_revision"
SUBMITTED_FIG3_CSV = WOM_DIR / "scripts" / "molecular_system_biology_fig3_results.csv"
SUBMITTED_FIG1B_CSV = (WOM_DIR / "manuscript_figures_nat_microbiology_first_submission"
                       / "figure1_core_results_consumed_4models_ordinal.csv")
OLD_MODEL = WOM_DIR / "trained_models" / "archive_submitted_lambda0.1" / "ordinal_bpr_consumed"
NEW_MODEL = WOM_DIR / "trained_models" / "ordinal_bpr_consumed"
LEVELS = ["isolate", "genus", "family", "order"]

# Values as printed in the mSystems manuscript
QUOTED_FIG3 = {
    "isolate": {"n": 375, "bpr": 0.895, "base": 0.899, "groups": 375},
    "genus": {"n": 213, "bpr": 0.843, "base": 0.802, "groups": 70, "p": "< 0.001"},
    "family": {"n": 325, "bpr": 0.833, "base": 0.785, "groups": 48, "p": "< 0.001"},
    "order": {"n": 320, "bpr": 0.751, "base": 0.707, "groups": 25},
}
# Submitted Supplementary Fig. 3 scatter header (figure8 output of the submitted run)
QUOTED_S3_SCATTER = {"gcs": (0.33, 0.35, 10), "emb_div": (-0.04, 0.91, 10)}


def f3(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def fp(p):
    if p is None or np.isnan(p):
        return "–"
    if p < 1e-300:
        return "< 1e-300"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- Fig. 2
def fig2_section(out_dir: Path) -> tuple[str, dict]:
    old = json.loads((out_dir / "reference_lambda0.1" / "fig2" / "stats_figure_biology.json").read_text())
    new = json.loads((out_dir / "fig2" / "stats_figure_biology.json").read_text())
    old_div = pd.read_csv(out_dir / "reference_lambda0.1" / "fig2" / "stats_gcs_emb_diversity.csv").set_index("genus")
    new_div = pd.read_csv(out_dir / "fig2" / "stats_gcs_emb_diversity.csv").set_index("genus")
    po, pn = old["permanova_embedding"], new["permanova_embedding"]
    go, gn = old["gcs_vs_embedding_diversity"], new["gcs_vs_embedding_diversity"]
    rows = [
        ["Within-genus mean pairwise Spearman ρ, measured profiles (model-independent)", "0.44",
         f3(old["within_genus_empirical_spearman"]["mean"]), f3(new["within_genus_empirical_spearman"]["mean"])],
        ["Within-genus mean cosine similarity, BPR embeddings", "0.59",
         f3(old["within_genus_embedding_cosine"]["mean"]), f3(new["within_genus_embedding_cosine"]["mean"])],
        ["PERMANOVA R² (embedding distance, genus)", "0.69", f3(po["r2"]), f3(pn["r2"])],
        ["PERMANOVA pseudo-F", "10.7", f3(po["pseudo_f"], 2), f3(pn["pseudo_f"], 2)],
        ["PERMANOVA d.f.", "65, 315", f"{po['df'][0]}, {po['df'][1]}", f"{pn['df'][0]}, {pn['df'][1]}"],
        ["PERMANOVA p (999 permutations)", "0.001", fp(po["p"]), fp(pn["p"])],
        ["GCS vs embedding diversity, Spearman ρ", "−0.73", f3(go["spearman_rho"], 2), f3(gn["spearman_rho"], 2)],
        ["GCS vs embedding diversity, p", "< 10⁻⁸", fp(go["p"]), fp(gn["p"])],
        ["Number of genera (≥ 3 isolates)", "48", go["n_genera"], gn["n_genera"]],
        ["*Methylobacterium* within-genus cosine distance", "0.166",
         f3(old_div.loc["Methylobacterium", "emb_diversity"]), f3(new_div.loc["Methylobacterium", "emb_diversity"])],
        ["*Sphingomonas* within-genus cosine distance", "0.77",
         f3(old_div.loc["Sphingomonas", "emb_diversity"], 2), f3(new_div.loc["Sphingomonas", "emb_diversity"], 2)],
    ]
    extremes = []
    for label, d in [("old", old_div), ("new", new_div)]:
        lo, hi = d["emb_diversity"].idxmin(), d["emb_diversity"].idxmax()
        extremes.append([label, f"*{lo}* ({f3(d.loc[lo, 'emb_diversity'])})",
                         f"*{hi}* ({f3(d.loc[hi, 'emb_diversity'])})",
                         f"{f3(d.loc['Caulobacter', 'emb_diversity'])} (rank {int(d['emb_diversity'].rank(ascending=False)['Caulobacter'])} of {len(d)})"
                         if "Caulobacter" in d.index else "–"])
    text = ("### Fig. 2 / genus conservation (`make_figure_biology.py`)\n\n"
            + md_table(["Number", "Quoted", "Old (λ=0.1)", "New (λ=0)"], rows)
            + "\n\nRange of within-genus embedding diversity across the genera in Fig. 2b:\n\n"
            + md_table(["Model", "Lowest (most consistent)", "Highest (most diverse)",
                        "*Caulobacter* (rank, 1 = most diverse)"], extremes))
    return text, {"old": old, "new": new}


# ----------------------------------------------------------------------------- Fig. 3
def fig3_summary(df: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon
    out = {}
    for L in LEVELS:
        sub = df[df.level == L]
        paired = sub.dropna(subset=["auc_bpr_scoreable", "auc_baseline_scoreable"])
        d = paired.auc_bpr_scoreable - paired.auc_baseline_scoreable
        out[L] = {"n_paired": len(paired), "n_groups": sub.group.nunique(),
                  "median_auc_bpr_scoreable": paired.auc_bpr_scoreable.median(),
                  "median_auc_baseline_scoreable": paired.auc_baseline_scoreable.median(),
                  "median_delta": d.median(),
                  "wilcoxon_p": wilcoxon(paired.auc_bpr_scoreable, paired.auc_baseline_scoreable).pvalue,
                  "median_auc_bpr_all": sub.auc_bpr_all.median()}
    return out


def fig3_section(out_dir: Path) -> tuple[str, dict]:
    old = fig3_summary(pd.read_csv(SUBMITTED_FIG3_CSV))
    new = json.loads((out_dir / "fig3" / "fig3_holdout_summary.json").read_text())
    rows = []
    for L in LEVELS:
        q, o, n = QUOTED_FIG3[L], old[L], new[L]
        rows += [
            [L, "n evaluable isolates", q["n"], o["n_paired"], n["n_paired"]],
            [L, "n held-out groups (coverage panel)", q["groups"], o["n_groups"], n["n_groups"]],
            [L, "median AUC, BPR (scoreable)", f3(q["bpr"]), f3(o["median_auc_bpr_scoreable"]), f3(n["median_auc_bpr_scoreable"])],
            [L, "median AUC, naive baseline", f3(q["base"]), f3(o["median_auc_baseline_scoreable"]), f3(n["median_auc_baseline_scoreable"])],
            [L, "median Δ (BPR − baseline)", "(figure)", f"{o['median_delta']:+.3f}", f"{n['median_delta']:+.3f}"],
            [L, "paired Wilcoxon p", q.get("p", "(figure)"), fp(o["wilcoxon_p"]), fp(n["wilcoxon_p"])],
            [L, "median AUC, BPR (all metabolites)", "–", f3(o["median_auc_bpr_all"]), f3(n["median_auc_bpr_all"])],
        ]
    abstract = [["Abstract: median ROC-AUC, isolate → order (all metabolites)", "0.90 → 0.76",
                 f"{f3(old['isolate']['median_auc_bpr_all'], 2)} → {f3(old['order']['median_auc_bpr_all'], 2)}",
                 f"{f3(new['isolate']['median_auc_bpr_all'], 2)} → {f3(new['order']['median_auc_bpr_all'], 2)}"],
                ["Same, scoreable metabolites (Fig. 3a)", "–",
                 f"{f3(old['isolate']['median_auc_bpr_scoreable'], 2)} → {f3(old['order']['median_auc_bpr_scoreable'], 2)}",
                 f"{f3(new['isolate']['median_auc_bpr_scoreable'], 2)} → {f3(new['order']['median_auc_bpr_scoreable'], 2)}"]]
    text = ("### Fig. 3 / hierarchical hold-out (`molecular_system_biology_fig3.py`)\n\n"
            "Old = the submitted results (`scripts/molecular_system_biology_fig3_results.csv`, λ=0.1), "
            "summarised with the same code as new.\n\n"
            + md_table(["Level", "Number", "Quoted", "Old (λ=0.1)", "New (λ=0)"], rows)
            + "\n\n" + md_table(["Number", "Quoted", "Old (λ=0.1)", "New (λ=0)"], abstract))
    return text, {"old": old, "new": new}


# ----------------------------------------------------------------------------- Fig. S3
def s3_scatter(table: pd.DataFrame, model_dir: Path, gcs_csv: Path) -> dict:
    """AUC vs GCS and vs embedding diversity across reference taxa, as in the submitted figure."""
    from core.io import load_model_artifacts
    from loaders.loaders import load_exometabolite
    import scripts.make_figures_for_manuscript_nature_microbiology_first_submission as fs3
    art = load_model_artifacts(model_dir)
    emp = load_exometabolite()
    gcs_lookup = pd.read_csv(gcs_csv).set_index("genus")["gcs"].to_dict()
    rows = []
    for _, r in table.iterrows():
        level, taxon = r["clean_holdout_reference_rank_used"], r[r["clean_holdout_reference_rank_used"]]
        emb, _ = fs3._compute_taxon_embedding_diversity_fig8(art.model, art.training_data, taxon, level,
                                                              min_isolates=2)
        if level == "genus" and taxon in gcs_lookup:
            gcs = gcs_lookup[taxon]
        else:
            gcs, _ = fs3._compute_taxon_empirical_consistency_fig8(taxon, level, emp, min_isolates=2)
        rows.append({"isolate": r["isolate_id"], "taxon": taxon, "auc": r["clean_holdout_roc_auc_mean"],
                     "gcs": gcs, "emb_div": emb})
    per_isolate = pd.DataFrame(rows)
    per_taxon = per_isolate.groupby("taxon", as_index=False).mean(numeric_only=True)
    out = {"per_taxon": per_taxon}
    for col in ["gcs", "emb_div"]:
        ok = per_taxon[col].notna()
        res = spearmanr(per_taxon.loc[ok, "auc"], per_taxon.loc[ok, col])
        out[col] = (float(res.statistic), float(res.pvalue), int(ok.sum()))
    return out


def s3_section(out_dir: Path) -> tuple[str, dict]:
    ref = out_dir / "reference_lambda0.1" / "figS3"
    old_sum = pd.read_csv(ref / "validation_12_isolates_summary.csv").set_index(["model_variant", "subset"])
    new_sum = pd.read_csv(out_dir / "figS3" / "validation_12_isolates_summary.csv").set_index(["model_variant", "subset"])
    old_tab = pd.read_csv(ref / "table_S_validation_12_isolates.csv")
    new_tab = pd.read_csv(out_dir / "figS3" / "table_S_validation_12_isolates.csv")
    key = ("clean_holdout_corrected_labels", "all_12")
    o, n = old_sum.loc[key], new_sum.loc[key]

    def n_ge(tab):
        return int((tab["clean_holdout_roc_auc_mean"] >= 0.8).sum())

    def caul(tab):
        r = tab.set_index("genus").loc["Caulobacter"]
        return f"{r['clean_holdout_roc_auc_mean']:.3f} ± {r['clean_holdout_roc_auc_sd']:.3f}"

    def lowest(tab):
        r = tab.loc[tab["clean_holdout_roc_auc_mean"].idxmin()]
        return f"{r['genus']} ({r['clean_holdout_roc_auc_mean']:.3f})"

    rows = [
        ["Mean ROC-AUC (12 isolates; mean over 20 seeds)", "0.87", f3(o["mean_roc_auc"]), f3(n["mean_roc_auc"])],
        ["SD of the mean across seeds", "–", f3(o["mean_roc_auc_sd_across_seeds"]), f3(n["mean_roc_auc_sd_across_seeds"])],
        ["Range of the mean across seeds", "–",
         f"{f3(o['mean_roc_auc_min_across_seeds'])}–{f3(o['mean_roc_auc_max_across_seeds'])}",
         f"{f3(n['mean_roc_auc_min_across_seeds'])}–{f3(n['mean_roc_auc_max_across_seeds'])}"],
        ["Median ROC-AUC (per-seed median, averaged)", "–", f3(o["median_roc_auc"]), f3(n["median_roc_auc"])],
        ["Isolates ≥ 0.80 (per-isolate 20-seed mean)", "11 of 12", f"{n_ge(old_tab)} of 12", f"{n_ge(new_tab)} of 12"],
        ["Isolates ≥ 0.80 (average over seeds)", "–", f3(o["n_roc_auc_ge_0.8"], 1), f3(n["n_roc_auc_ge_0.8"], 1)],
        ["*Caulobacter* ROC-AUC (mean ± SD, 20 seeds)", "0.77", caul(old_tab), caul(new_tab)],
        ["Lowest isolate", "*Caulobacter* (0.77)", lowest(old_tab), lowest(new_tab)],
    ]
    per = old_tab[["isolate_id", "genus", "clean_holdout_roc_auc_mean", "clean_holdout_roc_auc_sd"]].merge(
        new_tab[["isolate_id", "clean_holdout_roc_auc_mean", "clean_holdout_roc_auc_sd"]],
        on="isolate_id", suffixes=("_old", "_new"))
    per_rows = [[r["genus"], r["isolate_id"],
                 f"{r['clean_holdout_roc_auc_mean_old']:.3f} ± {r['clean_holdout_roc_auc_sd_old']:.3f}",
                 f"{r['clean_holdout_roc_auc_mean_new']:.3f} ± {r['clean_holdout_roc_auc_sd_new']:.3f}",
                 f"{r['clean_holdout_roc_auc_mean_new'] - r['clean_holdout_roc_auc_mean_old']:+.3f}"]
                for _, r in per.iterrows()]

    sc_old = s3_scatter(old_tab, OLD_MODEL, out_dir / "reference_lambda0.1" / "fig2" / "stats_gcs_emb_diversity.csv")
    sc_new = s3_scatter(new_tab, NEW_MODEL, out_dir / "fig2" / "stats_gcs_emb_diversity.csv")
    scatter_rows = []
    for col, label in [("gcs", "AUC vs GCS"), ("emb_div", "AUC vs embedding diversity")]:
        q = QUOTED_S3_SCATTER[col]
        scatter_rows.append([f"{label}: Spearman ρ, p (n taxa)", f"{q[0]:.2f}, p={q[1]:.2f} ({q[2]})",
                             f"{sc_old[col][0]:.2f}, p={sc_old[col][1]:.2f} ({sc_old[col][2]})",
                             f"{sc_new[col][0]:.2f}, p={sc_new[col][1]:.2f} ({sc_new[col][2]})"])
    def extremes(per_taxon):
        def pick(col, fn):
            sub = per_taxon.dropna(subset=[col])
            return sub.loc[getattr(sub[col], fn)(), 'taxon']
        return f"{pick('gcs', 'idxmin')} / {pick('emb_div', 'idxmax')} / {pick('auc', 'idxmin')}"

    scatter_rows.append(["Taxon with lowest GCS / highest embedding diversity / lowest AUC",
                         "*Caulobacter* (all three)", extremes(sc_old['per_taxon']),
                         extremes(sc_new['per_taxon'])])

    text = ("### Supplementary Fig. 3 + SI Table 2 (12 held-out isolates; clean hold-out, 20 seeds)\n\n"
            "Old = λ=0.1 with the archived submitted model and the same clean hold-out and code as new "
            "(`reference_lambda0.1/figS3/`; GW456-R20, FW507-4G11 and MT049 = MT49 removed from training). The quoted "
            "0.87 / 11 of 12 / 0.77 come from the submitted single unseeded run, which also included the "
            "in-training isolates (see `validation_12_isolates_REPORT.md`).\n\n"
            + md_table(["Number", "Quoted", "Old (λ=0.1)", "New (λ=0)"], rows)
            + "\n\nPer isolate (clean hold-out ROC-AUC, mean ± SD over 20 seeds):\n\n"
            + md_table(["Genus", "Isolate", "Old (λ=0.1)", "New (λ=0)", "Δ"], per_rows)
            + "\n\nSupplementary Fig. 3 scatter text (\"*Caulobacter* sits at the extremes … the trend is not "
              "statistically significant\"). Per reference taxon, AUC = clean 20-seed mean; GCS and embedding "
              "diversity as in the submitted figure code, with the saved model of each λ. Quoted = submitted "
              "figure header (single run including the in-training isolates):\n\n"
            + md_table(["Number", "Quoted", "Old (λ=0.1)", "New (λ=0)"], scatter_rows))
    return text, {"old": o, "new": n}


# ----------------------------------------------------------------------------- Fig. 1b
def fig1b_section(out_dir: Path) -> str:
    sub = pd.read_csv(SUBMITTED_FIG1B_CSV)
    sub32 = sub[sub.dim == 32].groupby("method")["auc"].agg(["mean", "std"])
    new = pd.read_csv(out_dir / "fig1b" / "fig1b_main_data.csv")
    new32 = new[new.dim == 32].set_index("model")
    old_label = {"BPR with measured-neutral negatives": "Ordinal-BPR (trained without known negatives, i.e. plain BPR)",
                 "BPR": None, "LMF": "LMF", "WR-MF": "WR-MF", "SVD": "SVD"}
    rows = []
    for model, r in new32.iterrows():
        o = old_label[model]
        old_val = f"{sub32.loc['Ordinal-BPR' if o and o.startswith('Ordinal') else o, 'mean']:.3f} ± " \
                  f"{sub32.loc['Ordinal-BPR' if o and o.startswith('Ordinal') else o, 'std']:.3f}" if o else "– (not in submitted figure)"
        rows.append([model, old_val, f"{r['mean']:.3f} ± {r['sd']:.3f}"])
    lam = new[new.model == "BPR with measured-neutral negatives"]
    rng = [["Abstract: \"ROC-AUC = 0.94\", robust to dimensionality",
            "0.94 (leave-one-out, mixed pool)",
            f"{lam['mean'].min():.3f}–{lam['mean'].max():.3f} across dims 4–128 "
            f"(peak {lam.loc[lam['mean'].idxmax(), 'mean']:.3f} at dim {int(lam.loc[lam['mean'].idxmax(), 'dim'])}); "
            f"{new32.loc['BPR with measured-neutral negatives', 'mean']:.3f} at dim 32"]]
    return ("### Fig. 1b (dim 32)\n\n"
            "Old = submitted Fig. 1b (leave-one-out vs all non-positive compounds; 'Ordinal-BPR' was trained "
            "without known negatives). New = strict hold-out per-microbe ROC-AUC (`fig1b/fig1b_main_data.csv`). "
            "The metric changed, so old and new are not directly comparable.\n\n"
            + md_table(["Model (new label)", "Old: submitted Fig. 1b", "New: strict hold-out"], rows)
            + "\n\n" + md_table(["Number", "Old", "New (λ=0)"], rng))


# ----------------------------------------------------------------------------- SI tables
def fmt_rho(value: str) -> str:
    """Format a stored {'rho': .., 'p': ..} string as 'ρ = x (p = y)'."""
    d = ast.literal_eval(value)
    return f"ρ = {d['rho']:.2f} (p = {d['p']:.3f})"


def si_section(out_dir: Path) -> str:
    d = out_dir / "si_tables"
    scores = pd.read_csv(d / "si_bpr_scores_old_vs_new.csv")
    genera = pd.read_csv(d / "si_prompt2_genus_old_vs_new.csv")
    ranks = pd.read_csv(d / "si_gemini_pseudomonas_bpr_rank_old_vs_new.csv")
    rank_stats = pd.read_csv(d / "si_gemini_pseudomonas_rank_stats.csv", index_col=0)["value"]

    def sd(v):
        return "" if pd.isna(v) else f" ± {v:.2f}"

    score_rows = [[r["table"], r["genus"], r["substrate"],
                   f"{r['quoted_score']:.2f}" + ("" if pd.isna(r["quoted_sd"]) else f" ± {r['quoted_sd']:.2f}"),
                   "yes" if r["old_reproduces_quote"] else "**no**",
                   f"{r['new_score']:.2f}{sd(r['new_sd'])}", f"{r['delta_score']:+.2f}"]
                  for _, r in scores.iterrows()]
    genus_rows = [[r["genus"], r["n_isolates"],
                   "–" if pd.isna(r["quoted_embedding_diversity"]) else f"{r['quoted_embedding_diversity']:.2f}",
                   "–" if pd.isna(r["new_embedding_diversity"]) else f"{r['new_embedding_diversity']:.2f}",
                   "yes" if r["old_top3_matches_quote"] else "no",
                   "unchanged" if r["new_top3_same_as_quote"] else r["new_model_top3"]]
                  for _, r in genera.iterrows()]
    rank_rows = [[r["metabolite"], r["experimental_timepoint"], f"{r['tsv_bpr_score']:.2f}", int(r["tsv_bpr_rank"]),
                  "yes" if r["old_rank_reproduces_tsv"] else "**no**", f"{r['new_score']:.2f}", int(r["new_rank"])]
                 for _, r in ranks.iterrows()]
    n_ok = int(scores["old_reproduces_quote"].sum())
    return ("### SI tables: BPR scores and ranks quoted from the WoM agent\n\n"
            f"The archived submitted model reproduces {n_ok} of {len(scores)} quoted scores (± SD) to the printed "
            f"precision, and {int(ranks['old_rank_reproduces_tsv'].sum())} of {len(ranks)} `bpr rank` values. "
            "New = genus mean ± isolate SD with the λ=0 model, same compounds (the agent's fuzzy matches "
            "Mannose → D-glucose and Arabinose → D-xylose are kept). Full values: `si_tables/`.\n\n"
            + md_table(["SI table", "Genus", "Substrate", "Quoted", "Old reproduced", "New (λ=0)", "Δ"], score_rows)
            + "\n\nPrompt 2 genus columns. \"Top-3\" is the genus's three highest raw-score metabolites over all "
              "metabolites, which is how the quoted top-3 were chosen if the old model reproduces them:\n\n"
            + md_table(["Genus", "Isolates", "Embedding diversity, quoted", "Embedding diversity, new",
                        "Old model reproduces quoted top-3", "New top-3"], genus_rows)
            + "\n\n`Supplementary Tables - GeminiRankingPseudomonas.tsv`, `bpr rank` column (competition rank "
              "among the 17 metabolites depleted within 12 h; the six never-depleted rows carry a placeholder "
              "rank of 16 and are not plotted):\n\n"
            + md_table(["Metabolite", "Depletion time (h)", "Quoted score", "Quoted rank", "Old reproduced",
                        "New score", "New rank"], rank_rows)
            + "\n\nSpearman correlation of BPR score with depletion time (17 metabolites; Supplementary Fig. 4b): "
              f"old {fmt_rho(rank_stats['spearman_score_vs_depletion_time_old'])}, "
              f"new {fmt_rho(rank_stats['spearman_score_vs_depletion_time_new'])}.")


def main(out_dir: Path):
    parts = ["## Tables (generated by `scripts/revision_lambda0_numbers.py`)"]
    parts.append(fig1b_section(out_dir))
    text, _ = fig2_section(out_dir); parts.append(text)
    text, _ = fig3_section(out_dir); parts.append(text)
    text, _ = s3_section(out_dir); parts.append(text)
    parts.append(si_section(out_dir))
    path = out_dir / "numbers_tables.md"
    path.write_text("\n\n".join(parts) + "\n")
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Old -> new manuscript numbers for the lambda_ord = 0 model")
    parser.add_argument("--output-dir", type=Path, default=REV / "lambda0")
    args = parser.parse_args()
    main(args.output_dir)
