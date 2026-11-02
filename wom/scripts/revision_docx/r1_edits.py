# r1_edits.py
"""Reviewer 1 comments 2-6 in the revision documents (mSystems revision).

Applies, as tracked changes (author Benjamin Bowen), the text for PR-AUC (comment 2),
calibration (3), the Fig. 1c schematic (4), what BPR and higher dimensions buy (5) and
the learning curve (6) to the manuscript and SI; adds Figure 1c and Supplementary
Figs. 6-8; adds a PR-AUC column to SI Table 2 and data sheets for Supplementary Figs.
6-8; writes the letter responses with Original/Modified pairs; fills the remaining
"Lines [__]" placeholders; and makes two small SI corrections found on the way (the
metabolite filter, R3 minor 2; the consumed label in the Fig. S3 caption,
R3 minor 3). Numbers are read from manuscript_figures/msystems_revision/lambda0/r1/.

Usage:
    python r1_edits.py --preview-dir /tmp/r1_preview
    python r1_edits.py --apply
"""

import io
import re
import sys
import json
import shutil
import zipfile
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import TrackedDocx, LetterDocx, q  # noqa: E402
from r3_major1_edits import (  # noqa: E402
    REPO, DOCS, MS, SI, LETTER, LINE_PDF, BACKUPS, AUTHOR, TABLES, LineNumbers, sheet_data_xml,
)

OUT = REPO / "wom/manuscript_figures/msystems_revision/lambda0"
R1 = OUT / "r1"
PRC = R1 / "pr_calibration"
LC = R1 / "learning_curve"
FIG1C = R1 / "fig1c" / "fig1c_schematic.png"
NEW_FIGS = [("image6.png", PRC / "figS6_pr_auc.png"), ("image7.png", PRC / "figS7_calibration.png"),
            ("image8.png", LC / "figS8_learning_curve.png")]
BPR = "Ordinal-BPR [lambda_ord=0]"
FIG1_HEIGHT_BEFORE = 783  # Figure 1 image (panels a and b) before panel c is added


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{x:.0f}%"


# ----------------------------------------------------------------------------- numbers
def numbers() -> dict:
    s = json.loads((PRC / "r1_pr_calibration_summary.json").read_text())
    dims = pd.DataFrame(s["fig1b_by_dim"]).set_index(["model", "dim"])
    base = pd.DataFrame(s["fig1b_baselines"]).set_index("baseline")
    cal = pd.DataFrame(s["calibration"]).set_index("method")
    lc = pd.read_csv(LC / "learning_curve_summary.csv").set_index("fraction")
    bins = pd.read_csv(PRC / "calibration_bins.csv")
    platt = bins[bins.method == "platt"].set_index("bin")
    pooled = pd.read_csv(OUT / "fig1b/fig1b_benchmark_results.csv")
    pooled = pooled[(pooled.model == BPR) & (pooled.protocol == "B_holdout") & (pooled.dim == 32)]

    bpr = dims.loc[BPR]
    best_dim = int(bpr.roc_auc_mean.idxmax())
    near = bpr[(bpr.roc_auc_mean >= bpr.roc_auc_mean.max() - 0.0125)].index
    wr = dims.loc["WR-MF"]
    others128 = [dims.loc[(m, 128), "roc_auc_mean"] for m in ["WR-MF", "SVD", "LMF"]]
    low = [dims.loc[(m, d), "roc_auc_mean"] for m in ["WR-MF", "SVD", "LMF", BPR] for d in (4, 8)]
    # learning curve: BPR above both baselines from `cross_hi` (and below at `cross_lo`)
    above = (lc.roc_auc_mean > lc.genus_roc_auc_mean) & (lc.roc_auc_mean > lc.popularity_roc_auc_mean)
    first = above.idxmax() if above.any() else None
    assert first is not None and above.loc[first:].all(), "BPR does not stay above the baselines"
    below = lc.index[lc.index < first].max()
    cal_ok = lc[(lc.cal_ols_slope_mean - 1).abs() <= 0.1].index.min()
    assert (best_dim, sorted(near)) == (16, [8, 16, 32, 64]), (best_dim, sorted(near))
    assert wr.loc[[4, 8], "pr_auc_mean"].max() > bpr.loc[32, "pr_auc_mean"], "WR-MF PR-AUC note no longer applies"
    return {
        "s": s, "dims": dims, "base": base, "cal": cal, "lc": lc, "platt": platt,
        "prev_pooled": float(pooled.prevalence.mean()), "prev_iso": float(bpr.loc[32, "prevalence_mean"]),
        "bpr32_roc": bpr.loc[32, "roc_auc_mean"], "bpr32_pr": bpr.loc[32, "pr_auc_mean"],
        "bpr16_roc": bpr.loc[16, "roc_auc_mean"], "bpr128_roc": bpr.loc[128, "roc_auc_mean"],
        "near_gap": bpr.roc_auc_mean.max() - bpr.loc[list(near), "roc_auc_mean"].min(),
        "wr_low_pr": (wr.loc[4, "pr_auc_mean"], wr.loc[8, "pr_auc_mean"]),
        "others128": (min(others128), max(others128)), "low_range": (min(low), max(low)),
        "lc_first": first, "lc_below": below, "lc_first_pct": lc.loc[first, "pct_of_matrix_mean"],
        "lc_below_pct": lc.loc[below, "pct_of_matrix_mean"], "cal_ok_pct": lc.loc[cal_ok, "pct_of_matrix_mean"],
    }


# ----------------------------------------------------------------------------- text
def texts(n: dict) -> dict:
    base, cal, lc, f3s, s3 = n["base"], n["cal"], n["lc"], n["s"]["fig3"], n["s"]["figS3"]
    b30 = n["platt"].loc[3]
    full, half = lc.loc[1.0], lc.loc[0.5]
    t = {}
    t["results_pr"] = (
        f" Because consumed pairs are the minority class ({pct(100 * n['prev_pooled'])} of withheld pairs), we also "
        f"computed the area under the precision–recall curve (PR-AUC), for which a random ranking scores the prevalence "
        f"of consumed pairs ({f2(n['prev_iso'])} averaged over isolates; Fig. S6a,b). BPR reached "
        f"PR-AUC = {f2(n['bpr32_pr'])} at 32 dimensions; by this measure WR-MF performed similarly at 4 and 8 "
        f"dimensions ({f2(n['wr_low_pr'][0])} and {f2(n['wr_low_pr'][1])}) but again declined at higher dimensions. "
        f"Larger embeddings did not make BPR more accurate: its ROC-AUC was highest at 16 dimensions "
        f"({f3(n['bpr16_roc'])}) and within {n['near_gap']:.3f} of this from 8 to 64 dimensions. Two learning-free "
        f"baselines evaluated on the same withheld pairs, the fraction of tested isolates that consumed each "
        f"metabolite and the same fraction within the isolate's genus, reached ROC-AUC = "
        f"{f2(base.loc['popularity', 'roc_auc_mean'])} and {f2(base.loc['genus', 'roc_auc_mean'])} (PR-AUC = "
        f"{f2(base.loc['popularity', 'pr_auc_mean'])} and {f2(base.loc['genus', 'pr_auc_mean'])}), so BPR's advantage "
        f"over simple prevalence is consistent but modest.")
    t["results_cal_lc"] = (
        f" BPR scores are ranking statistics, but a monotone calibration (Platt scaling) fitted on the withheld pairs "
        f"of other isolates converted them into well-calibrated probabilities of consumption (slope of the observed "
        f"outcome on the predicted probability = {f2(cal.loc['platt', 'ols_slope_mean'])}; Fig. S7). "
        f"In a learning curve on the same withheld pairs, accuracy increased steadily with the number of training "
        f"pairs and had not reached a plateau (ROC-AUC = {f2(half.roc_auc_mean)} with half and "
        f"{f2(full.roc_auc_mean)} with all training pairs), and BPR's ROC-AUC exceeded that of the learning-free "
        f"baselines only with {pct(n['lc_first_pct'])} or more of the isolate × metabolite matrix in training (it was "
        f"lower with {pct(n['lc_below_pct'])}; Fig. S8).")
    iso, gen, fam, order = (f3s[k] for k in ["isolate", "genus", "family", "order"])
    t["results_fig3"] = (
        f" PR-AUC led to the same conclusions (Fig. S6c): on the compounds both methods could score, "
        f"median PR-AUC was {f2(iso['median_ap_bpr_scoreable'])} for BPR and "
        f"{f2(iso['median_ap_baseline_scoreable'])} for the baseline at the isolate level, and "
        f"{f2(gen['median_ap_bpr_scoreable'])} vs. {f2(gen['median_ap_baseline_scoreable'])}, "
        f"{f2(fam['median_ap_bpr_scoreable'])} vs. {f2(fam['median_ap_baseline_scoreable'])}, and "
        f"{f2(order['median_ap_bpr_scoreable'])} vs. {f2(order['median_ap_baseline_scoreable'])} at the genus, "
        f"family, and order levels, compared with median prevalences of consumed metabolites of "
        f"{f2(min(f3s[k]['median_prevalence_scoreable'] for k in f3s))}–"
        f"{f2(max(f3s[k]['median_prevalence_scoreable'] for k in f3s))}.")
    t["results_s3"] = (
        f" Because most metabolites measured for these isolates were consumed (mean prevalence "
        f"{f2(s3['mean_prevalence'])}), PR-AUC is high even for a random ranking; BPR's mean PR-AUC was "
        f"{f2(s3['mean_pr_auc'])} and exceeded the prevalence for "
        + ("all 12 isolates" if s3["n_pr_auc_above_prevalence"] == 12 else f"{s3['n_pr_auc_above_prevalence']} of the 12 isolates")
        + " (SI Table 2; Fig. S6d).")
    t["methods_fig1c"] = "; Fig. 1c"
    t["methods_cal"] = (" Where a probability is needed, scores can be mapped to probabilities of consumption with a "
                        "monotone calibration fitted on held-out pairs (Fig. S7).")
    t["methods_pr"] = (" We also report the area under the precision–recall curve (PR-AUC, computed as average "
                       "precision), whose value for a random ranking equals the isolate's fraction of consumed "
                       "metabolites (prevalence).")
    t["discussion"] = (
        f" The pair-level benchmark showed the same pattern: learning-free prevalence baselines came within "
        f"{n['bpr32_roc'] - base.loc['genus', 'roc_auc_mean']:.2f}–{n['bpr32_roc'] - base.loc['popularity', 'roc_auc_mean']:.2f} "
        f"ROC-AUC of BPR when most of each isolate's profile was available for training, and outperformed it when "
        f"fewer pairs had been measured (Figs. S6 and S8). For coarse, genus-level questions or sparsely "
        f"profiled panels, such baselines are therefore a reasonable first approximation, and the advantage of BPR "
        f"lies in its complete, strain-level coverage.")
    t["caption_c"] = (
        " (c) Schematic of the BPR model. The only input is the isolate × metabolite assertion matrix (here the toy "
        "matrix of panel a); no taxonomic, chemical, or medium features are used. BPR learns a 32-value embedding for "
        "every isolate (u) and every metabolite (v) from sampled triples (i, j, k), in which isolate i depleted "
        "metabolite j and k is a measured-neutral (with probability 0.5) or an unobserved metabolite, by increasing "
        "the score u_i · v_j relative to u_i · v_k. The score of every isolate–metabolite pair is the dot product of "
        "the two embeddings, which ranks all metabolites for each isolate, including unobserved ones. Embedding values "
        "and scores in panel c are illustrative.")
    t["b30"] = b30
    return t


SI_METHODS = [
    "For the hold-out benchmark (Fig. 1b), the hierarchical hold-out (Fig. 3), and the 12 held-out isolates (Fig. S3), "
    "we also computed the area under the precision–recall curve (PR-AUC) as the average precision of consumed "
    "(positive) versus neutral (negative) metabolites for each isolate (scikit-learn average_precision_score). The "
    "PR-AUC of a random ranking equals the isolate's prevalence of consumed metabolites, which is shown as a reference "
    "in Fig. S6. For Fig. 3, PR-AUC was computed in the same run and on the same compounds as ROC-AUC. Because BPR "
    "is trained to rank metabolites within each isolate, PR-AUC was averaged over isolates; computed over all withheld "
    "pairs pooled across isolates, which also rewards differences in score scale between isolates, PR-AUC at 32 "
    "dimensions was {pooled}.",
    "As learning-free baselines for the hold-out benchmark, each withheld isolate–metabolite pair was scored by (i) the "
    "fraction of training isolates tested on the metabolite that consumed it (popularity) and (ii) the same fraction "
    "among training isolates of the isolate's genus, using (i) when no isolate of the genus was tested on the "
    "metabolite (genus prevalence). Only training pairs were used, so a withheld pair never contributed to its own "
    "score.",
    "To assess calibration, the BPR scores (32 dimensions) of the withheld pairs in each hold-out repeat were converted "
    "to probabilities of consumption by Platt scaling (logistic regression of the consumed/neutral label on the score) "
    "or by isotonic regression. Isolates were split into five folds, and the calibration map applied to each fold was "
    "fitted on the other four. We report reliability over ten equal-width bins of predicted probability, the slope and "
    "intercept of the least-squares regression of the unbinned label on the predicted probability (1 and 0 for perfect "
    "calibration), and the Brier score, and compare them with the logistic function of the raw score (Fig. S7).",
    "For the learning curve, BPR (32 dimensions) was trained on nested random subsets of 5, 10, 20, 30, 50, 75, and "
    "100% of the training pairs (consumed and neutral pairs together) of each of the five hold-out splits and "
    "evaluated on the fixed withheld pairs by ROC-AUC, PR-AUC, and the Brier score after cross-fitted Platt scaling. "
    "BPR draws one batch of 256 triplets per epoch, so the number of parameter updates did not depend on the number of "
    "training pairs. The learning-free baselines were recomputed from each subset (Fig. S8).",
]


def si_captions(n: dict) -> list:
    return [
        ("Supplementary Figure 6. Precision–recall analysis.",
         " (a) Hold-out PR-AUC (mean over isolates) for the four models of Fig. 1b across embedding dimensions; error "
         "bars, ±1 s.d. across five repeats. Horizontal lines show two learning-free baselines evaluated on the same "
         "withheld pairs (popularity: the fraction of tested training isolates that consumed the metabolite; genus "
         "prevalence: the same fraction within the isolate's genus) and the PR-AUC of a random ranking (mean "
         "prevalence of consumed pairs). (b) Precision–recall curves at 32 dimensions, averaged over isolates "
         "(precision interpolated at each recall level; first repeat); PR-AUC values in the legend are means over five "
         "repeats. (c) PR-AUC of BPR (teal) and the naive baseline (orange) on the "
         "metabolites both can score in the hierarchical hold-out of Fig. 3; dashed lines, median prevalence of "
         "consumed metabolites at each level. (d) PR-AUC of BPR for the 12 held-out isolates of Fig. S3 "
         "(mean over 20 models; stars) and the prevalence of consumed metabolites (open circles)."),
        ("Supplementary Figure 7. Calibration of BPR scores.",
         " (a) Reliability diagram for the withheld pairs of the Fig. 1b hold-out (BPR, 32 dimensions, five repeats "
         "pooled): observed fraction of consumed pairs against the mean predicted probability in ten equal-width bins, "
         "with Wilson 95% confidence intervals. Squares, the logistic function of the raw BPR score; stars, Platt-scaled "
         "scores, with the calibration map fitted on other isolates (five-fold cross-fitting). Slope and intercept, "
         "least-squares regression of the unbinned outcome on the predicted probability (1 and 0 for perfect "
         "calibration). (b) Distribution of Platt-scaled probabilities for withheld consumed and neutral pairs."),
        ("Supplementary Figure 8. Learning curve.",
         " BPR (32 dimensions) was trained on nested random subsets of the training pairs of the Fig. 1b hold-out and "
         "evaluated on the same withheld pairs; the x axis gives the number of training pairs as a percentage of the "
         f"433 × 311 isolate × metabolite matrix (the full training set covers {pct(n['lc'].loc[1.0, 'pct_of_matrix_mean'])}). "
         "(a) ROC-AUC and (b) PR-AUC, averaged over isolates, with the learning-free baselines recomputed from each "
         "subset and (b) the PR-AUC of a random ranking. (c) Brier score after cross-fitted Platt scaling (lower is "
         "better) and the Brier score of predicting the prevalence alone. Error bars, ±1 s.d. across five repeats."),
    ]


# ----------------------------------------------------------------------------- figure 1 and workbook
def figure1_with_c(old_png: bytes) -> bytes:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    top = Image.open(io.BytesIO(old_png)).convert("RGB")
    if top.height != FIG1_HEIGHT_BEFORE:
        raise SystemExit(f"Figure 1 image is {top.size}; panel c appears to be present already")
    c = Image.open(FIG1C).convert("RGB")
    c = c.resize((top.width, round(c.height * top.width / c.width)), Image.LANCZOS)
    out = Image.new("RGB", (top.width, top.height + c.height), "white")
    out.paste(top, (0, 0))
    out.paste(c, (0, top.height))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def sheet_rows(n: dict) -> dict:
    s = n["s"]
    dims = n["dims"].reset_index()
    label = {BPR: "BPR", "WR-MF": "WR-MF", "SVD": "SVD", "LMF": "LMF"}
    s6 = [["Data for Figure S6: PR-AUC (average precision) of consumed versus neutral metabolites."], [],
          ["(a) Fig. 1b hold-out, mean over isolates; mean and SD across five repeats"],
          ["model", "dim", "pr_auc_mean", "pr_auc_sd", "roc_auc_mean", "roc_auc_sd", "prevalence_mean"]]
    for m in [BPR, "WR-MF", "SVD", "LMF"]:
        for _, r in dims[dims.model == m].sort_values("dim").iterrows():
            s6.append([label[m], int(r.dim), round(r.pr_auc_mean, 4), round(r.pr_auc_sd, 4), round(r.roc_auc_mean, 4),
                       round(r.roc_auc_sd, 4), round(r.prevalence_mean, 4)])
    for name, r in n["base"].iterrows():
        s6.append([f"{name} baseline (no learning)", "", round(r.pr_auc_mean, 4), round(r.pr_auc_sd, 4),
                   round(r.roc_auc_mean, 4), round(r.roc_auc_sd, 4), round(r.prevalence_mean, 4)])
    s6 += [[], ["(c) Fig. 3 hierarchical hold-out, medians over held-out isolates"],
           ["level", "n_paired", "median_pr_auc_bpr_scoreable", "median_pr_auc_baseline_scoreable",
            "median_prevalence_scoreable", "median_paired_difference", "wilcoxon_p", "median_pr_auc_bpr_all",
            "median_prevalence_all"]]
    for level, r in s["fig3"].items():
        s6.append([level, r["n_paired"], round(r["median_ap_bpr_scoreable"], 4), round(r["median_ap_baseline_scoreable"], 4),
                   round(r["median_prevalence_scoreable"], 4), round(r["median_delta"], 4), f"{r['wilcoxon_p']:.2g}",
                   round(r["median_ap_bpr_all"], 4), round(r["median_prevalence_all"], 4)])
    s6 += [[], ["(d) 12 held-out isolates (Fig. S3), mean over 20 models"],
           ["isolate", "genus", "pr_auc_mean", "prevalence", "roc_auc_mean"]]
    for _, r in pd.read_csv(PRC / "figS3_pr.csv").iterrows():
        s6.append([r.isolate_id, r.genus, round(r.pr_auc, 4), round(r.prevalence, 4), round(r.roc_auc, 4)])

    bins = pd.read_csv(PRC / "calibration_bins.csv")
    s7 = [["Data for Figure S7: calibration of BPR scores (Fig. 1b hold-out, 32 dimensions, five repeats)."], [],
          ["(a) Reliability, pooled over repeats"],
          ["method", "bin_low", "bin_high", "n_pairs", "mean_predicted", "observed_fraction_consumed", "ci95_low",
           "ci95_high"]]
    names = {"raw": "logistic function of raw score", "platt": "Platt scaling (cross-fitted)",
             "isotonic": "isotonic regression (cross-fitted)"}
    for _, r in bins.iterrows():
        s7.append([names[r.method], round(r.bin_low, 2), round(r.bin_high, 2), int(r.n), round(r.mean_predicted, 4),
                   round(r.observed, 4), round(r.ci_low, 4), round(r.ci_high, 4)])
    s7 += [[], ["Calibration statistics, mean (SD) across five repeats"],
           ["method", "ols_slope", "ols_intercept", "logistic_slope", "brier", "expected_calibration_error"]]
    for method, r in n["cal"].iterrows():
        s7.append([names[method]] + [f"{r[f'{k}_mean']:.3f} ({r[f'{k}_std']:.3f})"
                                     for k in ["ols_slope", "ols_intercept", "logistic_slope", "brier", "ece"]])

    lc = n["lc"].reset_index()
    cols = ["fraction", "n_train_pairs_mean", "pct_of_matrix_mean", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean",
            "pr_auc_sd", "cal_brier_mean", "cal_brier_sd", "cal_ols_slope_mean", "brier_prevalence_only_mean",
            "popularity_roc_auc_mean", "genus_roc_auc_mean", "popularity_pr_auc_mean", "genus_pr_auc_mean",
            "prevalence_mean"]
    s8 = [["Data for Figure S8: BPR learning curve on the Fig. 1b hold-out (mean and SD across five repeats)."], [],
          ["fraction_of_training_pairs", "n_training_pairs", "pct_of_matrix", "roc_auc_mean", "roc_auc_sd",
           "pr_auc_mean", "pr_auc_sd", "brier_calibrated_mean", "brier_calibrated_sd", "calibration_slope_mean",
           "brier_prevalence_only", "popularity_roc_auc", "genus_roc_auc", "popularity_pr_auc", "genus_pr_auc",
           "prevalence"]]
    for _, r in lc.iterrows():
        s8.append([round(float(r[c]), 4) if c != "n_train_pairs_mean" else int(round(r[c])) for c in cols])
    return {"Data for Figure S6": s6, "Data for Figure S7": s7, "Data for Figure S8": s8}


def si_table2_rows(existing_rows: list) -> list:
    """SI Table 2 with a PR_AUC_mean column after ROC_AUC_SD."""
    pr = pd.read_csv(PRC / "figS3_pr.csv").set_index("isolate_id").pr_auc
    header = existing_rows[2]
    k = header.index("ROC_AUC_SD") + 1
    rows = existing_rows[:2] + [header[:k] + ["PR_AUC_mean"] + header[k:]]
    for r in existing_rows[3:]:
        rows.append(r[:k] + [round(float(pr.loc[r[0]]), 3)] + r[k:])
    return rows


def read_sheet_rows(xml: str) -> list:
    """Cell values of a sheet written by sheet_data_xml (inline strings and numbers)."""
    rows = []
    for row in re.findall(r"<row [^>]*>(.*?)</row>", xml, flags=re.S):
        cells = []
        for c in re.findall(r"<c [^>]*?(?:/>|>.*?</c>)", row, flags=re.S):
            text = re.search(r"<t[^>]*>(.*?)</t>", c, flags=re.S)
            value = re.search(r"<v>(.*?)</v>", c)
            if text:
                from xml.sax.saxutils import unescape
                cells.append(unescape(text.group(1)))
            elif value:
                v = float(value.group(1))
                cells.append(int(v) if v.is_integer() and "." not in value.group(1) else v)
            else:
                cells.append(None)
        rows.append(cells)
    return rows


def write_workbook(n: dict, out_path: Path, backup: bool):
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    wb = parts["xl/workbook.xml"].decode()
    if "Data for Figure S6" in wb:
        raise SystemExit("workbook already has the Supplementary Fig. 6-8 sheets")
    # SI Table 2: PR-AUC column
    sheet7 = parts["xl/worksheets/sheet7.xml"].decode()
    rows = si_table2_rows(read_sheet_rows(sheet7))
    parts["xl/worksheets/sheet7.xml"] = re.sub(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(rows), sheet7,
                                               flags=re.S).encode()
    # new sheets, cloned from the SI Table 2 sheet without its drawing
    template = re.sub(r'<drawing r:id="[^"]+"/>', "", sheet7)
    rels = parts["xl/_rels/workbook.xml.rels"].decode()
    types = parts["[Content_Types].xml"].decode()
    next_sheet = max(int(x) for x in re.findall(r"worksheets/sheet(\d+)\.xml", rels)) + 1
    next_rid = max(int(x) for x in re.findall(r'Id="rId(\d+)"', rels)) + 1
    next_id = max(int(x) for x in re.findall(r'sheetId="(\d+)"', wb)) + 1
    for k, (name, data) in enumerate(sheet_rows(n).items()):
        part = f"xl/worksheets/sheet{next_sheet + k}.xml"
        parts[part] = re.sub(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(data), template, flags=re.S).encode()
        info = zipfile.ZipInfo(part, date_time=infos[0].date_time)
        info.compress_type = zipfile.ZIP_DEFLATED
        infos.append(info)
        rid = f"rId{next_rid + k}"
        rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                            f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{next_sheet + k}.xml"/>'
                            "</Relationships>")
        wb = wb.replace("</sheets>", f'<sheet state="visible" name="{name}" sheetId="{next_id + k}" r:id="{rid}"/></sheets>')
        types = types.replace("</Types>", '<Override ContentType="application/vnd.openxmlformats-officedocument.'
                              f'spreadsheetml.worksheet+xml" PartName="/{part}"/></Types>')
    parts["xl/workbook.xml"], parts["xl/_rels/workbook.xml.rels"] = wb.encode(), rels.encode()
    parts["[Content_Types].xml"] = types.encode()
    if backup:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TABLES, BACKUPS / f"{TABLES.stem}.before-{datetime.now():%Y%m%d-%H%M%S}{TABLES.suffix}")
    tmp = out_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(out_path)


# ----------------------------------------------------------------------------- letter
def fill_line_placeholders(letter: LetterDocx, lines: LineNumbers):
    """Line numbers for main-text quotes; SI quotes have none, so the placeholder is dropped."""
    paragraphs = letter.paragraphs()
    for i, p in enumerate(paragraphs):
        label = letter.accepted_text(p)
        if "Lines [__]" not in label:
            continue
        quote = letter.accepted_text(paragraphs[i + 1]).strip()
        if re.search(r"\(SI|Supplementary|Fig\. S3 caption", label):
            new = label.replace(", Lines [__]", "")
        else:
            found = lines.lines(quote)
            if found == "__":  # line breaks reorder words in the PDF text: locate start and end separately
                key = lines._norm(quote)
                i0, i1 = lines.text.find(key[:40]), lines.text.find(key[-30:])
                assert i0 >= 0 and i1 > i0, f"cannot locate {quote[:60]!r}"
                nums = [x for x in lines.line_of[i0:i1 + 30] if x is not None]
                found = f"{min(nums)}–{max(nums)}"
            new = label.replace("Lines [__]", f"Lines {found}")
        letter.set_paragraph_text(p, new)


def main(apply: bool, preview_dir: Path):
    if apply and list(DOCS.glob("~$*")):
        raise SystemExit("Word lock files present; close the documents in Word first")
    n = numbers()
    t = texts(n)
    ms, si = TrackedDocx(MS, AUTHOR), TrackedDocx(SI, AUTHOR)
    lines = LineNumbers(LINE_PDF)

    # ---- manuscript
    ms.insert_after("also identified BPR as the most accurate method (Supplementary Methods).", t["results_pr"])
    ms.insert_after("with 32-dimensional embeddings (ROC-AUC = 0.865 ± 0.004) for all subsequent analyses.",
                    t["results_cal_lc"])
    ms.insert_after("at the isolate level and 0.02–0.04 at the genus, family, and order levels).", t["results_fig3"])
    ms.insert_after("(Agrobacterium, at 0.80, is close to this threshold).", t["results_s3"])
    ms.insert_after("the isolate × metabolite assertion matrix (433 isolates × 311 metabolites", t["methods_fig1c"])
    ms.insert_after("it is a ranking statistic and not a calibrated probability of consumption.", t["methods_cal"])
    ms.insert_after("than a randomly selected neutral metabolite (Fig. S3).", t["methods_pr"])
    ms.insert_after("into complete, prioritizable predictions that can guide which experiments to perform next.",
                    t["discussion"])
    ms.insert_after("Error bars, ±1 s.d. across five repeats.", t["caption_c"], context="Figure 1. Dataset architecture")
    fig1 = figure1_with_c(ms.read_part("word/media/image1.png"))
    ms.replace_part("word/media/image1.png", fig1)
    from PIL import Image
    w, h = Image.open(io.BytesIO(fig1)).size
    ms.set_image_extent("image1.png", 5943600, round(5943600 * h / w))
    (R1 / "fig1c").mkdir(parents=True, exist_ok=True)
    (R1 / "fig1c" / "figure1_revised.png").write_bytes(fig1)

    # ---- SI
    si.replace("Supplementary Figures 4-5", "Supplementary Figures 4-8", context="Supplementary Figures 4-5")
    si.replace("Metabolites consumed by fewer than three isolates across the entire dataset were removed",
               "Metabolites with a consumed or secreted assertion in fewer than three isolates across the entire "
               "dataset were removed")
    si.replace("consumed (experimental log2 fold-change ≤ −2; blue)", "consumed (assertion value ≤ −2; blue)")
    anchor = si.find("we selected BPR with 32-dimensional embeddings for all subsequent analyses.")
    body_rpr = si.text_runs(si.find("To benchmark predictive performance (Fig. 1b)"))[0][0].find(q("rPr"))
    after = anchor
    pooled = n["s"]["fig1b_pooled_pr_auc_dim32"]
    pooled_text = (f"{f2(pooled[BPR]['mean'])} for BPR, {f2(pooled['WR-MF']['mean'])} for WR-MF, "
                   f"{f2(pooled['SVD']['mean'])} for SVD, and {f2(pooled['LMF']['mean'])} for LMF (means over five repeats)")
    for text in SI_METHODS:
        after = si.insert_paragraph_after(after, [(body_rpr, text.format(pooled=pooled_text))])
    s5_caption = si.find("Supplementary Figure  5. WoM guided substrate selection")
    s5_image = s5_caption.getprevious().getprevious()
    assert s5_image.find(".//" + q("drawing")) is not None, "S5 image paragraph not found"
    cap_runs = s5_caption.findall(q("r"))
    bold_rpr = cap_runs[0].find(q("rPr"))
    plain_rpr = next(r.find(q("rPr")) for r in cap_runs if r.find(q("rPr")).find(q("b")) is None)
    after = s5_caption
    for (media, png), (title, rest) in zip(NEW_FIGS, si_captions(n)):
        after = si.insert_image_after(after, s5_image, media, png.read_bytes())
        after = si.insert_paragraph_after(after, [(bold_rpr, title), (plain_rpr, rest)], ppr_from=s5_caption)

    # ---- letter
    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    base, cal, lc, f3s, s3 = n["base"], n["cal"], n["lc"], n["s"]["fig3"], n["s"]["figS3"]
    dims = n["dims"]

    def mod_label(section, tail):
        return f'Modified ({section}): (no original text; new insertion immediately after the sentence ending "…{tail}")'

    letter.set_response("Comment 2. The data is heavily imbalanced", (
        "We thank the reviewer for this suggestion. We agree that ROC-AUC alone can hide a high false-positive rate "
        "when positives are rare, and we have added PR-AUC, together with its value for a random ranking (the "
        "prevalence of consumed metabolites), for all three evaluations. On the class balance: among tested "
        "isolate–metabolite pairs, 14.9% were consumed (11,093 of 74,604); the 8.6% figure in the submitted text "
        "divided a record count by the number of possible pairs (see Reviewer 3, Minor comment 2). In the Fig. 1b hold-out, where "
        f"{pct(100 * n['prev_pooled'])} of withheld pairs were consumed ({f2(n['prev_iso'])} on average per isolate), "
        f"BPR reached PR-AUC = {f2(n['bpr32_pr'])} at 32 dimensions (Fig. S6a,b). By this measure WR-MF "
        f"at 4 and 8 dimensions performed similarly ({f2(n['wr_low_pr'][0])} and {f2(n['wr_low_pr'][1])}), but WR-MF, "
        "SVD, and LMF declined at higher dimensions, as they did by ROC-AUC. In the hierarchical hold-out (Fig. 3), "
        "PR-AUC led to the same conclusions as ROC-AUC: BPR and the taxonomic baseline were similar at the isolate "
        f"level (median {f2(f3s['isolate']['median_ap_bpr_scoreable'])} vs. "
        f"{f2(f3s['isolate']['median_ap_baseline_scoreable'])}), and BPR was higher once genera, families, or orders "
        "were withheld (Fig. S6c). For the 12 held-out isolates most measured metabolites were consumed "
        f"(mean prevalence {f2(s3['mean_prevalence'])}), so even a random ranking scores high; BPR's mean PR-AUC was "
        f"{f2(s3['mean_pr_auc'])} and exceeded the prevalence for "
        + ("every isolate" if s3["n_pr_auc_above_prevalence"] == 12 else f"{s3['n_pr_auc_above_prevalence']} of 12 isolates")
        + " (Fig. S6d); PR-AUC is now also listed in SI Table 2. Changes were made to the Results "
        "sections \"Bayesian effectively learns bacterial substrate preferences\" and \"Phylogenetic conservation of "
        "metabolic profiles validates genus-level prediction\", the Methods section \"Complete hold-out validation\", "
        "the Supplementary Methods, SI Table 2, and the new Fig. S6."))
    p = letter.response_paragraph_after("Comment 2. The data is heavily imbalanced")
    letter.add_pairs(p, [
        (mod_label("Results, first section", "also identified BPR as the most accurate method (Supplementary Methods)."),
         t["results_pr"].strip()),
        (mod_label("Results, \"Phylogenetic conservation…\"", "at the genus, family, and order levels)."),
         t["results_fig3"].strip()),
        (mod_label("Results, \"Phylogenetic conservation…\", final paragraph", "is close to this threshold)."),
         t["results_s3"].strip()),
        (mod_label("Methods, Complete hold-out validation", "than a randomly selected neutral metabolite (Fig. S3)."),
         t["methods_pr"].strip()),
    ])

    letter.set_response("Comment 3. In the abstract the authors claim that the model is calibrated", (
        "We thank the reviewer for this request. As noted in the response to Reviewer 3, Minor comment 1, BPR scores "
        "are ranking statistics rather than probabilities, and we have removed \"calibrated\" from the Abstract. To "
        "answer the question directly, we tested whether the scores can be converted into calibrated probabilities. "
        "Using the withheld pairs of the Fig. 1b hold-out, we fitted a monotone calibration map (Platt scaling: a "
        "logistic regression of the consumed/neutral label on the score) on four-fifths of the isolates and applied it "
        "to the remaining fifth, rotating through all five folds (Fig. S7). Treating the raw score as a "
        f"log-odds gave poorly calibrated probabilities (slope of the label on the predicted probability = "
        f"{f2(cal.loc['raw', 'ols_slope_mean'])}; Brier score = {f3(cal.loc['raw', 'brier_mean'])}). After Platt "
        f"scaling, observed and predicted frequencies agreed across the ten bins (for example, "
        f"{pct(100 * t['b30'].observed)} of the pairs assigned to the 30–40% bin were consumed), the un-binned slope was "
        f"{f2(cal.loc['platt', 'ols_slope_mean'])} (intercept {f2(cal.loc['platt', 'ols_intercept_mean'])}), and the "
        f"Brier score was {f3(cal.loc['platt', 'brier_mean'])}; isotonic regression gave the same result (slope "
        f"{f2(cal.loc['isotonic', 'ols_slope_mean'])}). Calibrated probabilities therefore require this additional, "
        "data-derived step, and they apply to the measured metabolite panel and conditions. Calibration as a function "
        "of training-set size is shown in the response to Comment 6. Changes were made to the Results section "
        "\"Bayesian effectively learns bacterial substrate preferences\", the Methods section \"Model training\", the "
        "Supplementary Methods, and the new Fig. S7."))
    p = letter.response_paragraph_after("Comment 3. In the abstract the authors claim that the model is calibrated")
    letter.add_pairs(p, [
        (mod_label("Results, first section", "for all subsequent analyses."), t["results_cal_lc"].strip()),
        (mod_label("Methods, Model training", "not a calibrated probability of consumption."), t["methods_cal"].strip()),
    ])

    r4 = letter.response_text("Comment 4. I understand that your choice of representation")
    old = ("which did not describe Ordinal-BPR. ⟨PENDING: new Fig. 1c schematic of the input matrix → embeddings → "
           "dot-product scores.⟩")
    assert old in r4, "R1-4 response wording changed"
    letter.set_response("Comment 4. I understand that your choice of representation", r4.replace(old, (
        "which did not describe the model as implemented. We have also added Fig. 1c, a schematic that shows the input "
        "matrix and its three states, the sampled triples used in training, the isolate and metabolite embeddings, and "
        "how their dot product scores and ranks every isolate–metabolite pair, including untested ones. Changes were "
        "made to the Methods section \"Model training\" and to Figure 1 and its caption.")))
    p = letter.response_paragraph_after("Comment 4. I understand that your choice of representation")
    quote = p.getnext().getnext()  # the Methods quote after the 'Original' label
    while not letter.accepted_text(quote).startswith("We framed the prediction of microbial metabolite consumption as an implicit-feedback"):
        quote = quote.getnext()
    letter.add_pairs(quote, [
        (mod_label("Figure 1 caption", "Error bars, ±1 s.d. across five repeats."), t["caption_c"].strip()),
    ])

    others = n["others128"]
    letter.set_response("Comment 5. In fig 1B and the discussion in the paper", (
        "We thank the reviewer for this practical question. The submitted Fig. 1b was affected by the problems "
        "described in the response to Reviewer 3, Major comment 1, and in the corrected benchmark the four methods are "
        f"no longer equal. They are similar at 4–8 dimensions (ROC-AUC = {f2(n['low_range'][0])}–{f2(n['low_range'][1])}), "
        f"but only BPR keeps its accuracy at higher dimensions ({f2(n['bpr128_roc'])} at 128 dimensions, versus "
        f"{f2(others[0])}–{f2(others[1])} for the other three). Going to a higher-dimensional embedding does not buy "
        f"additional accuracy: BPR was most accurate at 16 dimensions (ROC-AUC = {f3(n['bpr16_roc'])}) and within "
        f"{n['near_gap']:.3f} of this from 8 to 64 dimensions; we use 32 dimensions, within this plateau, and the "
        "stability means that the choice is not critical. To address whether BPR is needed at all, we also evaluated "
        "two learning-free baselines on the same withheld pairs: the fraction of tested isolates that consumed each "
        f"metabolite (ROC-AUC = {f2(base.loc['popularity', 'roc_auc_mean'])}; PR-AUC = "
        f"{f2(base.loc['popularity', 'pr_auc_mean'])}) and the same fraction within the isolate's genus "
        f"({f2(base.loc['genus', 'roc_auc_mean'])}; {f2(base.loc['genus', 'pr_auc_mean'])}). BPR "
        f"({f2(n['bpr32_roc'])}; {f2(n['bpr32_pr'])} at 32 dimensions) is more accurate, but the gain is modest, and by "
        f"PR-AUC a low-dimensional WR-MF model is comparable ({f2(n['wr_low_pr'][1])} at 8 dimensions). The "
        "hierarchical hold-out (Fig. 3) shows the same pattern: BPR and taxonomic averaging are nearly identical within "
        "a genus, and BPR is modestly better when the reference set is more distant. With fewer measured pairs the "
        "baselines are more accurate than BPR (Fig. S8; see Comment 6). What BPR offers a practical user "
        "is therefore (i) accuracy that does not depend on tuning the embedding dimension, (ii) strain-level scores that "
        "use each isolate's own measurements, (iii) a score for every metabolite even when no close relative was tested "
        "on it, which a taxonomic average cannot provide (Fig. 3b), and (iv) released, pre-trained scores that can be "
        "used without fitting a model. For coarse, genus-level questions or sparsely profiled panels, a prevalence "
        "table is a reasonable alternative, and we now say so in the Discussion. Changes were made to the Results "
        "section \"Bayesian effectively learns bacterial substrate preferences\", the Discussion, and the Supplementary "
        "Methods, and the comparisons are shown in Figs. S6 and S8."))
    p = letter.response_paragraph_after("Comment 5. In fig 1B and the discussion in the paper")
    letter.add_pairs(p, [
        ("Modified (Results, first section): see the response to Comment 2, which quotes the new text on PR-AUC, "
         "embedding dimension, and learning-free baselines.", None),
        (mod_label("Discussion, third paragraph", "guide which experiments to perform next."), t["discussion"].strip()),
    ])

    first, below = lc.loc[n["lc_first"]], lc.loc[n["lc_below"]]
    smallest = lc.iloc[0]
    letter.set_response("Comment 6. I am also interested in seeing a learning curve", (
        "We thank the reviewer for this suggestion. We computed a learning curve on the Fig. 1b hold-out: keeping the "
        "withheld pairs fixed, we trained BPR on nested random subsets of 5–100% of the training pairs, which "
        f"correspond to {smallest.pct_of_matrix_mean:.0f}–{lc.loc[1.0, 'pct_of_matrix_mean']:.0f}% of the 433 × 311 "
        "isolate × metabolite matrix, and recorded ROC-AUC, PR-AUC, and the Brier score after cross-fitted Platt scaling "
        "(Fig. S8). All three improved steadily with more data and had not reached a plateau at the size "
        f"of the current dataset (ROC-AUC = {f2(smallest.roc_auc_mean)}, {f2(lc.loc[0.5, 'roc_auc_mean'])}, and "
        f"{f2(lc.loc[1.0, 'roc_auc_mean'])} with 5%, 50%, and 100% of the training pairs; PR-AUC = "
        f"{f2(smallest.pr_auc_mean)}, {f2(lc.loc[0.5, 'pr_auc_mean'])}, and {f2(lc.loc[1.0, 'pr_auc_mean'])}; Brier score "
        f"from {f3(smallest.cal_brier_mean)}, the value of predicting the prevalence alone, to "
        f"{f3(lc.loc[1.0, 'cal_brier_mean'])}). Once a Platt map was fitted, probabilities were well calibrated "
        f"(slope within 0.1 of 1) from {pct(n['cal_ok_pct'])} of the matrix onwards. For planning, the most useful "
        f"result is the comparison with the learning-free baselines: BPR was less accurate than both with up to "
        f"{pct(n['lc_below_pct'])} of the matrix in training and more accurate by ROC-AUC with "
        f"{pct(n['lc_first_pct'])} or more ({f3(first.roc_auc_mean)} vs. {f3(first.genus_roc_auc_mean)} for genus "
        f"prevalence at {pct(n['lc_first_pct'])}), because with few measurements per isolate the model cannot yet "
        f"learn isolate-specific embeddings. By PR-AUC, BPR matched genus prevalence at {pct(n['lc_first_pct'])} "
        f"({f3(first.pr_auc_mean)} vs. {f3(first.genus_pr_auc_mean)}) and exceeded it only with the full training "
        f"set ({f3(lc.loc[1.0, 'pr_auc_mean'])} vs. {f3(lc.loc[1.0, 'genus_pr_auc_mean'])}). Because pairs were "
        "sampled at random across isolates, this curve describes how densely the isolate × metabolite space must be "
        "sampled rather than how many isolates are needed; for a large microbe–metabolite universe it suggests profiling "
        "each isolate against a broad panel, and using prevalence-based predictions until roughly a quarter to a third "
        "of the matrix has been measured. Changes were made to the Results section \"Bayesian effectively learns bacterial substrate "
        "preferences\", the Discussion, the Supplementary Methods, and the new Fig. S8."))
    p = letter.response_paragraph_after("Comment 6. I am also interested in seeing a learning curve")
    letter.add_pairs(p, [
        ("Modified (Results, first section): see the response to Comment 3, which quotes the new text on calibration "
         "and the learning curve.", None),
        ("Modified (Discussion, third paragraph): see the response to Comment 5.", None),
    ])

    # R3 minor 2: the SI filter description
    r2 = letter.response_text("Minor comment 2. The manuscript reports 433 isolates")
    for old, new in [("we have corrected this in the Results and the Methods.",
                      "we have corrected this in the Results, the Methods, and the Supplementary Information."),
                     ("and the Methods section \"Exometabolite Dataset Construction\".",
                      "the Methods section \"Exometabolite Dataset Construction\", and the Supplementary Information "
                      "(Model training).")]:
        assert old in r2, old
        r2 = r2.replace(old, new)
    r2 = r2.replace("Changes were made to the Results section \"Bayesian effectively learns bacterial substrate "
                    "preferences\" the Methods", "Changes were made to the Results section \"Bayesian effectively learns "
                    "bacterial substrate preferences\", the Methods")
    letter.set_response("Minor comment 2. The manuscript reports 433 isolates", r2)
    minor3 = next(x for x in letter.paragraphs() if letter.accepted_text(x).startswith("Minor comment 3."))
    letter.add_pairs(minor3.getprevious(), [
        ("Original (Supplementary Information, Model training):",
         "Metabolites consumed by fewer than three isolates across the entire dataset were removed to exclude rarely "
         "observed or potentially spurious interactions."),
        ("Modified:", "Metabolites with a consumed or secreted assertion in fewer than three isolates across the entire "
         "dataset were removed to exclude rarely observed or potentially spurious interactions."),
    ])
    # R3 minor 3: verified SI quotes and the consumed label
    letter.set_text("Original (Supplementary Methods ⟨verify wording against submitted SI⟩, Lines [__]):",
                    "Original (Supplementary Information, Model training):", exact=True)
    letter.set_text("Original (Fig. S3 caption ⟨verify wording against submitted SI⟩, Lines [__]):",
                    "Original (Supplementary Figure 3 caption):", exact=True)
    letter.set_text("neutral (experimental log2 fold-change > -2; orange)",
                    "consumed (experimental log2 fold-change ≤ −2; blue) or neutral (experimental log2 fold-change > -2; "
                    "orange)", exact=True)
    letter.set_text("neutral (−2 < assertion value < +2; orange); secreted metabolites (assertion value ≥ +2) were excluded",
                    "consumed (assertion value ≤ −2; blue) or neutral (−2 < assertion value < +2; orange) based on "
                    "exometabolite measurements in NLDM (Northen Lab Defined Medium); secreted metabolites (assertion "
                    "value ≥ +2) were excluded", exact=True)
    # R3 minor 4: drop the garbled duplicate of the R3 minor 3 caption pair
    dup = letter.find("(−2 < assertion value < +2; orange)neutral (experimental log2 fold-change > -2; orange)")
    label, mod, mod_quote = dup.getprevious(), dup.getnext(), dup.getnext().getnext()
    assert letter.accepted_text(label).startswith("Original (SI, Supplementary Figure 3 caption")
    assert letter.accepted_text(mod) == "Modified:" and letter.accepted_text(mod_quote) == "(−2 < assertion value < +2; orange)"
    for x in (label, dup, mod, mod_quote):
        x.getparent().remove(x)
    fill_line_placeholders(letter, lines)

    if apply:
        for doc in (ms, si, letter):
            print("backup:", doc.backup(BACKUPS))
            doc.save()
        write_workbook(n, TABLES, backup=True)
        print("applied to", DOCS)
    else:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for doc in (ms, si, letter):
            doc.save(preview_dir / doc.path.name)
        write_workbook(n, preview_dir / TABLES.name, backup=False)
        print("preview written to", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/r1_preview"))
    args = parser.parse_args()
    main(args.apply, args.preview_dir)
