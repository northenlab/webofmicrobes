# lambda0_number_edits.py
"""Propagate the revised-model (BPR, lambda_ord = 0) numbers into the revision documents.

Follows r3_major1_edits.py. Updates every model-derived number for Fig. 2,
Fig. 3 and Supplementary Fig. 3 / SI Table 2 in the revised manuscript and SI
(tracked changes, author Benjamin Bowen), swaps in the regenerated Fig. 2,
Fig. 3, Supplementary Fig. 2 and Supplementary Fig. 3 images, replaces the
"Data For Figure 2a", "Data For Figure 2b" and "SI Table 2" sheets, and updates
the response letter (R1-1 and R3 minor 4 quotes, and Original/Modified pairs
for the number changes under R3 major 1). Supplementary Fig. 5 is left for its
remake. Numbers are read from manuscript_figures/msystems_revision/lambda0/.

Usage:
    python lambda0_number_edits.py --preview-dir /tmp/numbers_preview
    python lambda0_number_edits.py --apply
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import TrackedDocx, LetterDocx  # noqa: E402
from r3_major1_edits import (  # noqa: E402
    REPO, DOCS, MS, SI, LETTER, LINE_PDF, BACKUPS, AUTHOR, TABLES,
    LineNumbers, sentence_containing, sheet_data_xml,
)

OUT = REPO / "wom/manuscript_figures/msystems_revision/lambda0"
FIG2_PNG = OUT / "fig2/figure_biology_main.png"
FIG3_PNG = OUT / "fig3/fig3_holdout.png"
S2_PNG = OUT / "si_figures/figS2_genus_heatmap.png"
S3_PNG = OUT / "figS3/figS3_validation_clean_holdout.png"
IN_TRAINING = {"GW456-R20", "FW507-4G11", "MT49"}  # MT49 is MT049 in the training data


# ----------------------------------------------------------------------------- numbers
def numbers() -> dict:
    fig2 = json.loads((OUT / "fig2/stats_figure_biology.json").read_text())
    div = pd.read_csv(OUT / "fig2/stats_gcs_emb_diversity.csv").set_index("genus")
    alpha = div[div["class"] == "Alphaproteobacteria"].emb_diversity
    fig3 = json.loads((OUT / "fig3/fig3_holdout_summary.json").read_text())
    s3 = pd.read_csv(OUT / "figS3/validation_12_isolates_summary.csv").set_index(["model_variant", "subset"])
    retrain = s3.loc[("retrain_corrected_labels", "all_12"), "mean_roc_auc"]  # profiles kept in training
    s3 = s3.loc[("clean_holdout_corrected_labels", "all_12")]
    assert f"{retrain:.2f}" == f"{s3['mean_roc_auc']:.2f}", "removing the in-training profiles changed the mean"
    tab = pd.read_csv(OUT / "figS3/table_S_validation_12_isolates.csv").set_index("genus")

    assert alpha.idxmin() == "Methylobacterium" and alpha.idxmax() == "Sphingomonas", "alpha range genera changed"
    for level in ["isolate", "genus", "family", "order"]:
        assert fig3[level]["wilcoxon_p"] < 1e-3, f"{level} not significant at p < 0.001"
    assert fig2["gcs_vs_embedding_diversity"]["p"] < 1e-8 and fig2["permanova_embedding"]["df"] == [65, 315]

    # S3 sentence: values per reference taxon (scripts/revision_figS3_scatter_stats.py)
    pt = pd.read_csv(OUT / "figS3/figS3_scatter_per_taxon.csv").set_index("taxon")
    sc = {}
    for col in ["gcs", "emb_div"]:
        ok = pt[col].notna()
        res = spearmanr(pt.loc[ok, "auc"], pt.loc[ok, col])
        sc[col] = (float(res.statistic), float(res.pvalue), int(ok.sum()))
    genus_level = pt.drop(index=["Rhizobiaceae", "Enterobacterales"])
    assert pt.auc.idxmin() == "Caulobacter" and pt.emb_div.idxmax() == "Caulobacter"
    assert genus_level.gcs.idxmin() == "Caulobacter"
    assert pt.gcs.idxmax() == "Delftia" and pt.emb_div.idxmin() == "Delftia"
    deltas = [fig3[lv]["median_delta"] for lv in ["genus", "family", "order"]]
    return {
        "cos": fig2["within_genus_embedding_cosine"]["mean"],
        "r2": fig2["permanova_embedding"]["r2"], "F": fig2["permanova_embedding"]["pseudo_f"],
        "rho": fig2["gcs_vs_embedding_diversity"]["spearman_rho"],
        "methylo": alpha["Methylobacterium"], "sphingo": alpha["Sphingomonas"],
        "f3": fig3, "d_iso": fig3["isolate"]["median_delta"], "d_lo": min(deltas), "d_hi": max(deltas),
        "s3_mean": s3["mean_roc_auc"], "s3_sd": s3["mean_roc_auc_sd_across_seeds"], "s3_median": s3["median_roc_auc"],
        "agro": tab.loc["Agrobacterium", "clean_holdout_roc_auc_mean"],
        "caul": tab.loc["Caulobacter", "clean_holdout_roc_auc_mean"],
        "n_ge": int((tab.clean_holdout_roc_auc_mean >= 0.8).sum()),
        "caul_auc": pt.loc["Caulobacter", "auc"], "caul_div": pt.loc["Caulobacter", "emb_div"],
        "caul_gcs": pt.loc["Caulobacter", "gcs"], "delf_gcs": pt.loc["Delftia", "gcs"],
        "delf_div": pt.loc["Delftia", "emb_div"],
        "rho_gcs": sc["gcs"], "rho_div": sc["emb_div"],
    }


def from_sentence(text: str, start: str) -> str:
    """Text from the sentence beginning with `start` to the end of the paragraph."""
    return text[text.index(start):].strip()


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


# ----------------------------------------------------------------------------- edits
def manuscript_edits(n: dict) -> list:
    f = n["f3"]
    s3_text = (
        f"The isolate with the lowest ROC-AUC, Caulobacter ({f2(n['caul_auc'])}), was scored from the reference genus "
        f"with the highest embedding diversity ({f2(n['caul_div'])}) and, among the genus-level references, the "
        f"lowest genus consistency score ({f2(n['caul_gcs'])}), whereas the Delftia reference genus has the highest "
        f"consistency score ({f2(n['delf_gcs'])}) and the lowest embedding diversity ({f2(n['delf_div'])}). Across "
        f"the {n['rho_gcs'][2]} reference taxa for which both measures could be computed, however, ROC-AUC was not "
        f"significantly correlated with the genus consistency score (Spearman rho = {f2(n['rho_gcs'][0])}, "
        f"p = {f2(n['rho_gcs'][1])}) or with embedding diversity (rho = {f2(n['rho_div'][0])}, "
        f"p = {f2(n['rho_div'][1])}).")
    return [
        # Abstract
        ("from 0.90 to 0.76", f"from {f2(f['isolate']['median_auc_bpr_all'])} to {f2(f['order']['median_auc_bpr_all'])}",
         "In a hierarchical hold-out benchmark"),
        ("(mean ROC-AUC = 0.88)", f"(mean ROC-AUC = {f2(n['s3_mean'])})", "A prospective experiment on 12"),
        # Results, Fig. 2
        ("(mean cosine similarity = 0.59 within-genus)", f"(mean cosine similarity = {f2(n['cos'])} within-genus)", None),
        ("(R² = 0.69, pseudo-F = 10.7 on 65 and 315 d.f.;", f"(R² = {f2(n['r2'])}, pseudo-F = {n['F']:.1f} on 65 and 315 d.f.;", None),
        ("(Spearman rho = -0.73, p < 10^-8)", f"(Spearman rho = {f2(n['rho'])}, p < 10^-8)", None),
        # Results, Fig. 3
        ("with a median AUC of 0.90, declining", f"with a median AUC of {f2(f['isolate']['median_auc_bpr_all'])}, declining", None),
        ("essentially indistinguishable (0.895 vs. 0.899)",
         f"essentially indistinguishable ({f3(f['isolate']['median_auc_bpr_scoreable'])} vs. "
         f"{f3(f['isolate']['median_auc_baseline_scoreable'])})", None),
        ("(genus 0.843 vs. 0.802; family 0.833 vs. 0.785; order 0.751 vs. 0.707, Fig. 3a)",
         "(" + "; ".join(f"{lv} {f3(f[lv]['median_auc_bpr_scoreable'])} vs. {f3(f[lv]['median_auc_baseline_scoreable'])}"
                         for lv in ["genus", "family", "order"]) + ", Fig. 3a)", None),
        ("While these improvements were significant at the genus and family levels (paired Wilcoxon p < 0.001) they "
         "are small in absolute terms.",
         f"These differences were significant at all four levels (paired Wilcoxon p < 0.001) but small in absolute "
         f"terms (median paired differences of {f3(n['d_iso'])} at the isolate level and {f2(n['d_lo'])}–"
         f"{f2(n['d_hi'])} at the genus, family, and order levels).", None),
        # Results, Supplementary Fig. 3
        ("two of these isolates had earlier exometabolite profiles", "three of these isolates had earlier exometabolite profiles", None),
        ("the mean ROC-AUC was 0.88 (SD 0.004 across models; median 0.89)",
         f"the mean ROC-AUC was {f2(n['s3_mean'])} (SD {f3(n['s3_sd'])} across models; median {f2(n['s3_median'])})", None),
        ("(Agrobacterium, at 0.81, is close to this threshold)", f"(Agrobacterium, at {f2(n['agro'])}, is close to this threshold)", None),
        ("Caulobacter sits at the extremes of both panels with the lowest GCS, highest embedding diversity, and lowest "
         "AUC. Delftia is the opposite. Although these anchor points visually support a relationship between AUC and "
         "GCS or AUC and embedding diversity, the trend is not statistically significant.", s3_text, None),
        # Discussion
        ("found a mean ROC-AUC of 0.88, with 11 of the 12", f"found a mean ROC-AUC of {f2(n['s3_mean'])}, with {n['n_ge']} of the 12", None),
        ("range from high consistency (0.166, Methylobacterium) to much lower consistency/higher diversity of 0.77 (Sphingomonas)",
         f"range from high consistency ({f2(n['methylo'])}, Methylobacterium) to much lower consistency/higher "
         f"diversity of {f2(n['sphingo'])} (Sphingomonas)", None),
        # Methods
        ("10 of which had not previously been profiled (the two exceptions",
         "9 of which had not previously been profiled (the three exceptions", None),
        ("Two of them, Delftia GW456-R20 and Cupriavidus FW507-4G11, had exometabolite profiles from earlier, "
         "independent experiments in the training set;",
         "Three of them had exometabolite profiles in the training set: Delftia GW456-R20 and Cupriavidus "
         "FW507-4G11 from earlier, independent experiments, and Serratia MT49, recorded as MT049;", None),
        # Fig. 2 caption
        ("BPR mean cosine similarity = 0.59", f"BPR mean cosine similarity = {f2(n['cos'])}", None),
        ("negative correlation (Spearman rho = -0.73)", f"negative correlation (Spearman rho = {f2(n['rho'])})", None),
    ]


def si_edits(n: dict) -> list:
    return [("earlier profiles of two isolates (Delftia GW456-R20 and Cupriavidus FW507-4G11) were removed",
             "earlier profiles of three isolates (Delftia GW456-R20, Cupriavidus FW507-4G11, and Serratia MT49) were removed",
             None)]


# ----------------------------------------------------------------------------- images and sheets
def fit_png(new_png: Path, old_png: bytes) -> bytes:
    """Resize the new figure into the old image's pixel frame, letterboxing to keep its aspect."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # our own figures; the S2 heatmap is ~2.3e8 pixels
    old = Image.open(io.BytesIO(old_png))
    new = Image.open(new_png).convert("RGBA")
    scale = min(old.width / new.width, old.height / new.height)
    new = new.resize((round(new.width * scale), round(new.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGBA", old.size, (255, 255, 255, 0))
    canvas.alpha_composite(new, ((old.width - new.width) // 2, (old.height - new.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def panel_a_lines() -> list:
    """The metric definitions and both statistics tables of the panel A stats file, left-aligned,
    as in the submitted sheet (no banner, figure note or Notes section)."""
    lines = [line.strip() for line in (OUT / "fig2/figure_biology_panel_A_stats.txt").read_text().splitlines()]
    start = next(i for i, line in enumerate(lines) if line.startswith("Empirical :"))
    end = lines.index("Notes") - 1  # the rule above the Notes heading
    keep = [line for line in lines[start:end] if not line.startswith("Only distances present")]
    out = []
    for line in keep:
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def sheet_rows() -> dict:
    fig2a = [["Data for Figure 2a: Pairwise metabolic profile similarity as a function of taxonomic distance, "
              "computed on n > 90,000 isolate pairs"], []] + [[line] if line else [] for line in panel_a_lines()]
    div = pd.read_csv(OUT / "fig2/stats_gcs_emb_diversity.csv")
    fig2b = [["Data For Figure 2b: Genus Consistency Score (GCS; mean within-genus pairwise Spearman ρ) versus "
              "within-genus predicted metabolic diversity (mean pairwise cosine distance between BPR-predicted "
              "metabolic trait vectors), across 48 genera with ≥ 3 isolates."], [],
             ["genus", "gcs", "ci_low", "ci_high", "class", "n_isolates", "emb_diversity"]]
    fig2b += [[r.genus, r.gcs, r.ci_low, r.ci_high, r["class"], int(r.n_isolates), r.emb_diversity]
              for _, r in div.iterrows()]
    tab = pd.read_csv(OUT / "figS3/table_S_validation_12_isolates.csv")
    si2 = [["SI Table 2–Reference sets and evaluation metabolites for the 12 held-out isolates (Fig. S3)."], [],
           ["Isolate", "Genus", "Reference_rank", "Reference_taxon", "n_reference_isolates", "n_consumed",
            "n_neutral", "n_secreted_excluded", "Prevalence", "ROC_AUC_mean", "ROC_AUC_SD", "Prior_profile_removed"]]
    for _, r in tab.iterrows():
        si2.append([r.isolate_id, r.genus, r.clean_holdout_reference_rank_used, r.reference_taxon,
                    int(r.clean_holdout_n_source_training_isolates), int(r.n_consumed), int(r.n_neutral),
                    int(r.n_secreted_excluded), round(r.prevalence, 2), round(r.clean_holdout_roc_auc_mean, 3),
                    round(r.clean_holdout_roc_auc_sd, 3), "yes" if r.isolate_id in IN_TRAINING else "no"])
    return {"xl/worksheets/sheet2.xml": fig2a, "xl/worksheets/sheet3.xml": fig2b, "xl/worksheets/sheet7.xml": si2}


def write_sheets(out_path: Path, backup: bool):
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    for name, rows in sheet_rows().items():
        new, k = re.subn(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(rows),
                         parts[name].decode("utf-8"), flags=re.S)
        assert k == 1, name
        parts[name] = new.encode("utf-8")
    if backup:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TABLES, BACKUPS / f"{TABLES.stem}.before-{datetime.now():%Y%m%d-%H%M%S}{TABLES.suffix}")
    tmp = out_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(out_path)


# ----------------------------------------------------------------------------- main
def main(apply: bool, preview_dir: Path):
    if apply and list(DOCS.glob("~$*")):
        raise SystemExit("Word lock files present; close the documents in Word first")
    n = numbers()
    print({k: v for k, v in n.items() if k != "f3"})
    ms, si = TrackedDocx(MS, AUTHOR), TrackedDocx(SI, AUTHOR)
    lines = LineNumbers(LINE_PDF)

    def orig(doc, snippet, fragment=None):
        text = doc.rejected_text(doc.find(snippet))
        return sentence_containing(text, fragment or snippet)

    original = {
        "abstract": orig(ms, "with a median ROC-AUC declining", "median AUC declining from 0.90 to 0.76"),
        "fig2": orig(ms, "mean cosine similarity = 0.59 within-genus", "Isolates of the same genus were substantially")
        + " " + orig(ms, "mean cosine similarity = 0.59 within-genus", "We quantified conservation at the genus level"),
        "gcs": orig(ms, "Embedding diversity was negatively correlated", "Embedding diversity was negatively correlated"),
        "fig3": from_sentence(ms.rejected_text(ms.find("Across 375 isolate-level hold-outs")), "Across 375"),
        "s3": orig(ms, "Caulobacter sits at the extremes", "Caulobacter sits at the extremes")
        + " " + orig(ms, "Caulobacter sits at the extremes", "Delftia is the opposite")
        + " " + orig(ms, "Caulobacter sits at the extremes", "Although these anchor points"),
        "disc": orig(ms, "range from high consistency (0.166", "The predicted metabolic diversity of these genera"),
        "cap2a": orig(ms, "BPR mean cosine similarity = 0.59", "with same-genus pairs showing the highest similarity"),
        "cap2b": orig(ms, "negative correlation (Spearman rho = -0.73)", "The negative correlation"),
    }
    for old, new, ctx in manuscript_edits(n):
        ms.replace(old, new, ctx)
    for old, new, ctx in si_edits(n):
        si.replace(old, new, ctx)

    def acc(doc, snippet, fragment=None):
        return sentence_containing(doc.accepted_text(doc.find(snippet)), fragment or snippet)

    modified = {
        "abstract": acc(ms, "In a hierarchical hold-out benchmark"),
        "fig2": acc(ms, "Isolates of the same genus were substantially") + " "
        + acc(ms, "Isolates of the same genus were substantially", "We quantified conservation at the genus level"),
        "gcs": acc(ms, "Embedding diversity was negatively correlated"),
        "fig3": from_sentence(ms.accepted_text(ms.find("Across 375 isolate-level hold-outs")), "Across 375"),
        "s3": acc(ms, "The isolate with the lowest ROC-AUC") + " " + acc(ms, "The isolate with the lowest ROC-AUC", "Across the"),
        "disc": acc(ms, "range from high consistency", "The predicted metabolic diversity of these genera"),
        "cap2a": acc(ms, "BPR mean cosine similarity", "with same-genus pairs showing the highest similarity"),
        "cap2b": acc(ms, "The negative correlation (Spearman rho"),
    }
    L = {k: lines.lines(v) for k, v in original.items()}

    # images: Fig. 2, Fig. 3 (manuscript); Supplementary Figs. 2 and 3 (SI). Supplementary Fig. 5 untouched.
    ms.replace_part("word/media/image2.png", fit_png(FIG2_PNG, ms.read_part("word/media/image2.png")))
    ms.replace_part("word/media/image3.png", fit_png(FIG3_PNG, ms.read_part("word/media/image3.png")))
    si.replace_part("word/media/image2.png", fit_png(S2_PNG, si.read_part("word/media/image2.png")))
    si.replace_part("word/media/image5.png", fit_png(S3_PNG, si.read_part("word/media/image5.png")))

    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    # R1-1 quotes of the Abstract numbers
    letter.set_text("with a median ROC-AUC declining from 0.90 to 0.76",
                    f"with a median ROC-AUC declining from {f2(n['f3']['isolate']['median_auc_bpr_all'])} to "
                    f"{f2(n['f3']['order']['median_auc_bpr_all'])}")
    letter.set_text("(mean ROC-AUC = 0.87)", f"(mean ROC-AUC = {f2(n['s3_mean'])})", exact=True)
    # R3 minor 4: response and quotes
    text = letter.response_text("Minor comment 4.")
    old_sentence = ("two isolates (Delftia GW456-R20 and Cupriavidus FW507-4G11) had earlier profiles in the training "
                    "data; we removed them and retrained (20 models), which left the results essentially unchanged "
                    "(mean ROC-AUC 0.88).")
    assert old_sentence in text, "R3 minor 4 response wording changed"
    letter.set_response("Minor comment 4.", text.replace(old_sentence, (
        "three isolates (Delftia GW456-R20, Cupriavidus FW507-4G11, and Serratia MT49, listed as MT049) "
        "had earlier profiles in the training data; we removed them and retrained (20 models), which "
        f"left the results essentially unchanged (mean ROC-AUC {f2(n['s3_mean'])} with and without these profiles).")))
    letter.set_text("absent from training confirmed this generalization (mean ROC-AUC = 0.88).",
                    acc(ms, "A prospective experiment on 12"))
    results_quote = letter.set_text(
        "Here, a set of 12 isolates absent from the training data were grown",
        " ".join(acc(ms, "Here, a set of 12 isolates", frag) for frag in
                 ["Here, a set of 12 isolates", "We then compared these new data", "Across 20 independently"]))
    letter.add_pairs(results_quote, [
        (f"Original (Results, \"Phylogenetic conservation…\", final paragraph, Lines {L['s3']}):", original["s3"]),
        ("Modified:", modified["s3"]),
    ])
    letter.set_text("We next evaluated the performance of the model against 12 isolates absent",
                    " ".join(acc(ms, "We next evaluated the performance", frag) for frag in
                             ["We next evaluated the performance", "Caulobacter was the exception", "This is consistent with"]))
    culture = ms.accepted_text(ms.find("We selected 12 isolates that spanned"))
    letter.set_text("We selected 12 isolates that spanned different genera, 10 of which",
                    culture[culture.index("We selected 12"):culture.index("validation):") + len("validation):")])
    methods = ms.accepted_text(ms.find("To assess whether BPR predictions generalize")).strip()
    letter.set_text("To assess whether BPR predictions generalize to isolates absent",
                    methods[:methods.index("report the mean ± SD across models.") + len("report the mean ± SD across models.")])
    caption = si.accepted_text(si.find("None of these isolates were used to train the model")).strip()
    start = caption.index("None of these isolates")
    letter.set_text("None of these isolates were used to train the model; earlier profiles of two isolates",
                    caption[start:caption.index("(see SI Table 2).", start) + len("(see SI Table 2).")])
    # R3 major 1: the remaining number changes that follow from the retrained model
    minor1 = next(p for p in letter.paragraphs() if letter.accepted_text(p).startswith("Minor comment 1. The abstract"))
    letter.add_pairs(minor1.getprevious(), [
        (f"Original (Abstract, Lines {L['abstract']}):", original["abstract"]), ("Modified:", modified["abstract"]),
        (f"Original (Results, \"Phylogenetic conservation…\", Lines {L['fig2']}):", original["fig2"]),
        ("Modified:", modified["fig2"]),
        (f"Original (Results, \"Phylogenetic conservation…\", Lines {L['gcs']}):", original["gcs"]),
        ("Modified:", modified["gcs"]),
        (f"Original (Results, \"Phylogenetic conservation…\", Lines {L['fig3']}):", original["fig3"]),
        ("Modified:", modified["fig3"]),
        (f"Original (Discussion, second paragraph, Lines {L['disc']}):", original["disc"]),
        ("Modified:", modified["disc"]),
        (f"Original (Figure 2 caption, Lines {L['cap2a']}):", original["cap2a"]), ("Modified:", modified["cap2a"]),
        (f"Original (Figure 2 caption, Lines {L['cap2b']}):", original["cap2b"]), ("Modified:", modified["cap2b"]),
    ])

    if apply:
        for doc in (ms, si, letter):
            print("backup:", doc.backup(BACKUPS))
            doc.save()
        write_sheets(TABLES, backup=True)
        print("applied to", DOCS)
    else:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for doc in (ms, si, letter):
            doc.save(preview_dir / doc.path.name)
        write_sheets(preview_dir / TABLES.name, backup=False)
        print("preview written to", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/lambda0_numbers_preview"))
    args = parser.parse_args()
    main(args.apply, args.preview_dir)
