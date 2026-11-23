# calibration_curve_20261123.py
"""Reviewer 1, comment 3: show the calibration curve the reviewer asked for and nothing more (mSystems revision).

The reviewer asked for the model's predicted values against the observed labels in binned groups
(deciles, ventiles), with the slope of the un-binned labels on the predicted values. The first
revision instead rescaled the scores (Platt scaling, isotonic regression) and reported the rescaled
probabilities. This script puts the requested plot in as Supplementary Fig. 7 (made by
scripts/revision_r1_pr_calibration.py) and takes the rescaling out everywhere:

Manuscript
  * Results: the sentence on Platt-scaled probabilities is deleted.
  * Methods: "Where a probability is needed ... monotone calibration" is deleted; the statement that
    the score is not a calibrated probability now cites Fig. S7.
Supplementary Information
  * Supplementary Methods: the calibration paragraph is deleted; the learning curve is evaluated by
    ROC-AUC and PR-AUC only.
  * Fig. S7: new image and caption. Fig. S8: image without the Brier-score panel; caption shortened.
Supplementary tables
  * "Data for Figure S7": the ventiles and the un-binned slope. "Data for Figure S8": no Brier or
    calibration columns.
Response letter
  * Reviewer 1, comment 3: new response (the plot shows that the scores are not calibrated;
    calibration is a goal for future work). Its quoted Results text, now only the learning-curve
    sentence, moves to comment 6; comment 6 no longer reports calibration.
  * The quoted Methods paragraph (comment 4) and Results paragraph (Reviewer 3, major comment 1)
    follow the manuscript.

Plain (untracked) edits; all four documents are backed up first. Run once.

Usage:
    python calibration_curve_20261123.py            # dry run
    python calibration_curve_20261123.py --apply
"""

import io
import re
import sys
import zipfile
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, SI, LETTER, TABLES  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402
from r3_major1_edits import sheet_data_xml  # noqa: E402

R1 = Path(__file__).resolve().parents[3] / "wom/manuscript_figures/msystems_revision/lambda0/r1"
PRC, LC = R1 / "pr_calibration", R1 / "learning_curve"
SI_WIDTH = 5669280   # EMU; width of the other Supplementary figures
S7_WIDTH = 3429000   # a single square panel

RESCALING = ("BPR scores are ranking statistics, but a monotone calibration (Platt scaling) fitted on the withheld "
             "pairs of other isolates converted them into well-calibrated probabilities of consumption (slope of the "
             "observed outcome on the predicted probability = 1.00; Fig. S7). ")
METHODS_OLD = (". Where a probability is needed, scores can be mapped to probabilities of consumption with a monotone "
               "calibration fitted on held-out pairs (Fig. S7).")
METHODS_NEW = " (Fig. S7)."
METHODS_CONTEXT = "We framed the prediction of microbial metabolite consumption as an implicit-feedback"


def numbers() -> dict:
    groups = pd.read_csv(PRC / "calibration_ventiles.csv")
    slopes = pd.read_csv(PRC / "calibration_slope_by_repeat.csv")
    assert len(groups) == 20 and groups.n.max() - groups.n.min() <= 1, "not ventiles"
    assert groups.n.sum() == slopes.n_pairs.sum()
    g30 = groups.loc[(groups.mean_predicted - 0.30).abs().idxmin()]
    assert f"{g30.mean_predicted:.2f}" == "0.30", g30.mean_predicted
    slope = slopes.slope.mean()
    assert slope < 0.6 and (groups.observed < groups.mean_predicted).all(), "the text says the scores are not calibrated"
    return {"groups": groups, "slopes": slopes, "slope": f"{slope:.2f}", "n_pairs": f"{int(groups.n.sum()):,}",
            "pct30": f"{100 * g30.observed:.0f}%", "lc": pd.read_csv(LC / "learning_curve_summary.csv")}


def texts(n: dict) -> dict:
    return {
        "s7_title": "Calibration curve for BPR scores.",
        "s7_body": (" BPR does not output a probability of consumption; the logistic function of the BPR score was "
                    f"used as the predicted value. The withheld pairs of the Fig. 1b hold-out (BPR, 32 dimensions; "
                    f"{n['n_pairs']} pairs, five repeats pooled) were sorted by predicted value and divided into 20 "
                    "equal-sized groups (ventiles). Stars, observed fraction of consumed pairs against the mean "
                    "predicted value of each group, with Wilson 95% confidence intervals; dashed line, a calibrated "
                    "model. The slope of the least-squares regression of the un-binned labels (consumed = 1, "
                    f"neutral = 0) on the predicted values was {n['slope']} (1 for a calibrated model): BPR scores "
                    "are ranking statistics and not calibrated probabilities."),
        "r1c3": (" We have made the plot the reviewer describes (new Fig. S7). BPR does not output a "
                 "class-membership probability, so we used the closest equivalent, the logistic function of the BPR "
                 "score, as the predicted value. The withheld pairs of the Fig. 1b hold-out were divided into "
                 "ventiles of predicted value, and the observed fraction of consumed pairs in each ventile was "
                 "plotted against its mean predicted value. The scores are not calibrated: in the ventile with a "
                 f"mean predicted value of 0.30, {n['pct30']} of the pairs were consumed, and the slope of the "
                 f"un-binned labels on the predicted values was {n['slope']}, not 1. The scores should therefore be "
                 "read as a ranking and not as probabilities of consumption, and the Methods now state this with a "
                 "reference to Fig. S7. Calibrating the scores is a goal for future work. Changes were made to the "
                 "Abstract, the Methods section \"Model training\", and the new Fig. S7."),
        "r1c6_old": ("and recorded ROC-AUC, PR-AUC, and the Brier score after cross-fitted Platt scaling (Fig. S8). "
                     "All three improved steadily with more data and had not reached a plateau at the size of the "
                     "current dataset (ROC-AUC = 0.52, 0.78, and 0.86 with 5%, 50%, and 100% of the training pairs; "
                     "PR-AUC = 0.41, 0.59, and 0.69; Brier score from 0.132, the value of predicting the prevalence "
                     "alone, to 0.082). Once a Platt map was fitted, probabilities were well calibrated (slope within "
                     "0.1 of 1) from 8% of the matrix onwards."),
        "r1c6_new": ("and recorded ROC-AUC and PR-AUC (Fig. S8). Both improved steadily with more data and had not "
                     "reached a plateau at the size of the current dataset (ROC-AUC = 0.52, 0.78, and 0.86 with 5%, "
                     "50%, and 100% of the training pairs; PR-AUC = 0.41, 0.59, and 0.69). We do not report "
                     "calibration as a function of training-set size, because BPR scores are not calibrated "
                     "probabilities (see the response to Comment 3); calibration is a goal for future work."),
    }


def remove(p):
    p.getparent().remove(p)


# ----------------------------------------------------------------------------- manuscript
def fix_ms(ms):
    results = ms.find(RESCALING[:70])
    plain_replace(ms, RESCALING, "", p=results)
    plain_replace(ms, METHODS_OLD, METHODS_NEW, context=METHODS_CONTEXT)
    for p in (results, ms.find(METHODS_CONTEXT)):
        text = ms.accepted_text(p)
        k = text.index("In a learning curve") if p is results else text.index("The score s")
        print("   …" + text[max(0, k - 60):k + 230] + "…")


# ----------------------------------------------------------------------------- Supplementary Information
def png_size(png: bytes) -> tuple[int, int]:
    from PIL import Image
    return Image.open(io.BytesIO(png)).size


def fix_si(si, t: dict):
    remove(si.find("To assess calibration, the BPR scores (32 dimensions)"))
    plain_replace(si, "by ROC-AUC, PR-AUC, and the Brier score after cross-fitted Platt scaling.", "by ROC-AUC and PR-AUC.",
                  context="For the learning curve, BPR (32 dimensions)")
    s7 = si.find("Supplementary Figure 7.")
    body = si.accepted_text(s7).split("Calibration of BPR scores.")[1]
    plain_replace(si, "Calibration of BPR scores.", t["s7_title"], p=s7)
    plain_replace(si, body, t["s7_body"], p=s7)
    plain_replace(si, " (c) Brier score after cross-fitted Platt scaling (lower is better) and the Brier score of "
                  "predicting the prevalence alone.", "", context="Supplementary Figure 8.")
    for media, png, width in [("image7.png", PRC / "figS7_calibration.png", S7_WIDTH),
                              ("image8.png", LC / "figS8_learning_curve.png", SI_WIDTH)]:
        data = png.read_bytes()
        w, h = png_size(data)
        si.replace_part(f"word/media/{media}", data)
        si.set_image_extent(media, width, round(width * h / w))
        print(f"   {media}: {png.name} {w}×{h}")
    for key in ("For the learning curve, BPR (32 dimensions)", "Supplementary Figure 7.", "Supplementary Figure 8."):
        print("   " + si.accepted_text(si.find(key)))


# ----------------------------------------------------------------------------- Supplementary tables
def sheet_rows(n: dict) -> dict:
    s7 = [["Data for Figure S7: calibration curve for BPR scores (Fig. 1b hold-out, 32 dimensions). The predicted "
           "value of a withheld pair is the logistic function of its BPR score."], [],
          ["Ventiles of predicted value, five repeats pooled"],
          ["ventile", "predicted_low", "predicted_high", "n_pairs", "mean_predicted", "observed_fraction_consumed",
           "ci95_low", "ci95_high"]]
    for _, r in n["groups"].iterrows():
        s7.append([int(r["group"]), round(r.predicted_low, 4), round(r.predicted_high, 4), int(r.n),
                   round(r.mean_predicted, 4), round(r.observed, 4), round(r.ci_low, 4), round(r.ci_high, 4)])
    s7 += [[], ["Least-squares regression of the un-binned label (consumed = 1, neutral = 0) on the predicted value"],
           ["repeat", "slope", "intercept", "n_pairs"]]
    sl = n["slopes"]
    for _, r in sl.iterrows():
        s7.append([int(r["repeat"]), round(r.slope, 4), round(r.intercept, 4), int(r.n_pairs)])
    s7.append(["mean (SD)", f"{sl.slope.mean():.3f} ({sl.slope.std():.3f})",
               f"{sl.intercept.mean():.3f} ({sl.intercept.std():.3f})"])

    cols = ["fraction", "n_train_pairs_mean", "pct_of_matrix_mean", "roc_auc_mean", "roc_auc_sd", "pr_auc_mean",
            "pr_auc_sd", "popularity_roc_auc_mean", "genus_roc_auc_mean", "popularity_pr_auc_mean",
            "genus_pr_auc_mean", "prevalence_mean"]
    s8 = [["Data for Figure S8: BPR learning curve on the Fig. 1b hold-out (mean and SD across five repeats)."], [],
          ["fraction_of_training_pairs", "n_training_pairs", "pct_of_matrix", "roc_auc_mean", "roc_auc_sd",
           "pr_auc_mean", "pr_auc_sd", "popularity_roc_auc", "genus_roc_auc", "popularity_pr_auc", "genus_pr_auc",
           "prevalence"]]
    for _, r in n["lc"].iterrows():
        s8.append([round(float(r[c]), 4) if c != "n_train_pairs_mean" else int(round(r[c])) for c in cols])
    return {"Data for Figure S7": s7, "Data for Figure S8": s8}


def check_s8_unchanged(rows: list):
    """Every learning-curve value kept in the new sheet must equal the value in the current sheet."""
    import openpyxl
    old = [list(r) for r in openpyxl.load_workbook(TABLES, data_only=True)["Data for Figure S8"].iter_rows(values_only=True)]
    header = old[2]
    for new_row, old_row in zip(rows[3:], old[3:], strict=True):
        for name, value in zip(rows[2], new_row):
            assert old_row[header.index(name)] == value, (name, old_row[header.index(name)], value)


def fix_tables(n: dict, apply: bool):
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    wb, rels = parts["xl/workbook.xml"].decode(), parts["xl/_rels/workbook.xml.rels"].decode()
    new = sheet_rows(n)
    check_s8_unchanged(new["Data for Figure S8"])
    for name, rows in new.items():
        rid = re.search(rf'<sheet [^>]*name="{name}"[^>]*r:id="([^"]+)"', wb).group(1)
        rel = next(r for r in re.findall(r"<Relationship [^>]*/>", rels) if f'Id="{rid}"' in r)
        part = "xl/" + re.search(r'Target="([^"]+)"', rel).group(1)
        xml = parts[part].decode()
        assert xml.count("<sheetData>") == 1
        parts[part] = re.sub(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(rows), xml, flags=re.S).encode()
        print(f"   {name} ({part}): {len(rows)} rows; header {rows[3] if 'S7' in name else rows[2]}")
    if not apply:
        return
    tmp = TABLES.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(TABLES)


# ----------------------------------------------------------------------------- response letter
def fix_letter(letter, t: dict):
    ps = letter.paragraphs()
    resp = letter.find("Response: We thank the reviewer for this request. As noted in the response to Reviewer 3, Minor")
    i = ps.index(resp)
    results_label, results_quote, methods_label, methods_quote, nxt = ps[i + 1:i + 6]
    text = [letter.accepted_text(p) for p in (results_label, results_quote, methods_label, methods_quote, nxt)]
    assert text[0].startswith("Modified (Results, first section): (no original text; new insertion"), text[0]
    assert text[1].startswith(RESCALING) and text[1].endswith("Fig. S8)."), text[1]
    assert text[2].startswith("Modified (Methods, Model training): (no original text"), text[2]
    assert text[3].startswith("Where a probability is needed"), text[3]
    assert text[4].startswith("Comment 4."), text[4]

    # comment 3: new response after its first two sentences
    full = letter.accepted_text(resp)
    kept = re.search(r"and we have removed .calibrated. from the Abstract\.", full)
    plain_replace(letter, full[kept.end():], t["r1c3"], p=resp)

    # the quoted Results text is now only the learning-curve sentence: it belongs to comment 6
    cross = letter.find("Modified (Results, first section): see the response to Comment 3, which quotes")
    cross.addprevious(results_label)
    cross.addprevious(results_quote)
    remove(cross)
    quotes = [p for p in letter.paragraphs() if RESCALING in letter.accepted_text(p)]
    assert len(quotes) == 2, len(quotes)  # the moved quote and the full paragraph under Reviewer 3, major comment 1
    for p in quotes:
        plain_replace(letter, RESCALING, "", p=p)

    # comment 3: the Methods change is part of the paragraph quoted under comment 4
    prefix = "Modified (Methods, Model training): "
    plain_replace(letter, text[2][len(prefix):], "see the response to Comment 4, which quotes the rewritten paragraph.",
                  p=methods_label)
    remove(methods_quote)
    plain_replace(letter, METHODS_OLD, METHODS_NEW, context=METHODS_CONTEXT)

    # comment 6: no calibration
    c6 = "Response: We thank the reviewer for this suggestion. We computed a learning curve"
    plain_replace(letter, t["r1c6_old"], t["r1c6_new"], context=c6)

    ps = letter.paragraphs()
    for start in ("Comment 3. In the abstract", "Comment 6. I am also interested"):
        k = next(j for j, p in enumerate(ps) if letter.accepted_text(p).startswith(start))
        for p in ps[k + 1:k + 5]:
            s = letter.accepted_text(p)
            print("   " + (s if len(s) < 1500 else s[:700] + " […] " + s[-250:]))
        print()


# ----------------------------------------------------------------------------- main
def leftovers(docs: dict):
    for name, doc in docs.items():
        hits = [doc.accepted_text(p) for p in doc.paragraphs()]
        banned = [h[:80] for h in hits if re.search(r"Platt|isotonic|Brier", h)]
        assert not banned, (name, banned)
        for h in hits:
            for m in re.finditer(r"calibrat", h, flags=re.I):
                print(f"   [{name}] …{h[max(0, m.start() - 70):m.end() + 60]}…")


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    n = numbers()
    t = texts(n)
    ms, si, letter = (TrackedDocx(f, "unused") for f in (MS, SI, LETTER))
    print("Manuscript")
    fix_ms(ms)
    print("\nSupplementary Information")
    fix_si(si, t)
    print("\nResponse letter")
    fix_letter(letter, t)
    print("Remaining mentions of calibration")
    leftovers({"manuscript": ms, "SI": si, "letter": letter})
    if apply:
        print(f"\nbackups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    print("\nSupplementary tables")
    fix_tables(n, apply)
    if not apply:
        print("\ndry run: nothing written")
        return
    for doc in (ms, si, letter):
        doc.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
