# score_ranges_20261123.py
"""Supplementary Fig. 7 with the BPR score itself on the x axis (mSystems revision; Reviewer 1, comment 3).

calibration_curve_20261123.py put the logistic function of the BPR score on the x axis. That is a
derived quantity the paper does not otherwise use. The figure now shows the observed fraction of
consumed pairs in ranges of BPR score (made by scripts/revision_r1_pr_calibration.py), and this
script brings the documents in line with it:

Supplementary Information   Fig. S7: new image and caption.
Supplementary tables        "Data for Figure S7": one row per score range.
Response letter             Reviewer 1, comment 3: the response describes the new figure. The slope
                            the reviewer asked for is given in the letter only.

Plain (untracked) edits; all four documents are backed up first. Run once, after
calibration_curve_20261123.py.

Usage:
    python score_ranges_20261123.py            # dry run
    python score_ranges_20261123.py --apply
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
from revision_paths import SI, LETTER, TABLES  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402
from r3_major1_edits import sheet_data_xml  # noqa: E402

PRC = Path(__file__).resolve().parents[3] / "wom/manuscript_figures/msystems_revision/lambda0/r1/pr_calibration"
S7_WIDTH = 4572000  # EMU (5 inches)
S7_TITLE_OLD = "Calibration curve for BPR scores."
R1C3_KEEP = r"and we have removed .calibrated. from the Abstract\."


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def numbers() -> dict:
    r = pd.read_csv(PRC / "consumption_by_score_range.csv")
    slopes = pd.read_csv(PRC / "calibration_slope_by_repeat.csv")
    assert r.n.sum() == slopes.n_pairs.sum()
    assert list(r.score_range[[0, 1, 5, 8]]) == ["< −6", "−6 to −4", "2 to 4", "≥ 8"], list(r.score_range)
    below = r.n_consumed[:2].sum() / r.n[:2].sum()  # scores below −4
    assert below < 0.01 and (r.observed[1:].diff().dropna() > 0).all(), "the text says consumption rises with the score"
    return {"ranges": r, "n_pairs": f"{int(r.n.sum()):,}", "n_min": f"{int(r.n.min()):,}", "n_max": f"{int(r.n.max()):,}",
            "range_min": r.score_range[r.n.idxmin()], "range_max": r.score_range[r.n.idxmax()],
            "mid": pct(r.observed[5]), "top": pct(r.observed[8]), "slope": f"{slopes.slope.mean():.2f}"}


def texts(n: dict) -> dict:
    return {
        "s7_title": "Observed consumption by BPR score.",
        "s7_body": (f" The withheld pairs of the Fig. 1b hold-out (BPR, 32 dimensions; {n['n_pairs']} pairs, five "
                    "repeats pooled) were grouped by BPR score in ranges of two score units. Bars, fraction of the "
                    "pairs in each range that were consumed; error bars, Wilson 95% confidence intervals. The ranges "
                    f"contain between {n['n_min']} ({n['range_min']}) and {n['n_max']} ({n['range_max']}) pairs. BPR "
                    "scores are ranking statistics and not probabilities of consumption."),
        "r1c3": (" BPR does not output a class-membership probability, so we have plotted the observed labels "
                 "against the BPR score itself (new Fig. S7). The withheld pairs of the Fig. 1b hold-out were grouped "
                 "into ranges of BPR score, and the fraction of consumed pairs is shown for each range. The fraction "
                 "consumed increased with the score, from less than 1% of the pairs with a score below −4 to "
                 f"{n['mid']} of those with a score of 2 to 4 and {n['top']} of those with a score of 8 or more. "
                 "Because the scores are not probabilities, there is no 30% bin, and a slope of 1 is not expected; "
                 "for reference, if each score is converted to a value between 0 and 1 with the logistic function, "
                 f"the slope of the un-binned labels on that value is {n['slope']}. The scores should therefore be "
                 "read as a ranking and not as probabilities of consumption, and the Methods now state this with a "
                 "reference to Fig. S7. Calibrating the scores is a goal for future work. Changes were made to the "
                 "Abstract, the Methods section \"Model training\", and the new Fig. S7."),
    }


def fix_si(si, t: dict):
    s7 = si.find("Supplementary Figure 7.")
    body = si.accepted_text(s7).split(S7_TITLE_OLD)[1]
    plain_replace(si, S7_TITLE_OLD, t["s7_title"], p=s7)
    plain_replace(si, body, t["s7_body"], p=s7)
    from PIL import Image
    png = (PRC / "figS7_score_ranges.png").read_bytes()
    w, h = Image.open(io.BytesIO(png)).size
    si.replace_part("word/media/image7.png", png)
    si.set_image_extent("image7.png", S7_WIDTH, round(S7_WIDTH * h / w))
    print(f"   image7.png: figS7_score_ranges.png {w}×{h}")
    print("   " + si.accepted_text(s7))


def fix_tables(n: dict, apply: bool):
    rows = [["Data for Figure S7: observed consumption by BPR score (Fig. 1b hold-out, 32 dimensions, five repeats "
             "pooled). Each range includes its lower bound and excludes its upper bound."], [],
            ["bpr_score_range", "n_pairs", "n_consumed", "observed_fraction_consumed", "ci95_low", "ci95_high"]]
    for _, r in n["ranges"].iterrows():
        rows.append([r.score_range, int(r.n), int(r.n_consumed), round(r.observed, 4), round(r.ci_low, 4),
                     round(r.ci_high, 4)])
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    wb, rels = parts["xl/workbook.xml"].decode(), parts["xl/_rels/workbook.xml.rels"].decode()
    rid = re.search(r'<sheet [^>]*name="Data for Figure S7"[^>]*r:id="([^"]+)"', wb).group(1)
    rel = next(r for r in re.findall(r"<Relationship [^>]*/>", rels) if f'Id="{rid}"' in r)
    part = "xl/" + re.search(r'Target="([^"]+)"', rel).group(1)
    xml = parts[part].decode()
    assert "calibration curve for BPR scores" in xml, "the S7 sheet is not the one written by calibration_curve_20261123.py"
    parts[part] = re.sub(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(rows), xml, flags=re.S).encode()
    for row in rows[2:]:
        print("   ", row)
    if not apply:
        return
    tmp = TABLES.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(TABLES)


def fix_letter(letter, t: dict):
    resp = letter.find("Response: We thank the reviewer for this request. As noted in the response to Reviewer 3, Minor")
    full = letter.accepted_text(resp)
    assert "ventiles of predicted value" in full, "the response is not the one written by calibration_curve_20261123.py"
    plain_replace(letter, full[re.search(R1C3_KEEP, full).end():], t["r1c3"], p=resp)
    print("   " + letter.accepted_text(resp))


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    n = numbers()
    t = texts(n)
    si, letter = TrackedDocx(SI, "unused"), TrackedDocx(LETTER, "unused")
    print("Supplementary Information")
    fix_si(si, t)
    print("\nResponse letter")
    fix_letter(letter, t)
    for name, doc in (("SI", si), ("letter", letter)):
        stale = [doc.accepted_text(p)[:80] for p in doc.paragraphs()
                 if re.search(r"ventile|predicted value|Platt|isotonic|Brier", doc.accepted_text(p))
                 and not doc.accepted_text(p).startswith("Comment 3.")]
        assert not stale, (name, stale)
    if apply:
        print(f"\nbackups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    print("\nSupplementary tables")
    fix_tables(n, apply)
    if not apply:
        print("\ndry run: nothing written")
        return
    si.save()
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
