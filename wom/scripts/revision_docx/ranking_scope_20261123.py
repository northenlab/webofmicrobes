# ranking_scope_20261123.py
"""Keep the paper to ranking metabolites for individual isolates (mSystems revision).

1. The pooled PR-AUC (all withheld pairs of all isolates in one set) is removed: it was never part
   of the paper's question. Gone from the Supplementary Methods, the response to Reviewer 1,
   comment 2, and the "Data for Figure 1b" sheet (columns pr_auc_pooled and prevalence).
2. Reviewer 1, comment 3: the response says that "calibrated" was a poorly chosen word, that the
   paper is about ranking, that Fig. S7 is a first step towards calibrated use, and that calibration
   is future work. The logistic-function slope is no longer mentioned.
3. The one place where the paper applies a score threshold (the agreement counts of Fig. S4, which
   use a BPR threshold of 4.0) now says so in the main text and states, there and in the Fig. S4
   caption, that the threshold is an assumption for a ranking model.

Plain (untracked) edits; all four documents are backed up first. Run once.

Usage:
    python ranking_scope_20261123.py            # dry run
    python ranking_scope_20261123.py --apply
"""

import re
import sys
import copy
import zipfile
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, SI, LETTER, TABLES  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

SI_POOLED = ("PR-AUC was averaged over isolates; computed over all withheld pairs pooled across isolates, which also "
             "rewards differences in score scale between isolates, PR-AUC at 32 dimensions was 0.66 for BPR, 0.69 for "
             "WR-MF, 0.68 for SVD, and 0.51 for LMF (means over five repeats).")
LETTER_POOLED = ("Because BPR is trained to rank metabolites within each isolate, we averaged PR-AUC over isolates; "
                 "computed over all withheld pairs pooled across isolates, PR-AUC at 32 dimensions was lower for BPR "
                 "(0.66) than for WR-MF (0.69) and SVD (0.68), as stated in the Supplementary Methods. ")
MS_OLD1 = "Briefly, the integrated model's predictions matched"
MS_NEW1 = "Briefly, with a BPR score threshold of 4.0 used to predict consumption, the integrated model's predictions matched"
MS_OLD2 = ", as shown in Fig. S4. Additionally"
MS_ADDED = "Because BPR is a ranking model, this threshold is an assumption and not a calibrated value (Fig. S7)."
S4_OLD = "predictive accuracy was evaluated against a BPR threshold of 4.0."
S4_ADDED = ("The threshold is an assumption, because BPR scores are ranking statistics and not calibrated "
            "probabilities (Fig. S7).")
R1C3 = (" We thank the reviewer for this request. \"Calibrated\" was a poorly chosen word, and we have removed it from "
        "the Abstract (see also Reviewer 3, Minor comment 1). Our intention in this manuscript is to rank metabolites "
        "for individual isolates. BPR scores are ranking statistics and not probabilities, so there is no predicted "
        "probability to place on the x-axis and no slope to compare with 1. The new Fig. S7 instead shows the observed "
        "labels against the BPR score itself: the withheld pairs of the Fig. 1b hold-out were grouped into ranges of "
        "BPR score, and the fraction of consumed pairs increased with the score, from less than 1% of the pairs with a "
        "score below −4 to 37% of those with a score of 2 to 4 and 92% of those with a score of 8 or more. Fig. S7 is "
        "a first step towards how the model might be used in a calibrated way; calibration itself is a goal for future "
        "work. Where the manuscript applies a score threshold (the comparison with published data in Fig. S4), we now "
        "state that the threshold is an assumption. Changes were made to the Abstract, the Methods section \"Model "
        "training\", the Results section \"{section}\", the Fig. S4 caption, and the new Fig. S7.")


def fix_ms(ms) -> str:
    p = ms.find(MS_OLD1)
    plain_replace(ms, MS_OLD1, MS_NEW1, p=p)
    plain_replace(ms, MS_OLD2, f", as shown in Fig. S4. {MS_ADDED} Additionally", p=p)
    text = ms.accepted_text(p)
    print("   …" + text[text.index(MS_NEW1):text.index("Additionally")] + "…")
    ps = ms.paragraphs()
    heading = next(ms.accepted_text(x).strip() for x in reversed(ps[:ps.index(p)])
                   if 0 < len(ms.accepted_text(x).strip()) < 150)
    print("   section:", heading)
    return heading


def fix_si(si):
    plain_replace(si, SI_POOLED, "PR-AUC was averaged over isolates.")
    plain_replace(si, S4_OLD, f"{S4_OLD} {S4_ADDED}")
    for key in ("PR-AUC was averaged over isolates.", S4_ADDED):
        text = si.accepted_text(si.find(key))
        print("   …" + text[max(0, text.index(key) - 200):text.index(key) + len(key)])


def fix_tables(apply: bool):
    """Delete columns E (pr_auc_pooled) and F (prevalence) of the hold-out table of the Fig. 1b sheet
    and move its columns G-I left. The leave-one-out table further down the sheet is left as it is."""
    import openpyxl
    from lxml import etree
    rows = list(openpyxl.load_workbook(TABLES, read_only=True)["Data for Figure 1b"].iter_rows(values_only=True))
    header = [v for v in rows[2] if v is not None]
    assert header[4:6] == ["pr_auc_pooled", "prevalence"] and len(header) == 9, header
    last = next(i for i, r in enumerate(rows, start=1) if r and isinstance(r[0], str) and r[0].startswith("Leave-one-out")) - 1
    assert last > 100, last
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    wb, rels = parts["xl/workbook.xml"].decode(), parts["xl/_rels/workbook.xml.rels"].decode()
    rid = re.search(r'<sheet [^>]*name="Data for Figure 1b"[^>]*r:id="([^"]+)"', wb).group(1)
    rel = next(r for r in re.findall(r"<Relationship [^>]*/>", rels) if f'Id="{rid}"' in r)
    part = "xl/" + re.search(r'Target="([^"]+)"', rel).group(1)
    root = etree.fromstring(parts[part])
    ns = {"m": root.nsmap[None]}
    move, removed = {"G": "E", "H": "F", "I": "G"}, 0
    for cell in root.xpath("//m:sheetData/m:row/m:c", namespaces=ns):
        col, row = re.fullmatch(r"([A-Z]+)(\d+)", cell.get("r")).groups()
        if int(row) > last:
            continue
        assert col in "ABCDEFGHI", cell.get("r")
        if col in "EF":
            cell.getparent().remove(cell)
            removed += 1
        elif col in move:
            cell.set("r", move[col] + row)
    assert not root.xpath("//m:mergeCell", namespaces=ns), "merged cells would need moving"
    parts[part] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"   {part}: rows 3-{last}: {removed} cells removed (columns pr_auc_pooled and prevalence); "
          f"new header {header[:4] + header[6:]}")
    if not apply:
        return
    tmp = TABLES.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(TABLES)


def fix_letter(letter, section: str):
    plain_replace(letter, LETTER_POOLED, "")
    resp = letter.find("Response: We thank the reviewer for this request. As noted in the response to Reviewer 3, Minor")
    full = letter.accepted_text(resp)
    assert "for reference, if each score is converted" in full, "the response is not the one written earlier today"
    plain_replace(letter, full[len("Response:"):], R1C3.format(section=section), p=resp)

    # quotes of the two new sentences, formatted like an existing label and quote
    ps = letter.paragraphs()
    k = ps.index(letter.find("Modified (Results, first section): (no original text; new insertion immediately after the "
                             "sentence ending \"…for all subsequent analyses.\")"))
    label_t, quote_t = ps[k], ps[k + 1]
    anchor = letter.find("Modified (Methods, Model training): see the response to Comment 4, which quotes")
    pairs = [("Modified (Results, final section): (no original text; new insertion immediately after the sentence "
              "ending \"…as shown in Fig. S4.\")", MS_ADDED),
             ("Modified (Supplementary Figure 4 caption): (no original text; new insertion immediately after the "
              "sentence ending \"…against a BPR threshold of 4.0.\")", S4_ADDED)]
    for label_text, quote_text in pairs:
        for template, new_text in ((label_t, label_text), (quote_t, quote_text)):
            p = copy.deepcopy(template)
            anchor.addnext(p)
            plain_replace(letter, letter.accepted_text(p), new_text, p=p)
            anchor = p
    ps = letter.paragraphs()
    k = ps.index(resp)
    for p in ps[k:k + 7]:
        print("   " + letter.accepted_text(p)[:1600])


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    ms, si, letter = (TrackedDocx(f, "unused") for f in (MS, SI, LETTER))
    print("Manuscript")
    section = fix_ms(ms)
    print("\nSupplementary Information")
    fix_si(si)
    print("\nResponse letter")
    fix_letter(letter, section)
    for name, doc in (("manuscript", ms), ("SI", si), ("letter", letter)):
        stale = [doc.accepted_text(p)[:80] for p in doc.paragraphs()
                 if re.search(r"pooled across isolates|logistic function", doc.accepted_text(p))]
        assert not stale, (name, stale)
    if apply:
        print(f"\nbackups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    print("\nSupplementary tables")
    fix_tables(apply)
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
