# plain_bpr_20261123.py
"""Report BPR trained without the measured-neutral labels, the like-for-like comparison (mSystems revision).

The adopted BPR draws half of its negatives from each isolate's measured-neutral metabolites; WR-MF,
SVD and LMF are fitted to the consumed pairs only. The benchmark also ran BPR without the labels
(negatives drawn at random from every metabolite the isolate is not known to consume), which is the
comparison on equal information. It was not reported.

Supplementary Information   Model benchmarking: one sentence with the hold-out ROC-AUC of that variant.
Supplementary tables        "Data for Figure 1b": its 30 hold-out rows (6 dimensions x 5 repeats) and
                            a mention in the sheet's description.
Response letter             Reviewer 3, major comment 1: the quote of the benchmarking paragraph
                            follows the SI, and the response points to the comparison.

Plain (untracked) edits; all four documents are backed up first. Run once.

Usage:
    python plain_bpr_20261123.py            # dry run
    python plain_bpr_20261123.py --apply
"""

import re
import sys
import copy
import zipfile
import argparse
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import SI, LETTER, TABLES  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

RESULTS = (Path(__file__).resolve().parents[3] /
           "wom/manuscript_figures/msystems_revision/lambda0/fig1b/fig1b_benchmark_results.csv")
ADOPTED, PLAIN = "Ordinal-BPR [lambda_ord=0]", "BPR"  # names in the results file
SHEET_NAME = "BPR without measured-neutral labels (negatives drawn from all non-consumed metabolites)"

SI_ANCHOR = "and the mean ± standard deviation across repeats is reported."
SI_NEW = ("When BPR was trained without distinguishing measured-neutral from untested metabolites, so that negatives "
          "were drawn at random from all metabolites the isolate was not known to consume, hold-out ROC-AUC was "
          "{p32} at 32 dimensions and {p128} at 128 dimensions, compared with {a32} and {a128} for the model as used "
          "and {w32} and {w128} for WR-MF (SI Tables, Data for Figure 1b).")
RESP_ANCHOR = "so we removed it and now call the model BPR."
RESP_NEW = ("For a comparison with the other methods on equal information, the Supplementary Methods also report BPR "
            "trained without the measured-neutral labels ({p32} at 32 dimensions).")
TITLE_OLD = "the BPR variant with an additional ordinal term is reported for comparison (Supplementary Methods)."
TITLE_NEW = ("the BPR variant with an additional ordinal term and BPR trained without the measured-neutral labels are "
             "reported for comparison (Supplementary Methods).")


def holdout() -> pd.DataFrame:
    r = pd.read_csv(RESULTS)
    return r[r.protocol == "B_holdout"]


def numbers() -> dict:
    mean = holdout().groupby(["model", "dim"]).roc_auc.mean()
    n = {"p32": mean[PLAIN, 32], "p128": mean[PLAIN, 128], "a32": mean[ADOPTED, 32], "a128": mean[ADOPTED, 128],
         "w32": mean["WR-MF", 32], "w128": mean["WR-MF", 128]}
    assert n["a32"] > n["p32"] > n["w32"] and n["a128"] > n["p128"] > n["w128"]
    out = {k: f"{v:.2f}" for k, v in n.items()}
    assert out == {"p32": "0.83", "p128": "0.77", "a32": "0.86", "a128": "0.83", "w32": "0.77", "w128": "0.62"}, out
    return out


def fix_tables(apply: bool):
    plain = holdout().query("model == @PLAIN").sort_values(["dim", "repeat"])
    assert len(plain) == 30 and plain.known_negatives_passed.isna().all()
    values = [[int(x.dim), int(x.repeat), round(float(x.roc_auc), 6), int(x.n_microbes), int(x.n_pos), int(x.n_neg)]
              for x in plain.itertuples()]
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    shared = parts["xl/sharedStrings.xml"].decode("utf-8")
    assert shared.count(TITLE_OLD) == 1
    parts["xl/sharedStrings.xml"] = shared.replace(TITLE_OLD, TITLE_NEW).encode("utf-8")

    wb, rels = parts["xl/workbook.xml"].decode(), parts["xl/_rels/workbook.xml.rels"].decode()
    rid = re.search(r'<sheet [^>]*name="Data for Figure 1b"[^>]*r:id="([^"]+)"', wb).group(1)
    rel = next(r for r in re.findall(r"<Relationship [^>]*/>", rels) if f'Id="{rid}"' in r)
    part = "xl/" + re.search(r'Target="([^"]+)"', rel).group(1)
    root = etree.fromstring(parts[part])
    m = root.nsmap[None]
    ns = {"m": m}
    assert not root.xpath("//m:mergeCell", namespaces=ns)

    import openpyxl
    cells = [list(r) for r in openpyxl.load_workbook(TABLES, read_only=True)["Data for Figure 1b"].iter_rows(values_only=True)]
    title_row = next(i for i, r in enumerate(cells, start=1) if r and isinstance(r[0], str) and r[0].startswith("Leave-one-out"))
    last = title_row - 2  # last row of the hold-out table; a blank row follows it
    assert cells[last - 1][0].startswith("BPR with ordinal term") and cells[last - 1][1] == 128, cells[last - 1]
    assert not any(str(r[0]).startswith("BPR without") for r in cells if r and r[0]), "the rows are already present"

    rows = {int(r.get("r")): r for r in root.xpath("//m:sheetData/m:row", namespaces=ns)}
    template = rows[last]
    assert [re.match(r"[A-Z]+", c.get("r")).group(0) for c in template] == list("ABCDEFG"), "unexpected columns"
    for number in sorted(rows, reverse=True):  # move everything below the hold-out table down
        if number > last:
            row = rows[number]
            row.set("r", str(number + len(values)))
            for c in row:
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                c.set("r", f"{col}{number + len(values)}")
    anchor = template
    for k, vals in enumerate(values, start=1):
        row = copy.deepcopy(template)
        row.set("r", str(last + k))
        for c, value in zip(row, [SHEET_NAME] + vals):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            c.set("r", f"{col}{last + k}")
            for child in list(c):
                c.remove(child)
            if isinstance(value, str):
                c.set("t", "inlineStr")
                t = etree.SubElement(etree.SubElement(c, f"{{{m}}}is"), f"{{{m}}}t")
                t.text = value
            else:
                c.attrib.pop("t", None)
                etree.SubElement(c, f"{{{m}}}v").text = repr(value)
        anchor.addnext(row)
        anchor = row
    for dim in root.xpath("//m:dimension", namespaces=ns):
        dim.set("ref", re.sub(r":[A-Z]+\d+$", f":G{max(rows) + len(values)}", dim.get("ref")))
    parts[part] = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"   {part}: {len(values)} rows added after row {last}; first {[SHEET_NAME[:35] + '…'] + values[0]}")
    if not apply:
        return
    tmp = TABLES.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(TABLES)


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    n = numbers()
    si, letter = TrackedDocx(SI, "unused"), TrackedDocx(LETTER, "unused")
    sentence = SI_NEW.format(**n)
    for doc in (si, letter):
        hits = [p for p in doc.paragraphs() if SI_ANCHOR in doc.accepted_text(p)]
        assert len(hits) == 1, (doc.path.name, len(hits))  # in the letter: the blue quote of the revised paragraph
        plain_replace(doc, SI_ANCHOR, f"{SI_ANCHOR} {sentence}", p=hits[0])
    resp = letter.find("Response: We thank the reviewer for this careful reading of the benchmark and its code.")
    plain_replace(letter, RESP_ANCHOR, f"{RESP_ANCHOR} {RESP_NEW.format(**n)}", p=resp)
    print("Supplementary Information\n   …" + sentence)
    text = letter.accepted_text(resp)
    print("Response letter\n   …" + text[text.index("In the hold-out the ordinal term"):text.index("We retrained the model")])
    if apply:
        print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    print("Supplementary tables")
    fix_tables(apply)
    if not apply:
        print("dry run: nothing written")
        return
    si.save()
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
