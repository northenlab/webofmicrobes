# wording_fixes_20261116.py
"""Replace "accurately" for the agent examples with the measured agreement (mSystems revision).

With the revised model, the Prompt 1 predictions match 20 of 22 substrates (90.9%) and the Prompt 2
predictions 35 of 48 genus-compound pairs (72.9%) (Supplementary Fig. 4a and 4c; recomputed from the
two Fig. S4 data sheets before editing). The main text now gives these numbers instead of
"accurately predicted", and the Fig. S4 title and its two data-sheet titles are descriptive.
The response to Reviewer 3, major comment 1, mentions the numbers. Plain (untracked) edits to the
documents in ~/Downloads/revisions; back up the files before running.

Usage:
    python wording_fixes_20261116.py --apply
"""

import sys
import zipfile
import argparse
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import TrackedDocx, LetterDocx  # noqa: E402
from final_fixes_20261116 import DOCS, MS, SI, LETTER, TABLES, plain_replace  # noqa: E402

THRESHOLD = 4.0  # Supplementary Fig. 4: predicted consumed if the BPR score exceeds 4.0


def agreement() -> tuple[tuple[int, int], tuple[int, int]]:
    """(correct, total) for Fig. S4a (substrates) and Fig. S4c (genus-compound pairs)."""
    import openpyxl
    wb = openpyxl.load_workbook(TABLES, data_only=True)
    a = [r[:4] for r in wb["Data for Figure S4a and S4b"].iter_rows(min_row=9, values_only=True)]
    a = [r for r in a if r[0] and isinstance(r[1], (int, float)) and isinstance(r[3], (int, float)) and r[2] is not None]
    a = a[:22]
    groups = collections.defaultdict(list)
    for r in wb["Data for Figure S4c and S4d"].iter_rows(values_only=True):
        if r[0] and isinstance(r[3], (int, float)) and isinstance(r[2], (int, float)):
            groups[(r[0].split()[0], r[1])].append((r[2], r[4]))
    c = [(v[0][0], sum(x[1] for x in v) > len(v) / 2) for v in groups.values()]  # majority of the genus's isolates
    return ((sum((r[3] > THRESHOLD) == bool(r[1]) for r in a), len(a)),
            (sum((s > THRESHOLD) == obs for s, obs in c), len(c)))


def main(apply: bool):
    if list(DOCS.glob("~$*")):
        raise SystemExit("Word or Excel lock files present; close the documents first")
    (a_ok, a_n), (c_ok, c_n) = agreement()
    assert (a_ok, a_n, c_ok, c_n) == (20, 22, 35, 48), (a_ok, a_n, c_ok, c_n)
    pa, pc = f"{100 * a_ok / a_n:.1f}%", f"{100 * c_ok / c_n:.1f}%"

    ms, si = TrackedDocx(MS, "unused"), TrackedDocx(SI, "unused")
    ctx = "Briefly, we found that the integrated model"
    plain_replace(ms, "Briefly, we found that the integrated model accurately predicted observed substrate use from two "
                      "previous studies, Jenkins et al. ",
                  f"Briefly, the integrated model's predictions matched observed substrate use for {a_ok} of {a_n} "
                  f"substrates ({pa}) in Jenkins et al. ", context=ctx)
    ctx = "Briefly, the integrated model's predictions matched"
    plain_replace(ms, " (SI-Prompt 1) and Venturelli et al. ",
                  f" (SI-Prompt 1) and for {c_ok} of {c_n} genus–compound pairs ({pc}) in Venturelli et al. ", context=ctx)
    plain_replace(ms, " (SI-Prompt 2) as shown in ", " (SI-Prompt 2), as shown in ", context=ctx)

    plain_replace(si, "WoM accurately predicts and interprets substrate consumption by gut microbiota and temporal "
                      "utilization order for ",
                  "Comparison of WoM predictions with observed substrate consumption by gut microbiota and with the "
                  "temporal utilization order for ", context="Supplementary Figure 4.")

    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    r = letter.response_text("Major comment 1.")
    old = ("with the retrained model and updated their responses, Supplementary Fig. 4, and its data sheets.")
    new = ("with the retrained model and updated their responses, Supplementary Fig. 4, and its data sheets; the main "
           f"text now gives the agreement of these predictions with the published data ({a_ok} of {a_n} substrates for "
           f"Prompt 1 and {c_ok} of {c_n} genus–compound pairs for Prompt 2) in place of \"accurately predicted\".")
    assert old in r
    letter.set_response("Major comment 1.", r.replace(old, new))

    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    s = parts["xl/sharedStrings.xml"].decode("utf-8")
    for old, new in [
        ("The WoM Agent accurately predicts and interprets substrate consumption and temporal utilization order for "
         "Pseudomonas fluorescens",
         "Comparison of WoM Agent predictions with observed substrate consumption and temporal utilization order for "
         "Pseudomonas fluorescens"),
        ("The WoM Agent accurately predicts and interprets substrate consumption for gut microbiota",
         "Comparison of WoM Agent predictions with observed substrate consumption for gut microbiota"),
    ]:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    parts["xl/sharedStrings.xml"] = s.encode("utf-8")

    if not apply:
        print("dry run: all edits located; nothing written")
        return
    for doc in (ms, si, letter):
        doc.save()
    tmp = TABLES.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(TABLES)
    print("applied to", DOCS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
