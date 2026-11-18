# final_fixes_20261116.py
"""Small corrections found in the point-by-point audit of the response letter (mSystems revision).

Plain (untracked) edits to the documents in ~/Downloads/revisions; the marked-up copy is made at
the end with Word's document comparison. Back up the files before running.

Supplementary Information
  * Fig. S3 caption: "NLDM (Northen Lab Defined Medium (NLDM))" duplication.
  * Prompt 1: remove the agent's <developer> block that was pasted in after the table.
  * References: the Gemini citation carried the main-text number (25) and had no SI bibliography
    entry. It is now reference 6 (order of first citation); the three later references move up by one.
  * Fig. S4 caption: stray "(" before a citation. Fig. S4a image: x-axis label "BSR" -> "BPR"
    (the label of panel c, same font and position, is copied onto panel a).
Manuscript
  * "BRP model" -> "BPR model"; "Personalized Raking" -> "Personalized Ranking".
  * Fig. 3 caption: category definitions added (Reviewer 3, minor comment 3).
Supplementary tables (the two Fig. S4 sheets)
  * consumed is <= -2 (as in the data and the text), not < -2; "prompt 1" -> "prompt 2" on the
    S4c/d sheet; "Ventrullinet al." -> "Venturelli et al."; three spelling errors.
Response letter
  * Reviewer 3, minor comment 3: response and a new Original/Modified pair for the Fig. 3 caption.

Usage:
    python final_fixes_20261116.py --apply
"""

import io
import re
import sys
import copy
import zipfile
import argparse
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import TrackedDocx, LetterDocx, q, local, XML_SPACE  # noqa: E402
from revision_paths import DOCS, MS, SI, LETTER, TABLES, original_lines  # noqa: E402

FIG3_OLD = "(a) Hold-out AUC separating experimentally consumed from neutral metabolites for BPR (teal)"
FIG3_NEW = ("(a) Hold-out AUC separating experimentally consumed (assertion value ≤ −2) from neutral "
            "(−2 < assertion value < +2) metabolites, with secreted metabolites (assertion value ≥ +2) excluded, "
            "for BPR (teal)")
GEMINI_REF = ("Gemini 3 Flash Preview. Google AI for Developers. "
              "https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview. Retrieved 31 July 2026.")


def plain_replace(doc, old: str, new: str, context: str | None = None):
    """Untracked replacement of `old` by `new` in the paragraph containing `context` (or `old`)."""
    p = doc.find(context or old)
    text = doc.accepted_text(p)
    if text.count(old) != 1:
        raise ValueError(f"{text.count(old)} occurrences of {old[:60]!r} in the paragraph")
    start = text.index(old)
    covered = doc._isolate(p, start, start + len(old))
    if any(local(a) == "hyperlink" for r in covered for a in r.iterancestors()):
        raise ValueError(f"{old[:40]!r} lies inside a hyperlink; edit its run directly")
    t = covered[0].find(q("t"))
    t.text = new
    t.set(XML_SPACE, "preserve")
    for r in covered[1:] + ([covered[0]] if not new else []):
        r.getparent().remove(r)


def set_citation(doc, context: str, old: str, new: str):
    """Change the number shown by one citation hyperlink in the paragraph containing `context`."""
    p = doc.find(context)
    hits = [t for h in p.iter(q("hyperlink")) for t in h.iter(q("t")) if (t.text or "") == old]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} citations {old!r} in the paragraph")
    hits[0].text = new


def fix_si(si):
    plain_replace(si, "NLDM (Northen Lab Defined Medium (NLDM))", "NLDM (Northen Lab Defined Medium)")
    dev = si.find("<developer>")
    text = si.accepted_text(dev).strip()
    assert text.startswith("<developer>") and text.endswith("</developer>"), "developer block is not a whole paragraph"
    dev.getparent().remove(dev)

    # references in order of first citation: Gemini becomes 6; Kosina, Jenkins, Venturelli become 7, 8, 9
    set_citation(si, "Although we used Gemini 3 Flash", "(25)", "(6)")
    s4 = "Supplementary Figure 4. WoM accurately"
    set_citation(si, s4, "(6, 7)", "(7, 8)")
    set_citation(si, s4, "(6, 8)", "(7, 9)")
    plain_replace(si, "8 gut microbial genera(", "8 gut microbial genera", context=s4)
    entries = {}
    for p in si.paragraphs():
        m = re.match(r"(\d+)\. ", si.accepted_text(p))
        if m and p.find(q("hyperlink")) is not None and p.find(q("r")) is not None:
            entries[int(m.group(1))] = p
    assert sorted(entries) == list(range(1, 9)), sorted(entries)
    new = copy.deepcopy(entries[6])
    link = new.find(q("hyperlink"))
    run = link.find(q("r"))
    run.find(q("t")).text = GEMINI_REF
    link.addprevious(run)  # plain text: this entry is not managed by the reference manager
    new.remove(link)
    entries[6].addprevious(new)
    for n in (6, 7, 8):
        number = entries[n].find(q("r")).find(q("t"))
        assert number.text.strip() == f"{n}.", number.text
        number.text = f"{n + 1}. "
    si.replace_part("word/media/image4.png", fix_s4_label(si.read_part("word/media/image4.png")))


def fix_s4_label(png: bytes) -> bytes:
    """Panel a's x-axis label reads "BSR Model Score"; copy panel c's "BPR Model Score" over it."""
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    assert im.size == (2048, 1583), im.size
    a = np.array(im.convert("RGBA"))
    ink = (a[..., :3].astype(int).sum(-1) < 300) & (a[..., 3] > 128)

    def rows(y0, y1):
        r = np.where(ink[y0:y1, 450:850].any(1))[0]
        return y0 + r.min(), y0 + r.max()

    (a0, a1), (c0, c1) = rows(690, 730), rows(1455, 1500)
    assert a1 - a0 == c1 - c0, "labels differ in height"
    pad = 5
    a[a0 - pad:a1 + pad + 1, 500:800] = a[c0 - pad:c1 + pad + 1, 500:800]
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", dpi=im.info.get("dpi", (300, 300)))
    return buf.getvalue()


def fix_ms(ms):
    plain_replace(ms, "how the BRP model can be used", "how the BPR model can be used")
    plain_replace(ms, "Bayesian Personalized Raking", "Bayesian Personalized Ranking")
    plain_replace(ms, FIG3_OLD, FIG3_NEW, context="Figure 3. Comparison of BPR with naive taxonomic averaging")


def fix_tables():
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    s = parts["xl/sharedStrings.xml"].decode("utf-8")
    edits = [
        ("(log2 fold change &lt; −2)", "(log2 fold change ≤ −2)", 1),
        ("(log2 fold change =&gt; −2)", "(log2 fold change &gt; −2)", 1),
        ("(WoM score &lt; −2)", "(WoM score ≤ −2)", 1),
        ("(WoM score =&gt; −2)", "(WoM score &gt; −2)", 1),
        ("Ventrullinet al.", "Venturelli et al.", 1),
        ("prompt 1 (see manuscript and supplementary files)<", "prompt 2 (see manuscript and supplementary files)<", 1),
        ("oberservations", "observations", 3),
        ("substrae", "substrate", 2),
        ("Compund", "Compound", 1),
    ]
    for old, new, n in edits:
        assert s.count(old) == n, (old, s.count(old))
        s = s.replace(old, new)
    parts["xl/sharedStrings.xml"] = s.encode("utf-8")
    tmp = TABLES.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(TABLES)


def fix_letter(letter, ms):
    caption = ms.accepted_text(ms.find("Figure 3. Comparison of BPR with naive taxonomic averaging"))
    end = "at four hold-out levels of increasing phylogenetic distance."
    modified = caption[caption.index(FIG3_NEW):caption.index(end) + len(end)]
    original = modified.replace(FIG3_NEW, FIG3_OLD)
    found = original_lines(original, "fig3_caption_panel_a")
    assert found != "__", "Fig. 3 caption sentence not found in the line-numbered original"
    r = letter.response_text("Minor comment 3.")
    for old, new in [
        ("The category definitions for Fig. 1b are now stated in its caption and in the Supplementary Methods (see "
         "Major comment 1).",
         "The category definitions are now stated in the captions of Fig. 1b, Fig. 3, and Fig. S3, and for Fig. 1b "
         "also in the Supplementary Methods (see Major comment 1)."),
        ("the Supplementary Methods, the Fig. 1 caption, and the Fig. S3 caption.",
         "the Supplementary Methods, the Fig. 1 and Fig. 3 captions, and the Fig. S3 caption."),
    ]:
        assert old in r, old
        r = r.replace(old, new)
    letter.set_response("Minor comment 3.", r)
    minor4 = next(p for p in letter.paragraphs() if letter.accepted_text(p).startswith("Minor comment 4."))
    letter.add_pairs(minor4.getprevious(), [
        (f"Original (Figure 3 caption, Lines {found}):", original),
        ("Modified:", modified),
    ])


def main(apply: bool):
    if list(DOCS.glob("~$*")):
        raise SystemExit("Word or Excel lock files present; close the documents first")
    ms, si = TrackedDocx(MS, "unused"), TrackedDocx(SI, "unused")
    fix_ms(ms)
    fix_si(si)
    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    fix_letter(letter, ms)
    if not apply:
        print("dry run: all edits located; nothing written")
        return
    for doc in (ms, si, letter):
        doc.save()
    fix_tables()
    print("applied to", DOCS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
