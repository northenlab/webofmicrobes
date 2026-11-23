# line_numbers_20261123.py
"""Fill in the line numbers of the two "Original" labels added on 2026-11-23 (mSystems revision).

The line-numbered PDF of the original submission was lost on 2026-11-18. The user supplied the
original Word file, which has line numbering switched on; Word's PDF export of it is kept at
revision_paths.LINE_PDF. Before anything is written, every line range already in the letter is
recomputed from that PDF and must agree, which shows that the export reproduces the numbering of
the lost PDF. Plain (untracked) edit of the letter; the documents are backed up first.

Usage:
    python line_numbers_20261123.py            # dry run
    python line_numbers_20261123.py --apply
"""

import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import LETTER, LINE_PDF  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402
from r3_major1_edits import LineNumbers  # noqa: E402

# label as it stands -> (place, as it will read with the measured lines)
LABELS = {
    "Original (Discussion, first paragraph, first sentence):": "Original (Discussion, first paragraph, Lines {}):",
    "Original (Results, first section, final sentence of Lines 147–159):": "Original (Results, first section, Lines {}):",
}
WITH_LINES = re.compile(r"Original \((.*?), Lines? (\d+(?:–\d+)?)\):")


def check_existing(letter, lines: LineNumbers) -> int:
    """Every range already in the letter must be reproduced; returns how many were checked."""
    ps = [p for p in letter.paragraphs() if letter.accepted_text(p).strip()]
    checked, unmatched = 0, []
    for label, quote in zip(ps, ps[1:]):
        m = WITH_LINES.match(letter.accepted_text(label).strip())
        if not m:
            continue
        got = lines.lines(letter.accepted_text(quote).strip())
        if got == "__":
            unmatched.append(m.group(0))
            continue
        assert got == m.group(2), (m.group(0), got)
        checked += 1
    print(f"   {checked} existing line ranges reproduced exactly; quote not located in the PDF text: {unmatched}")
    assert checked >= 35 and len(unmatched) <= 1
    return checked


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(LETTER, "unused")
    lines = LineNumbers(LINE_PDF)
    print("Calibration")
    check_existing(letter, lines)
    print("Labels")
    ps = letter.paragraphs()
    for old, template in LABELS.items():
        label = letter.find(old, exact=True)
        quote = letter.accepted_text(ps[ps.index(label) + 1]).strip()
        found = lines.lines(quote)
        assert found != "__", quote[:60]
        plain_replace(letter, old, template.format(found), p=label)
        print(f"   {letter.accepted_text(label).strip()}  <-  {quote[:70]}…")
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
