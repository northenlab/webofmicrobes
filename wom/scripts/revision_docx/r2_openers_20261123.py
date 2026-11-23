# r2_openers_20261123.py
"""Vary the opening sentences of the four responses to Reviewer 2 (mSystems revision).

All four began "We thank the reviewer". Three now open differently; the fourth keeps its thanks.
Plain (untracked) edit of the letter; the documents are backed up first.

Usage:
    python r2_openers_20261123.py            # dry run
    python r2_openers_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

OPENERS = [
    (" We thank the reviewer for this important point. The model predicts end-point depletion",
     " The reviewer has identified an important gap in our description of what the model predicts. It predicts "
     "end-point depletion"),
    (" We thank the reviewer and agree that preferences measured under one set of conditions may not transfer to others.",
     " We agree: preferences measured under one set of conditions may not transfer to others."),
    (" We thank the reviewer for this careful observation. Substrate concentration is not used",
     " This is a good point. Substrate concentration is not used"),
]
KEPT = "Response: We thank the reviewer for this question. The model can score the more than 50 aromatic compounds"


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(revision_paths.LETTER, "unused")
    for old, new in OPENERS:
        p = letter.find("Response:" + old)
        plain_replace(letter, old, new, p=p)
        print("   " + letter.accepted_text(p))
    print("   " + letter.accepted_text(letter.find(KEPT)))
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
