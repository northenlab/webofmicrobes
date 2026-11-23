# vary_openers_20261123.py
"""Vary the opening sentences of the responses to Reviewers 1 and 3 (mSystems revision).

Twelve of the sixteen responses began "We thank the reviewer". Nine of them now open differently;
three keep their thanks (Reviewer 1 comment 6, Reviewer 2 comment 4, Reviewer 3 major comment 1).
Only the first sentence of each response changes. Plain (untracked) edit of the letter; the
documents are backed up first.

Usage:
    python vary_openers_20261123.py            # dry run
    python vary_openers_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

OPENERS = [
    # Reviewer 1, comments 1-4
    (' We thank the reviewer for noticing that "AUC" was ambiguous.',
     ' The reviewer is right that "AUC" was ambiguous.'),
    (" We thank the reviewer for this suggestion. We agree that ROC-AUC alone",
     " We agree that ROC-AUC alone"),
    (' We thank the reviewer for this request. "Calibrated" was a poorly chosen word',
     ' "Calibrated" was a poorly chosen word'),
    (" We thank the reviewer for pointing out that the embedding procedure could not be understood from the main text.",
     " We appreciate this comment: the embedding procedure could not be understood from the main text."),
    # Reviewer 3, minor comments 1-5
    (' We thank the reviewer for noticing that "calibrated" was not supported. We agree that embedding dot products '
     "are ranking statistics.",
     ' The reviewer is correct: "calibrated" was not supported, and embedding dot products are ranking statistics.'),
    (" We thank the reviewer for checking these numbers carefully. The 45% figure was incorrect.",
     " The reviewer's arithmetic is right, and the 45% figure was incorrect."),
    (" We thank the reviewer for noticing the inconsistent threshold notation. The released code",
     " The threshold notation was inconsistent, as the reviewer points out. The released code"),
    (" We thank the reviewer. The genus → family → order fallback",
     " We have added the information the reviewer requests. The genus → family → order fallback"),
    (" We thank the reviewer for this careful reading; the placeholder",
     " We are grateful for this careful reading; the placeholder"),
]


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(revision_paths.LETTER, "unused")
    for old, new in OPENERS:
        p = letter.find("Response:" + old)
        plain_replace(letter, old, new, p=p)
    starts = [letter.accepted_text(p) for p in letter.paragraphs() if letter.accepted_text(p).startswith("Response:")]
    for s in starts:
        print("   " + s[:118])
    n = sum(s.startswith("Response: We thank the reviewer") for s in starts)
    assert len(starts) == 16 and n == 3, (len(starts), n)
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
