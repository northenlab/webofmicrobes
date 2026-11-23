# r1c6_20261123.py
"""Reviewer 1, comment 6: a three-sentence response (mSystems revision).

The reviewer asked for an analysis that was not in the submission. The response now says that it
was done and what it shows (no plateau), and points to comment 3 for calibration; the detail is in
the Results sentence quoted below the response and in Fig. S8. Plain (untracked) edit of the letter;
the documents are backed up first.

Usage:
    python r1c6_20261123.py            # dry run
    python r1c6_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

START = "Response: We thank the reviewer for this suggestion. We computed a learning curve"
RESPONSE = (" We thank the reviewer for suggesting this analysis, which we had not done. The learning curve (new "
            "Fig. S8) shows that accuracy is still rising with the amount of data and has not reached a plateau. "
            "Calibration is addressed in the response to Comment 3.")
SUPPORT = "had not reached a plateau"  # the Results sentence quoted below the response must say this


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(revision_paths.LETTER, "unused")
    resp = letter.find(START)
    plain_replace(letter, letter.accepted_text(resp)[len("Response:"):], RESPONSE, p=resp)
    ps = letter.paragraphs()
    k = ps.index(resp)
    below = [letter.accepted_text(p) for p in ps[k:k + 4]]
    assert SUPPORT in below[2], below[2][:80]
    for text in below:
        print("   " + text[:420])
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
