# r3_minor5_response_20261123.py
"""Reviewer 3, minor comment 5: a shorter response (mSystems revision).

The response repeated what the Supplementary Methods, the Fig. S5 caption and the Prompt 3 note now
say, all of which are quoted below it. It keeps the admission, the four things the reviewer asked
for, and the one change the reviewer would not expect (Fig. S5 now shows succinate, not malate).
Plain (untracked) edit of the letter; the documents are backed up first.

Usage:
    python r3_minor5_response_20261123.py            # dry run
    python r3_minor5_response_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

START = "Response: We are grateful for this careful reading; the placeholder"
RESPONSE = (" We are grateful for this careful reading; the placeholder and the description of the experimental unit "
            "were both incomplete. The Supplementary Methods now state that the experiment was performed three times "
            "independently with three replicate wells per condition, identify the runs and the time window plotted "
            "(0–21 h), and say how the curves and error bands were computed. Because the corrected model (see Major "
            "comment 1) ranks succinate rather than malate first for Acinetobacter, Fig. S5 now shows the growth "
            "experiment with succinic acid.")
# what the response says must be in the quoted text below it
QUOTED = ["the experiment was performed three times independently", "three replicate wells",
          "the time axis ends at 21 h", "shaded bands", "succinate is the top-ranked of the 64 NLDM metabolites",
          "malate scores"]


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(revision_paths.LETTER, "unused")
    resp = letter.find(START)
    old = letter.accepted_text(resp)[len("Response:"):]
    plain_replace(letter, old, RESPONSE, p=resp)
    ps = letter.paragraphs()
    below = " ".join(letter.accepted_text(p) for p in ps[ps.index(resp) + 1:])
    for phrase in QUOTED:
        assert phrase in below, phrase
    print(f"   [{len(old.split())} -> {len(RESPONSE.split())} words]\n   Response:{RESPONSE}")
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
