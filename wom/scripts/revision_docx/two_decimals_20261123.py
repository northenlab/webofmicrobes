# two_decimals_20261123.py
"""Back to two decimals, the precision used everywhere else (mSystems revision; Reviewer 3, major comment 1).

three_decimals_20261123.py gave three decimals in one response and one Supplementary Methods
sentence so that two equal rounded values could be told apart. Mixed precision reads worse than the
coincidence, so both return to two decimals and the response says that the second 0.83 is a
coincidence. Plain (untracked) edits of the SI and the letter; the documents are backed up first.

Usage:
    python two_decimals_20261123.py            # dry run
    python two_decimals_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import SI, LETTER  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

SI_OLD = ("hold-out ROC-AUC was 0.832 at 32 dimensions and 0.773 at 128 dimensions, compared with 0.865 and 0.832 for "
          "the model as used and 0.766 and 0.621 for WR-MF")
SI_NEW = ("hold-out ROC-AUC was 0.83 at 32 dimensions and 0.77 at 128 dimensions, compared with 0.86 and 0.83 for the "
          "model as used and 0.77 and 0.62 for WR-MF")
RESPONSE = [("(ROC-AUC = 0.826 with it and 0.865 without it at 32 dimensions)",
             "(ROC-AUC = 0.83 with it and 0.86 without it at 32 dimensions)"),
            ("trained without the measured-neutral labels (0.832 at 32 dimensions).",
             "trained without the measured-neutral labels (coincidentally also 0.83 at 32 dimensions).")]


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    si, letter = TrackedDocx(SI, "unused"), TrackedDocx(LETTER, "unused")
    for doc in (si, letter):
        plain_replace(doc, SI_OLD, SI_NEW, p=doc.find(SI_OLD))
    resp = letter.find("Response: We thank the reviewer for this careful reading of the benchmark and its code.")
    for old, new in RESPONSE:
        plain_replace(letter, old, new, p=resp)
    text = letter.accepted_text(resp)
    print("   SI and its quote: …" + SI_NEW + "…")
    print("   response: …" + text[text.index("In the hold-out the ordinal term"):text.index("We retrained the model")])
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    si.save()
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
