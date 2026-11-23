# drop_median_20261123.py
"""Drop the redundant "median 0.89" from the 12-isolate result (mSystems revision).

The Results gave the mean ROC-AUC of the 12 held-out isolates (0.89), its SD across models and
also its median (0.89). The median repeats the mean and is removed from the manuscript and from the
letter's quote of that sentence (Reviewer 3, minor comment 4). Plain (untracked) edit; the
documents are backed up first.

Usage:
    python drop_median_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

OLD = "(SD 0.006 across models; median 0.89)"
NEW = "(SD 0.006 across models)"


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    docs = [TrackedDocx(f, "unused") for f in (revision_paths.MS, revision_paths.LETTER)]
    for doc in docs:
        p = doc.find(OLD)
        plain_replace(doc, OLD, NEW, p=p)
        text = doc.accepted_text(p)
        print(f"{doc.path.name}: …{text[text.index(NEW) - 70:text.index(NEW) + 60]}…")
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    for doc in docs:
        doc.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
