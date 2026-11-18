# reapply_20261118.py
"""Re-apply the 2026-11-16 edits to the documents restored from Google Drive (mSystems revision).

~/Downloads/revisions was deleted and restored on 2026-11-18 from the team's Google Drive copies,
which predate the three rounds of edits made on 2026-11-16. This script backs the restored files up
and runs those rounds again in their original order, each as a dry run followed by the real run:

  1. letter_sync_20261116    response letter brought in line with the manuscript and SI
  2. final_fixes_20261116    small corrections from the point-by-point audit
  3. wording_fixes_20261116  measured agreement in place of "accurately predicted"

The restored manuscript also lacks one comma that the 2026-11-16 copy had ("Thus, predictions that
span both ..."); it is put back first, because the letter quotes that sentence. All edits are plain
(untracked). Run once: the rounds are not idempotent.

Usage:
    python reapply_20261118.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
import letter_sync_20261116  # noqa: E402
import final_fixes_20261116  # noqa: E402
import wording_fixes_20261116  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402


def restore_comma():
    ms = TrackedDocx(revision_paths.MS, "unused")
    limits = ms.accepted_text(ms.find("There are a number of limitations to this study."))
    if "them. Thus, predictions that span both" in limits:
        return
    final_fixes_20261116.plain_replace(ms, "them. Thus predictions that span both",
                                       "them. Thus, predictions that span both",
                                       context="There are a number of limitations to this study.")
    ms.save()
    print("manuscript: comma restored in 'Thus, predictions that span both'")


def main():
    if list(revision_paths.DOCS.glob("~$*")):
        raise SystemExit("Word or Excel lock files present; close the documents first")
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    restore_comma()
    for step in (letter_sync_20261116, final_fixes_20261116, wording_fixes_20261116):
        print(f"\n===== {step.__name__}: dry run")
        step.main(False)
        print(f"===== {step.__name__}: apply")
        step.main(True)


if __name__ == "__main__":
    main()
