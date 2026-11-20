# ai_statement_20261120.py
"""Update the manuscript's AI-use statement (mSystems revision).

"criticise" (not the authors' word) becomes "critique", and Claude Fable 5.2, used during the
revision, is added to the models listed. Plain (untracked) edit; the manuscript is backed up first.

Usage:
    python ai_statement_20261120.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

OLD = "Claude Opus 4.5 and 4.8 were used to assist with coding and to criticise and recommend edits"
NEW = "Claude Opus 4.5, Opus 4.8, and Fable 5.2 were used to assist with coding and to critique and recommend edits"


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")):
        raise SystemExit("Word or Excel lock files present; close the documents first")
    ms = TrackedDocx(revision_paths.MS, "unused")
    plain_replace(ms, OLD, NEW)
    print(ms.accepted_text(ms.find(NEW)).strip())
    if not apply:
        print("dry run: nothing written")
        return
    print("backup:", ms.backup(revision_paths.BACKUPS))
    ms.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
