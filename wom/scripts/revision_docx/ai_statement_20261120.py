# ai_statement_20261120.py
"""Update the manuscript's AI-use statement (mSystems revision).

"criticise" (not the authors' word) becomes "critique", and Claude Fable 5.2, used during the
revision, is added to the models listed. The statement also now says that the models drafted text
(revised sentences, a caption, parts of the response letter) and that the authors reviewed and
approved it. Plain (untracked) edits; the manuscript is backed up first. Steps already present in
the document are skipped.

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

STEPS = [
    ("Claude Opus 4.5 and 4.8 were used to assist with coding and to criticise and recommend edits",
     "Claude Opus 4.5, Opus 4.8, and Fable 5.2 were used to assist with coding and to critique and recommend edits"),
    ("to critique and recommend edits to human written text.",
     "to critique, draft, and recommend edits to text, all of which was reviewed and approved by the authors."),
]
CONTEXT = "were used to assist with coding"


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")):
        raise SystemExit("Word or Excel lock files present; close the documents first")
    ms = TrackedDocx(revision_paths.MS, "unused")
    done = 0
    for old, new in STEPS:
        if old in ms.accepted_text(ms.find(CONTEXT)):
            plain_replace(ms, old, new, context=CONTEXT)
            done += 1
    print(ms.accepted_text(ms.find(CONTEXT)).strip())
    if not done:
        print("nothing to change")
        return
    if not apply:
        print("dry run: nothing written")
        return
    print("backup:", ms.backup(revision_paths.BACKUPS))
    ms.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
