# name_sigma_20261123.py
"""Name σ as the logistic function where it is defined (mSystems revision).

define_sigma_20261123.py gave σ by its formula only. The Methods, the Supplementary Methods and
the letter's quotes of both paragraphs now read "where σ is the logistic function,
σ(x) = 1/(1 + exp(−x))". Plain (untracked) edits; the documents are backed up first.

Usage:
    python name_sigma_20261123.py            # dry run
    python name_sigma_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, SI, LETTER  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

OLD = "where σ(x) = 1/(1 + exp(−x))"
NEW = "where σ is the logistic function, σ(x) = 1/(1 + exp(−x))"
EXPECTED = {MS: 1, SI: 1, LETTER: 2}  # the letter quotes the Methods and the Supplementary Methods paragraphs


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    docs = [TrackedDocx(f, "unused") for f in (MS, SI, LETTER)]
    for doc in docs:
        hits = [p for p in doc.paragraphs() if OLD in doc.accepted_text(p)]
        assert len(hits) == EXPECTED[doc.path], (doc.path.name, len(hits))
        for p in hits:
            plain_replace(doc, OLD, NEW, p=p)
            text = doc.accepted_text(p)
            print(f"{doc.path.name}: …{text[text.index(NEW) - 45:text.index(NEW) + len(NEW) + 60]}…")
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
