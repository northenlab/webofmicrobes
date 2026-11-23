# define_sigma_20261123.py
"""Define the symbol σ where the BPR loss is written out (mSystems revision).

The Methods ("Model training") and the Supplementary Methods give the loss as −log σ(...) without
saying what σ is. Both now define it, σ(x) = 1/(1 + exp(−x)), and the Methods add in words what
minimizing the loss does. The letter's quotes of the two paragraphs (Reviewer 1, comment 4;
Reviewer 3, major comment 1) follow. Plain (untracked) edits; the documents are backed up first.

Usage:
    python define_sigma_20261123.py            # dry run
    python define_sigma_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, SI, LETTER  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

SIGMA = "σ(x) = 1/(1 + exp(−x))"
METHODS = "We framed the prediction of microbial metabolite consumption as an implicit-feedback"
METHODS_OLD = "). Measured-neutral metabolites therefore enter training only as negatives"
METHODS_NEW = (f"), where {SIGMA}, so that the consumed metabolite scores above the negative. Measured-neutral "
               "metabolites therefore enter training only as negatives")
SI_PARAGRAPH = "Bayesian Personalized Ranking (BPR), as implemented here"
SI_OLD = "−log σ(score(positive) − score(negative)). Negatives are sampled"
SI_NEW = f"−log σ(score(positive) − score(negative)), where {SIGMA}. Negatives are sampled"


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    ms, si, letter = (TrackedDocx(f, "unused") for f in (MS, SI, LETTER))
    for doc in (ms, letter):
        p = doc.find(METHODS)
        plain_replace(doc, METHODS_OLD, METHODS_NEW, p=p)
        text = doc.accepted_text(p)
        print(f"{doc.path.name}: …{text[text.index('and minimizes'):text.index('only as negatives')]}…")
    for doc in (si, letter):
        p = doc.find(SI_PARAGRAPH)  # the revised paragraph; the letter's "Original" quote is left as submitted
        plain_replace(doc, SI_OLD, SI_NEW, p=p)
        text = doc.accepted_text(p)
        print(f"{doc.path.name}: …{text[text.index('Training minimizes'):text.index('Negatives are sampled')]}…")
    for doc in (ms, si):
        assert sum(doc.accepted_text(p).count("σ(x) =") for p in doc.paragraphs()) == 1, doc.path.name
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    for doc in (ms, si, letter):
        doc.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
