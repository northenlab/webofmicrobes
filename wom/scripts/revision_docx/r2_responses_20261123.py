# r2_responses_20261123.py
"""Shorter responses to the four comments of Reviewer 2 (mSystems revision).

Each response explained the new text at length. They now thank the reviewer, give the direct
answer where a question was asked, and point to the change; the quoted Original/Modified text
below each response carries the detail. Only the four response paragraphs of the letter change.
Plain (untracked) edits; the documents are backed up first.

Usage:
    python r2_responses_20261123.py            # dry run
    python r2_responses_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

# start of the current response -> (new response, a phrase the quoted Modified text below it must contain)
RESPONSES = {
    "Response: We thank the reviewer for raising the important point that co-metabolism": (
        " We thank the reviewer for this important point. The model predicts end-point depletion under the tested "
        "conditions, not the order of substrate use or metabolic potential. The limitations paragraph of the "
        "Discussion now says so, addresses each of the processes the reviewer lists, and names multi-omics "
        "integration as future work.",
        "It does not predict which substrates are used first"),
    "Response: We thank the reviewer for emphasizing that environmental conditions": (
        " We thank the reviewer and agree that preferences measured under one set of conditions may not transfer to "
        "others. The limitations paragraph of the Discussion now says so and describes the experiment the reviewer "
        "proposes, profiling under systematically varied conditions.",
        "Profiling the same isolates under systematically varied conditions"),
    "Response: We thank the reviewer for noting the wide range of starting concentrations": (
        " We thank the reviewer for this careful observation. Substrate concentration is not used in training or "
        "prediction. The Discussion now says so and proposes the equal-concentration experiment the reviewer "
        "suggests.",
        "The model also does not use substrate concentration"),
    "Response: We thank the reviewer for this question. The current model can score aromatic": (
        " We thank the reviewer for this question. The model can score the more than 50 aromatic compounds in its "
        "panel, but few isolates consumed them, and compounds outside the panel cannot be scored. The final "
        "paragraph of the Discussion now says this and describes how the model could be extended towards lignin "
        "utilization.",
        "would extend the approach to lignin utilization"),
}


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(revision_paths.LETTER, "unused")
    for start, (new, support) in RESPONSES.items():
        resp = letter.find(start)
        old = letter.accepted_text(resp)[len("Response:"):]
        plain_replace(letter, old, new, p=resp)
        ps = letter.paragraphs()
        k = ps.index(resp)
        below = " ".join(letter.accepted_text(p) for p in ps[k + 1:k + 5])
        assert support in below, (start[:50], support)
        print(f"   [{len(old.split())} -> {len(new.split())} words] Response:{new}\n")
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
