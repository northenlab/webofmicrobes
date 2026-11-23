# r1c6_sampling_20261123.py
"""Reviewer 1, comment 6: answer "how much must be sampled" from the hierarchical hold-out (mSystems revision).

The reviewer asked how much of a large microbe-metabolite space must be sampled. The hierarchical
hold-out of Fig. 3 already measures this for microbes, so it is restated in those terms; nothing is
extrapolated.

Manuscript
  * Results (hierarchical hold-out): one sentence giving the median ROC-AUC of an unprofiled isolate
    by how close its profiled relatives are (genus, family or order, class).
  * Discussion (third paragraph, at its end): one sentence on where new measurements are most valuable.
Response letter
  * Comment 6 response: two sentences added (microbes: Fig. 3; metabolites: only measured
    metabolites can be scored), and both manuscript sentences quoted below it.
  * The quote of the hold-out passage under Reviewer 3, major comment 1 follows the manuscript.

Plain (untracked) edits; the documents are backed up first. Run once.

Usage:
    python r1c6_sampling_20261123.py            # dry run
    python r1c6_sampling_20261123.py --apply
"""

import sys
import copy
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, LETTER  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

FIG3 = Path(__file__).resolve().parents[3] / "wom/manuscript_figures/msystems_revision/lambda0/fig3/fig3_holdout_results.csv"

RES_ANCHOR = "with a median AUC of 0.91, declining as the prediction source became more distant."
RES_NEW = ("In practical terms, an isolate that has not been profiled was ranked with a median ROC-AUC of 0.91 when "
           "other isolates of its genus were in the training data, 0.84–0.85 when only its family or order was "
           "represented, and 0.78 when only its class was represented.")
DISC_ANCHOR = "and the advantage of BPR lies in its complete, strain-level coverage."
DISC_NEW = ("The hierarchical hold-out also suggests where new measurements are most valuable: accuracy for an "
            "unprofiled isolate fell as its nearest profiled relatives became more distant, so profiling isolates from "
            "genera and families not yet represented should add more than further isolates of well-sampled genera.")
RESP_OLD = " Calibration is addressed in the response to Comment 3."
RESP_NEW = (" For a larger microbe–metabolite universe, the hierarchical hold-out (Fig. 3) gives the practical "
            "requirement for microbes: an unprofiled isolate is ranked with a median ROC-AUC of 0.91 when isolates of "
            "its genus have been profiled and 0.78 when only its class is represented. The metabolite universe is "
            "harder to search: the model can score only metabolites that have been measured, so testing more "
            "metabolites would extend its predictions independently of adding isolates. Calibration is addressed in "
            "the response to Comment 3.")


def check_claims():
    """Hold-out level -> rank of the reference set: isolate -> genus, genus -> family, family -> order, order -> class."""
    med = pd.read_csv(FIG3).groupby("level").auc_bpr_all.median()
    got = {k: f"{med[k]:.2f}" for k in ("isolate", "genus", "family", "order")}
    assert got == {"isolate": "0.91", "genus": "0.84", "family": "0.85", "order": "0.78"}, got
    assert med["isolate"] > med["genus"] and med["family"] > med["order"]


def add_pair(letter, anchor, label_t, quote_t, label_text: str, quote_text: str):
    for template, text in ((label_t, label_text), (quote_t, quote_text)):
        p = copy.deepcopy(template)
        anchor.addnext(p)
        plain_replace(letter, letter.accepted_text(p), text, p=p)
        anchor = p
    return anchor


def fix_letter(letter):
    resp = letter.find("Response: We thank the reviewer for suggesting this analysis, which we had not done.")
    plain_replace(letter, RESP_OLD, RESP_NEW, p=resp)
    hits = [p for p in letter.paragraphs() if RES_ANCHOR in letter.accepted_text(p)]
    assert len(hits) == 1, len(hits)  # the Modified quote under Reviewer 3, major comment 1 (the Original says 0.90)
    plain_replace(letter, RES_ANCHOR, f"{RES_ANCHOR} {RES_NEW}", p=hits[0])

    ps = letter.paragraphs()
    k = ps.index(resp)
    label_t, quote_t, cross = ps[k + 1], ps[k + 2], ps[k + 3]
    assert letter.accepted_text(label_t).startswith("Modified (Results, first section): (no original text"), letter.accepted_text(label_t)
    assert letter.accepted_text(cross).startswith("Modified (Discussion, third paragraph): see the response to Comment 5")
    add_pair(letter, quote_t, label_t, quote_t,
             "Modified (Results, \"Phylogenetic conservation…\"): (no original text; new insertion immediately after "
             "the sentence ending \"…as the prediction source became more distant.\")", RES_NEW)
    add_pair(letter, cross, label_t, quote_t,
             "Modified (Discussion, third paragraph): (no original text; new insertion immediately after the sentence "
             "ending \"…complete, strain-level coverage.\")", DISC_NEW)
    ps = letter.paragraphs()
    for p in ps[k:k + 9]:
        print("   " + letter.accepted_text(p)[:900])


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    check_claims()
    ms, letter = TrackedDocx(MS, "unused"), TrackedDocx(LETTER, "unused")
    plain_replace(ms, RES_ANCHOR, f"{RES_ANCHOR} {RES_NEW}")
    plain_replace(ms, DISC_ANCHOR, f"{DISC_ANCHOR} {DISC_NEW}")
    print("Manuscript")
    for new in (RES_NEW, DISC_NEW):
        text = ms.accepted_text(ms.find(new))
        print("   …" + text[max(0, text.index(new) - 110):text.index(new) + len(new) + 90] + "…")
    print("Response letter")
    fix_letter(letter)
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    ms.save()
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
