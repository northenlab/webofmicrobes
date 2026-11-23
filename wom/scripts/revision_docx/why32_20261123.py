# why32_20261123.py
"""Say why 32 dimensions are used, and drop "we selected BPR" (mSystems revision; Reviewer 1, comment 5).

The Results said that BPR with 32-dimensional embeddings was selected "based on its performance and
resistance to overfitting". Those are properties of BPR, not reasons for 32, and the benchmark puts
the highest ROC-AUC at 16 dimensions. The sentence is split: what the results show about BPR, and
why 32 is used (within the stable range, 0.010 below 16, and possibly useful as the dataset grows).
The Supplementary Methods sentence is brought in line. In the letter, the quotes of both sentences
follow, a label that pointed to the old sentence ending is updated, and the Results change is
quoted under Reviewer 1, comment 5. Plain (untracked) edits; the documents are backed up first.

Usage:
    python why32_20261123.py            # dry run
    python why32_20261123.py --apply
"""

import sys
import copy
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, SI, LETTER  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

BY_DIM = (Path(__file__).resolve().parents[3] /
          "wom/manuscript_figures/msystems_revision/lambda0/r1/pr_calibration/fig1b_pr_by_dim.csv")
BPR = "Ordinal-BPR [lambda_ord=0]"

MS_OLD = ("Based on its performance and resistance to overfitting, we selected BPR with 32-dimensional embeddings "
          "(ROC-AUC = 0.865 ± 0.004) for all subsequent analyses.")
MS_NEW = ("These results show that BPR performs comparably to or better than the other methods and is more resistant "
          "to overfitting. We use 32-dimensional embeddings (ROC-AUC = 0.865 ± 0.004) for all subsequent analyses: "
          "this is 0.010 below the value at 16 dimensions but within the range of stable accuracy (8 to 64 dimensions), "
          "and a larger embedding may be useful as more isolates and metabolites are added to the Web of Microbes.")
MS_ORIGINAL = ("Based on its performance and resistance to overfitting, we selected Ordinal-BPR with 32-dimensional "
               "embeddings for all subsequent analyses.")
SI_OLD = "Based on the hold-out results (Fig. 1b), we selected BPR with 32-dimensional embeddings for all subsequent analyses."
SI_NEW = ("BPR with 32-dimensional embeddings, which lie within the range of stable accuracy (Fig. 1b), was used for "
          "all subsequent analyses.")
LABEL_OLD = "…for all subsequent analyses.\")"
LABEL_NEW = "…are added to the Web of Microbes.\")"


def check_claims():
    roc = pd.read_csv(BY_DIM).query("model == @BPR").set_index("dim").roc_auc_mean
    assert f"{roc[16] - roc[32]:.3f}" == "0.010" and f"{roc[32]:.3f}" == "0.865", (roc[16], roc[32])
    assert roc.idxmax() == 16 and roc[[8, 16, 32, 64]].max() - roc[[8, 16, 32, 64]].min() <= 0.0125


def fix_letter(letter):
    for old, new in ((MS_OLD, MS_NEW), (SI_OLD, SI_NEW)):
        hits = [p for p in letter.paragraphs() if old in letter.accepted_text(p)]
        assert len(hits) == 1, (len(hits), old[:50])
        plain_replace(letter, old, new, p=hits[0])
    labels = [p for p in letter.paragraphs() if LABEL_OLD in letter.accepted_text(p)]
    assert len(labels) == 1 and letter.accepted_text(labels[0]).startswith("Modified (Results, first section)"), len(labels)
    plain_replace(letter, LABEL_OLD, LABEL_NEW, p=labels[0])
    originals = [p for p in letter.paragraphs() if letter.accepted_text(p).strip().endswith(MS_ORIGINAL)]
    assert len(originals) == 1, len(originals)  # the quote of Lines 147–159 under Reviewer 3, major comment 1

    # the change, quoted under Reviewer 1 comment 5 in the format of an existing Original/Modified pair
    ps = letter.paragraphs()
    k = ps.index(letter.find("Original (Discussion, first paragraph, Lines 254"))
    templates = ps[k:k + 4]
    assert letter.accepted_text(templates[2]).strip() == "Modified:"
    new = ["Original (Results, first section, final sentence of Lines 147–159):", MS_ORIGINAL, "Modified:", MS_NEW]
    anchor = letter.find("Modified (Results, first section): see the response to Comment 2, which quotes")
    for template, text in zip(templates, new):
        p = copy.deepcopy(template)
        anchor.addnext(p)
        plain_replace(letter, letter.accepted_text(p), text, p=p)
        anchor = p
    ps = letter.paragraphs()
    k = ps.index(letter.find("Comment 5. In fig 1B"))
    for p in ps[k + 1:k + 8]:
        print("   " + letter.accepted_text(p)[:470])
    print("   " + letter.accepted_text(labels[0]))


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    check_claims()
    ms, si, letter = (TrackedDocx(f, "unused") for f in (MS, SI, LETTER))
    plain_replace(ms, MS_OLD, MS_NEW)
    plain_replace(si, SI_OLD, SI_NEW)
    text = ms.accepted_text(ms.find(MS_NEW))
    print("Manuscript\n   …" + text[text.index(MS_NEW) - 90:text.index(MS_NEW) + len(MS_NEW) + 60] + "…")
    text = si.accepted_text(si.find(SI_NEW))
    print("Supplementary Information\n   …" + text[max(0, text.index(SI_NEW) - 80):text.index(SI_NEW) + len(SI_NEW)])
    print("Response letter")
    fix_letter(letter)
    if not apply:
        print("\ndry run: nothing written")
        return
    print(f"\nbackups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    for doc in (ms, si, letter):
        doc.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
