# r1c5_20261123.py
"""Reviewer 1, comment 5: a short response and two text changes that support it (mSystems revision).

Response letter
  * The response to comment 5 is replaced by four sentences: the question changed how the model is
    presented; higher dimensions do not improve accuracy; what BPR offers; what the changes do.
  * A new Original/Modified pair for the first sentence of the Discussion.
Manuscript
  * Results: the sentence on embedding size now ends "so the embedding dimension does not need to
    be tuned within this range" (ROC-AUC within 0.012 from 8 to 64 dimensions).
  * Discussion, first sentence: "outperformed" the three other methods becomes "by ROC-AUC, was
    more accurate than ... at every embedding dimension" (by PR-AUC it was not).
The letter's two quotes of the Results sentence follow the manuscript. The line numbers of the
Discussion sentence in the original submission cannot be measured (the line-numbered PDF is gone),
so its label gives none. Plain (untracked) edits; the documents are backed up first. Run once.

Usage:
    python r1c5_20261123.py            # dry run
    python r1c5_20261123.py --apply
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

BY_DIM = (Path(__file__).resolve().parents[3] /
          "wom/manuscript_figures/msystems_revision/lambda0/r1/pr_calibration/fig1b_pr_by_dim.csv")
BPR = "Ordinal-BPR [lambda_ord=0]"

TUNE_OLD = "and within 0.012 of this from 8 to 64 dimensions."
TUNE_NEW = ("and within 0.012 of this from 8 to 64 dimensions, so the embedding dimension does not need to be tuned "
            "within this range.")
DISC_START = "We found that Bayesian Personalized Ranking did indeed learn accurate representations of bacterial substrate preferences"
DISC_OLD = ("and outperformed Singular Value Decomposition, Weighted Regularized Matrix Factorization, and Logistic Matrix "
            "Factorization.")
DISC_NEW = ("and, by ROC-AUC, was more accurate than Singular Value Decomposition, Weighted Regularized Matrix "
            "Factorization, and Logistic Matrix Factorization at every embedding dimension.")
RESPONSE = (" The reviewer's question changed how we present the model. Higher-dimensional embeddings do not improve "
            "accuracy, and the Results now say so. What BPR offers is the highest ROC-AUC at every dimension, no need "
            "to tune the dimension, and, unlike a prevalence table, a score for every isolate–metabolite pair. The "
            "changes below state these points and remove a claim that the benchmark does not support.")


def check_claims():
    roc = pd.read_csv(BY_DIM).pivot(index="dim", columns="model", values="roc_auc_mean")
    assert (roc.idxmax(axis=1) == BPR).all(), "BPR does not have the highest ROC-AUC at every dimension"
    flat = roc.loc[[8, 16, 32, 64], BPR]
    assert flat.idxmax() == 16 and round(flat.max() - flat.min(), 3) <= 0.012, flat.to_dict()


def fix_ms(ms):
    plain_replace(ms, TUNE_OLD, TUNE_NEW)
    disc = ms.find(DISC_START)
    plain_replace(ms, DISC_OLD, DISC_NEW, p=disc)
    for p, key in ((ms.find(TUNE_NEW), "Larger embeddings"), (disc, "We found that")):
        text = ms.accepted_text(p)
        print("   …" + text[text.index(key):text.index(key) + 330] + "…")


def fix_letter(letter):
    hits = [p for p in letter.paragraphs() if TUNE_OLD in letter.accepted_text(p)]
    assert len(hits) == 2, len(hits)  # comment 2 quote and the full paragraph under Reviewer 3, major comment 1
    for p in hits:
        plain_replace(letter, TUNE_OLD, TUNE_NEW, p=p)

    resp = letter.find("Response: We thank the reviewer for this practical question.")
    plain_replace(letter, letter.accepted_text(resp)[len("Response:"):], RESPONSE, p=resp)

    # Original/Modified pair for the Discussion sentence, formatted like the pair that follows it in the Discussion
    ps = letter.paragraphs()
    k = ps.index(letter.find("Original (Discussion, first paragraph, Lines 254"))
    templates = ps[k:k + 4]
    assert letter.accepted_text(templates[2]).strip() == "Modified:", letter.accepted_text(templates[2])
    sentence = f"{DISC_START} {DISC_OLD}"
    new = ["Original (Discussion, first paragraph, first sentence):", sentence, "Modified:",
           sentence.replace(DISC_OLD, DISC_NEW)]
    anchor = letter.find("Modified (Results, first section): see the response to Comment 2, which quotes")
    for template, text in zip(templates, new):
        p = copy.deepcopy(template)
        anchor.addnext(p)
        plain_replace(letter, letter.accepted_text(p), text, p=p)
        anchor = p
    ps = letter.paragraphs()
    k = ps.index(resp)
    for p in ps[k - 1:k + 9]:
        print("   " + letter.accepted_text(p)[:520])


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    check_claims()
    ms, letter = TrackedDocx(MS, "unused"), TrackedDocx(LETTER, "unused")
    print("Manuscript")
    fix_ms(ms)
    print("\nResponse letter")
    fix_letter(letter)
    if not apply:
        print("\ndry run: nothing written")
        return
    print(f"\nbackups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    ms.save()
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
