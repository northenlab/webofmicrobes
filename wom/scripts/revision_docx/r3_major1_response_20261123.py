# r3_major1_response_20261123.py
"""Reviewer 3, major comment 1: a response of about a third of the length (mSystems revision).

The response states the two errors in a sentence, what replaced the benchmark, why the ordinal term
was removed, and the headline numbers before and after; the 17 Original/Modified pairs below it
carry the detail. The numbers in the response are checked against the benchmark results before
anything is written. Plain (untracked) edit of the letter; the documents are backed up first.

Usage:
    python r3_major1_response_20261123.py            # dry run
    python r3_major1_response_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

RESULTS = (Path(__file__).resolve().parents[3] /
           "wom/manuscript_figures/msystems_revision/lambda0/fig1b/fig1b_benchmark_results.csv")
START = "Response: We thank the reviewer for this careful reading of the benchmark and its code."
RESPONSE = (
    " We thank the reviewer for this careful reading of the benchmark and its code. Both concerns were correct, and "
    "addressing them changed the model used throughout the manuscript. The submitted benchmark compared each withheld "
    "consumed metabolite with a mixture of measured-neutral, secreted, and untested metabolites, and the ordinal term "
    "was not active in the run that produced Fig. 1b. We replaced the benchmark with a hold-out that matches the task "
    "described in the Results: withheld consumed versus withheld measured-neutral metabolites, with secreted "
    "metabolites never used as negatives. Leave-one-out results for three candidate sets (measured-neutral, untested, "
    "and all non-positive metabolites) are reported separately in the Supplementary Methods. In the hold-out the "
    "ordinal term lowered accuracy (ROC-AUC = {ordinal} with it and {bpr} without it at 32 dimensions), so we removed "
    "it and now call the model BPR. We retrained the model, regenerated every result that depends on it, and re-ran "
    "the example prompts. The ROC-AUC in Fig. 1b is now 0.86 (0.94 in the submission), the hierarchical hold-out "
    "medians are 0.91 to 0.78 (0.90 to 0.76), and the 12 held-out isolates average 0.89 (0.87). Code that reproduces "
    "the submitted benchmark, demonstrates both problems, and runs the corrected benchmark is in the project "
    "repository.")
# each before/after number in the response must appear in the quoted Original/Modified text below it
QUOTED = ["(AUC = 0.94)", "ROC-AUC = 0.86 for withheld isolate–metabolite pairs", "declining from 0.90 to 0.76",
          "declining from 0.91 to 0.78"]


def numbers() -> dict:
    r = pd.read_csv(RESULTS)
    hold = r[(r.protocol == "B_holdout") & (r.dim == 32)].groupby("model").roc_auc.mean()
    n = {"ordinal": f"{hold['Ordinal-BPR']:.2f}", "bpr": f"{hold['Ordinal-BPR [lambda_ord=0]']:.2f}"}
    assert n == {"ordinal": "0.83", "bpr": "0.86"}, n
    return n


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    letter = TrackedDocx(revision_paths.LETTER, "unused")
    resp = letter.find(START)
    old = letter.accepted_text(resp)[len("Response:"):]
    new = RESPONSE.format(**numbers())
    plain_replace(letter, old, new, p=resp)
    ps = letter.paragraphs()
    k = ps.index(resp)
    end = next(i for i in range(k + 1, len(ps)) if letter.accepted_text(ps[i]).startswith("Minor comment 1."))
    below = " ".join(letter.accepted_text(p) for p in ps[k + 1:end])
    for phrase in QUOTED:
        assert phrase in below, phrase
    everything = " ".join(letter.accepted_text(p) for p in ps)
    for phrase in ("(mean AUC = 0.87)", "(mean ROC-AUC = 0.89)"):  # the 12 isolates, quoted under other comments
        assert phrase in everything, phrase
    print(f"   [{len(old.split())} -> {len(new.split())} words]\n   Response:{new}")
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
