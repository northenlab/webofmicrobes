# pr_auc_wording_20261120.py
"""State the Fig. 1b model comparison separately for ROC-AUC and PR-AUC (mSystems revision).

The Results said BPR "was the most accurate method at every embedding dimension". That holds for
ROC-AUC only. By PR-AUC (mean over isolates) the best values of BPR, WR-MF and SVD are within 0.003
of one another, WR-MF and SVD are slightly higher than BPR at 4 and 8 dimensions, and BPR's
advantage is its stability at higher dimensions. The manuscript sentence is qualified, the PR-AUC
sentence rewritten with the numbers (read from fig1b_pr_by_dim.csv), and the response letter
updated: the two Modified quotes of that paragraph, the responses to Reviewer 1 comments 2 and 5
and Reviewer 3 major comment 1. The Reviewer 1 comment 2 response now also gives the PR-AUC pooled
across isolates, for which BPR is lower than WR-MF and SVD (already in the Supplementary Methods).
Plain (untracked) edits; the documents are backed up first.

Usage:
    python pr_auc_wording_20261120.py --apply
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from docx_track import TrackedDocx, _Package  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PR_CSV = REPO / "wom/manuscript_figures/msystems_revision/lambda0/r1/pr_calibration/fig1b_pr_by_dim.csv"
BPR = "Ordinal-BPR [lambda_ord=0]"  # results-CSV name of the adopted model

OLD_A = ("BPR was the most accurate method at every embedding dimension and the most stable across dimensions "
         "(ROC-AUC = 0.83–0.87 from 4 to 128 dimensions)")
NEW_A = ("By ROC-AUC, BPR was the most accurate method at every embedding dimension and the most stable across "
         "dimensions (0.83–0.87 from 4 to 128 dimensions)")
OLD_B = ("BPR reached PR-AUC = 0.69 at 32 dimensions; by this measure WR-MF performed similarly at 4 and 8 dimensions "
         "(0.69 and 0.70) but again declined at higher dimensions.")


def numbers() -> dict:
    pr = pd.read_csv(PR_CSV).pivot(index="dim", columns="model", values="pr_auc_mean")
    roc = pd.read_csv(PR_CSV).pivot(index="dim", columns="model", values="roc_auc_mean")
    others = [m for m in pr.columns if m != BPR]
    assert all(roc[BPR] > roc[others].max(axis=1)), "BPR is not the highest ROC-AUC at every dimension"
    assert (pr[BPR].idxmax(), pr["WR-MF"].idxmax(), pr["SVD"].idxmax()) == (16, 8, 8)
    best = {m: pr[m].max() for m in (BPR, "WR-MF", "SVD")}
    assert max(best.values()) - min(best.values()) < 0.003
    assert f"{best['WR-MF']:.3f}" == f"{best['SVD']:.3f}"
    for d in (4, 8):
        assert f"{pr.loc[d, 'WR-MF']:.3f}" == f"{pr.loc[d, 'SVD']:.3f}" and pr.loc[d, "WR-MF"] > pr.loc[d, BPR]
    for d in (16, 32, 64, 128):
        assert pr.loc[d, BPR] > pr.loc[d, ["WR-MF", "SVD", "LMF"]].max()
    assert f"{pr.loc[32, BPR]:.2f}" == "0.69"
    high = pr.loc[[16, 32, 64, 128], BPR]
    return dict(bpr_best=f"{best[BPR]:.3f}", other_best=f"{best['WR-MF']:.3f}",
                o4=f"{pr.loc[4, 'WR-MF']:.3f}", o8=f"{pr.loc[8, 'WR-MF']:.3f}",
                b4=f"{pr.loc[4, BPR]:.3f}", b8=f"{pr.loc[8, BPR]:.3f}",
                lo=f"{high.min():.2f}", hi=f"{high.max():.2f}",
                w128=f"{pr.loc[128, 'WR-MF']:.2f}", s128=f"{pr.loc[128, 'SVD']:.2f}")


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")):
        raise SystemExit("Word or Excel lock files present; close the documents first")
    n = numbers()
    best = (f"the best accuracies of BPR, WR-MF, and SVD were within 0.003 of one another ({n['bpr_best']} for BPR at "
            f"16 dimensions; {n['other_best']} for WR-MF and SVD at 8 dimensions)")
    new_b = (f"BPR reached PR-AUC = 0.69 at 32 dimensions. By this measure {best}, and WR-MF and SVD were slightly more "
             f"accurate than BPR at 4 and 8 dimensions ({n['o4']} and {n['o8']} vs. {n['b4']} and {n['b8']}). By PR-AUC, "
             f"the primary advantage of BPR is therefore its stability at higher dimensions: it stayed at "
             f"{n['lo']}–{n['hi']} from 16 to 128 dimensions, whereas WR-MF and SVD declined to {n['w128']} and "
             f"{n['s128']}.")

    ms = TrackedDocx(revision_paths.MS, "unused")
    ctx = "To benchmark model performance, we used a hold-out procedure"
    plain_replace(ms, OLD_A, NEW_A, context=ctx)
    plain_replace(ms, OLD_B, new_b, context=ctx)

    # the pooled values quoted in the letter must be the ones in the Supplementary Methods
    si = _Package(revision_paths.SI)
    pooled = "PR-AUC at 32 dimensions was 0.66 for BPR, 0.69 for WR-MF, 0.68 for SVD, and 0.51 for LMF"
    assert pooled in si.accepted_text(si.find("computed over all withheld pairs pooled across isolates"))

    letter = TrackedDocx(revision_paths.LETTER, "unused")
    # Modified quotes of the benchmark paragraph (Reviewer 1 comment 2; Reviewer 3 major comment 1)
    counts = {OLD_A: 0, OLD_B: 0}
    for p in letter.paragraphs():
        for old, new in ((OLD_A, NEW_A), (OLD_B, new_b)):
            if old in letter.accepted_text(p):
                plain_replace(letter, old, new, p=p)
                counts[old] += 1
    assert counts == {OLD_A: 1, OLD_B: 2}, counts

    responses = [
        # Reviewer 1, comment 2
        ("By this measure WR-MF at 4 and 8 dimensions performed similarly (0.69 and 0.70), but WR-MF, SVD, and LMF "
         "declined at higher dimensions, as they did by ROC-AUC.",
         f"By this measure BPR was not the most accurate method at every dimension, as it was by ROC-AUC: {best}, and "
         "WR-MF and SVD were slightly more accurate than BPR at 4 and 8 dimensions. By PR-AUC the primary advantage of "
         "BPR is its stability at higher dimensions, where WR-MF, SVD, and LMF declined as they did by ROC-AUC. Because "
         "BPR is trained to rank metabolites within each isolate, we averaged PR-AUC over isolates; computed over all "
         "withheld pairs pooled across isolates, PR-AUC at 32 dimensions was lower for BPR (0.66) than for WR-MF (0.69) "
         "and SVD (0.68), as stated in the Supplementary Methods."),
        # Reviewer 1, comment 5
        ("and by PR-AUC a low-dimensional WR-MF model is comparable (0.70 at 8 dimensions).",
         f"and by PR-AUC low-dimensional WR-MF and SVD models are comparable ({n['other_best']} at 8 dimensions, against "
         f"a best of {n['bpr_best']} for BPR at 16 dimensions)."),
        # Reviewer 3, major comment 1
        ("BPR was the most accurate model at every embedding dimension and the most stable (ROC-AUC = 0.83–0.87 from 4 "
         "to 128 dimensions; 0.865 ± 0.004 at 32 dimensions)",
         "By ROC-AUC, BPR was the most accurate model at every embedding dimension and the most stable (0.83–0.87 from "
         "4 to 128 dimensions; 0.865 ± 0.004 at 32 dimensions)"),
    ]
    for old, new in responses:
        hits = [p for p in letter.paragraphs() if old in letter.accepted_text(p)]
        assert len(hits) == 1 and letter.accepted_text(hits[0]).startswith("Response:"), (len(hits), old[:50])
        plain_replace(letter, old, new, p=hits[0])

    text = ms.accepted_text(ms.find(ctx))
    print(text[text.index("By ROC-AUC"):text.index("Larger embeddings")])
    if not apply:
        print("\ndry run: all edits located; nothing written")
        return
    print(f"\nbackups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    ms.save()
    letter.save()
    print("applied to", revision_paths.DOCS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
