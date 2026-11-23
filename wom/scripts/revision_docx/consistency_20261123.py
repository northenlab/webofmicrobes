# consistency_20261123.py
"""One name, one unit and one notation per concept in the revised text (mSystems revision).

A read-through of everything added in revision for places where the reader is given two versions
of the same thing. Plain (untracked) edits; all four documents are backed up first. Run once.

Manuscript (each change is mirrored in the letter's quotes)
  * Methods, Model training: the notation now matches Fig. 1c (isolate i, metabolite j, negative k;
    embeddings u_i and v_j; score s_ij), with real subscripts ("sui" and "pu" were plain text);
    "tested-but-neutral" becomes "measured-neutral", the term used in the Results; "triplets"
    becomes "triples", as in Fig. 1c.
  * Fig. 1c caption: "u_i" etc. typed with underscores become subscripts.
  * Results, learning curve: training-set size is given only as a percentage of the isolate ×
    metabolite matrix, the x axis of Fig. S8 (it was also given as a fraction of the training pairs).
  * Results, PR-AUC: one value for how rare consumed pairs are (0.29, the value a random ranking
    scores), not two (16% of all withheld pairs was the second).
Response letter
  * Reviewer 1, comments 2, 4 and 6: the same changes in the responses.
  * Reviewer 3, major comment 1: the comparison of the model with and without the ordinal term is
    given for the hold-out benchmark of Fig. 1b, as in the Supplementary Methods (the response used
    leave-one-out values, so "0.86" stood for two different models).

Usage:
    python consistency_20261123.py            # dry run
    python consistency_20261123.py --apply
"""

import sys
import copy
import argparse
from pathlib import Path

import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import MS, LETTER  # noqa: E402
from docx_track import TrackedDocx, q, XML_SPACE  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

L0 = Path(__file__).resolve().parents[3] / "wom/manuscript_figures/msystems_revision/lambda0"
METHODS = "We framed the prediction of microbial metabolite consumption as an implicit-feedback"
FIG1C = "(c) Schematic of the BPR model"
AFTER_VERTALIGN = ("rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath")


def sub(text: str) -> tuple:
    return (text, True)


# (old text, new text as plain strings and subscripts)
NOTATION = [
    ("consumed (a positive), tested but neutral (a measured negative), or untested",
     ["consumed (a positive), measured-neutral (tested but not consumed; a measured negative), or untested"]),
    ("Each isolate u and each metabolite i is assigned a learnable vector of length d (pu and qi;",
     ["Each isolate i and each metabolite j is assigned a learnable vector of length d (u", sub("i"), " and v",
      sub("j"), ";"]),
    ("the dot product sui = pu · qi reproduces",
     ["the dot product s", sub("ij"), " = u", sub("i"), " · v", sub("j"), " reproduces"]),
    ("The score sui is", ["The score s", sub("ij"), " is"]),
    ("training samples triplets (u, i, j) in which i is consumed by u and j is a negative, drawn with probability "
     "0.5 from u's tested-but-neutral metabolites",
     ["training samples triples (i, j, k) in which metabolite j is consumed by isolate i and k is a negative, drawn "
      "with probability 0.5 from the isolate's measured-neutral metabolites"]),
    ("−log σ(sui − suj)", ["−log σ(s", sub("ij"), " − s", sub("ik"), ")"]),
    ("256 triplets per batch", ["256 triples per batch"]),
]
CAPTION = [("the score u_i · v_j relative to u_i · v_k.",
            ["the score u", sub("i"), " · v", sub("j"), " relative to u", sub("i"), " · v", sub("k"), "."])]


def numbers() -> dict:
    lc = pd.read_csv(L0 / "r1/learning_curve/learning_curve_summary.csv").set_index("fraction")
    pct = lc.pct_of_matrix_mean.round().astype(int)
    assert (pct[0.05], pct[0.5], pct[0.75], pct[1.0]) == (2, 21, 32, 42), pct.to_dict()
    best_baseline = lc[["popularity_roc_auc_mean", "genus_roc_auc_mean"]].max(axis=1)
    above = lc.roc_auc_mean > best_baseline
    assert list(above[above].index) == [0.75, 1.0], "BPR exceeds both baselines only from 32% of the matrix"
    r = pd.read_csv(L0 / "fig1b/fig1b_benchmark_results.csv")
    hold = r[r.protocol == "B_holdout"].groupby(["model", "dim"]).roc_auc.mean()
    n = {"lc_roc": [f"{lc.roc_auc_mean[f]:.2f}" for f in (0.05, 0.5, 1.0)],
         "lc_pr": [f"{lc.pr_auc_mean[f]:.2f}" for f in (0.05, 0.5, 1.0)],
         "ord32": f"{hold['Ordinal-BPR', 32]:.2f}", "ord128": f"{hold['Ordinal-BPR', 128]:.2f}",
         "bpr32": f"{hold['Ordinal-BPR [lambda_ord=0]', 32]:.2f}", "bpr128": f"{hold['Ordinal-BPR [lambda_ord=0]', 128]:.2f}"}
    assert n["lc_roc"] == ["0.52", "0.78", "0.86"] and n["lc_pr"] == ["0.41", "0.59", "0.69"], n
    assert hold["Ordinal-BPR", 32] < hold["Ordinal-BPR [lambda_ord=0]", 32], "the ordinal term does not lower accuracy"
    return n


def texts(n: dict) -> dict:
    return {
        "lc_old": ("(ROC-AUC = 0.78 with half and 0.86 with all training pairs), and BPR's ROC-AUC exceeded that of "
                   "the learning-free baselines only with 32% or more of the isolate × metabolite matrix in training "
                   "(it was lower with 21%; Fig. S8)."),
        "lc_new": (f"(ROC-AUC = {n['lc_roc'][1]} with 21% of the isolate × metabolite matrix in training and "
                   f"{n['lc_roc'][2]} with 42%, the full training set), and BPR's ROC-AUC exceeded that of the "
                   "learning-free baselines only with 32% or more of the matrix in training (Fig. S8)."),
        "pr_old": "Because consumed pairs are the minority class (16% of withheld pairs), we also computed",
        "pr_new": "Because consumed pairs are the minority class, we also computed",
        "c2_old": ("In the Fig. 1b hold-out, where 16% of withheld pairs were consumed (0.29 on average per isolate), "
                   "BPR reached"),
        "c2_new": "In the Fig. 1b hold-out, where a random ranking scores 0.29, BPR reached",
        "c6_old": ("we trained BPR on nested random subsets of 5–100% of the training pairs, which correspond to 2–42% "
                   "of the 433 × 311 isolate × metabolite matrix, and recorded ROC-AUC and PR-AUC (Fig. S8). Both "
                   "improved steadily with more data and had not reached a plateau at the size of the current dataset "
                   "(ROC-AUC = 0.52, 0.78, and 0.86 with 5%, 50%, and 100% of the training pairs; PR-AUC = 0.41, 0.59, "
                   "and 0.69)."),
        "c6_new": ("we trained BPR on random subsets of the training pairs that covered 2–42% of the 433 × 311 isolate "
                   "× metabolite matrix (42% is the full training set), and recorded ROC-AUC and PR-AUC (Fig. S8). Both "
                   "improved steadily with more data and had not reached a plateau at the size of the current dataset "
                   f"(ROC-AUC = {', '.join(n['lc_roc'][:2])}, and {n['lc_roc'][2]} with 2%, 21%, and 42% of the matrix "
                   f"in training; PR-AUC = {', '.join(n['lc_pr'][:2])}, and {n['lc_pr'][2]})."),
        "r3_old1": ("the ordinal model performed worse rather than better: against measured-neutral compounds it "
                    "reached ROC-AUC = 0.79 at 32 dimensions, below standard BPR (0.86), and it fell to 0.54 at 128 "
                    "dimensions."),
        "r3_new1": ("the ordinal term did not help: in the hold-out benchmark described below, the ordinal model "
                    f"reached ROC-AUC = {n['ord32']} at 32 dimensions and fell to {n['ord128']} at 128 dimensions."),
        "r3_old2": "gave the most accurate and most stable model (0.90).",
        "r3_new2": f"gave the most accurate and most stable model ({n['bpr32']} at 32 dimensions and {n['bpr128']} at 128).",
    }


def replace_runs(doc, p, old: str, pieces: list):
    """Untracked replacement of `old` in paragraph `p` by runs of plain and subscript text."""
    text = doc.accepted_text(p)
    if text.count(old) != 1:
        raise ValueError(f"{text.count(old)} occurrences of {old[:60]!r} in the paragraph")
    start = text.index(old)
    covered = doc._isolate(p, start, start + len(old))
    for piece in pieces:
        piece_text, is_sub = piece if isinstance(piece, tuple) else (piece, False)
        run = copy.deepcopy(covered[0])
        t = run.find(q("t"))
        t.text = piece_text
        t.set(XML_SPACE, "preserve")
        if is_sub:
            rpr = run.find(q("rPr"))
            if rpr is None:
                rpr = etree.Element(q("rPr"))
                run.insert(0, rpr)
            v = etree.Element(q("vertAlign"))
            v.set(q("val"), "subscript")
            later = [c for c in rpr if c.tag in {q(x) for x in AFTER_VERTALIGN}]
            later[0].addprevious(v) if later else rpr.append(v)
        covered[0].addprevious(run)
    for r in covered:
        r.getparent().remove(r)


def show(doc, p, around: str, width: int = 330):
    text = doc.accepted_text(p)
    k = text.index(around)
    print("   …" + text[max(0, k - 40):k + width] + "…")


def fix_ms(ms, t: dict):
    methods = ms.find(METHODS)
    for old, pieces in NOTATION:
        replace_runs(ms, methods, old, pieces)
    for old, pieces in CAPTION:
        replace_runs(ms, ms.find(FIG1C), old, pieces)
    plain_replace(ms, t["lc_old"], t["lc_new"])
    plain_replace(ms, t["pr_old"], t["pr_new"])
    show(ms, methods, "takes one of three states", 200)
    show(ms, methods, "Each isolate i", 560)
    show(ms, methods, "For BPR, training samples", 330)
    show(ms, ms.find(FIG1C), "by increasing the score", 80)
    show(ms, ms.find(t["lc_new"][:40]), "In a learning curve", 480)
    show(ms, ms.find(t["pr_new"]), t["pr_new"], 250)


def fix_letter(letter, t: dict):
    methods = letter.find(METHODS)
    for old, pieces in NOTATION:
        replace_runs(letter, methods, old, pieces)
    for old, pieces in CAPTION:
        replace_runs(letter, letter.find(FIG1C), old, pieces)
    for old, new, count in [(t["lc_old"], t["lc_new"], 2), (t["pr_old"], t["pr_new"], 2)]:
        hits = [p for p in letter.paragraphs() if old in letter.accepted_text(p)]
        assert len(hits) == count, (len(hits), old[:50])  # the quote and the full paragraph under Reviewer 3, major 1
        for p in hits:
            plain_replace(letter, old, new, p=p)
    plain_replace(letter, "with consumed, tested-but-neutral, and untested states",
                  "with consumed, measured-neutral, and untested states")
    for key in ("c2", "c6"):
        plain_replace(letter, t[f"{key}_old"], t[f"{key}_new"])
        show(letter, letter.find(t[f"{key}_new"][:60]), t[f"{key}_new"][:60], len(t[f"{key}_new"]))
    r3 = letter.find(t["r3_old1"][:60])
    plain_replace(letter, t["r3_old1"], t["r3_new1"], p=r3)
    plain_replace(letter, t["r3_old2"], t["r3_new2"], p=r3)
    show(letter, r3, "When we passed", 760)


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    t = texts(numbers())
    ms, letter = TrackedDocx(MS, "unused"), TrackedDocx(LETTER, "unused")
    print("Manuscript")
    fix_ms(ms, t)
    print("\nResponse letter")
    fix_letter(letter, t)
    for name, doc in (("manuscript", ms), ("letter", letter)):
        for p in doc.paragraphs():
            s = doc.accepted_text(p)
            for stale in ("sui", "suj", "tested-but-neutral", "u_i", "v_j", "with half and"):
                assert stale not in s.replace("suit", "").replace("pursui", ""), (name, stale, s[:80])
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
