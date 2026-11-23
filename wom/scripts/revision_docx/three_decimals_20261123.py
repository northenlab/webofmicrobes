# three_decimals_20261123.py
"""Three decimals where two rounded values coincide (mSystems revision; Reviewer 3, major comment 1).

At two decimals the response gave "0.83" for two different models, and the new Supplementary Methods
sentence gave "0.77" twice. Both now use three decimals, computed from the benchmark results. Plain
(untracked) edits of the SI and the letter; the documents are backed up first.

Usage:
    python three_decimals_20261123.py            # dry run
    python three_decimals_20261123.py --apply
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import revision_paths  # noqa: E402
from revision_paths import SI, LETTER  # noqa: E402
from docx_track import TrackedDocx  # noqa: E402
from final_fixes_20261116 import plain_replace  # noqa: E402

RESULTS = (Path(__file__).resolve().parents[3] /
           "wom/manuscript_figures/msystems_revision/lambda0/fig1b/fig1b_benchmark_results.csv")
SI_OLD = ("hold-out ROC-AUC was 0.83 at 32 dimensions and 0.77 at 128 dimensions, compared with 0.86 and 0.83 for the "
          "model as used and 0.77 and 0.62 for WR-MF")
SI_NEW = ("hold-out ROC-AUC was {p32} at 32 dimensions and {p128} at 128 dimensions, compared with {a32} and {a128} for "
          "the model as used and {w32} and {w128} for WR-MF")
RESPONSE = [("(ROC-AUC = 0.83 with it and 0.86 without it at 32 dimensions)",
             "(ROC-AUC = {o32} with it and {a32} without it at 32 dimensions)"),
            ("trained without the measured-neutral labels (0.83 at 32 dimensions).",
             "trained without the measured-neutral labels ({p32} at 32 dimensions).")]


def numbers() -> dict:
    r = pd.read_csv(RESULTS)
    mean = r[r.protocol == "B_holdout"].groupby(["model", "dim"]).roc_auc.mean()
    n = {"p32": mean["BPR", 32], "p128": mean["BPR", 128], "a32": mean["Ordinal-BPR [lambda_ord=0]", 32],
         "a128": mean["Ordinal-BPR [lambda_ord=0]", 128], "w32": mean["WR-MF", 32], "w128": mean["WR-MF", 128],
         "o32": mean["Ordinal-BPR", 32]}
    out = {k: f"{v:.3f}" for k, v in n.items()}
    assert out["a32"] == "0.865" and out["o32"] != out["p32"] and out["p128"] != out["w32"], out
    return out


def main(apply: bool):
    if list(revision_paths.DOCS.glob("~$*")) and apply:
        raise SystemExit("Word or Excel lock files present; close the documents first")
    n = numbers()
    si, letter = TrackedDocx(SI, "unused"), TrackedDocx(LETTER, "unused")
    for doc in (si, letter):
        p = doc.find(SI_OLD)
        plain_replace(doc, SI_OLD, SI_NEW.format(**n), p=p)
    print("   SI and its quote: …" + SI_NEW.format(**n) + "…")
    resp = letter.find("Response: We thank the reviewer for this careful reading of the benchmark and its code.")
    for old, new in RESPONSE:
        plain_replace(letter, old, new.format(**n), p=resp)
    text = letter.accepted_text(resp)
    print("   response: …" + text[text.index("In the hold-out the ordinal term"):text.index("We retrained the model")])
    if not apply:
        print("dry run: nothing written")
        return
    print(f"backups: {revision_paths.BACKUPS}/*.before-{revision_paths.backup()}.*")
    si.save()
    letter.save()
    print("saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
