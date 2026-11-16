# letter_sync_20261116.py
"""Bring the response letter back in line with the current manuscript and SI (mSystems revision).

After the team re-ran the agent prompts and made small edits (documents in ~/Downloads/revisions,
all tracked changes accepted), a point-by-point audit found letter passages that no longer matched:
"Modified" quotes that had been overtaken by later edits, one "Original" quote that was not the
submitted wording, and two response sentences written before the prompts were re-run. This script
updates those passages from the documents themselves. It edits only the letter.

Usage:
    python letter_sync_20261116.py            # dry run: prints the changes
    python letter_sync_20261116.py --apply
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import LetterDocx, _Package  # noqa: E402
from r3_major1_edits import LINE_PDF, LineNumbers  # noqa: E402

DOCS = Path.home() / "Downloads/revisions"
LETTER = DOCS / "Response_to_reviewers.docx"
MS = DOCS / "Northen_msystems_BPR.docx"
SI = DOCS / "Northen_msystems_BPR_Supplementary Information.docx"
BACKUPS = Path.home() / "Downloads/1st-revision/backups"


def para(doc, snippet: str) -> str:
    return doc.accepted_text(doc.find(snippet)).strip()


def main(apply: bool):
    if list(DOCS.glob("~$*")):
        raise SystemExit("Word lock files present; close the documents in Word first")
    ms, si = _Package(MS), _Package(SI)
    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    lines = LineNumbers(LINE_PDF)
    log = []

    def set_text(snippet, new, exact=False, what=""):
        old = letter.accepted_text(letter.find(snippet, exact)).strip()
        if old != new:
            letter.set_text(snippet, new, exact)
            log.append((what, old, new))

    # R1 comment 4: Methods "Model training" paragraph (now also cites Fig. 1c and calibration)
    set_text("We framed the prediction of microbial metabolite consumption as an implicit-feedback",
             para(ms, "We framed the prediction of microbial metabolite consumption"), what="R1-4 Modified (Methods)")

    # R2 comment 2: limitations paragraph, from "Thus, predictions" to the end
    limits = para(ms, "There are a number of limitations to this study.")
    set_text("as should predictions for under-represented genera or rarely observed compounds. More generally",
             limits[limits.index("Thus, predictions"):], what="R2-2 Modified (Discussion)")

    # R3 major 1: benchmark paragraph (now includes the PR-AUC, baseline, calibration and learning-curve sentences)
    set_text("To benchmark model performance, we used a hold-out procedure repeated five times",
             para(ms, "To benchmark model performance, we used a hold-out procedure"), what="R3 major 1 Modified (Results)")

    # R3 major 1 response: the agent prompts were re-run
    r = letter.response_text("Major comment 1.")
    old = ("We retrained the final model with this objective and regenerated all model-derived results (Fig. 1b, Fig. 2, "
           "Fig. 3, Supplementary Figs. 2 and 3, SI Table 2, and the BPR scores quoted in the Supplementary Information); "
           "updated values are given in the revised text.")
    new = ("We retrained the final model with this objective and regenerated all model-derived results (Fig. 1b, Fig. 2, "
           "Fig. 3, Supplementary Figs. 2 and 3, and SI Table 2); updated values are given in the revised text. We also "
           "re-ran the three example agent prompts in the Supplementary Information with the retrained model and updated "
           "their responses, Supplementary Fig. 4, and its data sheets.")
    if old in r:
        letter.set_response("Major comment 1.", r.replace(old, new))
        log.append(("R3 major 1 response", old, new))

    # R3 minor 2: the manuscript sentence has no "(see Methods)"
    old = "corresponding to 74,604 unique isolate–metabolite pairs (see Methods), mainly from"
    if any(old in letter.accepted_text(p) for p in letter.paragraphs()):
        letter.replace_in_run(old, old.replace(" (see Methods)", ""))
        log.append(("R3 minor 2 Modified (Results)", old, old.replace(" (see Methods)", "")))

    # R3 minor 4: Abstract pair. The submitted sentence said "AUC"; the revised one starts "An experiment"
    orig = "A prospective experiment on 12 newly profiled isolates confirmed this generalization (mean AUC = 0.87)."
    found = lines.lines(orig)
    assert found != "__", "original Abstract sentence not found in the line-numbered PDF"
    stale_orig = "A prospective experiment on 12 newly profiled isolates confirmed this generalization (mean ROC-AUC = 0.87)."
    if any(letter.accepted_text(p).strip() == stale_orig for p in letter.paragraphs()):
        label = letter.find(stale_orig, exact=True).getprevious()
        assert letter.accepted_text(label).strip() == "Original (Abstract):"
        letter.set_paragraph_text(label, "Original (Abstract, Lines %s):" % found)
        set_text(stale_orig, orig, exact=True, what="R3 minor 4 Original (Abstract), now with lines %s" % found)
    abstract = para(ms, "Understanding and predicting bacterial substrate preferences has broad utility")
    start = abstract.index("An experiment on 12 newly profiled isolates")
    sentence = abstract[start:abstract.index("(mean ROC-AUC = 0.89).", start) + len("(mean ROC-AUC = 0.89).")]
    set_text("A prospective experiment on 12 newly profiled isolates absent from training confirmed this generalization",
             sentence, what="R3 minor 4 Modified (Abstract)")

    # R3 minor 5: the Prompt 3 response was regenerated with the revised model
    r = letter.response_text("Minor comment 5.")
    old = ("The Prompt 3 transcript, which was generated with the submitted model, is unchanged, but it now carries a "
           "note giving the revised-model scores of the tested substrates.")
    new = ("The Prompt 3 response was regenerated with the revised model: it now recommends succinate as the "
           "highest-scoring carbon source and arginine as a neutral-scoring one, both of which were tested in Fig. S5, "
           "and a note at the start of Prompt 3 gives the scores of all the tested substrates.")
    if old in r:
        letter.set_response("Minor comment 5.", r.replace(old, new))
        log.append(("R3 minor 5 response", old, new))
    set_text("Note added in revision: this response was generated with", para(si, "Note added in revision:"),
             what="R3 minor 5 Modified (SI, Prompt 3 note)")

    for what, old, new in log:
        print("\n## " + what)
        if len(old) > 400:
            import difflib
            for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
                if op != "equal":
                    print("   %s: %r -> %r" % (op, old[a1:a2][:160], new[b1:b2][:160]))
        else:
            print("   - " + old)
            print("   + " + new)
    print("\n%d passages %s" % (len(log), "updated" if apply else "would be updated (dry run)"))
    if apply and log:
        print("backup:", letter.backup(BACKUPS))
        letter.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
