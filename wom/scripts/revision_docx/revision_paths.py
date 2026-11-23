# revision_paths.py
"""Locations of the revision documents for the 2026-11-16 edit scripts (mSystems revision).

On 2026-11-18 ~/Downloads/revisions was restored from the team's Google Drive copies after the local
folder was deleted. The restored files carry the Drive file names, and ~/Downloads/1st-revision (the
earlier backups and the line-numbered PDF of the original submission) no longer exists. This module
accepts either set of file names, keeps new backups next to the documents' folder, and falls back to
the line numbers measured from the PDF on 2026-11-16 when the PDF is absent.

Usage:
    python revision_paths.py --backup     # copy the four documents to ~/Downloads/revisions_backups
"""

import shutil
import argparse
from datetime import datetime
from pathlib import Path

DOCS = Path.home() / "Downloads/revisions"
BACKUPS = Path.home() / "Downloads/revisions_backups"
# Word's PDF export (2026-11-23) of the original submission's .docx, which has line numbering switched on; it
# reproduces every line range that was measured from the journal's PDF before that file was lost
LINE_PDF = BACKUPS / "Original Maintext Submission ONLY FOR LINE NUMBERS (Word export).pdf"

# line ranges in the line-numbered original PDF, as measured by LineNumbers on 2026-11-16
RECORDED_LINES = {
    "abstract_generalization": "47–49",  # "A prospective experiment on 12 newly profiled isolates ... (mean AUC = 0.87)."
    "fig3_caption_panel_a": "486–489",   # "(a) Hold-out AUC separating ... increasing phylogenetic distance."
}


def _doc(*names: str) -> Path:
    found = [DOCS / n for n in names if (DOCS / n).exists()]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one of {names} in {DOCS}, found {[f.name for f in found]}")
    return found[0]


MS = _doc("Northen_msystems_BPR.docx", "msystems bpr maintext.docx")
SI = _doc("Northen_msystems_BPR_Supplementary Information.docx", "msystems bpr supplemental information.docx")
LETTER = _doc("Response_to_reviewers.docx", "msystems bpr response to reviewers.docx")
TABLES = _doc("Msystems BPR Supplementary Tables.xlsx", "msystems bpr si tables.xlsx")


def original_lines(quote: str, key: str) -> str:
    """Line range of `quote` in the original submission: from the PDF if present, else as recorded."""
    if LINE_PDF.exists():
        from r3_major1_edits import LineNumbers
        return LineNumbers(LINE_PDF).lines(quote)
    return RECORDED_LINES[key]


def backup() -> str:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for f in (MS, SI, LETTER, TABLES):
        shutil.copy2(f, BACKUPS / f"{f.stem}.before-{stamp}{f.suffix}")
    return stamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backup", action="store_true")
    if parser.parse_args().backup:
        print(f"backups: {BACKUPS}/*.before-{backup()}.*")
    else:
        for f in (MS, SI, LETTER, TABLES):
            print(f)
