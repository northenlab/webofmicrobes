# s5_edits.py
"""Supplementary Fig. 5 remade with the succinic acid growth experiment (R3 minor 5; mSystems revision).

Applies, as tracked changes (author Benjamin Bowen): the re-plotted Supplementary Fig. 5
(revision_figS5_growth.py, 0-21 h, all three experiments) and its caption, the SI Methods for the
growth experiment, and a note on the Prompt 3 transcript giving the revised-model scores of the
tested substrates (the transcript itself, generated with the submitted model, is kept). Replaces
the "Data for Figure S5" sheet with the three experiments and rewrites the R3 minor 5 response.

Usage:
    python s5_edits.py --preview-dir /tmp/s5_preview
    python s5_edits.py --apply
"""

import io
import re
import sys
import shutil
import zipfile
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import TrackedDocx, LetterDocx, q  # noqa: E402
from r3_major1_edits import REPO, DOCS, SI, LETTER, BACKUPS, AUTHOR, TABLES, sheet_data_xml  # noqa: E402

S5 = REPO / "wom/manuscript_figures/msystems_revision/figS5_succinate"
S5_PNG = S5 / "figS5_growth.png"
S5_DATA = S5 / "Growth_Data_Acinetobacter_Succinic_Acid.xlsx"
ORIGINAL_SI = DOCS.parent / "original_submission/Original Northen_msystems_BPR_Supplementary Information.docx"
S5_SHEET = "xl/worksheets/sheet6.xml"
RUNS = [("Exp1_AMD1706", "12 March 2026"), ("Exp2_AMD1706", "16 March 2026"), ("Exp3_AMD1706", "20 March 2026")]
# Revised-model genus scores for Acinetobacter (one training isolate, GW460-4) and ranks among the
# 64 NLDM metabolites in the model; computed with revision_lambda0_si_scores.GenusScorer
SCORES = {"succinate": (10.91, 1), "malate": (8.16, 8), "pyruvate": (6.95, 15), "D-glucose": (2.39, 36),
          "spermidine": (-1.46, 59), "L-ectoine": (-3.98, 64), "arginine": (3.31, 32),
          "N-acetyl-glutamate": (1.11, 51), "D-xylose": (0.47, 53)}


def od(summary: pd.DataFrame, condition: str) -> str:
    return f"{summary.set_index('condition').loc[condition, 'od_end_mean']:.2f}"


def caption_edits(s: pd.DataFrame) -> list:
    sc = SCORES
    return [
        ("over 36 hours on NLDM or sole carbon sources selected by the WoM model at equivalent",
         "over 21 hours on NLDM or on single carbon sources at equivalent"),
        ("Malic acid (the top-ranked BPR prediction, score = 13.06) supported the highest final biomass (OD ≈ 0.71), "
         "exceeding growth on NLDM complex medium (OD ≈ 0.60).",
         f"Succinic acid (succinate is the top-ranked of the 64 NLDM metabolites, BPR score = {sc['succinate'][0]:.2f}) "
         f"supported the highest biomass (OD ≈ {od(s, 'Succinic Acid')} at 21 h), exceeding growth on NLDM complex "
         f"medium (OD ≈ {od(s, 'NLDM')})."),
        ("Glucose (BPR score = 4.62), ectoine (BPR score = −0.89), spermidine (BPR score = −3.76), and water (negative "
         "control) all showed negligible growth.",
         f"N-acetyl-glutamate (BPR score = {sc['N-acetyl-glutamate'][0]:.2f}; rank {sc['N-acetyl-glutamate'][1]}), "
         f"arginine ({sc['arginine'][0]:.2f}; rank {sc['arginine'][1]}), and xylose ({sc['D-xylose'][0]:.2f}; rank "
         f"{sc['D-xylose'][1]}) supported much less growth (OD ≈ {od(s, 'N-Acetyl-Glutamate')}, {od(s, 'Arginine')}, "
         f"and {od(s, 'Xylose')} at 21 h), and water was the negative control (OD ≈ "
         f"{od(s, 'Water (Negative Control)')})."),
        ("Lines represent means of a single growth experiment; shaded bands indicate ± 1 s.d. across 3 biological "
         "replicates.",
         "Lines show the mean of three independent experiments, each the mean of three replicate wells; shaded bands, "
         "± 1 s.d. across the three experiments. The time axis ends at 21 h, when the shortest of the three "
         "experiments ended."),
    ]


METHODS_EDITS = [
    ("Growth Enhancedment - Prompt 3", "Growth Enhancement - Prompt 3"),
    ("containing malic acid, glucose, or ectoine as the sole carbon source at a final concentration of 1042 mg/L of "
     "carbon (21.66 mM, 14.44 mM, 14.45 mM, respectively.",
     "containing succinic acid, xylose, arginine, or N-acetyl-glutamate as the sole carbon source at a final "
     "concentration of 1042 mg/L of carbon."),
    ("The OD600 was measured in a 96 well plate at 15 min intervals for using BioTek Epoch 2 Reader (Agilent) setup to "
     "30C and continuous shaking at 282 cpm (3 mm). Each treatment group contained 3 biological replicates. The "
     "experiment was repeated 3 times (for a total of 4 growth experiments) with cultivation for 47, 47, 92 and 92 h.",
     "The OD600 was measured in a 96 well plate at 15 min intervals using a BioTek Epoch 2 reader (Agilent) at 30 °C "
     "with continuous double-orbital shaking. Each condition was grown in three replicate wells, and the experiment "
     "was performed three times independently (12, 16, and 20 March 2026), with cultivation for 47, 21, and 51 h. "
     "For Fig. S5, the replicate wells of each experiment were averaged and interpolated onto a common time grid; the "
     "curves show the mean of the three experiments and the shaded bands ± 1 s.d. (n − 1) across experiments, and the "
     "time axis ends at 21 h, when the second experiment ended."),
]

PROMPT3_NOTE = (
    "Note added in revision: this response was generated with the submitted version of the BPR model. With the revised "
    "model, succinate is the top-ranked of the 64 NLDM metabolites for Acinetobacter (BPR score {succinate[0]:.2f}); "
    "malate scores {malate[0]:.2f} (rank {malate[1]}), pyruvate {pyruvate[0]:.2f} (rank {pyruvate[1]}), D-glucose {D-glucose[0]:.2f} (rank {D-glucose[1]}), spermidine {spermidine[0]:.2f} (rank "
    "{spermidine[1]}), and L-ectoine {L-ectoine[0]:.2f} (rank {L-ectoine[1]}). Succinate, recommended in the response below, was used in "
    "the growth experiment of Fig. S5, together with arginine ({arginine[0]:.2f}, rank {arginine[1]}), "
    "N-acetyl-glutamate ({N-acetyl-glutamate[0]:.2f}, rank {N-acetyl-glutamate[1]}), and D-xylose ({D-xylose[0]:.2f}, "
    "rank {D-xylose[1]}).")


def prompt3_note() -> str:
    text = PROMPT3_NOTE
    for key, (score, rank) in SCORES.items():
        text = text.replace(f"{{{key}[0]:.2f}}", f"{score:.2f}").replace(f"{{{key}[1]}}", str(rank))
    assert "{" not in text, text
    return text


def s5_sheet_rows() -> list:
    rows = [["Acinetobacter guillouiae strain GW821-FHT01E09 grown in three independent experiments (12, 16, and 20 "
             "March 2026), each with three replicate wells per condition, on NLDM or on a defined medium with succinic "
             "acid, xylose, arginine, or N-acetyl-glutamate as the sole carbon source (1,040 ppm C), with water as the "
             "negative control; OD600 every 15 min at 30 °C. Supplementary Fig. 5 shows 0-21 h, the period covered by "
             "all three experiments (the second experiment ended at 21.2 h)."], []]
    for sheet, date in RUNS:
        df = pd.read_excel(S5_DATA, sheet_name=sheet).dropna(subset=["Time"])
        rows.append([f"OD600 readings, experiment {sheet[3]} ({date}; last read {df.Time.max():.2f} h)"])
        rows.append(["Time (h)"] + [str(c).split(".")[0] for c in df.columns[1:]])
        for _, r in df.iterrows():
            rows.append([round(float(v), 4) for v in r.values])
        rows.append([])
    return rows


def write_workbook(out_path: Path, backup: bool):
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    xml = parts[S5_SHEET].decode()
    if "Succinic" in xml:  # the original sheet keeps its text in sharedStrings; ours is written inline
        raise SystemExit("S5 sheet already holds the succinic acid experiment")
    new, k = re.subn(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(s5_sheet_rows()), xml, flags=re.S)
    assert k == 1
    parts[S5_SHEET] = new.encode()
    if backup:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TABLES, BACKUPS / f"{TABLES.stem}.before-{datetime.now():%Y%m%d-%H%M%S}{TABLES.suffix}")
    tmp = out_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(out_path)


def main(apply: bool, preview_dir: Path):
    if apply and list(DOCS.glob("~$*")):
        raise SystemExit("Word lock files present; close the documents in Word first")
    summary = pd.read_csv(S5 / "figS5_summary.csv")
    si = TrackedDocx(SI, AUTHOR)

    # figure: same width, height from the new image's aspect ratio
    from PIL import Image
    png = S5_PNG.read_bytes()
    w, h = Image.open(io.BytesIO(png)).size
    old_cx = 5669280
    si.replace_part("word/media/image4.png", png)
    si.set_image_extent("image4.png", old_cx, round(old_cx * h / w))

    cap = "Supplementary Figure  5. WoM guided substrate selection"
    for old, new in caption_edits(summary):
        si.replace(old, new, context=cap)
    for old, new in METHODS_EDITS:
        si.replace(old, new)
    heading = [p for p in si.paragraphs() if si.accepted_text(p).strip() == "Prompt 3: Growth enhancement"]
    heading = [p for p in heading if si.accepted_text(p.getnext()).strip() == "Prompt 3 - Initial user input"]
    assert len(heading) == 1
    body = si.find("We want to dive deeper into the substrate utilization of Acinetobacter")
    body_rpr = si.text_runs(body)[0][0].find(q("rPr"))
    si.insert_paragraph_after(heading[0], [(body_rpr, prompt3_note())], ppr_from=body)

    # letter: R3 minor 5
    from docx_track import _Package
    orig = _Package(ORIGINAL_SI)
    orig_caption = orig.accepted_text(orig.find("Supplementary Figure  5. WoM guided substrate selection")).strip()
    orig_methods = orig.accepted_text(orig.find("was cultured in a defined medium containing malic acid")).strip()
    new_caption = si.accepted_text(si.find(cap)).strip()
    new_methods = si.accepted_text(si.find("was cultured in a defined medium containing succinic acid")).strip()
    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    letter.set_response("Minor comment 5. The supplementary protocol", (
        "We thank the reviewer for this careful reading; the placeholder and the description of the experimental unit "
        "were both incomplete. In addition, because the corrected model (see Major comment 1) ranks succinate rather "
        "than malate first among the NLDM metabolites for Acinetobacter, Fig. S5 now shows the growth experiment with "
        "succinic acid, together with lower-ranked substrates (arginine, N-acetyl-glutamate, and xylose), NLDM, and "
        "water. This experiment was performed three times independently (12, 16, and 20 March 2026), with three "
        "replicate wells per condition in each experiment. The Supplementary Methods now distinguish the independent "
        "experiments from the replicate wells, give the cultivation time of each run (47, 21, and 51 h), and state how "
        "the curves and error bands were computed: replicate wells were averaged within each experiment, the curves "
        "show the mean of the three experiments, and the shaded bands show ± 1 s.d. across experiments. Because the "
        "second experiment ended after 21 h, the figure shows 0–21 h, so that every time point includes all three "
        "experiments. The Prompt 3 transcript, which was generated with the submitted model, is unchanged, but it now "
        "carries a note giving the revised-model scores of the tested substrates. Changes were made to the "
        "Supplementary Methods (Growth Enhancement - Prompt 3), the Fig. S5 caption and its data sheet, and the "
        "Prompt 3 section of the Supplementary Information."))
    resp = letter.response_paragraph_after("Minor comment 5. The supplementary protocol")
    stale = [resp.getnext(), resp.getnext().getnext()]
    assert letter.accepted_text(stale[0]) == "Modified:" and "cultivation for 47, 47, 92 and 92 h" in letter.accepted_text(stale[1])
    for x in stale:
        x.getparent().remove(x)
    letter.add_pairs(resp, [
        ("Original (Supplementary Information, Methods, Growth Enhancement - Prompt 3):", orig_methods),
        ("Modified:", new_methods),
        ("Original (Supplementary Figure 5 caption):", orig_caption),
        ("Modified:", new_caption),
        ("Modified (Supplementary Information, Prompt 3): (no original text; new note immediately after the heading "
         "\"Prompt 3: Growth enhancement\")", prompt3_note()),
    ])

    if apply:
        for doc in (si, letter):
            print("backup:", doc.backup(BACKUPS))
            doc.save()
        write_workbook(TABLES, backup=True)
        print("applied to", DOCS)
    else:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for doc in (si, letter):
            doc.save(preview_dir / doc.path.name)
        write_workbook(preview_dir / TABLES.name, backup=False)
        print("preview written to", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/s5_preview"))
    args = parser.parse_args()
    main(args.apply, args.preview_dir)
