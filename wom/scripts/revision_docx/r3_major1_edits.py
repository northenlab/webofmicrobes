# r3_major1_edits.py
"""Reviewer 3 major comment 1: tracked edits to the revised manuscript and SI, and
the response-letter entries (plus the R3 minor 3, R1-1 and R1-4 text it supersedes).

The adopted model (Ordinal-BPR code path with lambda_ord = 0) is called "BPR" in
the revised text. Numbers are computed from
manuscript_figures/msystems_revision/lambda0/fig1b/fig1b_benchmark_results.csv,
and the claims the text makes (BPR best at every dimension, most stable) are
checked against it. "Original" quotes are taken from the documents with all
tracked changes rejected; line numbers from the line-numbered original PDF.

Without --apply, edited copies go to --preview-dir. With --apply, the working
copies are backed up and edited in place (refused if Word has one open).

Usage:
    python r3_major1_edits.py --preview-dir /tmp/r3_preview
    python r3_major1_edits.py --apply
"""

import re
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from docx_track import TrackedDocx, LetterDocx  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "wom/manuscript_figures/msystems_revision/lambda0/fig1b/fig1b_benchmark_results.csv"
DOCS = Path.home() / "Downloads/1st-revision/revised version"
MS = DOCS / "Northen_msystems_BPR.docx"
SI = DOCS / "Northen_msystems_BPR_Supplementary Information.docx"
LETTER = DOCS / "Response_to_reviewers (1).docx"
LINE_PDF = Path.home() / ("Downloads/1st-revision/original_submission/"
                          "Original Maintext Submission ONLY FOR LINE NUMBERS.pdf")
BACKUPS = DOCS.parent / "backups"
AUTHOR = "Benjamin Bowen"
BPR = "Ordinal-BPR [lambda_ord=0]"  # results-CSV name of the adopted model
COMPARED = [BPR, "WR-MF", "SVD", "LMF"]


# ----------------------------------------------------------------------------- numbers
def benchmark_numbers(csv: Path) -> dict:
    r = pd.read_csv(csv)
    hold = r[r.protocol == "B_holdout"]
    loo = r[r.protocol == "A_loo"]
    per_dim = hold.groupby(["model", "dim"]).roc_auc.mean().unstack("model")
    loo_neutral = loo[loo.pool == "neutral"].groupby(["model", "dim"]).roc_auc.mean().unstack("model")

    # claims made in the text
    assert (per_dim[COMPARED].idxmax(axis=1) == BPR).all(), "BPR is not best at every dim (hold-out)"
    assert (loo_neutral[COMPARED].idxmax(axis=1) == BPR).all(), "BPR is not best at every dim (LOO neutral)"
    ranges = per_dim[COMPARED].max() - per_dim[COMPARED].min()
    assert ranges.idxmin() == BPR, "BPR is not the most stable model"

    def at32(df, model, col="roc_auc"):
        s = df[(df.model == model) & (df.dim == 32)][col]
        return s.mean(), s.std()

    n = {"bpr32": at32(hold, BPR)[0], "bpr32_sd": at32(hold, BPR)[1],
         "bpr_lo": per_dim[BPR].min(), "bpr_hi": per_dim[BPR].max(),
         "ord32": at32(hold, "Ordinal-BPR")[0], "plain32": at32(hold, "BPR")[0]}
    for key, model in [("wrmf", "WR-MF"), ("svd", "SVD"), ("lmf", "LMF")]:
        n[f"{key}4"], n[f"{key}128"] = per_dim[model][4], per_dim[model][128]
    for pool in ["neutral", "untested", "mixed"]:
        sub = loo[loo.pool == pool]
        for key, model in [("bpr", BPR), ("ord", "Ordinal-BPR"), ("plain", "BPR")]:
            n[f"loo_{pool}_{key}"] = at32(sub, model)[0]
    n["ord128_neutral"] = loo_neutral["Ordinal-BPR"][128]
    pools = loo[(loo.model == BPR) & (loo.dim == 32) & (loo.repeat == 0)].set_index("pool")
    n["n_neutral"] = int(pools.loc["neutral", "n_neg"])
    n["n_untested"] = int(pools.loc["untested", "n_neg"])
    n["n_secreted"] = int(pools.loc["mixed", "n_neg"]) - n["n_neutral"] - n["n_untested"]
    n["n_isolates"] = int(hold.n_microbes.iloc[0])
    n["n_loo_isolates"] = int(pools.loc["neutral", "n_microbes"])
    return n


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


# ----------------------------------------------------------------------------- edits
def manuscript_edits(n: dict) -> list:
    """(old, new, context) tracked replacements in the revised manuscript."""
    abstract = "Here, we evaluated several approaches for inferring bacterial substrate preferences"
    bench = "To benchmark model performance"
    return [
        ("was uniquely robust to embedding dimensionality",
         "was the most accurate of four ranking methods and the most robust to embedding dimensionality",
         abstract),
        ("ROC-AUC = 0.94)", f"ROC-AUC = {f2(n['bpr32'])} for withheld isolate–metabolite pairs)", abstract),
        # leftover comma from the R3 minor 1 edit ("continuous, calibrated" -> "continuous preference")
        ("continuous, preference (ranking) score", "continuous preference (ranking) score", abstract),
        ("Ordinal Bayesian Personalized Ranking (Ordinal-BPR)", "Bayesian Personalized Ranking (BPR)", None),
        ("Ordinal-BPR differs from standard BPR in that separate models are trained for depleted and "
         "produced metabolites and, within each, tested-but-inactive metabolites are explicitly "
         "distinguished from untested ones to extract more signal from the incomplete observation matrix.",
         "In our implementation of BPR, separate models are trained for depleted and produced metabolites, "
         "and half of the negative examples are drawn from metabolites that were measured but not "
         "depleted, so that measured non-consumption is used explicitly rather than being treated like an "
         "untested pair.", None),
        ("corresponding vectors implementation details", "corresponding vectors; implementation details", None),
        ("we used a repeated leave-one-out procedure across five independent trials",
         "we used a hold-out procedure repeated five times with different random seeds", bench),
        ("For each trial, one consumed substrate was withheld from training for every organism with at least "
         "two known preferences, and we computed the probability that the held-out substrate received a "
         "higher predicted rank than a randomly selected neutral substrate (AUC).",
         "In each repeat, 20% of every isolate's consumed metabolites and 20% of its measured-neutral "
         "metabolites were withheld from training for all models, and for each isolate we computed the "
         "probability that a withheld consumed metabolite received a higher score than a withheld neutral "
         "metabolite (the area under the receiver operating characteristic curve, ROC-AUC; throughout, AUC "
         "refers to ROC-AUC). Secreted metabolites were not used as negatives.", bench),
        ("this compound-level procedure withholds each compound only for a single organism at a time",
         "this procedure withholds each metabolite only for the isolate in question", bench),
        ("All four algorithms performed competitively at low embedding dimension, but SVD, WR-MF, and LMF "
         "declined at higher dimensions due to overfitting, whereas Ordinal-BPR maintained stable predictive "
         "power across the full range (AUC = 0.94).",
         f"BPR was the most accurate method at every embedding dimension and the most stable across "
         f"dimensions (ROC-AUC = {f2(n['bpr_lo'])}–{f2(n['bpr_hi'])} from 4 to 128 dimensions), whereas "
         f"WR-MF, SVD, and LMF declined at higher dimensions, consistent with overfitting (WR-MF from "
         f"{f2(n['wrmf4'])} at 4 dimensions to {f2(n['wrmf128'])} at 128, SVD from {f2(n['svd4'])} to "
         f"{f2(n['svd128'])}, and LMF from {f2(n['lmf4'])} to {f2(n['lmf128'])}). A leave-one-out comparison "
         f"of each withheld consumed metabolite with the isolate's measured-neutral metabolites also "
         f"identified BPR as the most accurate method (Supplementary Methods).", bench),
        ("we selected Ordinal-BPR with 32-dimensional embeddings",
         f"we selected BPR with 32-dimensional embeddings (ROC-AUC = {f3(n['bpr32'])} ± {f3(n['bpr32_sd'])})",
         bench),
        ("Ordinal-BPR maintained stable predictive power across the full range (AUC = 0.94, Fig. 1b) which is "
         "comparable for performance observed for Netflix data",
         f"BPR maintained stable predictive power across all embedding dimensions tested (ROC-AUC = "
         f"{f2(n['bpr32'])} for withheld isolate–metabolite pairs at 32 dimensions, Fig. 1b), comparable to "
         f"performance reported for Netflix data", None),
        ("For Ordinal-BPR, training samples", "For BPR, training samples", None),
        ("When j is a tested-but-neutral metabolite, a second term weighted by λ = 0.1, −log σ(suj − suk) "
         "with k an untested metabolite, ranks tested-but-neutral metabolites above untested ones, giving "
         "the ordering consumed > tested-but-neutral > untested.",
         "Measured-neutral metabolites therefore enter training only as negatives; adding a second term that "
         "ranks measured-neutral above untested metabolites reduced accuracy (Supplementary Methods), so no "
         "such term is used.", None),
        ("Leave-one-out cross-validation AUC for four collaborative-filtering models across six embedding "
         "dimensions, evaluated by a common leave-one-out protocol (Methods). Ordinal-BPR remains stable "
         "across dimensions (AUC = 0.94), whereas WR-MF, SVD, and LMF decline at higher dimensions; this "
         "panel establishes dimension-robustness, not a head-to-head accuracy claim. Error bars, +/-1 s.d. "
         "(five replicates).",
         "Hold-out ROC-AUC for four collaborative-filtering models across six embedding dimensions. In each "
         "of five repeats, 20% of every isolate's consumed and 20% of its neutral metabolites were withheld "
         "from training; ROC-AUC compares withheld consumed (assertion value ≤ −2) with withheld neutral "
         "(−2 < assertion value < +2) metabolites for each isolate and is averaged over isolates, and "
         "secreted metabolites (assertion value ≥ +2) are not used as negatives (Supplementary Methods). BPR "
         "is the most accurate model at every dimension and the most stable across dimensions, whereas "
         "WR-MF, SVD, and LMF decline at higher dimensions. Error bars, ±1 s.d. across five repeats.", None),
    ]


def si_edits(n: dict) -> list:
    model_par = "extends the standard BPR pairwise ranking objective"
    return [
        ("Ordinal-BPR scores discriminate", "BPR scores discriminate", None),
        ("Ordinal Bayesian Personalized Ranking (BPR) extends the standard BPR pairwise ranking objective to "
         "exploit the three-way ordinal structure of exometabolite data:",
         "Bayesian Personalized Ranking (BPR), as implemented here, uses the three states of the "
         "exometabolite data:", None),
        ("Training minimizes a two-term loss. The primary BPR term requires",
         "Training minimizes the BPR loss, which requires", "uses the three states of the exometabolite data"),
        ("The secondary ordinal term, weighted by λ_ord = 0.1, additionally requires that "
         "tested-but-not-consumed metabolites score higher than unobserved metabolites: "
         "−log σ(score(tested-not-consumed) − score(unobserved)). This term is only applied to triplets in "
         "which the primary negative was drawn from the tested-not-consumed set. The combined loss enforces "
         "a consistent preference ordering: consumed > tested-not-consumed > unobserved.",
         f"Adding a second term that requires tested-but-not-consumed metabolites to score above unobserved "
         f"ones (weight λ_ord = 0.1) lowered accuracy in the benchmark described below (hold-out ROC-AUC = "
         f"{f2(n['ord32'])} versus {f2(n['bpr32'])} at 32 dimensions), so the final model uses the BPR loss "
         f"alone (λ_ord = 0).", "uses the three states of the exometabolite data"),
        (SI_BENCHMARK_OLD, si_benchmark_new(n), None),
        ("Based on the leave-one-out cross-validation results (Fig 2)", "Based on the hold-out results (Fig. 1b)",
         None),
    ]


SI_BENCHMARK_OLD = (
    "To benchmark predictive performance (Fig. 1), we evaluated four matrix factorization approaches using a "
    "leave-one-out cross-validation scheme applied to the exometabolite consumption dataset. These models "
    "include: Ordinal Bayesian Personalized Ranking (Ordinal-BPR), Weighted Regularized Matrix Factorization "
    "(WR-MF), Singular Value Decomposition (SVD), and Logistic Matrix Factorization (LMF) For each microbe "
    "with two or more experimentally observed consumed compounds, one compound was randomly withheld from "
    "training; microbes with a single positive observation were retained in training only. Each model was "
    "then trained on the remaining observations and asked to rank all compounds for each microbe. Predictive "
    "performance was quantified per held-out pair as the probability that the model ranked the withheld "
    "compound above a randomly chosen unobserved compound (i.e., a pairwise AUC), excluding known training "
    "positives from the negative set. Mean AUC was averaged across all held-out pairs. The procedure was "
    "repeated five times with different random seeds, and mean AUC ± standard deviation across repeats is "
    "reported. Models were evaluated across embedding dimensions of 4, 8, 16, 32, 64, and 128 latent factors.")


def si_benchmark_new(n: dict) -> str:
    return (
        "To benchmark predictive performance (Fig. 1b), we compared four matrix factorization approaches on "
        "the exometabolite consumption dataset: BPR, Weighted Regularized Matrix Factorization (WR-MF), "
        "Singular Value Decomposition (SVD), and Logistic Matrix Factorization (LMF). Isolate–metabolite "
        "pairs were labelled as described in the Methods: consumed (assertion value ≤ −2), neutral "
        "(−2 < assertion value < +2), secreted (≥ +2), or untested. In each of five repeats, 20% of every "
        "isolate's consumed pairs and 20% of its neutral pairs (rounded, and at least one of each when the "
        "isolate had two or more) were withheld from training for every model; for BPR, the withheld neutral "
        "pairs were also removed from the measured negatives, so that they were untested during training. "
        "Each model was trained on the remaining pairs, and ROC-AUC was computed for each isolate from its "
        f"withheld consumed and withheld neutral pairs and averaged over isolates ({n['n_isolates']} "
        "isolates); secreted pairs were never used as negatives. Models were evaluated at embedding "
        "dimensions of 4, 8, 16, 32, 64, and 128 latent factors, all models used the same random seed within "
        "a repeat and dimension, and the mean ± standard deviation across repeats is reported. As a second "
        "evaluation, we used leave-one-out validation, withholding one consumed compound for each isolate "
        f"with at least two consumed compounds ({n['n_loo_isolates']} isolates), and computed the probability "
        "that the withheld compound was ranked above each compound in three candidate sets: the isolate's "
        "measured-neutral compounds, its untested compounds, or all compounds other than its training "
        f"positives (neutral, secreted, and untested). At 32 dimensions BPR reached ROC-AUC = "
        f"{f2(n['loo_neutral_bpr'])}, {f2(n['loo_untested_bpr'])}, and {f2(n['loo_mixed_bpr'])} against these "
        "three sets, and it was the most accurate of the four methods against measured-neutral compounds at "
        "every dimension.")


# ----------------------------------------------------------------------------- letter
def r3_major1_response(n: dict) -> str:
    return (
        "We thank the reviewer for this careful reading of the benchmark and its code. Both concerns were "
        "correct, and addressing them changed the model used throughout the manuscript. First, the run that "
        "produced the submitted Fig. 1b rebuilt the training data for each leave-one-out split without the "
        "measured-neutral compounds, so the model labelled Ordinal-BPR received no measured negatives: its "
        "ordinal modification was never active, and the curve shown was effectively standard BPR. Second, as "
        "the reviewer noted, compute_loo_auc compared each withheld consumed compound with every compound "
        f"that was not a training positive, which mixed {n['n_neutral']:,} measured-neutral, "
        f"{n['n_secreted']:,} secreted, and {n['n_untested']:,} untested candidates across the "
        f"{n['n_loo_isolates']} evaluated isolates, and so did not correspond to the comparison described in "
        "the Results. When we passed the measured-neutral compounds into training, the ordinal model "
        "performed worse rather than better: against measured-neutral compounds it reached ROC-AUC = "
        f"{f2(n['loo_neutral_ord'])} at 32 dimensions, below standard BPR ({f2(n['loo_neutral_plain'])}), and "
        f"it fell to {f2(n['ord128_neutral'])} at 128 dimensions. The ordinal term forces untested compounds "
        "below measured-neutral ones, and a withheld compound is untested during training, so the term "
        "penalizes the very predictions being evaluated. Using measured-neutral compounds as additional "
        "negatives without the ordinal term gave the most accurate and most stable model "
        f"({f2(n['loo_neutral_bpr'])}). We adopted this model, standard BPR in which half of the negative "
        "examples are drawn from the isolate's measured-neutral compounds, and refer to it simply as BPR "
        "throughout the revised manuscript. We also replaced the benchmark in Fig. 1b with a strict hold-out "
        "that matches the task described in the Results: in each of five repeats, 20% of every isolate's "
        "consumed pairs and 20% of its measured-neutral pairs were withheld from training for all models, "
        "ROC-AUC was computed for each isolate on the withheld consumed versus withheld neutral pairs, and "
        "secreted compounds were never used as negatives. BPR was the most accurate model at every embedding "
        f"dimension and the most stable (ROC-AUC = {f2(n['bpr_lo'])}–{f2(n['bpr_hi'])} from 4 to 128 "
        f"dimensions; {f3(n['bpr32'])} ± {f3(n['bpr32_sd'])} at 32 dimensions), whereas WR-MF fell from "
        f"{f2(n['wrmf4'])} to {f2(n['wrmf128'])}, SVD from {f2(n['svd4'])} to {f2(n['svd128'])}, and LMF "
        f"from {f2(n['lmf4'])} to {f2(n['lmf128'])}. The leave-one-out results for the three candidate sets "
        "(measured-neutral, untested, and all non-positive compounds) are reported separately in the "
        "Supplementary Methods. We retrained the final model with this objective and regenerated all "
        "model-derived results (Fig. 1b, Fig. 2, Fig. 3, Supplementary Figs. 2 and 3, SI Table 2, and the "
        "BPR scores quoted in the Supplementary Information); updated values are given in the revised text. "
        "Code that reproduces the submitted benchmark, demonstrates both problems, and runs the corrected "
        "benchmark is available in the project repository. Changes were made to the Abstract, the Results "
        "section \"Bayesian effectively learns bacterial substrate preferences\", the Discussion, the Methods "
        "section \"Model training\" (see Reviewer 1 Comment 4), the Fig. 1 caption, and the Supplementary "
        "Methods.")


class LineNumbers:
    """Locate quotes in the line-numbered original PDF (whitespace-insensitive)."""

    def __init__(self, pdf: Path):
        import fitz
        chars, line_of = [], []
        for page in fitz.open(pdf):
            words = page.get_text("words")
            margin = [w for w in words if re.fullmatch(r"\d{1,4}", w[4]) and w[0] < 70]
            body = sorted((w for w in words if w not in margin), key=lambda w: (round((w[1] + w[3]) / 2), w[0]))
            for w in body:
                y = (w[1] + w[3]) / 2
                near = [m for m in margin if abs((m[1] + m[3]) / 2 - y) < 4]
                num = int(near[0][4]) if near else None
                for c in self._norm(w[4]):
                    chars.append(c)
                    line_of.append(num)
        self.text, self.line_of = "".join(chars), line_of

    @staticmethod
    def _norm(s: str) -> str:
        s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return re.sub(r"\s+", "", s)

    def lines(self, quote: str) -> str:
        key = self._norm(quote)
        i = self.text.find(key)
        if i < 0:
            return "__"
        nums = [x for x in self.line_of[i:i + len(key)] if x is not None]
        lo, hi = min(nums), max(nums)
        return f"{lo}" if lo == hi else f"{lo}–{hi}"


# full stops that do not end a sentence (abbreviations used in the manuscript)
_NOT_END = re.compile(r"(?:\bFigs?|\be\.g|\bi\.e|\bet al|\bvs|\bs\.d|\bSupp)\.$")


def _sentence_ends(text: str) -> list:
    return [m.end() for m in re.finditer(r"\.(?=\s+[A-Z(]|\s*$)", text)
            if not _NOT_END.search(text[:m.end()])]


def sentence_containing(text: str, fragment: str) -> str:
    """The sentence(s) of `text` spanning `fragment`."""
    i = text.index(fragment)
    ends = _sentence_ends(text)
    start = max([e for e in ends if e <= i], default=0)
    end = min([e for e in ends if e >= i + len(fragment)], default=len(text))
    return text[start:end].strip()



# ----------------------------------------------------------------------------- figure 1 and SI tables
FIG1B_PANEL = REPO / "wom/manuscript_figures/msystems_revision/lambda0/fig1b/fig1b_main.png"
TABLES = DOCS / "Msystems BPR Supplementary Tables.xlsx"
PANEL_B_BOX = (1240, 100)  # old Figure 1 image: panel b lies right of x and below y


def figure1_composite(old_png: bytes, panel_png: Path) -> bytes:
    """Keep panel a (and the "b" label above y=100) of the old Figure 1 image and fit
    the new panel b into the bounding box of the old one, at the same image size."""
    import io
    import numpy as np
    from PIL import Image
    old = Image.open(io.BytesIO(old_png)).convert("RGBA")
    white = Image.alpha_composite(Image.new("RGBA", old.size, (255, 255, 255, 255)), old)
    x0, y0 = PANEL_B_BOX
    region = np.asarray(white.convert("L"))[y0:, x0:] < 245
    ys, xs = np.nonzero(region)
    left, top, right, bottom = x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1
    panel = Image.open(panel_png).convert("RGBA")
    scale = min((right - left) / panel.width, (bottom - top) / panel.height)
    panel = panel.resize((round(panel.width * scale), round(panel.height * scale)), Image.LANCZOS)
    out = old.copy()
    out.paste((255, 255, 255, 0), (x0, y0, old.width, old.height))
    px = left + ((right - left) - panel.width) // 2
    py = top + ((bottom - top) - panel.height) // 2
    out.alpha_composite(panel, (px, py))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _cell(ref: str, value, style: int) -> str:
    from xml.sax.saxutils import escape
    if isinstance(value, str):
        return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'
    if value is None:
        return f'<c r="{ref}" s="{style}"/>'
    return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'


def sheet_data_xml(rows: list) -> str:
    """<sheetData> with row 1 in the title style (1) and the rest in the cell style (2)."""
    cols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ["<sheetData>"]
    for i, row in enumerate(rows, start=1):
        cells = "".join(_cell(f"{cols[j]}{i}", v, 1 if i == 1 and j == 0 else 2) for j, v in enumerate(row))
        out.append(f'<row r="{i}">{cells}</row>')
    out.append("</sheetData>")
    return "".join(out)


def fig1b_sheet_rows(csv: Path) -> list:
    r = pd.read_csv(csv)
    names = {BPR: "BPR", "WR-MF": "WR-MF", "SVD": "SVD", "LMF": "LMF",
             "Ordinal-BPR": "BPR with ordinal term (lambda_ord = 0.1)"}
    r = r[r.model.isin(names)].assign(model=lambda d: d.model.map(names))
    order = {v: i for i, v in enumerate(names.values())}
    rows = [["Data for figure 1b: hold-out ROC-AUC for four collaborative-filtering models at six embedding "
             "dimensions (five repeats). In each repeat 20% of every isolate's consumed and 20% of its neutral "
             "metabolites were withheld from training; ROC-AUC compares withheld consumed with withheld neutral "
             "metabolites per isolate. BPR is the model used throughout the manuscript; the BPR variant with an "
             "additional ordinal term is reported for comparison (Supplementary Methods)."], [],
            ["model", "dim", "repeat", "roc_auc_per_isolate", "pr_auc_pooled", "prevalence", "n_isolates",
             "n_withheld_consumed", "n_withheld_neutral"]]
    hold = r[r.protocol == "B_holdout"].sort_values(["model", "dim", "repeat"], key=lambda c: c.map(order) if c.name == "model" else c)
    for _, x in hold.iterrows():
        rows.append([x["model"], int(x["dim"]), int(x["repeat"]), round(x["roc_auc"], 6),
                     round(x["pr_auc_pooled"], 6), round(x["prevalence"], 6), int(x["n_microbes"]),
                     int(x["n_pos"]), int(x["n_neg"])])
    rows += [[], ["Leave-one-out validation (Supplementary Methods): one consumed metabolite withheld per isolate; "
                  "ROC-AUC against three candidate sets"],
             ["model", "dim", "repeat", "candidate_set", "roc_auc", "n_isolates", "n_candidates"]]
    loo = r[r.protocol == "A_loo"].sort_values(["model", "dim", "repeat", "pool"], key=lambda c: c.map(order) if c.name == "model" else c)
    for _, x in loo.iterrows():
        rows.append([x["model"], int(x["dim"]), int(x["repeat"]),
                     {"neutral": "measured neutral", "untested": "untested", "mixed": "all non-positive"}[x["pool"]],
                     round(x["roc_auc"], 6), int(x["n_microbes"]), int(x["n_neg"])])
    return rows


def write_fig1b_sheet(csv: Path, out_path: Path, backup: bool):
    import re as _re
    import shutil
    import zipfile
    from datetime import datetime
    with zipfile.ZipFile(TABLES) as z:
        infos = z.infolist()
        parts = {i.filename: z.read(i.filename) for i in infos}
    sheet = parts["xl/worksheets/sheet1.xml"].decode("utf-8")
    new_sheet, n = _re.subn(r"<sheetData>.*</sheetData>", lambda m: sheet_data_xml(fig1b_sheet_rows(csv)),
                            sheet, flags=_re.S)
    assert n == 1, "sheetData not found in the Figure 1b sheet"
    parts["xl/worksheets/sheet1.xml"] = new_sheet.encode("utf-8")
    if backup:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TABLES, BACKUPS / f"{TABLES.stem}.before-{datetime.now():%Y%m%d-%H%M%S}{TABLES.suffix}")
    tmp = out_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, parts[info.filename])
    tmp.replace(out_path)

# ----------------------------------------------------------------------------- main
def main(apply: bool, preview_dir: Path, results: Path = RESULTS):
    locks = [p.name for p in DOCS.glob("~$*")]
    if apply and locks:
        raise SystemExit(f"Word lock files present ({locks}); close the documents in Word first")
    n = benchmark_numbers(results)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in n.items()})

    ms, si = TrackedDocx(MS, AUTHOR), TrackedDocx(SI, AUTHOR)
    # originals (all tracked changes rejected) before editing
    orig = {
        "abstract": sentence_containing(ms.rejected_text(ms.find("uniquely robust to embedding")), "uniquely robust"),
        "list": "and a pairwise ranking method, Ordinal Bayesian Personalized Ranking (Ordinal-BPR)(12).",
        "differs": ms.rejected_text(ms.find("Ordinal-BPR differs from standard BPR")).strip(),
        "bench": ms.rejected_text(ms.find("To benchmark model performance")).strip(),
        "disc": sentence_containing(ms.rejected_text(ms.find("comparable for performance observed for Netflix")),
                                    "Ordinal-BPR maintained stable"),
        "caption": "(b) " + ms.rejected_text(ms.find("Leave-one-out cross-validation AUC for four")).split("(b) ")[-1].strip()
        if "(b) " in ms.rejected_text(ms.find("Leave-one-out cross-validation AUC for four")) else
        ms.rejected_text(ms.find("Leave-one-out cross-validation AUC for four")),
        "si_model": si.rejected_text(si.find("extends the standard BPR pairwise ranking objective")).strip(),
        "si_bench": SI_BENCHMARK_OLD,
        "si_select": sentence_containing(si.rejected_text(si.find("Based on the leave-one-out cross-validation results")),
                                         "Based on the leave-one-out"),
        "s3_title": "Supplementary Figure 3. Ordinal-BPR scores discriminate consumed from neutral metabolites "
                    "in held-out bacterial isolates across diverse genera.",
    }
    for old, new, ctx in manuscript_edits(n):
        ms.replace(old, new, ctx)
    for old, new, ctx in si_edits(n):
        si.replace(old, new, ctx)
    new = {
        "abstract": sentence_containing(ms.accepted_text(ms.find("most accurate of four ranking methods")),
                                        "most accurate of four ranking methods"),
        "list": "and a pairwise ranking method, Bayesian Personalized Ranking (BPR)(12).",
        "differs": ms.accepted_text(ms.find("In our implementation of BPR")).strip(),
        "bench": ms.accepted_text(ms.find("To benchmark model performance")).strip(),
        "disc": sentence_containing(ms.accepted_text(ms.find("comparable to performance reported for Netflix")),
                                    "BPR maintained stable predictive power across all"),
        "caption": "(b) " + ms.accepted_text(ms.find("Hold-out ROC-AUC for four")).split("(b) ", 1)[1].strip(),
        "si_model": si.accepted_text(si.find("uses the three states of the exometabolite data")).strip(),
        "si_bench": si_benchmark_new(n),
        "si_select": sentence_containing(si.accepted_text(si.find("Based on the hold-out results")), "Based on the hold-out"),
        "s3_title": "Supplementary Figure 3. BPR scores discriminate consumed from neutral metabolites in "
                    "held-out bacterial isolates across diverse genera.",
        "methods": ms.accepted_text(ms.find("For BPR, training samples")).strip(),
    }

    lines = LineNumbers(LINE_PDF)
    L = lambda key: lines.lines(orig[key])
    letter = LetterDocx(LETTER, template_response="We thank the reviewer for noticing that \"calibrated\"")
    # R3 minor 3: Fig. 1b category definitions are now stated
    text = letter.response_text("Minor comment 3.")
    text = re.sub(r"⟨PENDING: the category definitions for Fig\. 1b[^⟩]*⟩",
                  "The category definitions for Fig. 1b are now stated in its caption and in the Supplementary "
                  "Methods (see Major comment 1).", text)
    text = text.replace("and the Fig. S3 caption.", "the Fig. 1 caption, and the Fig. S3 caption.")
    letter.set_response("Minor comment 3.", text)
    # R1-1 and R1-4 quote the passages rewritten above
    letter.set_text("uniquely robust to embedding dimensionality (area under the receiver operating characteristic "
                    "curve, ROC-AUC = 0.94)", new["abstract"][new["abstract"].index("found that"):])
    letter.set_text("than a randomly selected neutral substrate (AUC).",
                    sentence_containing(orig["bench"], "than a randomly selected neutral substrate (AUC)"))
    letter.set_text("than a randomly selected neutral substrate (the area under the receiver operating characteristic "
                    "curve, ROC-AUC; throughout, AUC refers to ROC-AUC).",
                    sentence_containing(new["bench"], "throughout, AUC refers to ROC-AUC"))
    letter.set_text("For Ordinal-BPR, training samples triplets", new["methods"])
    letter.replace_in_run("how the Ordinal-BPR objective uses the three states",
                          "how the BPR objective uses the three states")

    resp = letter.set_response("Major comment 1.", r3_major1_response(n))
    letter.add_pairs(resp, [
        (f"Original (Abstract, Lines {L('abstract')}):", orig["abstract"]), ("Modified:", new["abstract"]),
        (f"Original (Results, first section, Lines {L('list')}):", orig["list"]), ("Modified:", new["list"]),
        (f"Original (Results, first section, Lines {L('differs')}):", orig["differs"]), ("Modified:", new["differs"]),
        (f"Original (Results, first section, Lines {L('bench')}):", orig["bench"]), ("Modified:", new["bench"]),
        (f"Original (Discussion, first paragraph, Lines {L('disc')}):", orig["disc"]), ("Modified:", new["disc"]),
        (f"Original (Figure 1 caption, Lines {L('caption')}):", orig["caption"]), ("Modified:", new["caption"]),
        ("Original (Supplementary Information, Model training):", orig["si_model"]), ("Modified:", new["si_model"]),
        ("Original (Supplementary Information, Model benchmarking):", orig["si_bench"]), ("Modified:", new["si_bench"]),
        ("Original (Supplementary Information, Model benchmarking):", orig["si_select"]), ("Modified:", new["si_select"]),
        ("Original (Supplementary Figure 3 caption):", orig["s3_title"]), ("Modified:", new["s3_title"]),
    ])
    # Figure 1: new panel b in the embedded image (an image swap is not tracked by Word)
    ms.replace_part("word/media/image1.png", figure1_composite(ms.read_part("word/media/image1.png"), FIG1B_PANEL))

    if apply:
        for doc in (ms, si, letter):
            print("backup:", doc.backup(BACKUPS))
            doc.save()
        write_fig1b_sheet(results, TABLES, backup=True)
        print("applied to", DOCS)
    else:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for doc in (ms, si, letter):
            doc.save(preview_dir / doc.path.name)
        write_fig1b_sheet(results, preview_dir / TABLES.name, backup=False)
        print("preview written to", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--preview-dir", type=Path, default=Path("/tmp/r3_major1_preview"))
    parser.add_argument("--results", type=Path, default=RESULTS, help="benchmark results CSV")
    args = parser.parse_args()
    main(args.apply, args.preview_dir, args.results)
