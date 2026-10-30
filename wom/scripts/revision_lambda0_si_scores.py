# revision_lambda0_si_scores.py
"""Recompute BPR scores and ranks quoted in the mSystems SI with the lambda_ord = 0 model.

The quoted values were produced by the WoM agent (MCP) from the submitted
Ordinal-BPR model (lambda_ord = 0.1). They are genus-mean scores over the
genus's training isolates, with the sample SD (ddof = 1) across isolates and
mean pairwise cosine distance between isolate embeddings as "embedding
diversity". This script recomputes each quoted value with the archived
submitted model (to confirm the method reproduces the SI) and with the new
model. No LLM/agent prompt is re-run.

Sources (mSystems SI): Prompt 1 table (Pseudomonas, 23 substrates), Prompt 2
table (gut genera, top-3 substrates), Prompt 3 table and Supplementary Fig. 5
caption (Acinetobacter), and the "bpr rank" column of
benchmarking/output/Supplementary Tables - GeminiRankingPseudomonas.tsv.

Usage:
    python scripts/revision_lambda0_si_scores.py --output-dir <dir>
"""

import sys
import logging
import argparse
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.io import load_model_artifacts
from loaders.compound_mapper import map_compound_names_batch

logger = logging.getLogger(__name__)

# (table, genus, substrate as quoted, quoted score, quoted SD or None)
PROMPT1 = [("Serine", 10.13, 2.06), ("Glutamate", 9.75, 2.15), ("Guanine", 9.56, 2.15),
           ("Adenosine", 9.56, 2.88), ("Leucine", 9.53, 1.84), ("Arginine", 9.16, 2.22),
           ("Isoleucine", 9.01, 2.02), ("Lysine", 8.82, 2.05), ("Glycine", 8.43, 2.19),
           ("Hypoxanthine", 8.35, 2.62), ("Alanine", 7.82, 1.63), ("Uridine", 7.77, 2.42),
           ("Threonine", 7.44, 1.66), ("Phenylalanine", 7.40, 2.06), ("Proline", 7.35, 1.67),
           ("Glucose", 7.07, 2.31), ("Mannose", 7.07, 2.31), ("Trehalose", 3.54, 2.41),
           ("Mannitol", 3.47, 2.39), ("Arabinose", 2.74, 2.35),
           ("4-guanidinobutanoic acid", -1.42, 1.33), ("Maltose", -2.37, 2.03)]
PROMPT2 = {
    "Bacteroides": [("adenosine", 7.85, 2.69), ("3-(carbamoylamino)propanoate", 7.24, 1.99),
                    ("uridine", 7.23, 2.65)],
    "Blautia": [("S-methyl-5'-thioadenosine", 9.08, 0.73), ("choline", 6.91, 0.71),
                ("S-adenosyl-L-homocysteine", 6.81, 0.97)],
    "Clostridium": [("S-methyl-5'-thioadenosine", 8.09, 1.61), ("nicotinamide", 6.73, 2.42),
                    ("S-adenosyl-L-homocysteine", 6.34, 1.50)],
    "Collinsella": [("arginine", 7.65, 1.10), ("S-methyl-5'-thioadenosine", 7.09, 1.83),
                    ("N-benzoylglycine", 6.36, 2.55)],
    "Eggerthella": [("nicotinamide", 7.51, None), ("arginine", 7.41, None),
                    ("(9Z)-octadecenoate", 6.54, None)],
    "Eubacterium": [("S-methyl-5'-thioadenosine", 9.15, 2.52), ("S-adenosyl-L-homocysteine", 7.07, 2.41),
                    ("nicotinamide", 6.19, 3.74)],
    "Faecalibacterium": [("S-methyl-5'-thioadenosine", 8.35, 1.12), ("adenosine", 7.60, 3.05),
                         ("uridine", 6.19, 3.25)],
    "Prevotella": [("S-methyl-5'-thioadenosine", 9.56, 2.80), ("S-adenosyl-L-homocysteine", 7.96, 2.75),
                   ("3',5'-cyclic AMP", 6.25, 0.88)],
}
PROMPT2_DIVERSITY = {"Bacteroides": 0.42, "Blautia": 0.40, "Clostridium": 0.52, "Collinsella": 0.45,
                     "Eubacterium": 0.47, "Faecalibacterium": 0.48, "Prevotella": 0.47}
PROMPT3 = [("Malate", 13.06), ("Succinate", 10.98), ("Pyruvate", 9.44), ("D-glucose", 4.62),
           ("L-ectoine", -0.89), ("Spermidine", -3.76)]
GEMINI_TSV = WOM_DIR.parent / "benchmarking" / "output" / "Supplementary Tables - GeminiRankingPseudomonas.tsv"


class GenusScorer:
    """Genus-mean BPR scores, isolate SD and embedding diversity for one saved model."""

    def __init__(self, model_dir: Path):
        artifacts = load_model_artifacts(model_dir)
        self.td = artifacts.training_data
        self.E = artifacts.model.microbe_embeddings
        self.V = artifacts.model.metabolite_embeddings
        self.idx_to_metabolite = {i: m for m, i in self.td.metabolite_to_idx.items()}

    def isolates(self, genus: str) -> list[int]:
        return [self.td.microbe_to_idx[m] for m, t in self.td.microbe_taxonomy.items()
                if t.genus == genus and m in self.td.microbe_to_idx]

    def scores(self, genus: str) -> np.ndarray:
        """(n_isolates, n_metabolites) isolate scores."""
        return self.E[self.isolates(genus)] @ self.V.T

    def compound(self, genus: str, metabolite_idx: int) -> tuple[float, float | None, int]:
        s = self.scores(genus)[:, metabolite_idx]
        return float(s.mean()), (float(s.std(ddof=1)) if len(s) > 1 else None), len(s)

    def diversity(self, genus: str) -> float | None:
        E = self.E[self.isolates(genus)]
        if len(E) < 2:
            return None
        En = E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)
        sims = En @ En.T
        return float(1.0 - sims[np.triu_indices(len(E), k=1)].mean())

    def top(self, genus: str, k: int = 3) -> list[str]:
        mean = self.scores(genus).mean(axis=0)
        return [self.idx_to_metabolite[i] for i in np.argsort(-mean)[:k]]


def resolve(names: list[str], metabolite_to_idx: dict) -> dict:
    """Quoted name -> metabolite index, via the same compound mapper the agent uses."""
    mapping = map_compound_names_batch(names)
    out = {}
    for name in names:
        rep = mapping[name].representative_name if name in mapping else name
        if rep not in metabolite_to_idx:
            raise KeyError(f"{name!r} -> {rep!r} is not a model metabolite")
        out[name] = (rep, metabolite_to_idx[rep])
    return out


def score_rows(old: GenusScorer, new: GenusScorer) -> pd.DataFrame:
    entries = [("Prompt 1", "Pseudomonas", n, v, sd) for n, v, sd in PROMPT1]
    entries += [("Prompt 2", g, n, v, sd) for g, items in PROMPT2.items() for n, v, sd in items]
    entries += [("Prompt 3 / Fig. S5", "Acinetobacter", n, v, None) for n, v in PROMPT3]
    names = sorted({e[2] for e in entries})
    resolved = resolve(names, old.td.metabolite_to_idx)
    assert old.td.metabolite_to_idx == new.td.metabolite_to_idx

    rows = []
    for table, genus, name, quoted, quoted_sd in entries:
        rep, j = resolved[name]
        old_mean, old_sd, k = old.compound(genus, j)
        new_mean, new_sd, _ = new.compound(genus, j)
        rows.append({
            "table": table, "genus": genus, "substrate": name, "model_compound": rep,
            "n_isolates": k, "quoted_score": quoted, "quoted_sd": quoted_sd,
            "recomputed_old_score": round(old_mean, 4),
            "recomputed_old_sd": None if old_sd is None else round(old_sd, 4),
            "old_reproduces_quote": bool(abs(old_mean - quoted) <= 0.005 + 1e-9 and
                                         (quoted_sd is None or abs(old_sd - quoted_sd) <= 0.005 + 1e-9)),
            "new_score": round(new_mean, 4), "new_sd": None if new_sd is None else round(new_sd, 4),
            "delta_score": round(new_mean - old_mean, 4),
        })
    return pd.DataFrame(rows)


def prompt2_genus_rows(old: GenusScorer, new: GenusScorer) -> pd.DataFrame:
    """Embedding diversity and top-3 substrates (raw genus mean over all metabolites)."""
    rows = []
    for genus, items in PROMPT2.items():
        quoted_top = [n for n, _, _ in items]
        resolved = resolve(quoted_top, old.td.metabolite_to_idx)
        quoted_reps = [resolved[n][0] for n in quoted_top]
        old_div, new_div = old.diversity(genus), new.diversity(genus)
        rows.append({
            "genus": genus, "n_isolates": len(old.isolates(genus)),
            "quoted_embedding_diversity": PROMPT2_DIVERSITY.get(genus),
            "recomputed_old_embedding_diversity": None if old_div is None else round(old_div, 4),
            "new_embedding_diversity": None if new_div is None else round(new_div, 4),
            "quoted_top3": "; ".join(quoted_reps),
            "old_model_top3": "; ".join(old.top(genus)),
            "new_model_top3": "; ".join(new.top(genus)),
            "old_top3_matches_quote": old.top(genus) == quoted_reps,
            "new_top3_same_as_quote": new.top(genus) == quoted_reps,
        })
    return pd.DataFrame(rows)


def gemini_rank_rows(new_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Recompute the 'bpr rank' column: competition rank among the depleted metabolites,
    on scores rounded to 2 decimals as in the TSV (Guanine and Adenosine, both 9.56, tie at 3)."""
    tsv = pd.read_csv(GEMINI_TSV, sep="\t")
    depleted = tsv[tsv["experimental_timepoint"].isin(["3", "6", "12"])].copy()
    p1 = new_scores[new_scores["table"] == "Prompt 1"].set_index("substrate")
    depleted["recomputed_old_score"] = depleted["metabolite"].map(p1["recomputed_old_score"])
    depleted["new_score"] = depleted["metabolite"].map(p1["new_score"])
    depleted["recomputed_old_rank"] = (depleted["recomputed_old_score"].round(2)
                                       .rank(ascending=False, method="min").astype(int))
    depleted["new_rank"] = depleted["new_score"].round(2).rank(ascending=False, method="min").astype(int)
    depleted["old_rank_reproduces_tsv"] = depleted["recomputed_old_rank"] == depleted["bpr rank"]
    t = depleted["experimental_timepoint"].astype(float)
    stats = {
        "n_metabolites": int(len(depleted)),
        "spearman_score_vs_depletion_time_old": spearmanr(depleted["recomputed_old_score"], t),
        "spearman_score_vs_depletion_time_new": spearmanr(depleted["new_score"], t),
        "spearman_old_vs_new_score": spearmanr(depleted["recomputed_old_score"], depleted["new_score"]),
    }
    stats = {k: (v if isinstance(v, int) else {"rho": float(v.statistic), "p": float(v.pvalue)})
             for k, v in stats.items()}
    cols = ["metabolite", "experimental_timepoint", "BPR score", "bpr rank", "recomputed_old_score",
            "recomputed_old_rank", "old_rank_reproduces_tsv", "new_score", "new_rank"]
    return depleted[cols].rename(columns={"BPR score": "tsv_bpr_score", "bpr rank": "tsv_bpr_rank"}), stats


def main(old_model: Path, new_model: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    old, new = GenusScorer(old_model), GenusScorer(new_model)

    scores = score_rows(old, new)
    scores.to_csv(output_dir / "si_bpr_scores_old_vs_new.csv", index=False)
    n_bad = int((~scores["old_reproduces_quote"]).sum())
    logger.info(f"SI scores: {len(scores)} rows; archived model reproduces "
                f"{len(scores) - n_bad}/{len(scores)} quoted values")

    genera = prompt2_genus_rows(old, new)
    genera.to_csv(output_dir / "si_prompt2_genus_old_vs_new.csv", index=False)

    ranks, rank_stats = gemini_rank_rows(scores)
    ranks.to_csv(output_dir / "si_gemini_pseudomonas_bpr_rank_old_vs_new.csv", index=False)
    pd.Series({k: str(v) for k, v in rank_stats.items()}).to_csv(
        output_dir / "si_gemini_pseudomonas_rank_stats.csv", header=["value"])
    logger.info(f"Gemini table ranks: archived model reproduces "
                f"{int(ranks['old_rank_reproduces_tsv'].sum())}/{len(ranks)}; {rank_stats}")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        logger.info("\n" + scores[["table", "genus", "substrate", "quoted_score", "recomputed_old_score",
                                   "old_reproduces_quote", "new_score", "new_sd"]].to_string(index=False))
        logger.info("\n" + genera.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Recompute SI BPR scores with the lambda_ord = 0 model")
    parser.add_argument("--old-model-dir", type=Path,
                        default=WOM_DIR / "trained_models" / "archive_submitted_lambda0.1" / "ordinal_bpr_consumed")
    parser.add_argument("--new-model-dir", type=Path,
                        default=WOM_DIR / "trained_models" / "ordinal_bpr_consumed")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.old_model_dir, args.new_model_dir, args.output_dir)
