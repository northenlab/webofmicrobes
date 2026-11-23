# revision_r1_metrics.py
"""Metrics shared by the Reviewer 1 analyses (mSystems revision).

Per-isolate ROC-AUC and PR-AUC (average precision), learning-free baselines, and the
observed fraction of consumed pairs in ranges of BPR score (Reviewer 1, comment 3).
BPR scores are ranking statistics; the model does not output a probability of
consumption, so the score itself is binned. For the response letter only, the slope of
the unbinned labels on the logistic function of the score is also computed.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

SCORE_EDGES = [-6, -4, -2, 0, 2, 4, 6, 8]  # ranges of two score units; open-ended below -6 and from 8


def per_isolate(y: np.ndarray, s: np.ndarray, isolate: np.ndarray) -> pd.DataFrame:
    """ROC-AUC, PR-AUC and prevalence for every isolate with both classes present."""
    rows = []
    for m in pd.unique(isolate):
        sel = isolate == m
        ym, sm = y[sel], s[sel]
        if 0 < ym.sum() < len(ym):
            rows.append({"isolate": m, "n_pos": int(ym.sum()), "n_neg": int(len(ym) - ym.sum()),
                         "prevalence": float(ym.mean()), "roc_auc": float(roc_auc_score(ym, sm)),
                         "pr_auc": float(average_precision_score(ym, sm))})
    return pd.DataFrame(rows)


def predicted_values(s: np.ndarray) -> np.ndarray:
    """The logistic function of the BPR score: the score read as if it were a log-odds of consumption."""
    return expit(s)


def unbinned_slope(y: np.ndarray, p: np.ndarray) -> dict:
    """Least-squares slope and intercept of the unbinned 0/1 label on the predicted value
    (1 and 0 for a calibrated model)."""
    slope, intercept = np.polyfit(p, y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "prevalence": float(y.mean()),
            "n_pairs": int(len(y))}


def score_ranges(y: np.ndarray, s: np.ndarray, edges: list = SCORE_EDGES) -> pd.DataFrame:
    """Observed fraction consumed among the pairs whose score lies in each range [low, high),
    with a Wilson 95% interval. The first and last ranges are open-ended."""
    bounds = [-np.inf, *edges, np.inf]
    rows = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        sel = (s >= low) & (s < high)
        n, obs, z = int(sel.sum()), float(y[sel].mean()), 1.96
        centre = (obs + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        label = f"< {high:g}" if np.isinf(low) else f"≥ {low:g}" if np.isinf(high) else f"{low:g} to {high:g}"
        rows.append({"score_range": label.replace("-", "−"), "score_low": low, "score_high": high, "n": n,
                     "n_consumed": int(y[sel].sum()), "observed": obs, "ci_low": centre - half,
                     "ci_high": centre + half})
    return pd.DataFrame(rows)


def baseline_scores(positives: dict, known_negatives: dict, n_microbes: int, n_metabolites: int,
                    genus_of: dict) -> dict:
    """Learning-free scores from the training pairs only.

    popularity: fraction of the isolates tested on a metabolite that consumed it.
    genus: the same fraction among isolates of the isolate's genus; the popularity score
    where no isolate of the genus was tested on the metabolite (or the genus is unknown).
    """
    consumed = np.zeros((n_microbes, n_metabolites))
    tested = np.zeros((n_microbes, n_metabolites))
    for m, cs in positives.items():
        consumed[m, list(cs)] = 1
        tested[m, list(cs)] = 1
    for m, cs in known_negatives.items():
        tested[m, list(cs)] = 1
    n_tested = tested.sum(0)
    overall = consumed.sum() / tested.sum()
    popularity = np.where(n_tested > 0, consumed.sum(0) / np.maximum(n_tested, 1), overall)
    genus_score = np.tile(popularity, (n_microbes, 1))
    genera = sorted({g for g in genus_of.values() if g})
    for g in genera:
        members = [m for m, gm in genus_of.items() if gm == g]
        c, t = consumed[members].sum(0), tested[members].sum(0)
        genus_score[members] = np.where(t > 0, c / np.maximum(t, 1), popularity)
    return {"popularity": np.tile(popularity, (n_microbes, 1)), "genus": genus_score}
