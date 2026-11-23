# revision_r1_metrics.py
"""Metrics shared by the Reviewer 1 analyses (mSystems revision).

Per-isolate ROC-AUC and PR-AUC (average precision), learning-free baselines, and the
calibration curve Reviewer 1 asked for. BPR scores are ranking statistics; the model
does not output a probability of consumption. The closest equivalent, the logistic
function of the score, is taken as the predicted value and compared with the observed
labels in equal-sized groups, and by the slope of the unbinned labels on it.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

N_GROUPS = 20  # ventiles


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


def quantile_groups(y: np.ndarray, p: np.ndarray, n_groups: int = N_GROUPS) -> pd.DataFrame:
    """Pairs sorted by predicted value and cut into `n_groups` equal-sized groups: mean predicted
    value and observed fraction consumed of each group, with a Wilson 95% interval."""
    rows = []
    for k, idx in enumerate(np.array_split(np.argsort(p, kind="stable"), n_groups)):
        n, obs, z = len(idx), float(y[idx].mean()), 1.96
        centre = (obs + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append({"group": k + 1, "predicted_low": float(p[idx].min()), "predicted_high": float(p[idx].max()),
                     "n": n, "mean_predicted": float(p[idx].mean()), "observed": obs,
                     "ci_low": centre - half, "ci_high": centre + half})
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
