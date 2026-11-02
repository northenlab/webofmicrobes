# revision_r1_metrics.py
"""Metrics shared by the Reviewer 1 analyses (mSystems revision).

Per-isolate ROC-AUC and PR-AUC (average precision), and calibration of BPR scores:
BPR scores are ranking statistics, so probabilities are obtained with a monotone
map (Platt scaling, i.e. logistic regression of the label on the score, or isotonic
regression) fitted by cross-fitting over isolates, so that no isolate's pairs are
used to fit the map that is applied to them.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

N_FOLDS = 5
N_BINS = 10


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


def crossfit_probabilities(y: np.ndarray, s: np.ndarray, isolate: np.ndarray, seed: int = 0,
                           method: str = "platt") -> np.ndarray:
    """Out-of-fold P(consumed | score): the map is fitted on the pairs of the other folds of isolates."""
    rng = np.random.default_rng(seed)
    isolates = pd.unique(isolate)
    fold_of = dict(zip(isolates, rng.permutation(len(isolates)) % N_FOLDS))
    fold = np.array([fold_of[m] for m in isolate])
    p = np.empty(len(y), dtype=float)
    for k in range(N_FOLDS):
        train, test = fold != k, fold == k
        if method == "platt":
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(s[train, None], y[train])
            p[test] = lr.predict_proba(s[test, None])[:, 1]
        elif method == "isotonic":
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s[train], y[train])
            p[test] = iso.predict(s[test])
        else:
            raise ValueError(method)
    return p


def naive_probabilities(s: np.ndarray) -> np.ndarray:
    """The logistic function of the raw score, as if the BPR score were a log-odds."""
    return expit(s)


def calibration_stats(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Unbinned slope/intercept of the label on the prediction (1 and 0 when calibrated), the
    logistic calibration slope, Brier score and expected calibration error over equal-width bins."""
    ols = LinearRegression().fit(p[:, None], y)
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    logistic = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    bins = reliability(y, p, n_bins)
    ece = float((bins.n * (bins.observed - bins.mean_predicted).abs()).sum() / len(y))
    return {"ols_slope": float(ols.coef_[0]), "ols_intercept": float(ols.intercept_),
            "logistic_slope": float(logistic.coef_[0, 0]), "brier": float(np.mean((p - y) ** 2)),
            "ece": ece, "prevalence": float(y.mean()), "n_pairs": int(len(y))}


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Observed fraction consumed per equal-width bin of predicted probability, with Wilson 95% CI."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        sel = b == k
        n = int(sel.sum())
        if not n:
            continue
        obs = float(y[sel].mean())
        z = 1.96
        centre = (obs + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append({"bin": k, "bin_low": edges[k], "bin_high": edges[k + 1], "n": n,
                     "mean_predicted": float(p[sel].mean()), "observed": obs,
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
