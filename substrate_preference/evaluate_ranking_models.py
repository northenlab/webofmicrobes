#!/usr/bin/env python3
"""
Compare BPR vs Genus-Average baseline for substrate preference prediction.

Evaluation metrics:
1. AUC: Can model rank consumed > not-consumed?
2. Precision@K: Are top-K predictions actually consumed?
3. Rank correlation: Does predicted rank match consumption strength?

Date: 2026-01-14
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from scipy import stats
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_bpr import BPRModel, parse_assertion_code, load_organism_profiles
from baseline_genus_average import GenusAverageBaseline


# =============================================================================
# Model Wrappers for Unified Interface
# =============================================================================

class BPRWrapper:
    """Wrapper for BPR model with same interface as GenusAverageBaseline."""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / 'trained_bpr_model'
        model_dir = Path(model_dir)

        # Load vocabularies
        with open(model_dir / 'organism_vocab.json') as f:
            self.organism_vocab = json.load(f)
        with open(model_dir / 'metabolite_vocab.json') as f:
            self.metabolite_vocab = json.load(f)
        with open(model_dir / 'metadata.json') as f:
            self.metadata = json.load(f)

        self.idx_to_organism = {v: k for k, v in self.organism_vocab.items()}
        self.idx_to_metabolite = {v: k for k, v in self.metabolite_vocab.items()}

        # Load model
        self.device = torch.device('cpu')
        self.model = BPRModel(
            n_organisms=len(self.organism_vocab),
            n_metabolites=len(self.metabolite_vocab),
            embedding_dim=self.metadata['embedding_dim']
        )

        checkpoint = torch.load(model_dir / 'best_model.pt', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, organism: str) -> Dict[str, float]:
        """Get all metabolite scores for an organism."""
        if organism not in self.organism_vocab:
            return {}

        org_idx = torch.tensor(self.organism_vocab[organism])
        with torch.no_grad():
            scores = self.model.predict_all(org_idx).numpy()

        return {self.idx_to_metabolite[i]: float(scores[i]) for i in range(len(scores))}

    def get_top_consumed(self, organism: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get top consumed (highest score = most preferred)."""
        scores = self.predict(organism)
        if not scores:
            return []
        sorted_mets = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_mets[:n]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_auc(
    model,
    profiles: Dict[str, Dict[str, dict]],
    organisms: List[str]
) -> Dict[str, float]:
    """
    Compute AUC: fraction of (consumed, not-consumed) pairs correctly ranked.
    """
    all_aucs = []

    for org in organisms:
        if org not in profiles:
            continue

        scores = model.predict(org)
        if not scores:
            continue

        # Separate consumed vs not-consumed
        consumed_scores = []
        not_consumed_scores = []

        for met, info in profiles[org].items():
            if met not in scores:
                continue
            score = scores[met]

            if info['category'] == 'consumed':
                consumed_scores.append(score)
            elif info['category'] in ['neutral', 'produced']:
                not_consumed_scores.append(score)

        if not consumed_scores or not not_consumed_scores:
            continue

        # AUC: fraction of pairs correctly ranked
        n_correct = 0
        n_total = 0

        for pos_score in consumed_scores:
            for neg_score in not_consumed_scores:
                if pos_score > neg_score:
                    n_correct += 1
                elif pos_score == neg_score:
                    n_correct += 0.5
                n_total += 1

        if n_total > 0:
            all_aucs.append(n_correct / n_total)

    return {
        'mean_auc': np.mean(all_aucs) if all_aucs else 0.0,
        'std_auc': np.std(all_aucs) if all_aucs else 0.0,
        'n_organisms': len(all_aucs)
    }


def compute_precision_at_k(
    model,
    profiles: Dict[str, Dict[str, dict]],
    organisms: List[str],
    k: int = 10
) -> Dict[str, float]:
    """
    Precision@K: What fraction of top-K predictions are actually consumed?
    """
    all_precisions = []

    for org in organisms:
        if org not in profiles:
            continue

        top_k = model.get_top_consumed(org, n=k)
        if not top_k:
            continue

        predicted_mets = set(met for met, _ in top_k)

        # Get actual consumed
        actual_consumed = set(
            met for met, info in profiles[org].items()
            if info['category'] == 'consumed'
        )

        if not actual_consumed:
            continue

        # Precision = intersection / k
        precision = len(predicted_mets & actual_consumed) / k
        all_precisions.append(precision)

    return {
        'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'std_precision': np.std(all_precisions) if all_precisions else 0.0,
        'n_organisms': len(all_precisions)
    }


def compute_rank_correlation(
    model,
    profiles: Dict[str, Dict[str, dict]],
    organisms: List[str]
) -> Dict[str, float]:
    """
    Spearman correlation between predicted scores and true consumption strength.
    """
    all_rhos = []

    for org in organisms:
        if org not in profiles:
            continue

        scores = model.predict(org)
        if not scores:
            continue

        # Get consumed metabolites with their true scores
        pred_scores = []
        true_scores = []

        for met, info in profiles[org].items():
            if met not in scores:
                continue
            if info['category'] == 'consumed':
                pred_scores.append(scores[met])
                true_scores.append(abs(info['score']))  # More negative = more consumed

        if len(pred_scores) < 5:
            continue

        rho, _ = stats.spearmanr(pred_scores, true_scores)
        if not np.isnan(rho):
            all_rhos.append(rho)

    return {
        'mean_rho': np.mean(all_rhos) if all_rhos else 0.0,
        'std_rho': np.std(all_rhos) if all_rhos else 0.0,
        'n_organisms': len(all_rhos)
    }


# =============================================================================
# Main Comparison
# =============================================================================

def main():
    print("=" * 70)
    print("MODEL COMPARISON: BPR vs GENUS-AVERAGE BASELINE")
    print("=" * 70)

    # Load profiles
    print("\n[1] Loading data...")
    data_dir = Path(__file__).parent.parent / '04_exometabolomics_data'
    profiles = load_organism_profiles(data_dir)

    # Load BPR metadata for train/val split
    bpr_dir = Path(__file__).parent / 'trained_bpr_model'
    with open(bpr_dir / 'metadata.json') as f:
        bpr_meta = json.load(f)

    train_organisms = set(bpr_meta['train_organisms'])
    val_organisms = set(bpr_meta['val_organisms'])

    print(f"Total organisms: {len(profiles)}")
    print(f"Train: {len(train_organisms)}, Val: {len(val_organisms)}")

    # Load models
    print("\n[2] Loading models...")

    bpr = BPRWrapper()
    print("  BPR model loaded")

    genus_baseline = GenusAverageBaseline()
    genus_profiles_file = Path(__file__).parent / 'genus_profiles.json'
    if genus_profiles_file.exists():
        genus_baseline.load_profiles(str(genus_profiles_file))
    else:
        genus_baseline.train()
        genus_baseline.save_profiles(str(genus_profiles_file))
    print("  Genus-average baseline loaded")

    # Evaluate on validation set
    print("\n[3] Evaluating on validation organisms...")
    print("-" * 50)

    results = {}

    for name, model in [('BPR', bpr), ('Genus-Average', genus_baseline)]:
        print(f"\n{name}:")

        auc = compute_auc(model, profiles, list(val_organisms))
        p10 = compute_precision_at_k(model, profiles, list(val_organisms), k=10)
        rho = compute_rank_correlation(model, profiles, list(val_organisms))

        results[name] = {
            'auc': auc,
            'precision_at_10': p10,
            'rank_correlation': rho
        }

        print(f"  AUC:           {auc['mean_auc']:.3f} ± {auc['std_auc']:.3f} (n={auc['n_organisms']})")
        print(f"  Precision@10:  {p10['mean_precision']:.3f} ± {p10['std_precision']:.3f} (n={p10['n_organisms']})")
        print(f"  Rank Corr:     {rho['mean_rho']:.3f} ± {rho['std_rho']:.3f} (n={rho['n_organisms']})")

    # Comparison summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Metric | BPR | Genus-Average | Winner |")
    print("|--------|-----|---------------|--------|")

    bpr_auc = results['BPR']['auc']['mean_auc']
    genus_auc = results['Genus-Average']['auc']['mean_auc']
    winner = 'BPR' if bpr_auc > genus_auc else 'Genus-Average'
    print(f"| AUC | {bpr_auc:.3f} | {genus_auc:.3f} | {winner} |")

    bpr_p10 = results['BPR']['precision_at_10']['mean_precision']
    genus_p10 = results['Genus-Average']['precision_at_10']['mean_precision']
    winner = 'BPR' if bpr_p10 > genus_p10 else 'Genus-Average'
    print(f"| Precision@10 | {bpr_p10:.3f} | {genus_p10:.3f} | {winner} |")

    bpr_rho = results['BPR']['rank_correlation']['mean_rho']
    genus_rho = results['Genus-Average']['rank_correlation']['mean_rho']
    winner = 'BPR' if bpr_rho > genus_rho else 'Genus-Average'
    print(f"| Rank Corr | {bpr_rho:.3f} | {genus_rho:.3f} | {winner} |")

    # Save results
    output_file = Path(__file__).parent / 'model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if bpr_auc > genus_auc + 0.05:
        print("\nBPR significantly outperforms taxonomy-based baseline.")
        print("→ The model learns metabolic patterns beyond genus-level averages.")
    elif genus_auc > bpr_auc + 0.05:
        print("\nGenus-average baseline outperforms BPR!")
        print("→ Taxonomy alone is a strong predictor; NN may be overfitting.")
    else:
        print("\nBoth models perform similarly.")
        print("→ Consider: is the additional complexity of BPR worth it?")

    return results


if __name__ == '__main__':
    main()
