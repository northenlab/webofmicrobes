#!/usr/bin/env python3
"""
BPR (Bayesian Personalized Ranking) for Substrate Preference Prediction.

Treats metabolite consumption as implicit feedback - we observe what microbes
consume, but not explicit ratings. BPR learns to rank consumed metabolites
above non-consumed ones.

Reference: Rendle et al. 2009 "BPR: Bayesian Personalized Ranking from Implicit Feedback"

Date: 2026-01-14
"""

import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


# =============================================================================
# Data Loading and Triplet Generation
# =============================================================================

def parse_assertion_code(code) -> Tuple[float, bool, str]:
    """
    Parse WoM assertion code.

    Returns: (score, is_observed, category)
    Categories: 'consumed', 'produced', 'neutral', 'not_observed'
    """
    if pd.isna(code) or str(code).strip() == 'ND':
        return 0.0, False, 'not_observed'

    code = str(code).strip()
    if code == 'D':
        return 0.0, True, 'neutral'

    try:
        score = float(code)
        if score < -1:
            return score, True, 'consumed'
        elif score > 1:
            return score, True, 'produced'
        else:
            return score, True, 'neutral'
    except ValueError:
        return 0.0, False, 'not_observed'


def load_organism_profiles(data_dir: Path) -> Dict[str, Dict[str, dict]]:
    """
    Load organism metabolite profiles from WoM data.

    Returns: {organism: {metabolite: {'score': float, 'category': str, 'observed': bool}}}
    """
    files = [
        data_dir / 'processed' / 'biggs_nldm_wom_format.csv',
        data_dir / 'processed' / '17member_nldm_wom_format.csv',
    ]

    profiles = defaultdict(lambda: defaultdict(list))

    for f in files:
        if not f.exists():
            print(f"Warning: {f} not found")
            continue

        df = pd.read_csv(f)
        print(f"Loading {f.name}: {len(df)} records")

        for _, row in df.iterrows():
            org = row['Organism Id']
            met = row['Compound Id']
            score, observed, category = parse_assertion_code(row.get('Assertion Code', 'ND'))

            if observed:
                profiles[org][met].append({
                    'score': score,
                    'category': category,
                    'timepoint': row.get('Timepoint', None)
                })

    # Aggregate multiple observations per organism-metabolite pair
    aggregated = {}
    for org, mets in profiles.items():
        aggregated[org] = {}
        for met, observations in mets.items():
            # Take most extreme score (strongest signal)
            scores = [obs['score'] for obs in observations]
            categories = [obs['category'] for obs in observations]

            # Majority vote for category
            from collections import Counter
            cat_counts = Counter(categories)
            final_category = cat_counts.most_common(1)[0][0]

            # Use mean score
            aggregated[org][met] = {
                'score': np.mean(scores),
                'category': final_category,
                'n_obs': len(observations)
            }

    return aggregated


def build_triplets(
    profiles: Dict[str, Dict[str, dict]],
    organism_vocab: Dict[str, int],
    metabolite_vocab: Dict[str, int],
    n_negatives: int = 5
) -> List[Tuple[int, int, int, float]]:
    """
    Build (organism, positive_met, negative_met, weight) triplets for BPR training.

    Positive: consumed metabolites
    Negative: metabolites that were available but NOT consumed (or produced)
    Weight: based on consumption strength
    """
    triplets = []

    for org, org_profile in profiles.items():
        if org not in organism_vocab:
            continue
        org_idx = organism_vocab[org]

        # Separate consumed vs non-consumed
        consumed = []
        not_consumed = []

        for met, info in org_profile.items():
            if met not in metabolite_vocab:
                continue
            met_idx = metabolite_vocab[met]

            if info['category'] == 'consumed':
                consumed.append((met_idx, abs(info['score'])))  # weight by strength
            elif info['category'] in ['neutral', 'produced']:
                not_consumed.append(met_idx)

        # Generate triplets: consumed should rank above not_consumed
        for pos_idx, weight in consumed:
            # Sample negatives
            if not not_consumed:
                continue
            neg_samples = random.sample(not_consumed, min(n_negatives, len(not_consumed)))
            for neg_idx in neg_samples:
                triplets.append((org_idx, pos_idx, neg_idx, weight))

    print(f"Generated {len(triplets)} triplets")
    return triplets


def build_consumption_order_triplets(
    profiles: Dict[str, Dict[str, dict]],
    organism_vocab: Dict[str, int],
    metabolite_vocab: Dict[str, int]
) -> List[Tuple[int, int, int, float]]:
    """
    Build triplets based on consumption strength ranking.

    If metabolite A has score -5 and B has score -2, A should rank above B.
    """
    triplets = []

    for org, org_profile in profiles.items():
        if org not in organism_vocab:
            continue
        org_idx = organism_vocab[org]

        # Get all consumed metabolites with scores
        consumed = []
        for met, info in org_profile.items():
            if met not in metabolite_vocab:
                continue
            if info['category'] == 'consumed':
                consumed.append((metabolite_vocab[met], info['score']))

        # Sort by score (most negative = most consumed = higher rank)
        consumed.sort(key=lambda x: x[1])

        # Generate pairwise ranking triplets
        for i in range(len(consumed)):
            for j in range(i + 1, min(i + 5, len(consumed))):  # Compare with next 5
                pos_idx, pos_score = consumed[i]
                neg_idx, neg_score = consumed[j]
                weight = abs(pos_score - neg_score)  # Weight by score difference
                triplets.append((org_idx, pos_idx, neg_idx, weight))

    print(f"Generated {len(triplets)} consumption-order triplets")
    return triplets


# =============================================================================
# BPR Model
# =============================================================================

class BPRModel(nn.Module):
    """
    Bayesian Personalized Ranking model using embedding dot-product.

    Simple and effective for implicit feedback.
    """

    def __init__(
        self,
        n_organisms: int,
        n_metabolites: int,
        embedding_dim: int = 128,
        use_bias: bool = True
    ):
        super().__init__()

        self.n_organisms = n_organisms
        self.n_metabolites = n_metabolites
        self.embedding_dim = embedding_dim

        # Embeddings
        self.organism_emb = nn.Embedding(n_organisms, embedding_dim)
        self.metabolite_emb = nn.Embedding(n_metabolites, embedding_dim)

        # Optional biases
        self.use_bias = use_bias
        if use_bias:
            self.organism_bias = nn.Embedding(n_organisms, 1)
            self.metabolite_bias = nn.Embedding(n_metabolites, 1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.organism_emb.weight, std=0.01)
        nn.init.normal_(self.metabolite_emb.weight, std=0.01)
        if self.use_bias:
            nn.init.zeros_(self.organism_bias.weight)
            nn.init.zeros_(self.metabolite_bias.weight)

    def forward(self, org_idx: torch.Tensor, met_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute affinity score for (organism, metabolite) pairs.

        Higher score = organism prefers this metabolite more.
        """
        org_emb = self.organism_emb(org_idx)
        met_emb = self.metabolite_emb(met_idx)

        # Dot product
        score = (org_emb * met_emb).sum(dim=-1)

        if self.use_bias:
            score = score + self.organism_bias(org_idx).squeeze(-1)
            score = score + self.metabolite_bias(met_idx).squeeze(-1)

        return score

    def predict_all(self, org_idx: torch.Tensor) -> torch.Tensor:
        """Predict scores for one organism against all metabolites."""
        org_emb = self.organism_emb(org_idx)  # (embedding_dim,)
        all_met_emb = self.metabolite_emb.weight  # (n_metabolites, embedding_dim)

        scores = torch.matmul(all_met_emb, org_emb)  # (n_metabolites,)

        if self.use_bias:
            scores = scores + self.metabolite_bias.weight.squeeze(-1)
            scores = scores + self.organism_bias(org_idx).squeeze(-1)

        return scores


class BPRDataset(Dataset):
    """Dataset for BPR triplets."""

    def __init__(self, triplets: List[Tuple[int, int, int, float]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        org_idx, pos_idx, neg_idx, weight = self.triplets[idx]
        return {
            'org_idx': torch.tensor(org_idx, dtype=torch.long),
            'pos_idx': torch.tensor(pos_idx, dtype=torch.long),
            'neg_idx': torch.tensor(neg_idx, dtype=torch.long),
            'weight': torch.tensor(weight, dtype=torch.float32)
        }


# =============================================================================
# Training
# =============================================================================

def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    BPR loss: -log(sigmoid(pos_score - neg_score))

    Encourages positive items to score higher than negative items.
    """
    diff = pos_score - neg_score
    loss = -F.logsigmoid(diff)

    # Weight by importance
    loss = loss * weight

    return loss.mean()


def train_epoch(
    model: BPRModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    l2_reg: float = 0.001
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_bpr_loss = 0
    total_reg_loss = 0
    n_batches = 0

    for batch in dataloader:
        org_idx = batch['org_idx'].to(device)
        pos_idx = batch['pos_idx'].to(device)
        neg_idx = batch['neg_idx'].to(device)
        weight = batch['weight'].to(device)

        optimizer.zero_grad()

        # Compute scores
        pos_score = model(org_idx, pos_idx)
        neg_score = model(org_idx, neg_idx)

        # BPR loss
        loss_bpr = bpr_loss(pos_score, neg_score, weight)

        # L2 regularization on embeddings
        org_emb = model.organism_emb(org_idx)
        pos_emb = model.metabolite_emb(pos_idx)
        neg_emb = model.metabolite_emb(neg_idx)
        loss_reg = l2_reg * (org_emb.pow(2).mean() + pos_emb.pow(2).mean() + neg_emb.pow(2).mean())

        loss = loss_bpr + loss_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bpr_loss += loss_bpr.item()
        total_reg_loss += loss_reg.item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'bpr_loss': total_bpr_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }


def evaluate_auc(
    model: BPRModel,
    profiles: Dict[str, Dict[str, dict]],
    organism_vocab: Dict[str, int],
    metabolite_vocab: Dict[str, int],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model using AUC: can model rank consumed > not-consumed?
    """
    model.eval()

    all_aucs = []

    with torch.no_grad():
        for org, org_profile in profiles.items():
            if org not in organism_vocab:
                continue
            org_idx = torch.tensor(organism_vocab[org], device=device)

            # Get all scores
            all_scores = model.predict_all(org_idx).cpu().numpy()

            # Get consumed and not-consumed indices
            consumed_indices = []
            not_consumed_indices = []

            for met, info in org_profile.items():
                if met not in metabolite_vocab:
                    continue
                met_idx = metabolite_vocab[met]

                if info['category'] == 'consumed':
                    consumed_indices.append(met_idx)
                elif info['category'] in ['neutral', 'produced']:
                    not_consumed_indices.append(met_idx)

            if not consumed_indices or not not_consumed_indices:
                continue

            # Compute AUC: fraction of (consumed, not_consumed) pairs correctly ranked
            n_correct = 0
            n_total = 0

            for pos_idx in consumed_indices:
                for neg_idx in not_consumed_indices:
                    if all_scores[pos_idx] > all_scores[neg_idx]:
                        n_correct += 1
                    elif all_scores[pos_idx] == all_scores[neg_idx]:
                        n_correct += 0.5
                    n_total += 1

            if n_total > 0:
                all_aucs.append(n_correct / n_total)

    return {
        'mean_auc': np.mean(all_aucs) if all_aucs else 0.0,
        'std_auc': np.std(all_aucs) if all_aucs else 0.0,
        'n_organisms': len(all_aucs)
    }


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 70)
    print("BPR SUBSTRATE PREFERENCE TRAINING")
    print("=" * 70)

    # Config
    EMBEDDING_DIM = 128
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 0.01
    L2_REG = 0.001
    N_NEGATIVES = 10
    VAL_SPLIT = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n[1] Loading data...")
    data_dir = Path(__file__).parent.parent / '04_exometabolomics_data'
    profiles = load_organism_profiles(data_dir)
    print(f"Loaded {len(profiles)} organisms")

    # Build vocabularies
    all_organisms = list(profiles.keys())
    all_metabolites = set()
    for org_profile in profiles.values():
        all_metabolites.update(org_profile.keys())
    all_metabolites = sorted(all_metabolites)

    organism_vocab = {org: i for i, org in enumerate(all_organisms)}
    metabolite_vocab = {met: i for i, met in enumerate(all_metabolites)}

    print(f"Organisms: {len(organism_vocab)}")
    print(f"Metabolites: {len(metabolite_vocab)}")

    # Train/val split by organism
    random.seed(42)
    random.shuffle(all_organisms)
    n_val = int(len(all_organisms) * VAL_SPLIT)
    val_organisms = set(all_organisms[:n_val])
    train_organisms = set(all_organisms[n_val:])

    print(f"Train organisms: {len(train_organisms)}")
    print(f"Val organisms: {len(val_organisms)}")

    train_profiles = {org: profiles[org] for org in train_organisms}
    val_profiles = {org: profiles[org] for org in val_organisms}

    # Build triplets
    print("\n[2] Building triplets...")
    triplets = build_triplets(train_profiles, organism_vocab, metabolite_vocab, N_NEGATIVES)
    order_triplets = build_consumption_order_triplets(train_profiles, organism_vocab, metabolite_vocab)

    # Combine triplets
    all_triplets = triplets + order_triplets
    random.shuffle(all_triplets)
    print(f"Total triplets: {len(all_triplets)}")

    # Create dataset and dataloader
    dataset = BPRDataset(all_triplets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create model
    print("\n[3] Creating model...")
    model = BPRModel(
        n_organisms=len(organism_vocab),
        n_metabolites=len(metabolite_vocab),
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    print("\n[4] Training...")
    best_val_auc = 0
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        # Train
        train_metrics = train_epoch(model, dataloader, optimizer, device, L2_REG)

        # Evaluate
        train_eval = evaluate_auc(model, train_profiles, organism_vocab, metabolite_vocab, device)
        val_eval = evaluate_auc(model, val_profiles, organism_vocab, metabolite_vocab, device)

        scheduler.step(1 - val_eval['mean_auc'])  # Minimize 1-AUC

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_auc': train_eval['mean_auc'],
            'val_auc': val_eval['mean_auc']
        })

        if val_eval['mean_auc'] > best_val_auc:
            best_val_auc = val_eval['mean_auc']
            best_epoch = epoch

            # Save best model
            save_dir = Path(__file__).parent / 'trained_bpr_model'
            save_dir.mkdir(exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_auc': val_eval['mean_auc']
            }, save_dir / 'best_model.pt')

            with open(save_dir / 'organism_vocab.json', 'w') as f:
                json.dump(organism_vocab, f, indent=2)
            with open(save_dir / 'metabolite_vocab.json', 'w') as f:
                json.dump(metabolite_vocab, f, indent=2)
            with open(save_dir / 'metadata.json', 'w') as f:
                json.dump({
                    'embedding_dim': EMBEDDING_DIM,
                    'n_organisms': len(organism_vocab),
                    'n_metabolites': len(metabolite_vocab),
                    'best_val_auc': best_val_auc,
                    'best_epoch': best_epoch,
                    'train_organisms': list(train_organisms),
                    'val_organisms': list(val_organisms)
                }, f, indent=2)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d}: loss={train_metrics['loss']:.4f}, "
                  f"train_auc={train_eval['mean_auc']:.3f}, "
                  f"val_auc={val_eval['mean_auc']:.3f}")

    print(f"\nBest val AUC: {best_val_auc:.3f} at epoch {best_epoch}")

    # Save training history
    save_dir = Path(__file__).parent / 'trained_bpr_model'
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nModel saved to: {save_dir}")

    return model, history


if __name__ == '__main__':
    main()
