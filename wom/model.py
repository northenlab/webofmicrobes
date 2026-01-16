"""BPR recommendation models for metabolite preferences."""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import CACHE_FILES

logger = logging.getLogger(__name__)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class BPRModel(nn.Module):
    """Bayesian Personalized Ranking model for microbe-metabolite preferences."""
    
    def __init__(self, n_microbes: int, n_metabolites: int, embedding_dim: int = 32):
        super().__init__()
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        self.embedding_dim = embedding_dim
        
        self.microbe_embeddings = nn.Embedding(n_microbes, embedding_dim)
        self.metabolite_embeddings = nn.Embedding(n_metabolites, embedding_dim)
        
        nn.init.normal_(self.microbe_embeddings.weight, std=0.01)
        nn.init.normal_(self.metabolite_embeddings.weight, std=0.01)
    
    def forward(self, microbe_idx, pos_idx, neg_idx):
        microbe_emb = self.microbe_embeddings(microbe_idx)
        pos_emb = self.metabolite_embeddings(pos_idx)
        neg_emb = self.metabolite_embeddings(neg_idx)
        
        pos_score = (microbe_emb * pos_emb).sum(dim=1)
        neg_score = (microbe_emb * neg_emb).sum(dim=1)
        
        return pos_score, neg_score
    
    def predict_all(self, microbe_idx: int) -> np.ndarray:
        """Predict scores for all metabolites given a microbe."""
        with torch.no_grad():
            microbe_emb = self.microbe_embeddings(torch.tensor([microbe_idx]))
            scores = torch.matmul(microbe_emb, self.metabolite_embeddings.weight.T).squeeze()
        return scores.numpy()
    
    def predict_single(self, microbe_idx: int, metabolite_idx: int) -> float:
        """Predict score for a single microbe-metabolite pair."""
        with torch.no_grad():
            microbe_emb = self.microbe_embeddings(torch.tensor([microbe_idx]))
            metabolite_emb = self.metabolite_embeddings(torch.tensor([metabolite_idx]))
            score = (microbe_emb * metabolite_emb).sum()
        return score.item()


class BPRModelWrapper:
    """Wrapper providing high-level API for BPR model."""
    
    def __init__(self, model: BPRModel, metadata: dict):
        self.model = model
        self.model.eval()
        
        self.microbe_to_idx = metadata["microbe_to_idx"]
        self.metabolite_to_idx = metadata["metabolite_to_idx"]
        self.idx_to_microbe = metadata["idx_to_microbe"]
        self.idx_to_metabolite = metadata["idx_to_metabolite"]
        self.microbe_taxonomy = metadata["microbe_taxonomy"]
        self.positives = metadata["positives"]
        self.preference_type = metadata.get("preference_type", "consumed")
    
    def predict(self, microbe_id: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Predict top metabolites for a microbe.
        
        Returns:
            List of (chebi_id, score) tuples, sorted by score descending.
        """
        if microbe_id not in self.microbe_to_idx:
            raise ValueError(f"Unknown microbe: {microbe_id}")
        
        m_idx = self.microbe_to_idx[microbe_id]
        scores = self.model.predict_all(m_idx)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chebi_id = self.idx_to_metabolite[idx]
            results.append((chebi_id, float(scores[idx])))
        
        return results
    
    def similar_microbes(self, microbe_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find microbes with similar metabolite preferences."""
        if microbe_id not in self.microbe_to_idx:
            raise ValueError(f"Unknown microbe: {microbe_id}")
        
        m_idx = self.microbe_to_idx[microbe_id]
        embeddings = self.model.microbe_embeddings.weight.detach().numpy()
        
        query_emb = embeddings[m_idx]
        # Cosine similarity
        norms = np.linalg.norm(embeddings, axis=1)
        similarities = embeddings @ query_emb / (norms * np.linalg.norm(query_emb) + 1e-8)
        
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # exclude self
        results = []
        for idx in top_indices:
            other_id = self.idx_to_microbe[idx]
            results.append((other_id, float(similarities[idx])))
        
        return results
    
    def get_microbe_embedding(self, microbe_id: str) -> np.ndarray:
        """Get raw embedding vector for a microbe."""
        m_idx = self.microbe_to_idx[microbe_id]
        return self.model.microbe_embeddings.weight[m_idx].detach().numpy()
    
    def get_metabolite_embedding(self, chebi_id: int) -> np.ndarray:
        """Get raw embedding vector for a metabolite."""
        c_idx = self.metabolite_to_idx[chebi_id]
        return self.model.metabolite_embeddings.weight[c_idx].detach().numpy()


def _sample_triplets(positives: dict, n_metabolites: int, batch_size: int):
    """Sample (microbe, positive, negative) triplets for BPR training."""
    microbe_indices, pos_indices, neg_indices = [], [], []
    microbe_list = [m for m in positives.keys() if len(positives[m]) > 0]
    
    for _ in range(batch_size):
        m_idx = np.random.choice(microbe_list)
        pos_idx = np.random.choice(list(positives[m_idx]))
        
        neg_idx = np.random.randint(n_metabolites)
        while neg_idx in positives[m_idx]:
            neg_idx = np.random.randint(n_metabolites)
        
        microbe_indices.append(m_idx)
        pos_indices.append(pos_idx)
        neg_indices.append(neg_idx)
    
    return (torch.tensor(microbe_indices), 
            torch.tensor(pos_indices), 
            torch.tensor(neg_indices))


def _bpr_loss(pos_scores, neg_scores):
    """BPR loss: maximize margin between positive and negative."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def train_bpr_model(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
    n_epochs: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 0.01,
) -> tuple[BPRModel, dict]:
    """
    Train a BPR model on exometabolite data.
    
    Args:
        df: Exometabolite dataframe with organism_id, chebi_id, assertion columns
        preference_type: "consumed" or "secreted"
        embedding_dim: Size of embedding vectors
        n_epochs: Training epochs
        batch_size: Triplets per batch
        learning_rate: Adam learning rate
    
    Returns:
        (model, metadata) tuple
    """
    # Filter to rows with valid chebi_id
    df = df[df["chebi_id"].notna()].copy()
    
    # Build mappings
    microbes = df["organism_id"].unique()
    metabolites = df["chebi_id"].unique()
    
    microbe_to_idx = {m: i for i, m in enumerate(microbes)}
    metabolite_to_idx = {int(m): i for i, m in enumerate(metabolites)}
    idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}
    idx_to_metabolite = {i: int(m) for m, i in metabolite_to_idx.items()}
    
    n_microbes = len(microbes)
    n_metabolites = len(metabolites)
    
    logger.info(f"  Training BPR ({preference_type}): {n_microbes} microbes, {n_metabolites} metabolites")
    
    # Build taxonomy lookup
    microbe_taxonomy = {}
    tax_cols = ["kingdom", "class", "order", "family", "genus"]
    for _, row in df.drop_duplicates("organism_id").iterrows():
        microbe_taxonomy[row["organism_id"]] = {
            col: row.get(col) for col in tax_cols
        }
    
    # Build positive set
    positives = defaultdict(set)
    for _, row in df.iterrows():
        if row["assertion"] == preference_type:
            m_idx = microbe_to_idx[row["organism_id"]]
            c_idx = metabolite_to_idx[int(row["chebi_id"])]
            positives[m_idx].add(c_idx)
    
    n_positive_pairs = sum(len(v) for v in positives.values())
    logger.info(f"  Positive pairs: {n_positive_pairs}")
    
    if n_positive_pairs == 0:
        raise ValueError(f"No positive examples found for preference_type='{preference_type}'")
    
    # Train
    model = BPRModel(n_microbes, n_metabolites, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        microbe_batch, pos_batch, neg_batch = _sample_triplets(positives, n_metabolites, batch_size)
        pos_scores, neg_scores = model(microbe_batch, pos_batch, neg_batch)
        loss = _bpr_loss(pos_scores, neg_scores)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            logger.info(f"    Epoch {epoch}: loss={loss.item():.4f}")
    
    logger.info(f"    Final loss: {loss.item():.4f}")
    
    metadata = {
        "microbe_to_idx": microbe_to_idx,
        "metabolite_to_idx": metabolite_to_idx,
        "idx_to_microbe": idx_to_microbe,
        "idx_to_metabolite": idx_to_metabolite,
        "microbe_taxonomy": microbe_taxonomy,
        "positives": dict(positives),
        "n_microbes": n_microbes,
        "n_metabolites": n_metabolites,
        "embedding_dim": embedding_dim,
        "preference_type": preference_type,
    }
    
    return model, metadata


def build_bpr_models(df: pd.DataFrame, force: bool = False) -> dict[str, BPRModelWrapper]:
    """
    Build BPR models for consumption and secretion.
    
    Returns:
        Dict with keys "consumed", "secreted" -> BPRModelWrapper
    """
    models = {}
    
    for pref_type in ["consumed", "secreted"]:
        model_path = CACHE_FILES["bpr_model"].parent / f"bpr_{pref_type}_model.pt"
        meta_path = CACHE_FILES["bpr_model"].parent / f"bpr_{pref_type}_metadata.pkl"
        
        if not force and model_path.exists() and meta_path.exists():
            logger.info(f"Loading cached BPR model ({pref_type})")
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            model = BPRModel(
                metadata["n_microbes"],
                metadata["n_metabolites"],
                metadata["embedding_dim"]
            )
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            logger.info(f"Training BPR model ({pref_type})...")
            try:
                model, metadata = train_bpr_model(df, preference_type=pref_type)
                torch.save(model.state_dict(), model_path)
                with open(meta_path, "wb") as f:
                    pickle.dump(metadata, f)
            except ValueError as e:
                logger.warning(f"  Skipping {pref_type}: {e}")
                continue
        
        models[pref_type] = BPRModelWrapper(model, metadata)
    
    return models


def load_bpr_models() -> dict[str, BPRModelWrapper]:
    """Load cached BPR models."""
    return build_bpr_models(None, force=False)


def plot_microbe_embeddings(
    wrapper: BPRModelWrapper,
    color_by: str = "order",
    perplexity: int = 30,
    figsize: tuple = (12, 10),
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Generate t-SNE visualization of microbe embeddings.
    
    Args:
        wrapper: BPRModelWrapper instance
        color_by: Taxonomy level to color by (kingdom, class, order, family, genus)
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
        output_dir: Directory for output files (default: current dir)
    
    Returns:
        DataFrame with embedding coordinates and taxonomy
    """
    if output_dir is None:
        output_dir = Path(".")
    
    # Extract embeddings
    embeddings = wrapper.model.microbe_embeddings.weight.detach().numpy()
    
    # Run t-SNE
    from sklearn import __version__ as sklearn_version
    sklearn_minor = int(sklearn_version.split('.')[1])
    
    if sklearn_minor >= 4:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Build dataframe
    plot_df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "microbe_id": [wrapper.idx_to_microbe[i] for i in range(len(embeddings))]
    })
    
    # Add taxonomy
    for col in ["kingdom", "class", "order", "family", "genus"]:
        plot_df[col] = plot_df["microbe_id"].map(
            lambda m: wrapper.microbe_taxonomy.get(m, {}).get(col)
        )
    
    plot_df[color_by] = plot_df[color_by].fillna("Unknown")
    
    # Colors
    categories = plot_df[color_by].unique()
    cmap = plt.cm.get_cmap("tab20", len(categories))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for category in categories:
        subset = plot_df[plot_df[color_by] == category]
        ax.scatter(
            subset["x"],
            subset["y"],
            c=[color_map[category]],
            label=category,
            alpha=0.7,
            s=50
        )
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    pref_type = wrapper.preference_type
    ax.set_title(f"Microbe Embeddings (BPR {pref_type})\nColored by {color_by}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    
    base_name = f"microbe_embeddings_{pref_type}_{color_by}"
    plt.savefig(output_dir / f"{base_name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"{base_name}.pdf", bbox_inches="tight")
    logger.info(f"Saved: {base_name}.png/pdf")
    
    plt.close()
    
    return plot_df


def plot_all_embeddings(
    models: dict[str, BPRModelWrapper],
    output_dir: Path = None,
    tax_levels: list[str] = None,
):
    """Generate t-SNE plots for all models at multiple taxonomy levels."""
    if tax_levels is None:
        tax_levels = ["order", "class", "family"]
    
    for pref_type, wrapper in models.items():
        for level in tax_levels:
            logger.info(f"Plotting {pref_type} embeddings colored by {level}...")
            plot_microbe_embeddings(wrapper, color_by=level, output_dir=output_dir)