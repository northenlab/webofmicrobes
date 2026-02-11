# model.py
"""Embedding models for metabolite preference prediction.

Provides a unified interface for BPR, WR-MF, and SVD matrix factorization
approaches, plus taxonomic aggregation utilities.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding-based preference models.
    
    All models learn embeddings for microbes and metabolites such that
    the dot product approximates preference strength.
    """
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.n_microbes = None
        self.n_metabolites = None
        self._microbe_embeddings = None
        self._metabolite_embeddings = None
        self._fitted = False
    
    @abstractmethod
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        **kwargs
    ) -> "BaseEmbeddingModel":
        """Fit the model to training data.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            **kwargs: Model-specific parameters
        
        Returns:
            self
        """
        pass
    
    @property
    def microbe_embeddings(self) -> np.ndarray:
        """Microbe embedding matrix (n_microbes, embedding_dim)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._microbe_embeddings
    
    @property
    def metabolite_embeddings(self) -> np.ndarray:
        """Metabolite embedding matrix (n_metabolites, embedding_dim)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._metabolite_embeddings
    
    def predict_scores(self, microbe_idx: int) -> np.ndarray:
        """Predict preference scores for all metabolites given a microbe.
        
        Args:
            microbe_idx: Index of the microbe
        
        Returns:
            Array of scores (n_metabolites,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._microbe_embeddings[microbe_idx] @ self._metabolite_embeddings.T
    
    def predict_scores_from_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Predict scores using a custom embedding (e.g., aggregated taxon embedding).
        
        Args:
            embedding: Embedding vector (embedding_dim,)
        
        Returns:
            Array of scores (n_metabolites,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return embedding @ self._metabolite_embeddings.T
    
    def predict_matrix(self) -> np.ndarray:
        """Return full prediction matrix (n_microbes, n_metabolites)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet")
        return self._microbe_embeddings @ self._metabolite_embeddings.T


# =============================================================================
# BPR Model
# =============================================================================

class BPRModel(BaseEmbeddingModel):
    """Bayesian Personalized Ranking via matrix factorization.
    
    Learns embeddings by optimizing pairwise ranking loss: observed positives
    should score higher than unobserved items.
    
    Reference: Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from
    Implicit Feedback"
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
        self._torch_model = None
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        n_epochs: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> "BPRModel":
        """Fit BPR model using SGD.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            n_epochs: Number of training epochs
            batch_size: Triplets per batch
            learning_rate: Adam learning rate
            verbose: Whether to log training progress
        
        Returns:
            self
        """
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build PyTorch model
        model = _BPRTorchModel(n_microbes, n_metabolites, self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        microbe_list = [m for m, pos in positives.items() if len(pos) > 0]
        
        if len(microbe_list) == 0:
            raise ValueError("No microbes with positive examples")
        
        for epoch in range(n_epochs):
            # Sample triplets
            m_batch, pos_batch, neg_batch = [], [], []
            
            for _ in range(batch_size):
                m_idx = np.random.choice(microbe_list)
                pos_idx = np.random.choice(list(positives[m_idx]))
                
                neg_idx = np.random.randint(n_metabolites)
                while neg_idx in positives[m_idx]:
                    neg_idx = np.random.randint(n_metabolites)
                
                m_batch.append(m_idx)
                pos_batch.append(pos_idx)
                neg_batch.append(neg_idx)
            
            pos_scores, neg_scores = model(
                torch.tensor(m_batch),
                torch.tensor(pos_batch),
                torch.tensor(neg_batch),
            )
            
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if verbose and epoch % 200 == 0:
                logger.info(f"  BPR epoch {epoch}: loss={loss.item():.4f}")
        
        # Extract embeddings
        self._torch_model = model
        self._microbe_embeddings = model.microbe_emb.weight.detach().numpy()
        self._metabolite_embeddings = model.metabolite_emb.weight.detach().numpy()
        self._fitted = True
        
        return self
    
    def get_torch_model(self) -> nn.Module:
        """Return underlying PyTorch model (for saving/loading)."""
        return self._torch_model
    
    def load_from_torch(self, torch_model: nn.Module) -> "BPRModel":
        """Load embeddings from a PyTorch model."""
        self._torch_model = torch_model
        self._microbe_embeddings = torch_model.microbe_emb.weight.detach().numpy()
        self._metabolite_embeddings = torch_model.metabolite_emb.weight.detach().numpy()
        self.n_microbes = self._microbe_embeddings.shape[0]
        self.n_metabolites = self._metabolite_embeddings.shape[0]
        self._fitted = True
        return self


class _BPRTorchModel(nn.Module):
    """Internal PyTorch module for BPR."""
    
    def __init__(self, n_microbes: int, n_metabolites: int, embedding_dim: int):
        super().__init__()
        self.microbe_emb = nn.Embedding(n_microbes, embedding_dim)
        self.metabolite_emb = nn.Embedding(n_metabolites, embedding_dim)
        nn.init.normal_(self.microbe_emb.weight, std=0.01)
        nn.init.normal_(self.metabolite_emb.weight, std=0.01)
    
    def forward(self, m_idx, pos_idx, neg_idx):
        m = self.microbe_emb(m_idx)
        pos = self.metabolite_emb(pos_idx)
        neg = self.metabolite_emb(neg_idx)
        return (m * pos).sum(1), (m * neg).sum(1)


# =============================================================================
# WR-MF Model (Weighted Regularized Matrix Factorization)
# =============================================================================

class WRMFModel(BaseEmbeddingModel):
    """Weighted Regularized Matrix Factorization using ALS.
    
    Uses the implicit library for efficient ALS optimization.
    
    Reference: Hu et al. (2008) "Collaborative Filtering for Implicit Feedback
    Datasets"
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        iterations: int = 50,
        regularization: float = 0.1,
        verbose: bool = False,
    ) -> "WRMFModel":
        """Fit WR-MF model using ALS.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
            iterations: Number of ALS iterations
            regularization: L2 regularization strength
            verbose: Whether to show progress
        
        Returns:
            self
        """
        try:
            from implicit.als import AlternatingLeastSquares
            from scipy.sparse import csr_matrix
        except ImportError:
            raise ImportError("WR-MF requires 'implicit' library: pip install implicit")
        
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build sparse matrix
        rows, cols, data = [], [], []
        for m_idx, pos_set in positives.items():
            for c_idx in pos_set:
                rows.append(m_idx)
                cols.append(c_idx)
                data.append(1.0)
        
        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_microbes, n_metabolites),
        )
        
        model = AlternatingLeastSquares(
            factors=self.embedding_dim,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )
        
        model.fit(sparse_matrix, show_progress=verbose)
        
        self._microbe_embeddings = model.user_factors
        self._metabolite_embeddings = model.item_factors
        self._fitted = True
        
        return self


# =============================================================================
# SVD Model
# =============================================================================

class SVDModel(BaseEmbeddingModel):
    """Truncated SVD for matrix factorization baseline.
    
    Simple baseline that factorizes the binary interaction matrix directly.
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__(embedding_dim)
    
    def fit(
        self,
        positives: dict[int, set[int]],
        n_microbes: int,
        n_metabolites: int,
        **kwargs
    ) -> "SVDModel":
        """Fit truncated SVD.
        
        Args:
            positives: Dict mapping microbe_idx -> set of positive metabolite_idx
            n_microbes: Total number of microbes
            n_metabolites: Total number of metabolites
        
        Returns:
            self
        """
        from sklearn.decomposition import TruncatedSVD
        
        self.n_microbes = n_microbes
        self.n_metabolites = n_metabolites
        
        # Build dense matrix
        M = np.zeros((n_microbes, n_metabolites))
        for m_idx, pos_set in positives.items():
            for c_idx in pos_set:
                M[m_idx, c_idx] = 1.0
        
        # Fit SVD
        n_components = min(self.embedding_dim, min(M.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # microbe_embeddings = U * sqrt(S)
        # metabolite_embeddings = V * sqrt(S)
        # so that M ≈ microbe_embeddings @ metabolite_embeddings.T
        U = svd.fit_transform(M)  # (n_microbes, n_components)
        S = svd.singular_values_
        V = svd.components_  # (n_components, n_metabolites)
        
        sqrt_S = np.sqrt(S)
        self._microbe_embeddings = U * sqrt_S  # broadcast
        self._metabolite_embeddings = (V.T * sqrt_S).T.T  # (n_metabolites, n_components)
        
        # Simpler: just use U and V directly scaled
        self._microbe_embeddings = U
        self._metabolite_embeddings = V.T  # (n_metabolites, n_components)
        
        self._fitted = True
        
        return self


# =============================================================================
# Taxonomic Aggregation
# =============================================================================

def compute_taxonomy_embeddings(
    microbe_embeddings: np.ndarray,
    microbe_to_idx: dict[str, int],
    taxonomy_df: pd.DataFrame,
    levels: list[str] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate microbe embeddings by taxonomic level.
    
    Args:
        microbe_embeddings: Embedding matrix (n_microbes, embedding_dim)
        microbe_to_idx: Mapping from organism_id to embedding index
        taxonomy_df: DataFrame with organism_id and taxonomy columns
        levels: Taxonomic levels to aggregate (default: genus, family, order, class)
    
    Returns:
        Nested dict: level -> taxon_name -> embedding
    """
    if levels is None:
        levels = ["genus", "family", "order", "class"]
    
    # Build organism_id -> taxonomy lookup
    taxonomy_df = taxonomy_df.drop_duplicates("organism_id")
    org_to_tax = {}
    for _, row in taxonomy_df.iterrows():
        org_id = row["organism_id"]
        org_to_tax[org_id] = {level: row.get(level) for level in levels}
    
    tax_embeddings = {level: {} for level in levels}
    
    for level in levels:
        # Group organisms by taxon
        taxon_to_orgs = defaultdict(list)
        for org_id, idx in microbe_to_idx.items():
            taxon = org_to_tax.get(org_id, {}).get(level)
            if taxon and pd.notna(taxon):
                taxon_to_orgs[taxon].append(idx)
        
        # Average embeddings per taxon
        for taxon, indices in taxon_to_orgs.items():
            tax_embeddings[level][taxon] = microbe_embeddings[indices].mean(axis=0)
        
        logger.info(f"  {level}: {len(tax_embeddings[level])} taxa")
    
    return tax_embeddings


def get_taxon_embedding(
    taxonomy_embeddings: dict[str, dict[str, np.ndarray]],
    taxon_name: str,
    level: str,
) -> np.ndarray | None:
    """Get embedding for a specific taxon.
    
    Args:
        taxonomy_embeddings: Output from compute_taxonomy_embeddings
        taxon_name: Name of the taxon
        level: Taxonomic level
    
    Returns:
        Embedding vector or None if not found
    """
    return taxonomy_embeddings.get(level, {}).get(taxon_name)


def get_embedding_for_organism(
    organism_id: str,
    taxonomy: dict[str, str],
    microbe_to_idx: dict[str, int],
    microbe_embeddings: np.ndarray,
    taxonomy_embeddings: dict[str, dict[str, np.ndarray]],
    exclude_self: bool = False,
    fallback_levels: list[str] = None,
) -> tuple[np.ndarray | None, str]:
    """Get best available embedding for an organism, with taxonomic fallback.
    
    Args:
        organism_id: The organism to get embedding for
        taxonomy: Dict with genus, family, order, class for this organism
        microbe_to_idx: Mapping from organism_id to embedding index
        microbe_embeddings: Full embedding matrix
        taxonomy_embeddings: Aggregated taxonomy embeddings
        exclude_self: If True, don't use organism's own embedding
        fallback_levels: Ordered list of levels to try (default: genus, family, order, class)
    
    Returns:
        (embedding, source) where source is 'isolate', 'genus', 'family', etc.
    """
    if fallback_levels is None:
        fallback_levels = ["genus", "family", "order", "class"]
    
    # Try own embedding first
    if not exclude_self and organism_id in microbe_to_idx:
        idx = microbe_to_idx[organism_id]
        return microbe_embeddings[idx], "isolate"
    
    # Fallback through taxonomy
    for level in fallback_levels:
        taxon = taxonomy.get(level)
        if taxon and pd.notna(taxon):
            emb = get_taxon_embedding(taxonomy_embeddings, taxon, level)
            if emb is not None:
                return emb, level
    
    return None, "none"


# =============================================================================
# Model Wrapper (High-level API)
# =============================================================================

class ModelWrapper:
    """High-level wrapper providing prediction API for any embedding model."""
    
    def __init__(
        self,
        model: BaseEmbeddingModel,
        microbe_to_idx: dict[str, int],
        metabolite_to_idx: dict[str, int],
        microbe_taxonomy: dict[str, dict[str, str]],
        positives: dict[int, set[int]],
        preference_type: str = "consumed",
        taxonomy_embeddings: dict[str, dict[str, np.ndarray]] = None,
    ):
        self.model = model
        self.microbe_to_idx = microbe_to_idx
        self.metabolite_to_idx = metabolite_to_idx
        self.idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}
        self.idx_to_metabolite = {i: m for m, i in metabolite_to_idx.items()}
        self.microbe_taxonomy = microbe_taxonomy
        self.positives = positives
        self.preference_type = preference_type
        self.taxonomy_embeddings = taxonomy_embeddings
    
    def predict_for_microbe(self, microbe_id: str, top_k: int = None) -> list[tuple[str, float]]:
        """Predict metabolite preferences for a known microbe.
        
        Returns:
            List of (metabolite_name, score) sorted by score descending
        """
        if microbe_id not in self.microbe_to_idx:
            raise ValueError(f"Unknown microbe: {microbe_id}")
        
        m_idx = self.microbe_to_idx[microbe_id]
        scores = self.model.predict_scores(m_idx)
        
        results = [
            (self.idx_to_metabolite[i], float(scores[i]))
            for i in range(len(scores))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def predict_for_taxon(
        self,
        taxon_name: str,
        level: str,
        top_k: int = None,
    ) -> list[tuple[str, float]]:
        """Predict metabolite preferences for a taxonomic group.
        
        Returns:
            List of (metabolite_name, score) sorted by score descending
        """
        if self.taxonomy_embeddings is None:
            raise ValueError("No taxonomy embeddings computed")
        
        emb = get_taxon_embedding(self.taxonomy_embeddings, taxon_name, level)
        if emb is None:
            raise ValueError(f"Unknown taxon: {taxon_name} at level {level}")
        
        scores = self.model.predict_scores_from_embedding(emb)
        
        results = [
            (self.idx_to_metabolite[i], float(scores[i]))
            for i in range(len(scores))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def get_microbe_embedding(self, microbe_id: str) -> np.ndarray:
        """Get embedding for a specific microbe."""
        m_idx = self.microbe_to_idx[microbe_id]
        return self.model.microbe_embeddings[m_idx]
    
    def get_metabolite_embedding(self, metabolite_name: str) -> np.ndarray:
        """Get embedding for a specific metabolite."""
        c_idx = self.metabolite_to_idx[metabolite_name]
        return self.model.metabolite_embeddings[c_idx]


# =============================================================================
# Data Preparation Utilities
# =============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    preference_type: str = "consumed",
) -> dict:
    """Prepare data for model training.
    
    Args:
        df: DataFrame with organism_id, representative_compound, assertion columns
        preference_type: "consumed" or "secreted"
    
    Returns:
        Dict with mappings, positives, and metadata
    """
    df = df[df["representative_compound"].notna()].copy()
    
    microbes = df["organism_id"].unique()
    metabolites = df["representative_compound"].unique()
    
    microbe_to_idx = {m: i for i, m in enumerate(microbes)}
    metabolite_to_idx = {m: i for i, m in enumerate(metabolites)}
    
    # Build positive set
    positives = defaultdict(set)
    for _, row in df.iterrows():
        if row["assertion"] == preference_type:
            m_idx = microbe_to_idx[row["organism_id"]]
            c_idx = metabolite_to_idx[row["representative_compound"]]
            positives[m_idx].add(c_idx)
    
    # Build taxonomy lookup
    microbe_taxonomy = {}
    tax_cols = ["kingdom", "class", "order", "family", "genus"]
    for _, row in df.drop_duplicates("organism_id").iterrows():
        microbe_taxonomy[row["organism_id"]] = {
            col: row.get(col) for col in tax_cols if col in df.columns
        }
    
    return {
        "n_microbes": len(microbes),
        "n_metabolites": len(metabolites),
        "microbe_to_idx": microbe_to_idx,
        "metabolite_to_idx": metabolite_to_idx,
        "positives": dict(positives),
        "microbe_taxonomy": microbe_taxonomy,
        "preference_type": preference_type,
    }


def train_model(
    model_class: type,
    df: pd.DataFrame,
    preference_type: str = "consumed",
    embedding_dim: int = 32,
    **model_kwargs,
) -> ModelWrapper:
    """Train a model and return wrapped instance.
    
    Args:
        model_class: BPRModel, WRMFModel, or SVDModel
        df: Training data
        preference_type: "consumed" or "secreted"
        embedding_dim: Embedding dimension
        **model_kwargs: Additional arguments for model.fit()
    
    Returns:
        ModelWrapper instance
    """
    data = prepare_training_data(df, preference_type)
    
    model = model_class(embedding_dim=embedding_dim)
    model.fit(
        data["positives"],
        data["n_microbes"],
        data["n_metabolites"],
        **model_kwargs,
    )
    
    # Compute taxonomy embeddings
    tax_embeddings = compute_taxonomy_embeddings(
        model.microbe_embeddings,
        data["microbe_to_idx"],
        df,
    )
    
    return ModelWrapper(
        model=model,
        microbe_to_idx=data["microbe_to_idx"],
        metabolite_to_idx=data["metabolite_to_idx"],
        microbe_taxonomy=data["microbe_taxonomy"],
        positives=data["positives"],
        preference_type=preference_type,
        taxonomy_embeddings=tax_embeddings,
    )


# =============================================================================
# Model I/O
# =============================================================================

def save_model(wrapper: ModelWrapper, base_path: Path) -> None:
    """Save model to disk.
    
    Creates:
        - {base_path}_embeddings.npz: Microbe and metabolite embeddings
        - {base_path}_metadata.json: Mappings and config
        - {base_path}_taxonomy.npz: Taxonomy embeddings
    """
    base_path = Path(base_path)
    
    # Save embeddings
    np.savez_compressed(
        base_path.with_name(f"{base_path.name}_embeddings.npz"),
        microbe_embeddings=wrapper.model.microbe_embeddings,
        metabolite_embeddings=wrapper.model.metabolite_embeddings,
    )
    
    # Save metadata
    metadata = {
        "microbe_to_idx": wrapper.microbe_to_idx,
        "metabolite_to_idx": {str(k): v for k, v in wrapper.metabolite_to_idx.items()},
        "idx_to_microbe": {str(k): v for k, v in wrapper.idx_to_microbe.items()},
        "idx_to_metabolite": {str(k): v for k, v in wrapper.idx_to_metabolite.items()},
        "microbe_taxonomy": wrapper.microbe_taxonomy,
        "positives": {str(k): list(v) for k, v in wrapper.positives.items()},
        "preference_type": wrapper.preference_type,
        "embedding_dim": wrapper.model.embedding_dim,
        "n_microbes": wrapper.model.n_microbes,
        "n_metabolites": wrapper.model.n_metabolites,
    }
    
    with open(base_path.with_name(f"{base_path.name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save taxonomy embeddings
    if wrapper.taxonomy_embeddings:
        tax_arrays = {}
        for level, embeddings in wrapper.taxonomy_embeddings.items():
            for name, emb in embeddings.items():
                safe_name = name.replace("__", "___")
                tax_arrays[f"{level}__{safe_name}"] = emb
        
        np.savez_compressed(
            base_path.with_name(f"{base_path.name}_taxonomy.npz"),
            **tax_arrays,
        )
    
    logger.info(f"Saved model to {base_path}")


def load_model(base_path: Path, model_class: type = BPRModel) -> ModelWrapper:
    """Load model from disk.
    
    Args:
        base_path: Base path used in save_model
        model_class: Model class to instantiate
    
    Returns:
        ModelWrapper instance
    """
    base_path = Path(base_path)
    
    # Load embeddings
    emb_data = np.load(base_path.with_name(f"{base_path.name}_embeddings.npz"))
    microbe_embeddings = emb_data["microbe_embeddings"]
    metabolite_embeddings = emb_data["metabolite_embeddings"]
    
    # Load metadata
    with open(base_path.with_name(f"{base_path.name}_metadata.json")) as f:
        metadata = json.load(f)
    
    # Restore types
    positives = {int(k): set(v) for k, v in metadata["positives"].items()}
    idx_to_microbe = {int(k): v for k, v in metadata["idx_to_microbe"].items()}
    idx_to_metabolite = {int(k): v for k, v in metadata["idx_to_metabolite"].items()}
    
    # Create model and inject embeddings
    model = model_class(embedding_dim=metadata["embedding_dim"])
    model._microbe_embeddings = microbe_embeddings
    model._metabolite_embeddings = metabolite_embeddings
    model.n_microbes = metadata["n_microbes"]
    model.n_metabolites = metadata["n_metabolites"]
    model._fitted = True
    
    # Load taxonomy embeddings
    tax_path = base_path.with_name(f"{base_path.name}_taxonomy.npz")
    taxonomy_embeddings = {"genus": {}, "family": {}, "order": {}, "class": {}}
    
    if tax_path.exists():
        tax_data = np.load(tax_path)
        for key in tax_data.files:
            level, name = key.split("__", 1)
            name = name.replace("___", "__")
            taxonomy_embeddings[level][name] = tax_data[key]
    
    return ModelWrapper(
        model=model,
        microbe_to_idx=metadata["microbe_to_idx"],
        metabolite_to_idx=metadata["metabolite_to_idx"],
        microbe_taxonomy=metadata["microbe_taxonomy"],
        positives=positives,
        preference_type=metadata["preference_type"],
        taxonomy_embeddings=taxonomy_embeddings,
    )