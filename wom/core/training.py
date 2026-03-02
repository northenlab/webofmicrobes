"""Training functions for embedding models."""

import logging
from typing import Optional

import numpy as np

from core.types import TrainingData, ModelArtifacts, Taxonomy
from core.models import BaseEmbeddingModel, BPRModel, ImplicitBPRModel, LMFModel, WRMFModel, SVDModel

logger = logging.getLogger(__name__)


def train_model(
    training_data: TrainingData,
    model_type: str = "bpr",
    embedding_dim: int = 32,
    verbose: bool = True,
    **model_kwargs,
) -> ModelArtifacts:
    """Train an embedding model on training data.
    
    Args:
        training_data: TrainingData object with microbes, metabolites, and interactions
        model_type: One of "bpr", "implicit_bpr", "lmf", "wrmf", "svd"
        embedding_dim: Embedding dimensionality
        verbose: Whether to show training progress
        **model_kwargs: Model-specific training parameters
            BPR: n_epochs, batch_size, learning_rate
            Implicit_BPR: iterations, learning_rate, regularization
            LMF: iterations, learning_rate, regularization
            WR-MF: iterations, regularization
            SVD: (none)
    
    Returns:
        ModelArtifacts with fitted model and metadata
    """
    if verbose:
        logger.info(f"Training {model_type.upper()} model with {embedding_dim}-dim embeddings")
        logger.info(f"Data: {training_data.n_microbes} microbes, {training_data.n_metabolites} metabolites")
        logger.info(f"Positive interactions: {sum(len(v) for v in training_data.positives.values())}")
    
    # Create model
    model_type = model_type.lower()
    if model_type == "bpr":
        model = BPRModel(embedding_dim=embedding_dim)
    elif model_type == "implicit_bpr":
        model = ImplicitBPRModel(embedding_dim=embedding_dim)
    elif model_type == "lmf":
        model = LMFModel(embedding_dim=embedding_dim)
    elif model_type == "wrmf":
        model = WRMFModel(embedding_dim=embedding_dim)
    elif model_type == "svd":
        model = SVDModel(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(
        positives=training_data.positives,
        n_microbes=training_data.n_microbes,
        n_metabolites=training_data.n_metabolites,
        verbose=verbose,
        **model_kwargs,
    )
    
    if verbose:
        logger.info("Training complete")
    
    # Create artifacts
    return ModelArtifacts(
        model=model,
        training_data=training_data,
        embedding_dim=embedding_dim,
        model_type=model_type.upper(),
    )


def compute_taxonomy_embeddings(
    artifacts: ModelArtifacts,
    taxon_level: str = "genus",
) -> dict[str, np.ndarray]:
    """Compute aggregate embeddings for taxonomic groups.
    
    For each taxon at the specified level, compute the average embedding
    of all microbes belonging to that taxon.
    
    Args:
        artifacts: ModelArtifacts with trained model and taxonomy
        taxon_level: Taxonomic level to aggregate at
            (domain, phylum, class, order, family, genus, species)
    
    Returns:
        Dict mapping taxon name -> aggregate embedding
    """
    taxon_embeddings = {}
    taxon_members = {}  # taxon -> list of microbe indices
    
    # Group microbes by taxon
    for microbe_id, idx in artifacts.training_data.microbe_to_idx.items():
        taxonomy = artifacts.training_data.microbe_taxonomy[microbe_id]
        taxon_name = taxonomy.get_level(taxon_level)
        
        if taxon_name is None:
            continue
        
        if taxon_name not in taxon_members:
            taxon_members[taxon_name] = []
        taxon_members[taxon_name].append(idx)
    
    # Compute average embeddings
    microbe_embeddings = artifacts.model.microbe_embeddings
    
    for taxon, indices in taxon_members.items():
        embeddings = microbe_embeddings[indices]
        taxon_embeddings[taxon] = embeddings.mean(axis=0)
    
    logger.info(f"Computed {len(taxon_embeddings)} {taxon_level}-level embeddings")
    return taxon_embeddings


def compute_distinctive_scores(
    artifacts: ModelArtifacts,
    microbe_or_embedding: str | np.ndarray,
    taxon_level: str = "genus",
    top_k: int = 20,
) -> dict[str, float]:
    """Compute distinctive metabolite scores for a microbe or taxon.
    
    Distinctive score = raw score - average score across taxon peers
    
    Args:
        artifacts: ModelArtifacts with trained model
        microbe_or_embedding: Either microbe_id (str) or embedding vector (np.ndarray)
        taxon_level: Taxonomic level for background distribution
        top_k: Return top K metabolites
    
    Returns:
        Dict mapping metabolite_id -> distinctive_score (sorted by score)
    """
    # Get embedding
    if isinstance(microbe_or_embedding, str):
        microbe_id = microbe_or_embedding
        idx = artifacts.training_data.microbe_to_idx.get(microbe_id)
        if idx is None:
            raise ValueError(f"Unknown microbe: {microbe_id}")
        embedding = artifacts.model.microbe_embeddings[idx]
        taxonomy = artifacts.training_data.microbe_taxonomy[microbe_id]
    else:
        embedding = microbe_or_embedding
        taxonomy = None
    
    # Predict raw scores
    raw_scores = artifacts.model.predict_scores_from_embedding(embedding)
    
    # Compute background distribution
    if taxonomy is not None:
        taxon = taxonomy.get_level(taxon_level)
        if taxon is not None:
            # Average scores of microbes in the same taxon
            taxon_indices = [
                idx for mid, idx in artifacts.training_data.microbe_to_idx.items()
                if artifacts.training_data.microbe_taxonomy[mid].get_level(taxon_level) == taxon
            ]
            background_embeddings = artifacts.model.microbe_embeddings[taxon_indices]
            background_scores = (background_embeddings @ artifacts.model.metabolite_embeddings.T).mean(axis=0)
        else:
            background_scores = raw_scores.mean()
    else:
        # Use global average
        background_scores = raw_scores.mean()
    
    # Compute distinctive scores
    distinctive_scores = raw_scores - background_scores
    
    # Get top K
    top_indices = np.argsort(distinctive_scores)[::-1][:top_k]
    
    idx_to_metabolite = {idx: mid for mid, idx in artifacts.training_data.metabolite_to_idx.items()}
    
    return {
        idx_to_metabolite[idx]: float(distinctive_scores[idx])
        for idx in top_indices
    }
