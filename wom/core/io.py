"""Save and load model artifacts."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.types import ModelArtifacts, Taxonomy, TrainingData
from core.models import BaseEmbeddingModel, BPRModel, WRMFModel, SVDModel

logger = logging.getLogger(__name__)


def save_model_artifacts(
    artifacts: ModelArtifacts,
    output_dir: str | Path,
) -> None:
    """Save model artifacts to disk.
    
    Saves:
    - model.pt: PyTorch state dict (for BPR) or numpy embeddings (for others)
    - metadata.json: Model type, dimensions, mappings, taxonomy
    
    Args:
        artifacts: ModelArtifacts to save
        output_dir: Directory to save to (will be created if needed)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = artifacts.model
    model_type = type(model).__name__
    
    # Save model weights
    if isinstance(model, BPRModel):
        torch_model = model.get_torch_model()
        torch.save(
            {
                "state_dict": torch_model.state_dict(),
                "embedding_dim": model.embedding_dim,
            },
            output_dir / "model.pt",
        )
    else:
        # For WR-MF and SVD, save embeddings directly
        np.savez(
            output_dir / "model.npz",
            microbe_embeddings=model.microbe_embeddings,
            metabolite_embeddings=model.metabolite_embeddings,
            embedding_dim=model.embedding_dim,
        )
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "embedding_dim": model.embedding_dim,
        "n_microbes": model.n_microbes,
        "n_metabolites": model.n_metabolites,
        "microbe_to_idx": artifacts.training_data.microbe_to_idx,
        "metabolite_to_idx": artifacts.training_data.metabolite_to_idx,
        "microbe_taxonomy": {
            mid: {
                "kingdom": tax.kingdom,
                "class_": tax.class_,
                "order": tax.order,
                "family": tax.family,
                "genus": tax.genus,
                "species": tax.species,
            }
            for mid, tax in artifacts.training_data.microbe_taxonomy.items()
        },
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {model_type} artifacts to {output_dir}")


def load_model_artifacts(
    input_dir: str | Path,
) -> ModelArtifacts:
    """Load model artifacts from disk.
    
    Args:
        input_dir: Directory containing model.pt/npz and metadata.json
    
    Returns:
        ModelArtifacts with loaded model and metadata
    """
    input_dir = Path(input_dir)
    
    # Load metadata
    with open(input_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    model_type = metadata["model_type"]
    embedding_dim = metadata["embedding_dim"]
    n_microbes = metadata["n_microbes"]
    n_metabolites = metadata["n_metabolites"]
    
    # Reconstruct taxonomy
    microbe_taxonomy = {
        mid: Taxonomy(
            kingdom=tax["kingdom"],
            class_=tax["class_"],
            order=tax["order"],
            family=tax["family"],
            genus=tax["genus"],
            species=tax["species"],
        )
        for mid, tax in metadata["microbe_taxonomy"].items()
    }
    
    # Load model
    if model_type == "BPRModel":
        model_path = input_dir / "model.pt"
        checkpoint = torch.load(model_path, weights_only=True)
        
        from core.models import _BPRTorchModel
        torch_model = _BPRTorchModel(n_microbes, n_metabolites, embedding_dim)
        torch_model.load_state_dict(checkpoint["state_dict"])
        
        model = BPRModel(embedding_dim=embedding_dim)
        model.load_from_torch(torch_model)
        
    elif model_type == "WRMFModel":
        model_path = input_dir / "model.npz"
        data = np.load(model_path)
        
        model = WRMFModel(embedding_dim=embedding_dim)
        model.n_microbes = n_microbes
        model.n_metabolites = n_metabolites
        model._microbe_embeddings = data["microbe_embeddings"]
        model._metabolite_embeddings = data["metabolite_embeddings"]
        model._fitted = True
        
    elif model_type == "SVDModel":
        model_path = input_dir / "model.npz"
        data = np.load(model_path)
        
        model = SVDModel(embedding_dim=embedding_dim)
        model.n_microbes = n_microbes
        model.n_metabolites = n_metabolites
        model._microbe_embeddings = data["microbe_embeddings"]
        model._metabolite_embeddings = data["metabolite_embeddings"]
        model._fitted = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Loaded {model_type} from {input_dir}")
    
    # Reconstruct TrainingData (without positives - those aren't saved)
    # Note: positives dict is not needed for inference, only for training
    training_data = TrainingData(
        n_microbes=n_microbes,
        n_metabolites=n_metabolites,
        microbe_to_idx=metadata["microbe_to_idx"],
        metabolite_to_idx=metadata["metabolite_to_idx"],
        positives={},  # Empty for loaded models
        microbe_taxonomy=microbe_taxonomy,
    )
    
    return ModelArtifacts(
        model=model,
        training_data=training_data,
        embedding_dim=embedding_dim,
        model_type=model_type,
    )
