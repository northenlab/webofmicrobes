"""BPR recommendation models for metabolite preferences."""

import json
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
        self.taxonomy_embeddings = metadata.get("taxonomy_embeddings")
        
    
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


def save_metadata(metadata: dict, base_path: Path) -> None:
    """
    Save metadata to JSON + NPZ format.
    
    Args:
        metadata: Model metadata dict
        base_path: Path without extension (will create .json and .npz)
    """
    simple = {}
    arrays = {}
    
    for k, v in metadata.items():
        if k == "taxonomy_embeddings" and v is not None:
            # Flatten nested dict of arrays
            for rank, embeddings in v.items():
                for name, arr in embeddings.items():
                    # Use double underscore as separator since names might have underscores
                    safe_name = name.replace("__", "___")  # Escape existing double underscores
                    arrays[f"tax__{rank}__{safe_name}"] = arr
        elif k == "positives":
            # Convert int keys to str, sets to lists
            simple[k] = {str(key): list(val) for key, val in v.items()}
        elif k == "idx_to_microbe":
            simple[k] = {str(key): val for key, val in v.items()}
        elif k == "idx_to_metabolite":
            simple[k] = {str(key): val for key, val in v.items()}
        elif k == "metabolite_to_idx":
            # Int keys (chebi_id)
            simple[k] = {str(key): val for key, val in v.items()}
        elif isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            simple[k] = v
    
    json_path = base_path.with_suffix(".json")
    npz_path = base_path.with_suffix(".npz")
    
    with open(json_path, "w") as f:
        json.dump(simple, f, indent=2)
    
    if arrays:
        np.savez_compressed(npz_path, **arrays)
    
    logger.info(f"  Saved metadata to {json_path.name} + {npz_path.name}")


def load_metadata(base_path: Path) -> dict:
    """
    Load metadata from JSON + NPZ format.
    
    Args:
        base_path: Path without extension
    
    Returns:
        Reconstructed metadata dict
    """
    json_path = base_path.with_suffix(".json")
    npz_path = base_path.with_suffix(".npz")
    
    with open(json_path) as f:
        metadata = json.load(f)
    
    # Restore int keys
    if "positives" in metadata:
        metadata["positives"] = {
            int(k): set(v) for k, v in metadata["positives"].items()
        }
    
    if "idx_to_microbe" in metadata:
        metadata["idx_to_microbe"] = {
            int(k): v for k, v in metadata["idx_to_microbe"].items()
        }
    
    if "idx_to_metabolite" in metadata:
        metadata["idx_to_metabolite"] = {
            int(k): v for k, v in metadata["idx_to_metabolite"].items()
        }
    
    if "metabolite_to_idx" in metadata:
        metadata["metabolite_to_idx"] = {
            k: v for k, v in metadata["metabolite_to_idx"].items()
        }
    
    # Load arrays and reconstruct taxonomy_embeddings
    taxonomy_embeddings = {"genus": {}, "family": {}, "order": {}, "class": {}}
    
    if npz_path.exists():
        arrays = np.load(npz_path)
        
        for key in arrays.files:
            if key.startswith("tax__"):
                # Parse: tax__rank__name
                _, rank, name = key.split("__", 2)
                name = name.replace("___", "__")  # Unescape
                taxonomy_embeddings[rank][name] = arrays[key]
            else:
                metadata[key] = arrays[key]
    
    # Only include taxonomy_embeddings if we found any
    has_tax_embeddings = any(len(v) > 0 for v in taxonomy_embeddings.values())
    metadata["taxonomy_embeddings"] = taxonomy_embeddings if has_tax_embeddings else None
    
    return metadata


def train_bpr_model(
    df: pd.DataFrame,
    preference_type: str = "consumed",
    embedding_dim: int = 16,
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
    df = df[df["representative_compound"].notna()].copy()
    
    # Build mappings
    microbes = df["organism_id"].unique()
    metabolites = df["representative_compound"].unique()
    
    microbe_to_idx = {m: i for i, m in enumerate(microbes)}
    metabolite_to_idx = {m: i for i, m in enumerate(metabolites)}
    idx_to_microbe = {i: m for m, i in microbe_to_idx.items()}
    idx_to_metabolite = {i: m for m, i in metabolite_to_idx.items()}
    
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
            c_idx = metabolite_to_idx[row["representative_compound"]]
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

def compute_taxonomy_embeddings(wrapper: BPRModelWrapper, exo: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Average microbe embeddings by taxonomic level."""
    embeddings = wrapper.model.microbe_embeddings.weight.detach().numpy()
    
    tax_embeddings = {}
    for rank in ["genus", "family", "order", "class"]:
        if rank not in exo.columns:
            logger.info(f"    {rank}: column not present, skipping")
            continue
            
        tax_embeddings[rank] = {}
        
        for tax_name in exo[rank].dropna().unique():
            org_ids = exo[exo[rank] == tax_name]["organism_id"].unique()
            indices = [
                wrapper.microbe_to_idx[o] 
                for o in org_ids 
                if o in wrapper.microbe_to_idx
            ]
            
            if indices:
                tax_embeddings[rank][tax_name] = embeddings[indices].mean(axis=0)
        
        logger.info(f"    {rank}: {len(tax_embeddings[rank])} embeddings")
    
    return tax_embeddings

# def score_microbes_on_medium(
#     wrapper: BPRModelWrapper,
#     compound_list: list[str],
#     normalize: bool = True
# ) -> pd.Series:
#     """
#     Score microbes by how well a medium matches their preferences.
    
#     Returns Series of microbe_id -> score, sorted descending.
#     """
#     records = []
    
#     for microbe_id in wrapper.microbe_to_idx.keys():
#         m_idx = wrapper.microbe_to_idx[microbe_id]
#         all_scores = wrapper.model.predict_all(m_idx)
#         n_metabolites = len(all_scores)
        
#         percentiles = {}
#         for compound in compound_list:
#             if compound not in wrapper.metabolite_to_idx:
#                 continue
#             c_idx = wrapper.metabolite_to_idx[compound]
#             score = all_scores[c_idx]
#             percentiles[compound] = (all_scores < score).sum() / n_metabolites
        
#         if percentiles:
#             records.append({'microbe_id': microbe_id, **percentiles})
    
#     df = pd.DataFrame(records).set_index('microbe_id')
    
#     if normalize:
#         df = df.T
#         df = df.subtract(df.min(axis=0), axis=1)
#         df = df.divide(df.max(axis=0) + 1e-8, axis=1)
#         df = df.T
    
#     scores = df.sum(axis=1).sort_values(ascending=False)
#     scores.name = 'medium_fit_score'
    
#     return scores

# def check_microbe_preferences(
#     wrapper: BPRModelWrapper,
#     microbe_id: str,
#     compound_list: list[str],
#     top_k: int = 20,
#     return_all_scores: bool = False
# ) -> dict:
#     """
#     For a microbe, show its top predicted compounds and 
#     how many from compound_list appear in the top rankings.
#     """
#     if microbe_id not in wrapper.microbe_to_idx:
#         raise ValueError(f"Unknown microbe: {microbe_id}")
    
#     m_idx = wrapper.microbe_to_idx[microbe_id]
    
#     all_scores = wrapper.model.predict_all(m_idx)
#     if return_all_scores:
#         out = {}
#         for i,s in enumerate(all_scores):
#             representative_compound = wrapper.idx_to_metabolite[i]
#             out[representative_compound] = float(s)
#         return out
    
#     # Get top predicted compounds
#     top_indices = np.argsort(all_scores)[::-1][:top_k]
#     top_compounds = [wrapper.idx_to_metabolite[i] for i in top_indices]
    
#     # Find overlap with provided list
#     compound_set = set(compound_list)
#     overlap = [c for c in top_compounds if c in compound_set]
    
#     # Where does each compound in the list rank?
#     rankings = {}
#     for name in compound_list:
#         if name in wrapper.metabolite_to_idx:
#             c_idx = wrapper.metabolite_to_idx[name]
#             rank = int((all_scores > all_scores[c_idx]).sum() + 1)
#             percentile = float((all_scores < all_scores[c_idx]).sum() / len(all_scores))
#             rankings[name] = {'rank': rank, 'percentile': percentile}
#         else:
#             rankings[name] = {'rank': None, 'percentile': None}
    
#     return {
#         'microbe_id': microbe_id,
#         'top_predicted': top_compounds,
#         'overlap_in_top': overlap,
#         'overlap_count': len(overlap),
#         'compound_rankings': rankings
#     }

# def predict_taxa_for_mixture(
#     wrapper: BPRModelWrapper,
#     compound_names: list[str],
#     rank: str = "genus",
#     weights: list[float] = None,
#     method: str = "mean",
#     top_k: int = 10
# ) -> list[tuple[str, float]]:
#     """
#     Predict which taxa prefer a mixture of compounds.
    
#     Args:
#         wrapper: trained BPRModelWrapper
#         compound_names: list of representative_compound names
#         rank: taxonomic level ('genus', 'family', 'order', 'class')
#         weights: optional weights per compound (e.g., concentrations)
#         method: scoring method
#             - 'mean': average score across compounds
#             - 'min': minimum score (taxa that tolerate all components)
#             - 'geometric': geometric mean (rewards consistent preference)
#             - 'embedding': average embeddings then score (equivalent to mean)
#         top_k: number of results to return
    
#     Returns:
#         List of (taxon_name, score) tuples, sorted descending
#     """
#     if wrapper.taxonomy_embeddings is None or rank not in wrapper.taxonomy_embeddings:
#         raise ValueError(f"No taxonomy embeddings for rank: {rank}")
    
#     # Validate compounds
#     valid_compounds = []
#     for name in compound_names:
#         if name not in wrapper.metabolite_to_idx:
#             logger.warning(f"Unknown compound '{name}', skipping")
#         else:
#             valid_compounds.append(name)
    
#     if not valid_compounds:
#         raise ValueError("No valid compounds in mixture")
    
#     # Handle weights
#     if weights is None:
#         weights = [1.0] * len(valid_compounds)
#     else:
#         # Filter weights to match valid compounds
#         weights = [w for w, name in zip(weights, compound_names) if name in wrapper.metabolite_to_idx]
    
#     # Normalize weights
#     total_weight = sum(weights)
#     weights = [w / total_weight for w in weights]
    
#     # Get metabolite embeddings
#     met_embeddings = []
#     for name in valid_compounds:
#         c_idx = wrapper.metabolite_to_idx[name]
#         emb = wrapper.model.metabolite_embeddings.weight[c_idx].detach().numpy()
#         met_embeddings.append(emb)
#     met_embeddings = np.array(met_embeddings)
    
#     scores = []
    
#     if method == "embedding":
#         # Average the embeddings, then score
#         mixture_emb = np.average(met_embeddings, axis=0, weights=weights)
#         for taxon_name, tax_emb in wrapper.taxonomy_embeddings[rank].items():
#             score = float(np.dot(mixture_emb, tax_emb))
#             scores.append((taxon_name, score))
    
#     else:
#         # Score each compound individually, then combine
#         for taxon_name, tax_emb in wrapper.taxonomy_embeddings[rank].items():
#             compound_scores = [np.dot(emb, tax_emb) for emb in met_embeddings]
            
#             if method == "mean":
#                 score = np.average(compound_scores, weights=weights)
#             elif method == "min":
#                 score = np.min(compound_scores)
#             elif method == "geometric":
#                 # Shift scores to be positive for geometric mean
#                 shifted = np.array(compound_scores) - np.min(compound_scores) + 1e-6
#                 score = np.exp(np.average(np.log(shifted), weights=weights))
#             else:
#                 raise ValueError(f"Unknown method: {method}")
            
#             scores.append((taxon_name, float(score)))
    
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return scores[:top_k]

# def predict_taxa_for_compound(
#     wrapper: BPRModelWrapper,
#     compound_name: str,
#     rank: str = "genus",
#     top_k: int = 10
# ) -> list[tuple[str, float]]:
#     """
#     Predict which taxa prefer a given compound.
    
#     Args:
#         wrapper: trained BPRModelWrapper
#         compound_name: representative_compound name (e.g., 'glucose')
#         rank: taxonomic level ('genus', 'family', 'order', 'class')
#         top_k: number of results to return
    
#     Returns:
#         List of (taxon_name, score) tuples, sorted descending
#     """
#     if compound_name not in wrapper.metabolite_to_idx:
#         raise ValueError(f"Unknown compound: {compound_name}")
    
#     if wrapper.taxonomy_embeddings is None or rank not in wrapper.taxonomy_embeddings:
#         raise ValueError(f"No taxonomy embeddings for rank: {rank}")
    
#     # Get metabolite embedding
#     c_idx = wrapper.metabolite_to_idx[compound_name]
#     met_emb = wrapper.model.metabolite_embeddings.weight[c_idx].detach().numpy()
    
#     # Score against all taxa at this rank
#     scores = []
#     for taxon_name, tax_emb in wrapper.taxonomy_embeddings[rank].items():
#         score = float(np.dot(met_emb, tax_emb))
#         scores.append((taxon_name, score))
    
#     scores.sort(key=lambda x: x[1], reverse=True)
#     return scores[:top_k]

def build_bpr_models(df: pd.DataFrame, force: bool = False) -> dict[str, BPRModelWrapper]:
    """
    Build BPR models for consumption and secretion.
    
    Returns:
        Dict with keys "consumed", "secreted" -> BPRModelWrapper
    """
    models = {}
    
    for pref_type in ["consumed", "secreted"]:
        model_path = CACHE_FILES["bpr_model"].parent / f"bpr_{pref_type}_model.pt"
        meta_base = CACHE_FILES["bpr_model"].parent / f"bpr_{pref_type}_metadata"
        json_path = meta_base.with_suffix(".json")
        
        # Check for legacy pickle format and migrate
        legacy_pkl = CACHE_FILES["bpr_model"].parent / f"bpr_{pref_type}_metadata.pkl"
        if legacy_pkl.exists() and not json_path.exists():
            logger.info(f"Migrating {pref_type} metadata from pickle to JSON+NPZ...")
            with open(legacy_pkl, "rb") as f:
                metadata = pickle.load(f)
            save_metadata(metadata, meta_base)
            legacy_pkl.unlink()  # Remove old pickle
            logger.info(f"  Removed legacy {legacy_pkl.name}")
        
        if not force and model_path.exists() and json_path.exists():
            logger.info(f"Loading cached BPR model ({pref_type})")
            metadata = load_metadata(meta_base)
            model = BPRModel(
                metadata["n_microbes"],
                metadata["n_metabolites"],
                metadata["embedding_dim"]
            )
            model.load_state_dict(torch.load(model_path, weights_only=True))
            models[pref_type] = BPRModelWrapper(model, metadata)
        else:
            if df is None:
                raise ValueError(f"No cached model for {pref_type} and no dataframe provided")
            
            logger.info(f"Training BPR model ({pref_type})...")
            try:
                model, metadata = train_bpr_model(df, preference_type=pref_type)
                
                # Create wrapper first so we can compute taxonomy embeddings
                wrapper = BPRModelWrapper(model, metadata)
                
                # Compute taxonomy embeddings
                logger.info(f"  Computing taxonomy embeddings...")
                tax_embeddings = compute_taxonomy_embeddings(wrapper, df)
                
                # Add to metadata and update wrapper
                metadata["taxonomy_embeddings"] = tax_embeddings
                wrapper.taxonomy_embeddings = tax_embeddings
                
                # Save
                torch.save(model.state_dict(), model_path)
                save_metadata(metadata, meta_base)
                
                models[pref_type] = wrapper
                
            except ValueError as e:
                logger.warning(f"  Skipping {pref_type}: {e}")
                continue
    
    return models


def load_bpr_models() -> dict[str, BPRModelWrapper]:
    """Load cached BPR models."""
    return build_bpr_models(None, force=False)

def plot_metabolite_embeddings(
    wrapper: BPRModelWrapper,
    exo: pd.DataFrame = None,
    perplexity: int = 30,
    figsize: tuple = (14, 12),
    output_dir: Path = None,
    label_top_n: int = None,
    min_font_size: int = 6,
    max_font_size: int = 10,
) -> pd.DataFrame:
    """
    Generate t-SNE visualization of metabolite embeddings.
    """
    if output_dir is None:
        output_dir = Path(".")
    
    # Extract embeddings
    embeddings = wrapper.model.metabolite_embeddings.weight.detach().numpy()
    n_metabolites = len(embeddings)
    
    # Debug: check what we received
    logger.info(f"  exo is None: {exo is None}")
    if exo is not None:
        logger.info(f"  exo columns: {list(exo.columns)}")
        logger.info(f"  exo shape: {exo.shape}")
        if "representative_compound" in exo.columns:
            logger.info(f"  representative_compound dtype: {exo['representative_compound'].dtype}")
            logger.info(f"  representative_compound sample: {exo['representative_compound'].dropna().head(3).tolist()}")
            logger.info(f"  representative_compound nunique: {exo['representative_compound'].nunique()}")

    

    
    
    
    # Run t-SNE
    from sklearn.manifold import TSNE
    from sklearn import __version__ as sklearn_version
    sklearn_minor = int(sklearn_version.split('.')[1])
    
    effective_perplexity = min(perplexity, (n_metabolites - 1) // 3)
    
    if sklearn_minor >= 4:
        tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, max_iter=1000)
    else:
        tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, n_iter=1000)
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Build dataframe
    plot_df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "metabolite_idx": range(n_metabolites),
    })
    
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    
    
    scatter = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        alpha=1,
        edgecolors="white",
        linewidths=0.5,
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("# organisms using metabolite")
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    pref_type = wrapper.preference_type
    ax.set_title(f"Metabolite Embeddings (BPR {pref_type})\nSize/color = # organisms")
    
    plt.tight_layout()
    
    base_name = f"metabolite_embeddings_{pref_type}"
    plt.savefig(output_dir / f"{base_name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / f"{base_name}.pdf", bbox_inches="tight")
    logger.info(f"Saved: {base_name}.png/pdf")
    
    plt.close()
    
    return plot_df


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

def plot_microbe_embeddings_class_genus(
    model, microbe_taxonomy, idx_to_microbe,
    perplexity=30, figsize=(14, 10), pref_type=None, output_dir=None, 
    title_suffix="", major_grouping='class'
):
    """
    Plot microbe embeddings with class as color (tab20) and genus as shape.
    
    Shapes are reset within each major_grouping, so the same shape can appear
    in different colors. Legend is hierarchical: class header with genera nested below.
    """
    from sklearn.manifold import TSNE
    from sklearn import __version__ as sklearn_version
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import pandas as pd
    import numpy as np
    
    # Solid markers first, then hollow
    SOLID_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'P', 'X', '<', '>', 'd', 'H', '8']
    
    # Extract embeddings
    embeddings = model.microbe_embeddings.weight.detach().numpy()
    
    # Run t-SNE
    if int(sklearn_version.split('.')[1]) >= 4:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Build dataframe for plotting
    plot_df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "microbe_id": [idx_to_microbe[i] for i in range(len(embeddings))]
    })
    
    # Add taxonomy
    plot_df[major_grouping] = plot_df["microbe_id"].map(
        lambda m: microbe_taxonomy.get(m, {}).get(major_grouping)
    )
    plot_df["genus"] = plot_df["microbe_id"].map(
        lambda m: microbe_taxonomy.get(m, {}).get("genus")
    )
    
    # Handle NaN
    plot_df[major_grouping] = plot_df[major_grouping].fillna("Unknown")
    plot_df["genus"] = plot_df["genus"].fillna("Unknown")
    
    # Assign colors to major groupings (tab20)
    groups = sorted(plot_df[major_grouping].unique())
    cmap = plt.cm.get_cmap("tab20", max(len(groups), 20))
    group_colors = {grp: cmap(i % 20) for i, grp in enumerate(groups)}
    
    # Assign shapes to genera WITHIN each group (reset per group)
    # This creates a nested dict: group -> genus -> (marker, is_hollow)
    genus_markers = {}
    for grp in groups:
        genera_in_group = sorted(plot_df[plot_df[major_grouping] == grp]["genus"].unique())
        genus_markers[grp] = {}
        for i, genus in enumerate(genera_in_group):
            marker_idx = i % len(SOLID_MARKERS)
            is_hollow = i >= len(SOLID_MARKERS)
            genus_markers[grp][genus] = (SOLID_MARKERS[marker_idx], is_hollow)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for grp in groups:
        color = group_colors[grp]
        group_data = plot_df[plot_df[major_grouping] == grp]
        
        for genus in sorted(group_data["genus"].unique()):
            marker, is_hollow = genus_markers[grp][genus]
            subset = group_data[group_data["genus"] == genus]
            
            if is_hollow:
                ax.scatter(
                    subset["x"], subset["y"],
                    marker=marker,
                    facecolors='none',
                    edgecolors=color,
                    linewidths=2.0,
                    s=80,
                    alpha=0.9,
                )
            else:
                ax.scatter(
                    subset["x"], subset["y"],
                    marker=marker,
                    c=[color],
                    edgecolors='white',
                    linewidths=0.5,
                    s=70,
                    alpha=0.85,
                )
    
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title(
        f"Microbe Embeddings{title_suffix}\nColor = {major_grouping.capitalize()}, Shape = Genus",
        fontsize=12
    )
    
    # Build hierarchical legend: group header + indented genera with colored shapes
    legend_handles = []
    for grp in groups:
        color = group_colors[grp]
        
        # Group header (bold text via a patch with the group name)
        header = mpatches.Patch(color=color, label=grp)
        legend_handles.append(header)
        
        # Genera within this group (indented with "  " prefix)
        genera_in_group = sorted(genus_markers[grp].keys())
        for genus in genera_in_group:
            marker, is_hollow = genus_markers[grp][genus]
            if is_hollow:
                handle = mlines.Line2D(
                    [], [], color='none', marker=marker, linestyle='',
                    markeredgecolor=color, markeredgewidth=2,
                    markersize=7, label=f"   {genus}"
                )
            else:
                handle = mlines.Line2D(
                    [], [], color=color, marker=marker, linestyle='',
                    markeredgecolor='white', markeredgewidth=0.5,
                    markersize=7, label=f"   {genus}"
                )
            legend_handles.append(handle)
    
    # Determine legend columns based on total entries
    total_entries = len(legend_handles)
    if total_entries > 40:
        ncol = 3
    elif total_entries > 20:
        ncol = 2
    else:
        ncol = 1
    
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, ncol=ncol, columnspacing=1.0,
        handletextpad=0.3, labelspacing=0.3
    )
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        base = output_dir / f"microbe_embeddings_{major_grouping}_genus_{pref_type}"
    else:
        base = Path(f"microbe_embeddings_{major_grouping}_genus_{pref_type}")
    
    plt.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    logger.info(f"Saved: {base}.png/pdf")
    
    plt.close()
    
    return plot_df


def plot_all_embeddings(
    models: dict[str, BPRModelWrapper],
    exo: pd.DataFrame = None,
    output_dir: Path = None,
    tax_levels: list[str] = None,
):
    """Generate t-SNE plots for all models."""
    if tax_levels is None:
        tax_levels = ["order", "class", "family"]
    
    for pref_type, wrapper in models.items():
        # Microbe embeddings
        for level in tax_levels:
            logger.info(f"Plotting {pref_type} microbe embeddings colored by {level}...")
            plot_microbe_embeddings(wrapper, color_by=level, output_dir=output_dir)
        
        # Metabolite embeddings
        logger.info(f"Plotting {pref_type} metabolite embeddings...")
        plot_metabolite_embeddings(wrapper, exo=exo, output_dir=output_dir)
        
    
    for pref_type, wrapper in models.items():
        for level in tax_levels:
            logger.info(f"Plotting {pref_type} embeddings colored by {level}...")
            plot_microbe_embeddings(wrapper, color_by=level, output_dir=output_dir)
        plot_microbe_embeddings_class_genus(
            wrapper.model,
            wrapper.microbe_taxonomy,
            wrapper.idx_to_microbe,
            output_dir=output_dir,
            pref_type=pref_type,
            major_grouping="class",
            title_suffix=f" (BPR {pref_type})"
        )
        plot_microbe_embeddings_class_genus(
            wrapper.model,
            wrapper.microbe_taxonomy,
            wrapper.idx_to_microbe,
            output_dir=output_dir,
            pref_type=pref_type,
            major_grouping="order",
            title_suffix=f" (BPR {pref_type})"
        )