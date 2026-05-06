"""Core data types and dataclasses for Web of Microbes.

This module eliminates parameter drilling by providing well-typed containers
for related data that gets passed through the system.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np


# =============================================================================
# Enums
# =============================================================================

class AssertionType(Enum):
    """Metabolite interaction assertions."""
    CONSUMED = "consumed"
    SECRETED = "secreted"
    NEUTRAL = "neutral"


class TaxonomicLevel(Enum):
    """Taxonomic hierarchy levels."""
    ISOLATE = "isolate"
    SPECIES = "species"
    GENUS = "genus"
    FAMILY = "family"
    ORDER = "order"
    CLASS = "class"
    KINGDOM = "kingdom"


# =============================================================================
# Data Containers
# =============================================================================

@dataclass
class Taxonomy:
    """Taxonomic information for a microbe."""
    kingdom: Optional[str] = None
    class_: Optional[str] = None  # 'class' is reserved keyword
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    
    def get_level(self, level: str) -> Optional[str]:
        """Get taxonomy at specified level."""
        level_map = {
            "kingdom": self.kingdom,
            "class": self.class_,
            "order": self.order,
            "family": self.family,
            "genus": self.genus,
            "species": self.species,
        }
        return level_map.get(level)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "kingdom": self.kingdom,
            "class": self.class_,
            "order": self.order,
            "family": self.family,
            "genus": self.genus,
            "species": self.species,
        }


@dataclass
class CompoundMapping:
    """Result of mapping a compound name to ChEBI."""
    query_name: str
    representative_name: Optional[str]
    chebi_id: Optional[str]
    match_score: float  # 0.0 to 1.0
    match_type: str  # 'exact', 'fuzzy', 'none'


@dataclass
class TrainingData:
    """Complete training dataset with all mappings.
    
    This replaces the dictionary that was previously passed around with keys like
    'n_microbes', 'microbe_to_idx', etc. Now we have a typed object.
    """
    # Dimensions
    n_microbes: int
    n_metabolites: int
    
    # Forward mappings (name -> index)
    microbe_to_idx: dict[str, int]
    metabolite_to_idx: dict[str, int]
    
    # Reverse mappings (index -> name)
    idx_to_microbe: dict[int, str] = field(default_factory=dict)
    idx_to_metabolite: dict[int, str] = field(default_factory=dict)
    
    # Interaction data
    positives: dict[int, set[int]] = field(default_factory=dict)  # microbe_idx -> metabolite_idx set
    known_negatives: dict[int, set[int]] = field(default_factory=dict)  # microbe_idx -> metabolite_idx set (tested, not consumed)
    
    # Metadata
    microbe_taxonomy: dict[str, Taxonomy] = field(default_factory=dict)  # organism_id -> Taxonomy
    preference_type: AssertionType = AssertionType.CONSUMED
    
    def __post_init__(self):
        """Auto-generate reverse mappings if not provided."""
        if not self.idx_to_microbe:
            self.idx_to_microbe = {v: k for k, v in self.microbe_to_idx.items()}
        if not self.idx_to_metabolite:
            self.idx_to_metabolite = {v: k for k, v in self.metabolite_to_idx.items()}
    
    def get_microbe_positives(self, organism_id: str) -> set[str]:
        """Get positive metabolite names for an organism."""
        microbe_idx = self.microbe_to_idx.get(organism_id)
        if microbe_idx is None:
            return set()
        metabolite_indices = self.positives.get(microbe_idx, set())
        return {self.idx_to_metabolite[idx] for idx in metabolite_indices}
    
    def get_metabolite_positives(self, metabolite_name: str) -> set[str]:
        """Get organisms that have this metabolite as positive."""
        metabolite_idx = self.metabolite_to_idx.get(metabolite_name)
        if metabolite_idx is None:
            return set()
        organism_ids = set()
        for microbe_idx, pos_set in self.positives.items():
            if metabolite_idx in pos_set:
                organism_ids.add(self.idx_to_microbe[microbe_idx])
        return organism_ids


@dataclass
class ModelArtifacts:
    """Complete trained model with all metadata.
    
    This wraps the trained model along with all necessary context, eliminating
    the need to pass around separate model, mappings, and taxonomy objects.
    """
    # Core model
    model: 'BaseEmbeddingModel'  # Forward reference since models not yet defined
    
    # Training context
    training_data: TrainingData
    
    # Aggregated embeddings
    taxonomy_embeddings: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # level -> taxon -> embedding
    
    # Model metadata
    embedding_dim: int = 32
    model_type: str = "BPR"
    
    def get_embedding(self, level: str, name: str) -> np.ndarray:
        """Get embedding at any taxonomic level including isolate.
        
        Args:
            level: "isolate", "species", "genus", "family", "order", "class", "kingdom"
            name: The name of the isolate or taxon
        
        Returns:
            Embedding vector
        """
        if level == "isolate":
            idx = self.training_data.microbe_to_idx.get(name)
            if idx is None:
                raise ValueError(f"Unknown isolate: {name}")
            return self.model.microbe_embeddings[idx]
        else:
            if level not in self.taxonomy_embeddings:
                raise ValueError(f"No embeddings computed for level '{level}'. Call compute_taxonomy_embeddings() first.")
            embedding = self.taxonomy_embeddings[level].get(name)
            if embedding is None:
                raise ValueError(f"Unknown {level}: {name}")
            return embedding
    
    def get_metabolite_embedding(self, metabolite_name: str) -> np.ndarray:
        """Get the embedding vector for a specific metabolite."""
        idx = self.training_data.metabolite_to_idx.get(metabolite_name)
        if idx is None:
            raise ValueError(f"Unknown metabolite: {metabolite_name}")
        return self.model.metabolite_embeddings[idx]
    
    def _taxon_score_sd(self, level: str, name: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute per-compound SD and SE of BPR scores arising from isolate embedding variance.

        For a genus (or any aggregated taxon) with K isolate embeddings E (K×d):
          - Sample covariance: C = Cov(E)  [d×d, ddof=1]
          - For all compounds V (n_mets×d) simultaneously:
              var_isolate[c] = v_c^T C v_c = diag(V C V^T)[c]
              score_sd[c]   = sqrt(var_isolate[c])   — SD across isolates
              score_se[c]   = score_sd[c] / sqrt(K)  — SE of the genus mean

        Returns (score_sd, score_se) arrays of shape (n_metabolites,), or (None, None)
        for isolate-level queries or taxa with only one member.
        """
        if level == "isolate":
            return None, None

        isolate_ids = [
            org_id for org_id, tax in self.training_data.microbe_taxonomy.items()
            if tax.get_level(level) == name and org_id in self.training_data.microbe_to_idx
        ]
        K = len(isolate_ids)
        if K < 2:
            return None, None

        indices = [self.training_data.microbe_to_idx[oid] for oid in isolate_ids]
        E = self.model.microbe_embeddings[indices]          # (K, d)
        V = self.model.metabolite_embeddings                # (n_mets, d)

        C = np.cov(E.T)                                    # (d, d), ddof=1
        VC = V @ C                                         # (n_mets, d)
        var_isolate = np.maximum((VC * V).sum(axis=1), 0) # (n_mets,) — clamp numerical noise
        score_sd = np.sqrt(var_isolate)
        score_se = score_sd / np.sqrt(K)
        return score_sd, score_se

    def _taxon_embedding_diversity(self, level: str, name: str) -> Optional[float]:
        """Mean pairwise cosine distance between isolate embeddings within a taxon.

        Measures how heterogeneous the genus's isolates are in BPR embedding space.
        Validated against empirical within-genus metabolic consistency (GCS):
        Spearman ρ = -0.73, p < 10⁻⁸ across 48 genera — genera with high
        embedding_diversity correspond to metabolically heterogeneous genera where
        genus-level predictions are less reliable.

        Complementary to _taxon_score_sd: score_sd captures per-compound variance
        (V_i projected through Cov(E)), while embedding_diversity is a genus-level
        scalar summarising overall isolate heterogeneity in embedding space.

        Returns:
            float in [0, 1]: 0 = all isolates identical in embedding space,
                             1 = maximally diverse (orthogonal embeddings).
            None for isolate-level queries or taxa with < 2 members in the model.
        """
        if level == "isolate":
            return None

        isolate_ids = [
            org_id for org_id, tax in self.training_data.microbe_taxonomy.items()
            if tax.get_level(level) == name and org_id in self.training_data.microbe_to_idx
        ]
        K = len(isolate_ids)
        if K < 2:
            return None

        indices = [self.training_data.microbe_to_idx[oid] for oid in isolate_ids]
        E = self.model.microbe_embeddings[indices]          # (K, d)

        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)
        E_norm = E / norms                                  # (K, d)

        # Pairwise cosine similarity matrix, upper-triangle mean
        sims = E_norm @ E_norm.T                            # (K, K)
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        mean_sim = float(sims[mask].mean())

        return round(1.0 - mean_sim, 4)                    # cosine distance

    def predict_for_taxon(self, level: str, name: str) -> np.ndarray:
        """Get prediction scores at any taxonomic level including isolate.
        
        Args:
            level: "isolate", "species", "genus", "family", "order", "class", "kingdom"
            name: The name of the isolate or taxon
        
        Returns:
            Array of prediction scores for all metabolites
        """
        embedding = self.get_embedding(level, name)
        return self.model.predict_scores_from_embedding(embedding)
    
    def predict_with_stats(
        self,
        level: str,
        name: str,
        compute_distinctive: bool = True,
        background_level: Optional[str] = None,
    ) -> 'PredictionResult':
        """Get comprehensive prediction results with statistics.
        
        Args:
            level: "isolate", "species", "genus", "family", "order", "class", "kingdom"
            name: Name of the isolate or taxon
            compute_distinctive: Whether to compute distinctive scores
            background_level: Level for background comparison (e.g., "genus" for isolate)
                            If None, uses parent level of query level
        
        Returns:
            PredictionResult with comprehensive statistics
        """
        # Get raw scores
        raw_scores = self.predict_for_taxon(level, name)
        
        # Get metabolite names
        idx_to_metabolite = {idx: name for name, idx in self.training_data.metabolite_to_idx.items()}
        metabolite_names = [idx_to_metabolite[i] for i in range(len(raw_scores))]
        
        # Compute rankings
        ranks = np.argsort(np.argsort(raw_scores)[::-1]) + 1  # 1-indexed
        percentiles = 100 * (1 - (ranks - 1) / len(raw_scores))
        
        # Basic statistics
        mean_score = float(np.mean(raw_scores))
        median_score = float(np.median(raw_scores))
        std_score = float(np.std(raw_scores))
        
        # Get training positives count
        n_positive_training = 0
        if level == "isolate":
            positives = self.training_data.positives.get(
                self.training_data.microbe_to_idx.get(name, -1),
                set()
            )
            n_positive_training = len(positives)
        
        # Compute distinctive scores and background stats
        distinctive_scores = None
        background_mean = None
        background_std = None
        stderr_score = None
        n_isolates_aggregated = 1
        
        if compute_distinctive:
            # Determine background
            if background_level is None and level == "isolate":
                # Use genus as default background for isolates
                taxonomy = self.training_data.microbe_taxonomy.get(name)
                if taxonomy and taxonomy.genus:
                    background_level = "genus"
                    background_taxon = taxonomy.genus
                else:
                    background_level = None
            elif background_level and level == "isolate":
                taxonomy = self.training_data.microbe_taxonomy.get(name)
                if taxonomy:
                    background_taxon = taxonomy.get_level(background_level)
                else:
                    background_level = None
            else:
                background_level = None
            
            if background_level and background_level in self.taxonomy_embeddings:
                # Compute background scores
                if level == "isolate" and background_taxon:
                    # Get all isolates in the same background taxon
                    taxon_isolates = [
                        org_id for org_id, tax in self.training_data.microbe_taxonomy.items()
                        if tax.get_level(background_level) == background_taxon
                    ]
                    
                    if len(taxon_isolates) > 1:
                        taxon_indices = [
                            self.training_data.microbe_to_idx[org_id]
                            for org_id in taxon_isolates
                            if org_id in self.training_data.microbe_to_idx
                        ]
                        
                        # Compute background scores
                        background_embeddings = self.model.microbe_embeddings[taxon_indices]
                        background_scores = background_embeddings @ self.model.metabolite_embeddings.T
                        
                        background_mean_vec = background_scores.mean(axis=0)
                        background_std_vec = background_scores.std(axis=0)
                        
                        distinctive_scores = raw_scores - background_mean_vec
                        background_mean = float(np.mean(background_mean_vec))
                        background_std = float(np.mean(background_std_vec))
                        stderr_score = float(std_score / np.sqrt(len(taxon_isolates)))
                        n_isolates_aggregated = len(taxon_isolates)
            
            # If we couldn't compute distinctive scores, use global background
            if distinctive_scores is None:
                global_mean = np.mean(raw_scores)
                distinctive_scores = raw_scores - global_mean
                background_mean = float(global_mean)
                background_std = float(std_score)
        
        # Per-compound uncertainty from isolate embedding variance
        score_sd, score_se = self._taxon_score_sd(level, name)

        return PredictionResult(
            level=level,
            name=name,
            metabolite_names=metabolite_names,
            metabolite_indices=np.arange(len(raw_scores)),
            raw_scores=raw_scores,
            distinctive_scores=distinctive_scores,
            ranks=ranks,
            percentiles=percentiles,
            mean_score=mean_score,
            median_score=median_score,
            std_score=std_score,
            stderr_score=stderr_score,
            n_metabolites=len(raw_scores),
            n_positive_training=n_positive_training,
            n_isolates_aggregated=n_isolates_aggregated,
            background_mean=background_mean,
            background_std=background_std,
            score_sd=score_sd,
            score_se=score_se,
        )
    
    def get_taxa_at_level(self, level: str) -> list[str]:
        """Get all taxon names at a given taxonomic level."""
        taxa = set()
        for org_id, taxonomy in self.training_data.microbe_taxonomy.items():
            taxon = taxonomy.get_level(level)
            if taxon and isinstance(taxon, str):  # Filter out None and non-strings
                taxa.add(taxon)
        return sorted(taxa)


@dataclass
class PredictionResult:
    """Rich prediction results with statistics and rankings.
    
    Contains comprehensive information about predictions including raw scores,
    distinctive scores, rankings, and statistics.
    """
    # Query context (required)
    level: str  # "isolate", "genus", "family", etc.
    name: str   # Name of isolate or taxon
    
    # Metabolite information (required)
    metabolite_names: list[str] = field(default_factory=list)  # Names in score order
    metabolite_indices: np.ndarray = field(default_factory=lambda: np.array([]))  # Original indices
    raw_scores: np.ndarray = field(default_factory=lambda: np.array([]))  # Raw prediction scores
    
    # Counts and basic statistics (with defaults)
    n_metabolites: int = 0
    mean_score: float = 0.0
    median_score: float = 0.0
    std_score: float = 0.0
    
    # Optional fields
    distinctive_scores: Optional[np.ndarray] = None  # vs background
    ranks: Optional[np.ndarray] = None  # 1-indexed ranks
    percentiles: Optional[np.ndarray] = None  # 0-100 percentiles
    stderr_score: Optional[float] = None  # For aggregated predictions
    n_positive_training: int = 0  # Known positives in training data
    n_isolates_aggregated: int = 1  # For taxon-level predictions
    background_mean: Optional[float] = None
    background_std: Optional[float] = None
    # Per-compound uncertainty from isolate embedding variance (None for isolate-level queries)
    score_sd: Optional[np.ndarray] = None  # SD of individual isolate scores for each compound
    score_se: Optional[np.ndarray] = None  # SE of the genus-mean score (score_sd / sqrt(K))
    
    def top_k(self, k: int = 20, by: str = "raw") -> "PredictionResult":
        """Return top K metabolites by specified score type.
        
        Args:
            k: Number of top results
            by: "raw" or "distinctive"
        
        Returns:
            New PredictionResult with only top K metabolites
        """
        scores = self.raw_scores if by == "raw" else self.distinctive_scores
        if scores is None:
            raise ValueError(f"No {by} scores available")
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        return PredictionResult(
            level=self.level,
            name=self.name,
            metabolite_names=[self.metabolite_names[i] for i in top_indices],
            metabolite_indices=self.metabolite_indices[top_indices],
            raw_scores=self.raw_scores[top_indices],
            distinctive_scores=self.distinctive_scores[top_indices] if self.distinctive_scores is not None else None,
            ranks=self.ranks[top_indices] if self.ranks is not None else None,
            percentiles=self.percentiles[top_indices] if self.percentiles is not None else None,
            mean_score=self.mean_score,
            median_score=self.median_score,
            std_score=self.std_score,
            stderr_score=self.stderr_score,
            n_metabolites=self.n_metabolites,
            n_positive_training=self.n_positive_training,
            n_isolates_aggregated=self.n_isolates_aggregated,
            background_mean=self.background_mean,
            background_std=self.background_std,
            score_sd=self.score_sd[top_indices] if self.score_sd is not None else None,
            score_se=self.score_se[top_indices] if self.score_se is not None else None,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level,
            "name": self.name,
            "metabolites": [
                {
                    "name": name,
                    "raw_score": float(self.raw_scores[i]),
                    "distinctive_score": float(self.distinctive_scores[i]) if self.distinctive_scores is not None else None,
                    "rank": int(self.ranks[i]) if self.ranks is not None else None,
                    "percentile": float(self.percentiles[i]) if self.percentiles is not None else None,
                    "score_sd": float(self.score_sd[i]) if self.score_sd is not None else None,
                    "score_se": float(self.score_se[i]) if self.score_se is not None else None,
                }
                for i, name in enumerate(self.metabolite_names)
            ],
            "statistics": {
                "mean_score": self.mean_score,
                "median_score": self.median_score,
                "std_score": self.std_score,
                "stderr_score": self.stderr_score,
                "n_metabolites": self.n_metabolites,
                "n_positive_training": self.n_positive_training,
                "n_isolates_aggregated": self.n_isolates_aggregated,
                "background_mean": self.background_mean,
                "background_std": self.background_std,
            }
        }

    
    # Metadata
    query_id: str = ""  # e.g., "genus:Burkholderia"
    metabolite_names: list[str] = field(default_factory=list)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for easy analysis."""
        import pandas as pd
        
        data = {
            'metabolite': self.metabolite_names,
            'raw_score': self.raw_scores,
        }
        
        if self.distinctive_scores is not None:
            data['distinctive_score'] = self.distinctive_scores
        
        if self.uncertainty is not None:
            data['uncertainty'] = self.uncertainty
        
        df = pd.DataFrame(data)
        return df.sort_values('raw_score', ascending=False)


@dataclass
class QueryRequest:
    """Unified query format for CLI and REST APIs."""
    command: str  # 'microbes', 'compounds', 'selective', etc.
    
    # Model selection
    model_type: str = 'bpr'
    preference_type: str = 'consumed'
    
    # Query parameters
    compounds: Optional[list[str]] = None
    microbes: Optional[list[str]] = None  # Format: "level:name"
    
    # Aggregation
    level: str = 'genus'
    agg_method: str = 'mean'  # 'mean', 'median', 'max'
    
    # Output
    top_k: int = 10
    include_uncertainty: bool = False
    include_distinctive: bool = False


@dataclass
class QueryResponse:
    """Unified response format for CLI and REST APIs."""
    success: bool
    query_type: str
    results: list[dict]
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'success': self.success,
            'query_type': self.query_type,
            'results': self.results,
            'error': self.error,
            'metadata': self.metadata,
        }
