"""Query configuration and request types using dataclasses.

Defines all query parameters with defaults, eliminating argument drilling
and providing a single source of truth for CLI, API, and programmatic usage.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


# =============================================================================
# Query Configuration
# =============================================================================

@dataclass
class QueryConfig:
    """Global configuration for query system."""
    model_type: str = "bpr"  # "bpr", "wrmf", "svd"
    model_dir: str = "trained_models/latest"
    interaction_type: Literal["auto", "consumed", "secreted"] = "auto"
    output_format: Literal["json", "table", "dataframe"] = "json"
    verbose: bool = False
    explain: bool = False  # Show detailed computation steps
    
    # Aggregation defaults
    default_level: str = "genus"
    default_top_k: int = 10
    default_agg_method: Literal["mean", "sum", "max"] = "mean"
    
    # Score computation
    compute_distinctive: bool = True
    background_level: Optional[str] = None  # Auto-determine if None


# =============================================================================
# Query Request Types
# =============================================================================

@dataclass
class CompoundsForMicrobesQuery:
    """Query: What compounds are consumed by these microbes?"""
    # Required
    microbes: list[tuple[str, str]]  # [(level, name), ...] e.g., [("genus", "Burkholderia")]
    
    # Optional parameters
    top_k: int = 20
    compute_distinctive: bool = True
    background_level: Optional[str] = None
    min_score: Optional[float] = None
    explain: bool = False  # Show computation details
    
    # Metadata
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_string(cls, microbes_str: str, **kwargs) -> "CompoundsForMicrobesQuery":
        """Parse from CLI format: 'genus:Burkholderia,isolate:XYZ'."""
        microbes = []
        for part in microbes_str.split(','):
            part = part.strip()
            if not part:
                continue
            
            if ':' in part:
                level, name = part.split(':', 1)
                microbes.append((level.strip().lower(), name.strip()))
            else:
                microbes.append(('isolate', part))
        
        return cls(microbes=microbes, **kwargs)


@dataclass
class MicrobesForCompoundsQuery:
    """Query: Which microbes consume these compounds?"""
    # Required
    compounds: list[str]  # Compound names to query
    
    # Optional parameters
    level: str = "genus"  # Taxonomic level to aggregate at
    top_k: int = 10
    agg_method: Literal["mean", "sum", "max"] = "mean"
    min_score: Optional[float] = None
    explain: bool = False  # Show computation details
    
    # Metadata
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_string(cls, compounds_str: str, **kwargs) -> "MicrobesForCompoundsQuery":
        """Parse from CLI format: 'glutamate,serine,guanine'."""
        compounds = [c.strip() for c in compounds_str.split(',') if c.strip()]
        return cls(compounds=compounds, **kwargs)


@dataclass
class SelectiveMetabolitesQuery:
    """Query: Find metabolites that favor target over competitors."""
    # Required
    target: tuple[str, str]  # (level, name)
    competitors: list[tuple[str, str]]  # [(level, name), ...]
    
    # Optional parameters
    top_k: int = 20
    method: Literal["difference", "ratio"] = "difference"
    min_selectivity: Optional[float] = None
    use_distinctive: bool = False  # Use distinctive scores instead of raw
    explain: bool = False
    
    # Metadata
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_strings(
        cls,
        target_str: str,
        competitors_str: str,
        **kwargs
    ) -> "SelectiveMetabolitesQuery":
        """Parse from CLI format."""
        # Parse target
        if ':' in target_str:
            level, name = target_str.split(':', 1)
            target = (level.strip().lower(), name.strip())
        else:
            target = ('isolate', target_str)
        
        # Parse competitors
        competitors = []
        for part in competitors_str.split(','):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                level, name = part.split(':', 1)
                competitors.append((level.strip().lower(), name.strip()))
            else:
                competitors.append(('isolate', part))
        
        return cls(target=target, competitors=competitors, **kwargs)


@dataclass
class MetaboliteEnvironmentFitnessQuery:
    """Query: Rank microbes by fitness in a metabolite environment."""
    # Required
    metabolites: list[str]  # Available metabolites in environment
    
    # Optional parameters
    level: str = "genus"
    top_k: int = 10
    fitness_method: Literal["sum", "mean", "weighted"] = "sum"
    use_distinctive: bool = False  # Use distinctive scores instead of raw
    explain: bool = False
    
    # Optional: restrict to specific microbes
    microbes: Optional[list[tuple[str, str]]] = None
    
    # Metadata
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_strings(
        cls,
        metabolites_str: str,
        microbes_str: Optional[str] = None,
        **kwargs
    ) -> "MetaboliteEnvironmentFitnessQuery":
        """Parse from CLI format."""
        metabolites = [m.strip() for m in metabolites_str.split(',') if m.strip()]
        
        microbes = None
        if microbes_str:
            microbes = []
            for part in microbes_str.split(','):
                part = part.strip()
                if not part:
                    continue
                if ':' in part:
                    level, name = part.split(':', 1)
                    microbes.append((level.strip().lower(), name.strip()))
                else:
                    microbes.append(('isolate', part))
        
        return cls(metabolites=metabolites, microbes=microbes, **kwargs)


@dataclass
class NicheMetabolitesQuery:
    """Query: Find unique metabolite preferences vs background."""
    # Required
    microbe: tuple[str, str]  # (level, name)
    
    # Optional parameters
    top_k: int = 20
    background_level: Optional[str] = None  # Auto-determine if None
    min_distinctive_score: Optional[float] = None
    explain: bool = False  # Show computation details
    
    # Metadata
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_string(cls, microbe_str: str, **kwargs) -> "NicheMetabolitesQuery":
        """Parse from CLI format."""
        if ':' in microbe_str:
            level, name = microbe_str.split(':', 1)
            microbe = (level.strip().lower(), name.strip())
        else:
            microbe = ('isolate', microbe_str)
        
        return cls(microbe=microbe, **kwargs)


@dataclass
class CompetitionAnalysisQuery:
    """Query: Analyze competition between microbes for metabolites."""
    microbes: list[tuple[str, str]]  # [(level, name), ...]
    metabolites: list[str]  # Compound names
    use_distinctive: bool = False  # Use distinctive scores instead of raw
    explain: bool = False
    
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_string(cls, microbes_str: str, metabolites_str: str, **kwargs) -> "CompetitionAnalysisQuery":
        """Parse from CLI format."""
        microbes = []
        for part in microbes_str.split(','):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                level, name = part.split(':', 1)
                microbes.append((level.strip().lower(), name.strip()))
            else:
                microbes.append(('isolate', part))
        
        metabolites = [m.strip() for m in metabolites_str.split(',') if m.strip()]
        
        return cls(microbes=microbes, metabolites=metabolites, **kwargs)


@dataclass
class RankSubstratesQuery:
    """Query: Rank specific substrates for specific microbes."""
    microbes: list[tuple[str, str]]  # [(level, name), ...]
    substrates: list[str]  # Compound names to rank
    top_k: int = 10
    use_distinctive: bool = False  # Use distinctive scores instead of raw
    agg_method: Optional[Literal["mean", "median", "min", "max"]] = None  # For aggregating across microbes
    explain: bool = False
    
    config: QueryConfig = field(default_factory=QueryConfig)
    
    @classmethod
    def from_string(cls, microbes_str: str, substrates_str: str, **kwargs) -> "RankSubstratesQuery":
        """Parse from CLI format."""
        microbes = []
        for part in microbes_str.split(','):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                level, name = part.split(':', 1)
                microbes.append((level.strip().lower(), name.strip()))
            else:
                microbes.append(('isolate', part))
        
        substrates = [s.strip() for s in substrates_str.split(',') if s.strip()]
        
        return cls(microbes=microbes, substrates=substrates, **kwargs)


@dataclass
class SearchCompoundQuery:
    """Query: Search for compound name matches."""
    query: str
    fuzzy: bool = True
    max_results: int = 10
    min_score: float = 0.7  # Fuzzy matching score threshold
    
    config: QueryConfig = field(default_factory=QueryConfig)


@dataclass
class ListTaxaQuery:
    """Query: List available taxa at a given level."""
    level: str = "genus"
    include_counts: bool = True  # Include organism counts
    
    config: QueryConfig = field(default_factory=QueryConfig)


@dataclass
class ModelInfoQuery:
    """Query: Get model information and statistics."""
    include_stats: bool = True
    
    config: QueryConfig = field(default_factory=QueryConfig)


# =============================================================================
# Query Response Types
# =============================================================================

@dataclass
class QueryResponse:
    """Base response type for all queries."""
    success: bool
    query_type: str
    results: list[dict]
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "query_type": self.query_type,
            "results": self.results,
            "metadata": self.metadata,
            "error": self.error,
        }
