# gtdb.py
"""GTDB taxonomy integration for phylogenetic-aware predictions."""

import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from config import CACHE_FILES

logger = logging.getLogger(__name__)

# GTDB release 220 (April 2024)
GTDB_RELEASE = "latest"
GTDB_BASE_URL = f"https://data.gtdb.ecogenomic.org/releases/{GTDB_RELEASE}"
GTDB_BAC_TAX_URL = f"{GTDB_BASE_URL}/bac120_taxonomy.tsv.gz"
GTDB_ARC_TAX_URL = f"{GTDB_BASE_URL}/ar53_taxonomy.tsv.gz"

GTDB_CACHE_DIR = CACHE_FILES["bpr_model"].parent / "gtdb"
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


def download_gtdb(force: bool = False) -> dict[str, Path]:
    """
    Download GTDB taxonomy files.
    
    Args:
        force: Re-download even if cached
    
    Returns:
        Dict with paths to downloaded files
    """
    GTDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    files = {
        "bacteria": GTDB_CACHE_DIR / "bac120_taxonomy.tsv.gz",
        "archaea": GTDB_CACHE_DIR / "ar53_taxonomy.tsv.gz",
    }
    urls = {
        "bacteria": GTDB_BAC_TAX_URL,
        "archaea": GTDB_ARC_TAX_URL,
    }
    
    for key, path in files.items():
        if force or not path.exists():
            logger.info(f"Downloading GTDB {key} taxonomy...")
            try:
                urllib.request.urlretrieve(urls[key], path)
                logger.info(f"  Saved to {path}")
            except Exception as e:
                logger.warning(f"  Failed to download {key}: {e}")
    
    return files


def parse_gtdb_lineage(tax_string: str) -> dict[str, str]:
    """
    Parse GTDB taxonomy string into dict.
    
    Input: "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;..."
    Output: {"domain": "Bacteria", "phylum": "Proteobacteria", ...}
    """
    result = {}
    prefixes = {"d": "domain", "p": "phylum", "c": "class", 
                "o": "order", "f": "family", "g": "genus", "s": "species"}
    
    for part in tax_string.split(";"):
        if "__" in part:
            prefix, name = part.split("__", 1)
            rank = prefixes.get(prefix)
            if rank and name:  # Skip empty names
                result[rank] = name
    
    return result


def load_gtdb_taxonomy(force_download: bool = False) -> pd.DataFrame:
    """
    Load GTDB taxonomy as DataFrame.
    
    Returns:
        DataFrame with columns: genome_id, domain, phylum, class, order, family, genus, species
    """
    cache_path = GTDB_CACHE_DIR / "gtdb_taxonomy_parsed.parquet"
    
    if not force_download and cache_path.exists():
        logger.info("Loading cached GTDB taxonomy...")
        return pd.read_parquet(cache_path)
    
    files = download_gtdb(force=force_download)
    
    all_records = []
    
    for domain_key, path in files.items():
        if not path.exists():
            logger.warning(f"  Missing {path}, skipping")
            continue
        
        logger.info(f"Parsing {path.name}...")
        
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                
                genome_id, tax_string = parts
                lineage = parse_gtdb_lineage(tax_string)
                lineage["genome_id"] = genome_id
                all_records.append(lineage)
    
    df = pd.DataFrame(all_records)
    
    # Reorder columns
    cols = ["genome_id"] + RANKS
    df = df[[c for c in cols if c in df.columns]]
    
    logger.info(f"  Loaded {len(df)} genomes")
    
    # Cache parsed result
    df.to_parquet(cache_path)
    logger.info(f"  Cached to {cache_path}")
    
    return df


def build_taxonomy_lookup(gtdb_df: pd.DataFrame) -> dict[str, dict[str, set]]:
    """
    Build lookup tables for fast taxonomy queries.
    
    Returns:
        {
            "genus": {"Burkholderia": {"Burkholderiales", ...parent taxa...}},
            "family": {...},
            ...
        }
    """
    lookup = {rank: {} for rank in RANKS if rank != "species"}
    
    for _, row in gtdb_df.iterrows():
        lineage = {r: row.get(r) for r in RANKS if pd.notna(row.get(r))}
        
        # For each rank, store its parent lineage
        for rank in lookup.keys():
            name = lineage.get(rank)
            if name and name not in lookup[rank]:
                lookup[rank][name] = lineage.copy()
    
    return lookup


def get_lineage(
    taxon_name: str,
    rank: str,
    gtdb_df: pd.DataFrame,
) -> dict[str, str] | None:
    """
    Get full lineage for a taxon.
    
    Args:
        taxon_name: e.g., "Burkholderia"
        rank: e.g., "genus"
        gtdb_df: GTDB taxonomy dataframe
    
    Returns:
        {"domain": "Bacteria", "phylum": "Proteobacteria", ...} or None
    """
    matches = gtdb_df[gtdb_df[rank] == taxon_name]
    
    if matches.empty:
        return None
    
    row = matches.iloc[0]
    return {r: row[r] for r in RANKS if pd.notna(row.get(r))}


def taxonomic_distance(lineage1: dict, lineage2: dict) -> int:
    """
    Compute taxonomic distance between two lineages.
    
    Distance = number of ranks from most specific shared rank to genus level.
    0 = same genus, 1 = same family different genus, etc.
    """
    # Find most specific shared rank
    for i, rank in enumerate(reversed(RANKS[:-1])):  # genus -> domain
        if lineage1.get(rank) and lineage1.get(rank) == lineage2.get(rank):
            # Distance is how far from genus
            genus_idx = RANKS.index("genus")
            rank_idx = RANKS.index(rank)
            return genus_idx - rank_idx
    
    return len(RANKS)  # No shared taxonomy


def find_nearest_measured(
    query_lineage: dict[str, str],
    measured_taxa: dict[str, dict[str, str]],  # {taxon_name: lineage}
    max_distance: int = 4,
) -> list[dict]:
    """
    Find measured taxa closest to query in taxonomic tree.
    
    Args:
        query_lineage: {"domain": "Bacteria", "phylum": ..., "genus": ...}
        measured_taxa: Dict mapping taxon names to their lineages
        max_distance: Maximum taxonomic distance to consider
    
    Returns:
        List of {"taxon": str, "rank": str, "distance": int, "shared_rank": str}
        sorted by distance ascending.
    """
    results = []
    
    for taxon_name, taxon_lineage in measured_taxa.items():
        distance = taxonomic_distance(query_lineage, taxon_lineage)
        
        if distance <= max_distance:
            # Find the shared rank
            shared_rank = None
            for rank in reversed(RANKS[:-1]):
                if query_lineage.get(rank) and query_lineage.get(rank) == taxon_lineage.get(rank):
                    shared_rank = rank
                    break
            
            results.append({
                "taxon": taxon_name,
                "rank": "genus",  # We're matching at genus level in measured_taxa
                "distance": distance,
                "shared_rank": shared_rank,
                "shared_value": query_lineage.get(shared_rank) if shared_rank else None,
            })
    
    results.sort(key=lambda x: x["distance"])
    return results


class GTDBTaxonomy:
    """
    High-level interface for GTDB taxonomy queries.
    
    Usage:
        gtdb = GTDBTaxonomy()
        gtdb.load()
        
        lineage = gtdb.get_lineage("Burkholderia", "genus")
        neighbors = gtdb.find_neighbors(lineage, measured_genera)
    """
    
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.lookup: dict | None = None
        self._loaded = False
    
    def load(self, force_download: bool = False) -> "GTDBTaxonomy":
        """Load GTDB taxonomy data."""
        if self._loaded and not force_download:
            return self
        
        self.df = load_gtdb_taxonomy(force_download=force_download)
        self.lookup = build_taxonomy_lookup(self.df)
        self._loaded = True
        
        return self
    
    def get_lineage(self, taxon_name: str, rank: str = "genus") -> dict[str, str] | None:
        """Get full lineage for a taxon."""
        if not self._loaded:
            self.load()
        
        return get_lineage(taxon_name, rank, self.df)
    
    def find_neighbors(
        self,
        query_lineage: dict[str, str],
        measured_taxa: dict[str, dict[str, str]],
        max_distance: int = 4,
    ) -> list[dict]:
        """Find nearest measured taxa to query."""
        if not self._loaded:
            self.load()
        
        return find_nearest_measured(query_lineage, measured_taxa, max_distance)
    
    def search(
        self,
        query: str,
        rank: str = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for taxa by name (exact substring match).
        
        Args:
            query: Search string
            rank: Limit to specific rank, or None for all
            limit: Max results
        
        Returns:
            List of {"name": str, "rank": str, "lineage": dict}
        """
        if not self._loaded:
            self.load()
        
        query_lower = query.lower()
        results = []
        
        ranks_to_search = [rank] if rank else ["genus", "family", "order", "class", "phylum"]
        
        for r in ranks_to_search:
            if r not in self.lookup:
                continue
            
            for name, lineage in self.lookup[r].items():
                if query_lower in name.lower():
                    results.append({
                        "name": name,
                        "rank": r,
                        "lineage": lineage,
                    })
                    
                    if len(results) >= limit:
                        return results
        
        return results
    
    def get_all_at_rank(self, rank: str) -> list[str]:
        """Get all unique taxa at a given rank."""
        if not self._loaded:
            self.load()
        
        if rank not in self.lookup:
            return []
        
        return list(self.lookup[rank].keys())
    
    def get_children(
        self,
        taxon_name: str,
        taxon_rank: str,
        child_rank: str,
    ) -> list[str]:
        """
        Get all taxa at child_rank that belong to taxon_name.
        
        Example:
            get_children("Burkholderiales", "order", "genus")
            -> ["Burkholderia", "Paraburkholderia", "Caballeronia", ...]
        """
        if not self._loaded:
            self.load()
        
        matches = self.df[self.df[taxon_rank] == taxon_name]
        return matches[child_rank].dropna().unique().tolist()


# Singleton instance for convenience
_gtdb_instance: GTDBTaxonomy | None = None


def get_gtdb() -> GTDBTaxonomy:
    """Get or create singleton GTDBTaxonomy instance."""
    global _gtdb_instance
    if _gtdb_instance is None:
        _gtdb_instance = GTDBTaxonomy()
    return _gtdb_instance


def build_measured_taxa_lookup(
    wrapper,  # BPRModelWrapper
    gtdb: GTDBTaxonomy,
) -> dict[str, dict[str, str]]:
    """
    Build lookup mapping measured genera to their GTDB lineages.
    
    Args:
        wrapper: BPRModelWrapper with taxonomy_embeddings
        gtdb: Loaded GTDBTaxonomy instance
    
    Returns:
        {"Burkholderia": {"domain": "Bacteria", "phylum": ...}, ...}
    """
    measured = {}
    
    if wrapper.taxonomy_embeddings is None:
        return measured
    
    for genus_name in wrapper.taxonomy_embeddings.get("genus", {}).keys():
        lineage = gtdb.get_lineage(genus_name, "genus")
        if lineage:
            measured[genus_name] = lineage
        else:
            # Taxon not in GTDB, use what we have from our data
            # Find an organism with this genus
            for org_id, tax in wrapper.microbe_taxonomy.items():
                if tax.get("genus") == genus_name:
                    measured[genus_name] = {
                        "genus": tax.get("genus"),
                        "family": tax.get("family"),
                        "order": tax.get("order"),
                        "class": tax.get("class"),
                        "kingdom": tax.get("kingdom"),
                    }
                    break
    
    return measured


def predict_embedding_phylogenetic(
    taxonomy: dict[str, str],
    wrapper,  # BPRModelWrapper
    gtdb: GTDBTaxonomy = None,
    max_distance: int = 4,
    top_k_neighbors: int = 10,
) -> dict:
    """
    Predict embedding for unmeasured taxon using phylogenetic neighbors.
    
    Args:
        taxonomy: {"genus": "SomeGenus", "family": "SomeFamily", ...}
        wrapper: BPRModelWrapper with taxonomy_embeddings
        gtdb: GTDBTaxonomy instance (loaded automatically if None)
        max_distance: Maximum taxonomic distance for neighbors
        top_k_neighbors: Max neighbors to use for averaging
    
    Returns:
        {
            "embedding": np.ndarray or None,
            "neighbors": [{"taxon": str, "distance": int, "weight": float}, ...],
            "confidence": float,  # 0-1
            "method": str,  # "exact", "phylogenetic", "none"
        }
    """
    # Check if we have exact match first
    query_genus = taxonomy.get("genus")
    
    if query_genus and wrapper.taxonomy_embeddings:
        if query_genus in wrapper.taxonomy_embeddings.get("genus", {}):
            return {
                "embedding": wrapper.taxonomy_embeddings["genus"][query_genus],
                "neighbors": [{"taxon": query_genus, "distance": 0, "weight": 1.0}],
                "confidence": 1.0,
                "method": "exact",
            }
    
    # Load GTDB if needed
    if gtdb is None:
        gtdb = get_gtdb()
        gtdb.load()
    
    # Get query lineage from GTDB or use provided taxonomy
    if query_genus:
        query_lineage = gtdb.get_lineage(query_genus, "genus")
    
    if not query_lineage:
        # Fall back to provided taxonomy
        query_lineage = taxonomy
    
    if not query_lineage:
        return {
            "embedding": None,
            "neighbors": [],
            "confidence": 0.0,
            "method": "none",
        }
    
    # Build measured taxa lookup
    measured = build_measured_taxa_lookup(wrapper, gtdb)
    
    if not measured:
        return {
            "embedding": None,
            "neighbors": [],
            "confidence": 0.0,
            "method": "none",
        }
    
    # Find neighbors
    neighbors = find_nearest_measured(query_lineage, measured, max_distance)
    
    if not neighbors:
        return {
            "embedding": None,
            "neighbors": [],
            "confidence": 0.0,
            "method": "none",
        }
    
    # Limit to top_k
    neighbors = neighbors[:top_k_neighbors]
    
    # Compute weighted average embedding
    embeddings = []
    weights = []
    neighbor_info = []
    
    for n in neighbors:
        taxon = n["taxon"]
        distance = n["distance"]
        
        emb = wrapper.taxonomy_embeddings["genus"].get(taxon)
        if emb is not None:
            # Weight: inverse of (distance + 1), squared for stronger locality
            weight = 1.0 / ((distance + 1) ** 2)
            embeddings.append(emb)
            weights.append(weight)
            neighbor_info.append({
                "taxon": taxon,
                "distance": distance,
                "shared_rank": n["shared_rank"],
                "shared_value": n["shared_value"],
                "weight": weight,  # Will normalize later
            })
    
    if not embeddings:
        return {
            "embedding": None,
            "neighbors": [],
            "confidence": 0.0,
            "method": "none",
        }
    
    # Normalize weights
    weights = np.array(weights)
    weights /= weights.sum()
    
    for i, w in enumerate(weights):
        neighbor_info[i]["weight"] = float(w)
    
    # Weighted average
    combined = np.average(np.array(embeddings), axis=0, weights=weights)
    
    # Confidence score
    # - Higher with more neighbors
    # - Higher with closer neighbors
    # - Max 1.0
    min_distance = min(n["distance"] for n in neighbor_info)
    n_neighbors = len(neighbor_info)
    
    distance_factor = 1.0 / (min_distance + 1)  # 1.0 at distance 0, 0.5 at distance 1, etc.
    neighbor_factor = min(1.0, n_neighbors / 3)  # Saturates at 3 neighbors
    confidence = distance_factor * neighbor_factor
    
    return {
        "embedding": combined,
        "neighbors": neighbor_info,
        "confidence": float(confidence),
        "method": "phylogenetic",
    }


def predict_metabolites_phylogenetic(
    taxonomy: dict[str, str],
    wrapper,  # BPRModelWrapper
    gtdb: GTDBTaxonomy = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Predict metabolites for unmeasured taxon using phylogenetic inference.
    
    Args:
        taxonomy: {"genus": "SomeGenus", "family": "SomeFamily", ...}
        wrapper: BPRModelWrapper
        gtdb: GTDBTaxonomy instance
        top_k: Number of metabolites to return
    
    Returns:
        DataFrame with chebi_id, score, plus prediction metadata
    """
    result = predict_embedding_phylogenetic(taxonomy, wrapper, gtdb)
    
    if result["embedding"] is None:
        return pd.DataFrame(columns=["chebi_id", "score", "confidence", "method"])
    
    # Score all metabolites
    metabolite_embs = wrapper.model.metabolite_embeddings.weight.detach().numpy()
    scores = metabolite_embs @ result["embedding"]
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    rows = []
    for idx in top_indices:
        chebi_id = wrapper.idx_to_metabolite[idx]
        rows.append({
            "chebi_id": chebi_id,
            "score": float(scores[idx]),
            "confidence": result["confidence"],
            "method": result["method"],
            "n_neighbors": len(result["neighbors"]),
            "min_distance": min(n["distance"] for n in result["neighbors"]) if result["neighbors"] else None,
        })
    
    return pd.DataFrame(rows)