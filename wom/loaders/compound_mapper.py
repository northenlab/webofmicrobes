"""ChEBI compound name mapping and normalization.

Provides fuzzy matching of compound names to ChEBI entries with InChI-based canonicalization.
"""

import gzip
import json
import logging
import re
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from rapidfuzz import process, fuzz

from .config import (
    CACHE_FILES,
    CHEBI_SOURCES,
    CHEBI_SOURCE_DIR,
    FUZZY_MATCH_CUTOFF,
)
from core.types import CompoundMapping

logger = logging.getLogger(__name__)


# =============================================================================
# ChEBI Download and Caching
# =============================================================================

def _download_chebi_file(url: str, filename: str) -> Path:
    """Download a ChEBI file if not already present."""
    dest = CHEBI_SOURCE_DIR / filename
    if dest.exists():
        logger.info(f"ChEBI file already exists: {dest}")
        return dest
    
    logger.info(f"Downloading {url} -> {dest}")
    req = Request(url, headers={"User-Agent": "wom/0.1"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    return dest


def _decompress_gz(gz_path: Path) -> Path:
    """Decompress a .gz file, return path to decompressed file."""
    out_path = gz_path.with_suffix("")
    if out_path.exists():
        return out_path
    
    logger.info(f"Decompressing {gz_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        f_out.write(f_in.read())
    return out_path


# =============================================================================
# Name Normalization
# =============================================================================

def normalize_compound_name(name: str) -> str:
    """Normalize compound name for matching.
    
    Applies consistent transformations:
    - Lowercase
    - Remove spaces, dashes, underscores
    - Normalize Unicode characters
    - Remove common mass spec artifacts
    """
    if pd.isna(name):
        return ""
    
    s = str(name).strip().lower()

    # Remove XML/HTML tags that occasionally appear in names
    s = re.sub(r"<[^>]+>", "", s)
    
    # Normalize whitespace
    s = " ".join(s.split())
    
    # Normalize dashes and quotes
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("′", "'").replace("'", "'")
    
    # Normalize Unicode stereochemistry letters
    s = s.replace("ᴅ", "d").replace("ᴸ", "l").replace("ʟ", "l").replace("ᴰ", "d")
    
    # Remove spaces, dashes, underscores for matching
    s = s.replace(' ', '').replace('-', '').replace('_', '')
    
    # Remove common mass spec artifacts
    s = s.replace('[m+h]+', '').replace('[2m+h]+', '').replace('[mh]', '')
    
    # Handle known problematic compounds
    compound_fixes = {
        'cystine(dimerofcysteine)': 'cysteine',
        'cysteine/cystine(dimerofcysteine)': 'cysteine',
        'cysteine/cystine': 'cysteine',
        'cysteine/cysteine': 'cysteine',
        'cystine/cysteine': 'cysteine',
        'isomersisonicotinicacid//nicotinicacid': 'nicotinicacid',
        'threonineisomers(coeluters:threonine,homoserine,allothreonine)': 'threonine',
        'leucine/norleucine': 'leucine',
        'isomers2hydroxycinnamicacid//4coumaricacid': 'coumaricacid',
    }
    
    return compound_fixes.get(s, s)


def clean_html_tags(name: str) -> str:
    """Remove HTML/XML tags from display names while preserving formatting.
    
    Specifically designed for display purposes - keeps spaces and text intact.
    E.g., <em>D</em>-glucose -> D-glucose
    """
    if pd.isna(name) or not isinstance(name, str):
        return name
    
    # Remove HTML/XML tags but preserve the content inside
    cleaned = re.sub(r"<[^>]+>", "", name)
    
    # Clean up any excess whitespace
    cleaned = " ".join(cleaned.split())
    
    return cleaned


# =============================================================================
# ChEBI Database Building
# =============================================================================

def build_chebi_database(force: bool = False) -> pd.DataFrame:
    """Download and process ChEBI database.
    
    Returns DataFrame with columns:
        - compound_id: ChEBI ID (e.g., "CHEBI:15377")
        - name: Original compound name
        - name_normalized: Normalized name for matching
        - best_name: Preferred display name
        - smiles: SMILES structure
        - inchi: InChI string
        - inchi_key: InChIKey
    
    Args:
        force: If True, rebuild even if cache exists
    
    Returns:
        DataFrame with one row per compound name (multiple names per compound_id)
    """
    cache_path = CACHE_FILES["chebi"]
    
    if not force and cache_path.exists():
        logger.info("Loading ChEBI from cache")
        return pd.read_parquet(cache_path)
    
    logger.info("Building ChEBI database...")
    
    # Download and decompress files
    compounds_gz = _download_chebi_file(
        CHEBI_SOURCES["compounds"]["url"],
        CHEBI_SOURCES["compounds"]["filename"]
    )
    compounds_tsv = _decompress_gz(compounds_gz)
    
    names_gz = _download_chebi_file(
        CHEBI_SOURCES["names"]["url"],
        CHEBI_SOURCES["names"]["filename"]
    )
    names_tsv = _decompress_gz(names_gz)
    
    struct_gz = _download_chebi_file(
        CHEBI_SOURCES["structures"]["url"],
        CHEBI_SOURCES["structures"]["filename"]
    )
    struct_tsv = _decompress_gz(struct_gz)
    
    # Load compounds (for primary names)
    logger.info("Loading compounds...")
    compounds = pd.read_csv(compounds_tsv, sep="\t", dtype=str)
    # Handle case-insensitive column names
    compounds.columns = compounds.columns.str.upper()
    compounds = compounds[["ID", "NAME"]].copy()
    compounds.columns = ["compound_id", "primary_name"]
    compounds["compound_id"] = "CHEBI:" + compounds["compound_id"].astype(str)
    
    # Load all names (including synonyms)
    logger.info("Loading names...")
    names = pd.read_csv(names_tsv, sep="\t", dtype=str)
    # Handle case-insensitive column names
    names.columns = names.columns.str.upper()
    names = names[["COMPOUND_ID", "NAME", "TYPE"]].copy()
    names.columns = ["compound_id", "name", "name_type"]
    names["compound_id"] = "CHEBI:" + names["compound_id"].astype(str)
    
    # Load structures
    logger.info("Loading structures...")
    structures = pd.read_csv(struct_tsv, sep="\t", dtype=str)
    # Handle case-insensitive column names
    structures.columns = structures.columns.str.lower()
    
    # New ChEBI format has smiles, standard_inchi, standard_inchi_key columns directly
    if 'smiles' in structures.columns:
        # New format
        structures = structures[["compound_id", "smiles", "standard_inchi", "standard_inchi_key"]].copy()
        structures.columns = ["compound_id", "smiles", "inchi", "inchi_key"]
        structures["compound_id"] = "CHEBI:" + structures["compound_id"].astype(str)
        struct_merged = structures
    else:
        # Old format (pivoting)
        structures = structures[["COMPOUND_ID", "TYPE", "STRUCTURE"]].copy()
        structures.columns = ["compound_id", "structure_type", "structure"]
        structures["compound_id"] = "CHEBI:" + structures["compound_id"].astype(str)
        
        # Pivot structures to wide format
        smiles = structures[structures["structure_type"] == "SMILES"][["compound_id", "structure"]]
        smiles.columns = ["compound_id", "smiles"]
        
        inchi = structures[structures["structure_type"] == "InChI"][["compound_id", "structure"]]
        inchi.columns = ["compound_id", "inchi"]
        
        inchi_key = structures[structures["structure_type"] == "InChIKey"][["compound_id", "structure"]]
        inchi_key.columns = ["compound_id", "inchi_key"]
        
        # Merge structures
        struct_merged = smiles.merge(inchi, on="compound_id", how="outer")
        struct_merged = struct_merged.merge(inchi_key, on="compound_id", how="outer")
    
    # Merge everything
    logger.info("Merging data...")
    chebi = names.merge(compounds, on="compound_id", how="left")
    chebi = chebi.merge(struct_merged, on="compound_id", how="left")
    
    # Clean display names (remove XML/HTML tags)
    chebi["name"] = chebi["name"].fillna("").astype(str).str.replace(r"<[^>]+>", "", regex=True).str.strip()
    chebi["primary_name"] = chebi["primary_name"].fillna("").astype(str).str.replace(r"<[^>]+>", "", regex=True).str.strip()

    # Keep only entries with InChIKey and derive first-block grouping key
    chebi = chebi[pd.notna(chebi["inchi_key"])].copy()
    chebi["first_block_inchi_key"] = chebi["inchi_key"].str.split("-").str[0]

    # Prefer unspecific stereochemistry names as representative when available,
    # then preferred names, then synonyms, then IUPAC/brand-like names.
    chebi["name_rank"] = 4
    if "name_type" in chebi.columns:
        type_rank = {
            "SYNONYM": 3,
            "IUPAC NAME": 4,
            "UNIPROT NAME": 1,
            "INN": 3,
            "BRAND NAME": 4,
        }
        chebi["name_rank"] = chebi["name_type"].map(type_rank).fillna(4).astype(int)

    # Rows where name equals primary_name are preferred names
    is_preferred = (chebi["primary_name"] != "") & (chebi["name"] == chebi["primary_name"])
    chebi.loc[is_preferred, "name_rank"] = 2

    # Unspecific stereochemistry markers in InChIKey are best representatives
    unspecific = chebi["inchi_key"].str.contains("UHFFFAOYSA", na=False)
    chebi.loc[unspecific, "name_rank"] = 1

    # Add name length for sorting (prefer shorter names)
    chebi["name_length"] = chebi["name"].str.len()

    # Create a secondary rank for type preference: UNIPROT > Preferred > others
    # Use name_rank as primary, and a secondary criterion that favors UNIPROT/Preferred names
    type_secondary = {
        "UNIPROT NAME": 0,  # Highest priority
        "PREFERRED NAME": 1,
        "SYNONYM": 2,
        "INN": 2,
        "IUPAC NAME": 3,
        "BRAND NAME": 3,
    }
    chebi["type_secondary"] = chebi["name_type"].map(type_secondary).fillna(4).astype(int)

    # Choose representative name per first-block InChIKey
    # Strategy: Prefer "Goldilocks zone" names (5-60 chars) to avoid both acronyms and IUPAC names
    # Sort by: name_rank (lower is better): UNIPROT>Preferred>Synonym>IUPAC
    #          type_secondary (UNIPROT > Preferred > others - secondary tiebreaker)
    #          name_length (shorter is better)
    #          compound_id (lower ChEBI ID preferred - more established)
    rep_candidates = chebi[["first_block_inchi_key", "name", "name_rank", "type_secondary", "name_length", "compound_id"]].copy()
    rep_candidates = rep_candidates[rep_candidates["name"] != ""]
    
    # First, try to select from "Goldilocks zone" names (5-60 chars)
    goldilocks_candidates = rep_candidates[(rep_candidates["name_length"] >= 5) & (rep_candidates["name_length"] <= 60)].copy()
    goldilocks_candidates = goldilocks_candidates.sort_values(
        ["name_rank", "type_secondary", "name_length", "compound_id"],
        ascending=[True, True, True, True]
    )
    goldilocks_names = goldilocks_candidates.drop_duplicates(subset=["first_block_inchi_key"], keep="first")
    goldilocks_names = goldilocks_names[["first_block_inchi_key", "name"]].rename(columns={"name": "best_name"})
    
    # For compounds without Goldilocks names, fall back to any name (sorted by same logic)
    all_candidates = rep_candidates.sort_values(
        ["name_rank", "type_secondary", "name_length", "compound_id"],
        ascending=[True, True, True, True]
    )
    all_names = all_candidates.drop_duplicates(subset=["first_block_inchi_key"], keep="first")
    all_names = all_names[["first_block_inchi_key", "name"]].rename(columns={"name": "best_name"})
    
    # Prefer Goldilocks names, but use fallback if none exists
    rep_names = goldilocks_names.merge(all_names, on="first_block_inchi_key", how="outer", suffixes=("", "_fallback"))
    rep_names["best_name"] = rep_names["best_name"].fillna(rep_names["best_name_fallback"])
    rep_names = rep_names[["first_block_inchi_key", "best_name"]]
    
    chebi = chebi.merge(rep_names, on="first_block_inchi_key", how="left")

    # Fallback if any representative is still missing
    chebi["best_name"] = chebi["best_name"].fillna(chebi["primary_name"]).fillna(chebi["name"])
    
    # Clean HTML tags from best_name for display
    chebi["best_name"] = chebi["best_name"].apply(clean_html_tags)

    # Add normalized names for matching (synonyms + preferred names)
    chebi["name_for_matching"] = chebi["name"].apply(normalize_compound_name)
    
    # Remove empty names
    chebi = chebi[chebi["name_for_matching"] != ""].copy()
    
    # Save to cache
    logger.info(f"Saving to cache: {cache_path}")
    chebi.to_parquet(cache_path)
    
    logger.info(
        f"ChEBI database built: {len(chebi):,} name entries, "
        f"{chebi['compound_id'].nunique():,} unique compounds"
    )
    
    return chebi


def load_chebi() -> pd.DataFrame:
    """Load ChEBI database, building if necessary."""
    cache_path = CACHE_FILES["chebi"]
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)

        # Rebuild if cache predates first-block grouping or still has XML tags.
        missing_grouping = "first_block_inchi_key" not in cached.columns
        dirty_names = False
        if "best_name" in cached.columns:
            dirty_names = cached["best_name"].astype(str).str.contains(r"<[^>]+>", regex=True, na=False).any()

        if not missing_grouping and not dirty_names:
            return cached

        logger.info("ChEBI cache is outdated; rebuilding with first-block grouping and cleaned names")

    return build_chebi_database(force=True)


# =============================================================================
# Compound Name Mapping
# =============================================================================

def map_compound_name(
    name: str,
    chebi: pd.DataFrame = None,
    score_cutoff: int = None
) -> CompoundMapping:
    """Map a compound name to ChEBI using exact or fuzzy matching.
    
    Args:
        name: Compound name to map
        chebi: ChEBI database (will load if not provided)
        score_cutoff: Minimum fuzzy match score (0-100)
    
    Returns:
        CompoundMapping with results
    """
    if chebi is None:
        chebi = load_chebi()
    
    if score_cutoff is None:
        score_cutoff = FUZZY_MATCH_CUTOFF
    
    # Normalize input
    normalized = normalize_compound_name(name)
    
    if not normalized:
        return CompoundMapping(
            query_name=name,
            representative_name=None,
            chebi_id=None,
            match_score=0.0,
            match_type='none'
        )
    
    # Try exact match
    matches = chebi[chebi['name_for_matching'] == normalized]
    if len(matches) > 0:
        matches = matches.copy()
        matches['_is_self_rep'] = (matches['best_name'] == matches['name']).astype(int)
        best = (
            matches
            .sort_values(['_is_self_rep', 'compound_id'], ascending=[False, True])
            .drop_duplicates('best_name')
            .iloc[0]
        )
        return CompoundMapping(
            query_name=name,
            representative_name=best['best_name'],
            chebi_id=best['compound_id'],
            match_score=1.0,
            match_type='exact'
        )
    
    # Fall back to fuzzy matching
    unique_names = chebi['name_for_matching'].unique().tolist()
    result = process.extractOne(
        normalized,
        unique_names,
        scorer=fuzz.ratio,
        score_cutoff=score_cutoff
    )
    
    if result:
        match_name, score, _ = result
        matches = chebi[chebi['name_for_matching'] == match_name].copy()
        matches['_is_self_rep'] = (matches['best_name'] == matches['name']).astype(int)
        best = (
            matches
            .sort_values(['_is_self_rep', 'compound_id'], ascending=[False, True])
            .drop_duplicates('best_name')
            .iloc[0]
        )
        
        return CompoundMapping(
            query_name=name,
            representative_name=best['best_name'],
            chebi_id=best['compound_id'],
            match_score=score / 100.0,
            match_type='fuzzy'
        )
    
    # No match found
    return CompoundMapping(
        query_name=name,
        representative_name=None,
        chebi_id=None,
        match_score=0.0,
        match_type='none'
    )


def map_compound_names_batch(
    names: list[str],
    chebi: pd.DataFrame = None,
    score_cutoff: int = None
) -> dict[str, CompoundMapping]:
    """Map multiple compound names in batch.
    
    More efficient than calling map_compound_name repeatedly.
    
    Args:
        names: List of compound names to map
        chebi: ChEBI database (will load if not provided)
        score_cutoff: Minimum fuzzy match score
    
    Returns:
        Dict mapping query name to CompoundMapping
    """
    if chebi is None:
        chebi = load_chebi()
    
    results = {}
    for name in names:
        results[name] = map_compound_name(name, chebi, score_cutoff)
    
    return results
