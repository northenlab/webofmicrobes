"""Exometabolite data loading and processing.

Loads metabolite preference data from TSV/CSV files and integrates with taxonomy.
"""

import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from .config import (
    EXOMETABOLITE_SOURCES,
    PHYLOGENY_FILE,
    CACHE_FILES,
    MIN_CONSUMERS,
    MIN_METABOLITES,
    CONSUMED_THRESHOLD,
    SECRETED_THRESHOLD,
)
from .compound_mapper import load_chebi, map_compound_names_batch, normalize_compound_name
from core.types import AssertionType, Taxonomy, TrainingData

logger = logging.getLogger(__name__)


# =============================================================================
# Phylogeny Loading
# =============================================================================

def load_phylogeny() -> pd.DataFrame:
    """Load phylogeny mapping: isolate -> taxonomy.
    
    Returns:
        DataFrame with columns: organism_id, kingdom, class, order, family, genus
    """
    if not PHYLOGENY_FILE.exists():
        raise FileNotFoundError(f"Phylogeny file not found: {PHYLOGENY_FILE}")
    
    df = pd.read_csv(PHYLOGENY_FILE, sep="\t")
    df.columns = df.columns.str.lower().str.strip()
    
    # Rename 'isolate' to 'organism_id' for consistency
    if "isolate" in df.columns:
        df = df.rename(columns={"isolate": "organism_id"})
    
    return df


# =============================================================================
# Exometabolite Data Loading
# =============================================================================

def _load_and_melt_exometabolite(path: Path) -> pd.DataFrame:
    """Load a single exometabolite TSV/CSV and melt to long format.
    
    Returns:
        DataFrame with columns: organism_id, compound_name, assertion_value, source_file
    """
    if isinstance(path, str):
        path = Path(path)
    
    # Determine separator
    sep = "," if path.suffix == ".csv" else "\t"
    
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path, sep=sep)
    
    # First column is organism ID
    id_col = df.columns[0]
    
    # Find metadata columns (non-compound columns)
    meta_cols = [id_col]
    sort_by_time = False
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower.startswith("time ") or col_lower.startswith("timepoint"):
            meta_cols.append(col)
            sort_by_time = True
            time_col = col
    
    # Handle time-series data (take latest timepoint)
    if sort_by_time:
        logger.info(f"  Sorting by time column: {time_col}")
        df = df.sort_values(by=[time_col], ascending=False)
        df = df.drop_duplicates(subset=[id_col], keep="first")
        df = df.drop(columns=[time_col])
        df = df.reset_index(drop=True)
    
    # Everything else is a compound
    compound_cols = [c for c in df.columns if c not in meta_cols]
    
    # Melt to long format
    df_long = df.melt(
        id_vars=id_col,
        value_vars=compound_cols,
        var_name="compound_name",
        value_name="assertion_value"
    )
    
    # Rename ID column
    df_long = df_long.rename(columns={id_col: "organism_id"})
    df_long["source_file"] = path.name
    
    return df_long


def _assertion_label(val: float) -> str:
    """Convert assertion value to label."""
    if pd.isna(val):
        return "neutral"
    
    val = float(val)
    if val <= CONSUMED_THRESHOLD:
        return "consumed"
    elif val >= SECRETED_THRESHOLD:
        return "secreted"
    else:
        return "neutral"


def build_exometabolite_dataframe(force: bool = False) -> pd.DataFrame:
    """Load and process all exometabolite data sources.
    
    Returns:
        DataFrame with columns:
            - organism_id: Isolate identifier
            - compound_name: Original compound name from data
            - representative_compound: Mapped ChEBI best name
            - chebi_id: ChEBI identifier
            - assertion_value: Numeric fold-change value
            - assertion: Categorical label (consumed, secreted, neutral)
            - kingdom, class, order, family, genus: Taxonomy
            - source_file: Original data file
    
    Args:
        force: If True, rebuild even if cache exists
    """
    cache_path = CACHE_FILES["exometabolite"]
    
    if not force and cache_path.exists():
        logger.info("Loading exometabolite data from cache")
        return pd.read_parquet(cache_path)
    
    logger.info("Building exometabolite dataframe...")
    
    # Load ChEBI for mapping
    chebi = load_chebi()
    
    # Load and merge all source files
    dfs = []
    for path in EXOMETABOLITE_SOURCES:
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            df = _load_and_melt_exometabolite(path)
            dfs.append(df)
        else:
            logger.warning(f"Source file not found: {path}")
    
    if not dfs:
        raise FileNotFoundError("No exometabolite source files found")
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df):,} observations from {len(dfs)} files")
    
    # Drop missing compound names
    df = df[df['compound_name'].notna()].copy()
    logger.info(f"After dropping missing compound names: {len(df):,} observations")
    
    # Filter out known bad compounds
    bad_compounds = ['C7H8O3S', '_RB_BiocrustExp', 'Unk_']
    for bad in bad_compounds:
        before = len(df)
        df = df[~df['compound_name'].str.contains(bad, na=False)]
        if len(df) < before:
            logger.info(f"Filtered out '{bad}': removed {before - len(df):,} observations")
    
    # Normalize compound names
    df['compound_name_normalized'] = df['compound_name'].apply(normalize_compound_name)
    
    # Map compound names to ChEBI
    logger.info("Mapping compound names to ChEBI...")
    unique_names = df['compound_name_normalized'].unique().tolist()
    mappings = map_compound_names_batch(unique_names, chebi)
    
    # Create mapping dict
    name_to_repr = {name: mapping.representative_name for name, mapping in mappings.items()}
    name_to_chebi = {name: mapping.chebi_id for name, mapping in mappings.items()}
    
    df['representative_compound'] = df['compound_name_normalized'].map(name_to_repr)
    df['chebi_id'] = df['compound_name_normalized'].map(name_to_chebi)
    
    # Log mapping stats
    exact = sum(1 for m in mappings.values() if m.match_type == 'exact')
    fuzzy = sum(1 for m in mappings.values() if m.match_type == 'fuzzy')
    unmatched = sum(1 for m in mappings.values() if m.match_type == 'none')
    logger.info(
        f"Compound mapping: {exact} exact, {fuzzy} fuzzy, {unmatched} unmatched "
        f"out of {len(unique_names)} unique names"
    )
    
    # Drop unmapped compounds
    df = df[pd.notna(df['representative_compound'])].copy()
    logger.info(f"After dropping unmapped compounds: {len(df):,} observations")
    
    # Add taxonomy
    logger.info("Adding taxonomy...")
    taxonomy = load_phylogeny()
    df = df.merge(taxonomy, on='organism_id', how='left')
    
    # Convert assertion values to labels
    df["assertion"] = df["assertion_value"].apply(_assertion_label)
    
    # Filter to compounds consumed by at least MIN_CONSUMERS organisms
    if MIN_CONSUMERS > 0:
        consumer_counts = (
            df[df["assertion"] == "consumed"]
            .groupby("representative_compound")["organism_id"]
            .nunique()
        )
        valid_compounds = consumer_counts[consumer_counts >= MIN_CONSUMERS].index.tolist()
        df = df[df["representative_compound"].isin(valid_compounds)]
        logger.info(
            f"After filtering to compounds consumed by ≥{MIN_CONSUMERS} organisms: "
            f"{len(df):,} observations, {len(valid_compounds)} compounds"
        )
    
    # Filter to organisms that consume at least MIN_METABOLITES compounds
    if MIN_METABOLITES > 0:
        metabolite_counts = (
            df[df["assertion"] == "consumed"]
            .groupby("organism_id")["representative_compound"]
            .nunique()
        )
        valid_organisms = metabolite_counts[metabolite_counts >= MIN_METABOLITES].index.tolist()
        df = df[df["organism_id"].isin(valid_organisms)]
        logger.info(
            f"After filtering to organisms consuming ≥{MIN_METABOLITES} metabolites: "
            f"{len(df):,} observations, {len(valid_organisms)} organisms"
        )
    
    # Save to cache
    logger.info(f"Saving to cache: {cache_path}")
    df.to_parquet(cache_path)
    
    logger.info(f"Exometabolite dataframe built with {len(df):,} observations")
    
    return df


def load_exometabolite() -> pd.DataFrame:
    """Load exometabolite data, building if necessary."""
    return build_exometabolite_dataframe(force=False)


# =============================================================================
# Training Data Preparation
# =============================================================================

def prepare_training_data(
    df: pd.DataFrame = None,
    preference_type: AssertionType = AssertionType.CONSUMED,
    include_secreted: bool = True,
) -> TrainingData:
    """Prepare data for model training using new dataclass structure.
    
    Args:
        df: Exometabolite DataFrame (will load if not provided)
        preference_type: Primary interaction type to use as positive
        include_secreted: If True (DEFAULT), include both consumed AND secreted as positives.
                         Set to False to train only on consumed metabolites.
    
    Returns:
        TrainingData object with all mappings and indices
    """
    if df is None:
        df = load_exometabolite()
    
    if include_secreted:
        logger.info("Preparing training data for BOTH consumed AND secreted (produced) metabolites")
        preference_types = ["consumed", "secreted"]
    else:
        logger.info(f"Preparing training data for preference type: {preference_type.value}")
        preference_types = [preference_type.value]
    
    # Get unique microbes and metabolites
    microbes = sorted(df["organism_id"].unique())
    metabolites = sorted(df["representative_compound"].unique())
    
    logger.info(f"  {len(microbes)} unique organisms")
    logger.info(f"  {len(metabolites)} unique metabolites")
    
    # Create index mappings
    microbe_to_idx = {m: i for i, m in enumerate(microbes)}
    metabolite_to_idx = {m: i for i, m in enumerate(metabolites)}
    
    # Build positive interaction set
    positives = defaultdict(set)
    for pref_type in preference_types:
        for _, row in df[df["assertion"] == pref_type].iterrows():
            m_idx = microbe_to_idx[row["organism_id"]]
            c_idx = metabolite_to_idx[row["representative_compound"]]
            positives[m_idx].add(c_idx)
    
    logger.info(f"  {len(positives)} organisms with positive interactions")
    total_interactions = sum(len(s) for s in positives.values())
    logger.info(f"  {total_interactions:,} total positive interactions")
    
    # Build taxonomy lookup
    microbe_taxonomy = {}
    tax_cols = ["kingdom", "class", "order", "family", "genus"]
    
    for _, row in df.drop_duplicates("organism_id").iterrows():
        microbe_taxonomy[row["organism_id"]] = Taxonomy(
            kingdom=row.get("kingdom"),
            class_=row.get("class"),  # Note the underscore
            order=row.get("order"),
            family=row.get("family"),
            genus=row.get("genus"),
        )
    
    # Create TrainingData object
    training_data = TrainingData(
        n_microbes=len(microbes),
        n_metabolites=len(metabolites),
        microbe_to_idx=microbe_to_idx,
        metabolite_to_idx=metabolite_to_idx,
        positives=dict(positives),
        microbe_taxonomy=microbe_taxonomy,
        preference_type=preference_type,
    )
    
    return training_data


# =============================================================================
# Convenience Functions
# =============================================================================

def get_organism_preferences(
    organism_id: str,
    df: pd.DataFrame = None
) -> pd.DataFrame:
    """Get all metabolite preferences for an organism.
    
    Args:
        organism_id: Organism identifier
        df: Exometabolite DataFrame (will load if not provided)
    
    Returns:
        DataFrame filtered to this organism
    """
    if df is None:
        df = load_exometabolite()
    
    return df[df["organism_id"] == organism_id].copy()


def get_compound_preferences(
    compound_name: str,
    df: pd.DataFrame = None
) -> pd.DataFrame:
    """Get all organism preferences for a compound.
    
    Args:
        compound_name: Representative compound name
        df: Exometabolite DataFrame (will load if not provided)
    
    Returns:
        DataFrame filtered to this compound
    """
    if df is None:
        df = load_exometabolite()
    
    return df[df["representative_compound"] == compound_name].copy()
