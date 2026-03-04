"""Configuration, paths, and data sources for Web of Microbes."""

from pathlib import Path
import os

# =============================================================================
# Paths
# =============================================================================

# Base directory for wom_refactor
BASE_DIR = Path(__file__).parent.parent

# New directory structure - clearer naming
CHEBI_SOURCE_DIR = BASE_DIR / "chebi_source"  # Downloaded ChEBI files (was "raw")
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"  # Processed outputs and models (was "cache")
EXOMETABOLITE_DATA_DIR = BASE_DIR / "exometabolite_data"  # Source TSV/CSV files (was "data")
MAPPINGS_DIR = EXOMETABOLITE_DATA_DIR / "mappings"

# Big data location (NERSC) - use same naming convention
BIG_DATA_DIR = Path('/pscratch/sd/b/bpb/wom_data_cache')
if BIG_DATA_DIR.exists():
    CHEBI_SOURCE_DIR = BIG_DATA_DIR / "chebi_source"
    PROCESSED_DATA_DIR = BIG_DATA_DIR / "processed_data"
    # Note: EXOMETABOLITE_DATA_DIR stays local as it's version-controlled source data

# Ensure directories exist
for d in [CHEBI_SOURCE_DIR, PROCESSED_DATA_DIR]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data Sources
# =============================================================================

# ChEBI database sources
CHEBI_SOURCES = {
    "compounds": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/compounds.tsv.gz",
        "filename": "chebi_compounds.tsv.gz",
    },
    "names": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/names.tsv.gz",
        "filename": "chebi_names.tsv.gz",
    },
    "structures": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/structures.tsv.gz",
        "filename": "chebi_structures.tsv.gz",
    },
}

# Exometabolite data sources
EXOMETABOLITE_SOURCES = [
    EXOMETABOLITE_DATA_DIR / "wom_1.tsv",
    # EXOMETABOLITE_DATA_DIR / "wom_2.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_3.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_4.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_5.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_6.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_7.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_8.tsv",
    EXOMETABOLITE_DATA_DIR / "wom_9.csv",
    EXOMETABOLITE_DATA_DIR / "wom_10.csv",
    EXOMETABOLITE_DATA_DIR / "wom_11.csv",
    EXOMETABOLITE_DATA_DIR / "nature_2021_wom_format.tsv"
]

# Phylogeny mapping
PHYLOGENY_FILE = EXOMETABOLITE_DATA_DIR / "phylogeny.tsv"

# =============================================================================
# Cache Files
# =============================================================================

CACHE_FILES = {
    "chebi": PROCESSED_DATA_DIR / "chebi.parquet",
    "compound_map": PROCESSED_DATA_DIR / "compound_name_to_chebi.json",
    "exometabolite": PROCESSED_DATA_DIR / "exometabolite.parquet",
    # Model files will be added as: {model_type}_{preference_type}_*.npz
}

# =============================================================================
# Processing Parameters
# =============================================================================

# Minimum number of organisms that must consume a compound for it to be included
MIN_CONSUMERS = 3

# Minimum number of metabolites an organism must consume to be included
MIN_METABOLITES = 3

# Fuzzy matching score cutoff (0-100)
FUZZY_MATCH_CUTOFF = 90

# Assertion value thresholds
CONSUMED_THRESHOLD = -2  # Values <= this are "consumed"
SECRETED_THRESHOLD = 2   # Values >= this are "secreted"
