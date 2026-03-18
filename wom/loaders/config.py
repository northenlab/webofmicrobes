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
    EXOMETABOLITE_DATA_DIR / "nature_2021_wom_format.tsv",
    # EXOMETABOLITE_DATA_DIR / "wom_12_released.csv",
    # EXOMETABOLITE_DATA_DIR / "wom_13_released.csv",
]

# Phylogeny mapping
PHYLOGENY_FILE = EXOMETABOLITE_DATA_DIR / "phylogeny.tsv"

# Environment data
ENVIRONMENTS_FILE = EXOMETABOLITE_DATA_DIR / "environments.tsv"

# Optional UniProt source TSV (gzip-compressed; if present, used instead of API streaming)
UNIPROT_ACTIVITY_TSV_GZ = EXOMETABOLITE_DATA_DIR / "uniprot_activity.tsv.gz"

# Per-genus TSV cache directory — each genus saved as {genus}.tsv.gz for restartable builds
UNIPROT_GENUS_CACHE_DIR = EXOMETABOLITE_DATA_DIR / "uniprot_genus_cache"

# Per-genus stream URL template — avoids the stream endpoint's result-count limit.
# The all-bacteria query (taxonomy_id:2) returns ~16 M entries which exceeds UniProt's limit;
# querying one genus at a time keeps each response well under the threshold.
UNIPROT_GENUS_STREAM_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv"
    "&fields=accession%2Corganism_id%2Corganism_name%2Clineage%2Cec%2Ccc_catalytic_activity"
    "&query=%28cc_catalytic_activity%3A%2A%29+AND+%28taxonomy_name%3A{genus}%29"
)

# =============================================================================
# Cache Files
# =============================================================================

CACHE_FILES = {
    "chebi": PROCESSED_DATA_DIR / "chebi.parquet",
    "compound_map": PROCESSED_DATA_DIR / "compound_name_to_chebi.json",
    "exometabolite": PROCESSED_DATA_DIR / "exometabolite.parquet",
    "environments": PROCESSED_DATA_DIR / "environments.parquet",
    "function_proteins": PROCESSED_DATA_DIR / "function_proteins.parquet",
    "function_protein_ec": PROCESSED_DATA_DIR / "function_protein_ec.parquet",
    "function_protein_reactions": PROCESSED_DATA_DIR / "function_protein_reactions.parquet",
    "function_reaction_catalog": PROCESSED_DATA_DIR / "function_reaction_catalog.parquet",
    "taxon_function_enrichment": PROCESSED_DATA_DIR / "taxon_function_enrichment.parquet",
    "taxon_function_compound_support": PROCESSED_DATA_DIR / "taxon_function_compound_support.parquet",
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
