"""Configuration, paths, and data structures for Web of Microbes."""

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Optional
import os

# ============================================================
# Paths
# ============================================================

# ROOT_DIR = Path(__file__).parent

BIG_DATA_DIR = Path('/pscratch/sd/b/bpb/wom_data_cache')
# RAW_DIR = BIG_DATA_DIR / "raw"
RAW_DIR = Path("./raw")
# CACHE_DIR = BIG_DATA_DIR / "cache"
CACHE_DIR = Path("./cache")
DATA_DIR = Path("./data")

MAPPINGS_DIR = DATA_DIR / "mappings"

# Ensure directories exist
for d in [RAW_DIR, CACHE_DIR, MAPPINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Source URLs
# ============================================================

KG_SOURCES = {
    "kg_microbe": {
        "url": "https://github.com/Knowledge-Graph-Hub/kg-microbe/releases/download/2025-03-07/kg-microbe-core.tar.gz",
        "archive_type": "tar.gz",
        "nodes_file": "merged-kg_nodes.tsv",  # adjust after inspecting archive
        "edges_file": "merged-kg_edges.tsv",
    },
    # "metagenomic_kg": {
    #     "url": "https://zenodo.org/records/17546861/files/MetagenomicKG.zip?download=1",
    #     "archive_type": "zip",
    #     "nodes_file": "MetagenomicKG_nodes.tsv",  # adjust after inspecting
    #     "edges_file": "MetagenomicKG_edges.tsv",
    # },
}

CHEBI_SOURCES = {
    "compounds":{
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

EXOMETABOLITE_SOURCES = [
    os.path.join(DATA_DIR, "Biggs_isolates.tsv"),
    os.path.join(DATA_DIR, "NLDM_isolates.tsv"),
    os.path.join(DATA_DIR, "mCAFES_NLDM_17ExoTSNLDMS2_WoM_style_data_Dunnet.csv"),
    os.path.join(DATA_DIR, "mCAFES_NLDM_ExometabTimeSeries_WoM_style_data_Dunnet.csv"),
    os.path.join(DATA_DIR, "NLDM_20220202_AG_YW_ENIGMA_SyncomTSer_WoM_style_data_Dunnet.csv"),
    os.path.join(DATA_DIR, "WoM_isolates_Exomet_data_WoM_style.tsv"),
    os.path.join(DATA_DIR, "ECON_NLDM_WoM_style_data_Dunnet.tsv"),
    os.path.join(DATA_DIR, "WoM_isolates_Exomet_data_BG11+6crustiso_WoM_style.tsv"),
    os.path.join(DATA_DIR, "JGI_Merchant_WoM_style_data_Dunnet.tsv"),
    os.path.join(DATA_DIR, "Substrate_Utilization_Set1_and_Set2_WoM_style_data_Dunnet.tsv"),
    ]

# ============================================================
# Cache file names
# ============================================================

CACHE_FILES = {
    "kg_nodes": CACHE_DIR / "kg_nodes.parquet",
    "kg_edges": CACHE_DIR / "kg_edges.parquet",
    "exometabolite": CACHE_DIR / "exometabolite.parquet",
    "chebi": CACHE_DIR / "chebi.parquet",
    "compound_map": CACHE_DIR / "compound_name_to_chebi.json", 
    "bpr_model": CACHE_DIR / "bpr_model.pt",
    "manifest": CACHE_DIR / "manifest.json",
}

MAPPING_FILES = {
    # "organism_to_genus": MAPPINGS_DIR / "organism_to_genus.json",
    # "compound_synonyms": MAPPINGS_DIR / "compound_synonyms.json",
    "compound_manual": MAPPINGS_DIR / "compound_manual_mappings.json",
    }

# ============================================================
# Enums
# ============================================================

class AssertionCode(Enum):
    """Exometabolite assertion codes indicating uptake/release confidence."""
    NOT_DETECTED = "ND"
    SLIGHT = "2"
    MODERATE = "4"
    STRONG = "5"
    UNKNOWN = "?"

    @classmethod
    def from_value(cls, val) -> "AssertionCode":
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return cls.UNKNOWN
        val_str = str(val).strip()
        for member in cls:
            if member.value == val_str:
                return member
        return cls.UNKNOWN



class NodeType(Enum):
    """Node types in the KG."""
    MICROBE = "microbe"
    METABOLITE = "metabolite"
    PHENOTYPE = "phenotype"
    ENVIRONMENT = "environment"
    ENZYME = "enzyme"
    PROCESS = "process"
    OTHER = "other"


# ============================================================
# Dataclasses
# ============================================================

@dataclass(slots=True)
class KGNode:
    """A node in the unified knowledge graph."""
    id: str
    name: str
    node_type: NodeType
    source: str  # which KG it came from
    synonyms: list[str] = field(default_factory=list)
    xrefs: dict[str, str] = field(default_factory=dict)  # e.g., {"chebi": "CHEBI:1234"}
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class KGEdge:
    """An edge in the unified knowledge graph."""
    subject: str
    predicate: str
    object: str
    source: str
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class ExometaboliteObs:
    """A single exometabolite observation."""
    id: int
    organism_id: str
    organism_id_mapped: Optional[str]  # resolved KG ID or "unmapped"
    compound_id: str
    compound_id_mapped: Optional[str]  # resolved KG ID or "unmapped"
    environment: str
    project_id: str
    fold_change: Optional[float]
    log2_fold_change: Optional[float]
    assertion: AssertionCode
    timepoint: str
    p_value: Optional[float]
    q_value: Optional[float]


# ============================================================
# Columns to keep from each KG (tune after inspecting data)
# ============================================================

KG_NODE_COLS = ["id", "name", "category", "synonym", "xref"]
KG_EDGE_COLS = ["subject", "predicate", "object"]



# Index(['id', 'subject', 'predicate', 'object', 'relation', 'knowledge_source',
#        'meta', 'primary_knowledge_source', 'primary_knowledge_source.1'],
#       dtype='object')

# Index(['source_node', 'target_node', 'predicate', 'description',
#        'knowledge_source'],
#       dtype='object')

# Index(['id', 'category', 'name', 'description', 'xref', 'provided_by',
#        'synonym', 'deprecated', 'iri', 'object', 'predicate', 'relation',
#        'same_as', 'subject', 'subsets'],
#       dtype='object')

# Index(['node_id', 'node_type', 'all_names', 'description', 'knowledge_source',
#        'link', 'synonyms', 'is_pathogen'],
#       dtype='object')