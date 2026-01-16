"""Knowledge graph loading, harmonization, and caching."""

import hashlib
import json
import logging
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd

from config import (
    KG_SOURCES,
    RAW_DIR,
    CACHE_FILES,
    KG_NODE_COLS,
    KG_EDGE_COLS,
    NodeType,
)

logger = logging.getLogger(__name__)


# ============================================================
# Download and extraction
# ============================================================

def _download(url: str, dest: Path) -> Path:
    """Download a file if not already present."""
    if dest.exists():
        logger.info(f"Already downloaded: {dest}")
        return dest
    logger.info(f"Downloading {url} -> {dest}")
    req = Request(url, headers={"User-Agent": "wom/0.1"})
    with urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    return dest


# Replace the _extract function with this:

def _extract(archive_path: Path, archive_type: str, dest_dir: Path) -> Path:
    """Extract archive to dest_dir. Handles nested archives. Returns extraction folder."""
    extract_to = dest_dir / archive_path.stem.replace(".tar", "")
    if extract_to.exists():
        logger.info(f"Already extracted: {extract_to}")
    else:
        logger.info(f"Extracting {archive_path} -> {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)
        if archive_type == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_to)
        elif archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(extract_to)
    
    # Check for nested archives and extract them
    for nested in extract_to.rglob("*.tar.gz"):
        nested_dest = nested.parent / nested.stem.replace(".tar", "")
        if not nested_dest.exists():
            logger.info(f"Extracting nested archive: {nested}")
            nested_dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(nested, "r:gz") as tar:
                tar.extractall(nested_dest)
    
    for nested in extract_to.rglob("*.zip"):
        nested_dest = nested.parent / nested.stem
        if not nested_dest.exists():
            logger.info(f"Extracting nested archive: {nested}")
            nested_dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(nested, "r") as z:
                z.extractall(nested_dest)
    
    return extract_to


def _find_file(folder: Path, pattern: str) -> Path:
    """Recursively find a file matching pattern."""
    matches = list(folder.rglob(pattern))
    if not matches:
        # Try looser match
        matches = list(folder.rglob(f"*{pattern.split('_')[-1]}"))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {folder}")
    return matches[0]


# ============================================================
# Loading and filtering
# ============================================================

def _load_tsv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load a TSV, keeping only specified columns if they exist."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if usecols:
        keep = [c for c in usecols if c in df.columns]
        df = df[keep]
    return df


def _infer_node_type(category: str | None) -> NodeType:
    """Map KG category string to our NodeType enum."""
    if category is None or pd.isna(category):
        return NodeType.OTHER
    
    cat = str(category).lower()
    
    # Explicit mappings based on kg-microbe categories
    if "organismtaxon" in cat:
        return NodeType.MICROBE
    if "chemicalentity" in cat or "chemicalsubstance" in cat or "chemicalrole" in cat:
        return NodeType.METABOLITE
    if "enzyme" in cat:
        return NodeType.ENZYME
    if "phenotypicquality" in cat or "attribute" in cat:
        return NodeType.PHENOTYPE
    if "environmentalfeature" in cat:
        return NodeType.ENVIRONMENT
    if "biologicalprocess" in cat:
        return NodeType.PROCESS
    
    return NodeType.OTHER

def _process_nodes(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardize node dataframe."""
    df = df.copy()
    df["source"] = source
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure required columns
    if "id" not in df.columns:
        raise ValueError(f"No 'id' column in nodes from {source}")
    if "name" not in df.columns:
        df["name"] = df["id"]
    
    # Add node_type as new column (don't replace category)
    cat_col = next((c for c in df.columns if c == "category"), None)
    if cat_col:
        df["node_type"] = df[cat_col].apply(_infer_node_type).apply(lambda x: x.value)
    else:
        df["node_type"] = NodeType.OTHER.value
    
    return df

def _process_edges(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardize edge dataframe."""
    df = df.copy()
    df["source"] = source
    df.columns = df.columns.str.lower().str.strip()
    
    required = ["subject", "predicate", "object"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in edges from {source}")
    
    return df

# ============================================================
# Cache management
# ============================================================

def _compute_manifest_hash() -> str:
    """Hash of source URLs to detect config changes."""
    data = json.dumps(KG_SOURCES, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _cache_is_valid() -> bool:
    """Check if cache exists and was built with current config."""
    manifest_path = CACHE_FILES["manifest"]
    if not manifest_path.exists():
        return False
    if not CACHE_FILES["kg_nodes"].exists() or not CACHE_FILES["kg_edges"].exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("hash") == _compute_manifest_hash()


def _write_manifest():
    """Write cache manifest."""
    manifest = {"hash": _compute_manifest_hash()}
    CACHE_FILES["manifest"].write_text(json.dumps(manifest))


# ============================================================
# Public API
# ============================================================

def build_kg(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download, process, and cache the unified knowledge graph.
    
    Returns (nodes_df, edges_df).
    """
    if not force and _cache_is_valid():
        logger.info("Loading KG from cache")
        nodes = pd.read_parquet(CACHE_FILES["kg_nodes"])
        edges = pd.read_parquet(CACHE_FILES["kg_edges"])
        return nodes, edges
    
    all_nodes = []
    all_edges = []
    
    for source_name, cfg in KG_SOURCES.items():
        logger.info(f"Processing {source_name}")
        
        # Download
        archive_name = cfg["url"].split("/")[-1].split("?")[0]
        archive_path = RAW_DIR / archive_name
        _download(cfg["url"], archive_path)
        
        # Extract
        extract_dir = _extract(archive_path, cfg["archive_type"], RAW_DIR)
        
        # Find and load files
        nodes_path = _find_file(extract_dir, cfg["nodes_file"])
        edges_path = _find_file(extract_dir, cfg["edges_file"])
        
        nodes_df = _load_tsv(nodes_path)
        edges_df = _load_tsv(edges_path)
        
        logger.info(f"  {source_name}: {len(nodes_df)} nodes, {len(edges_df)} edges")
        
        # Process
        nodes_df = _process_nodes(nodes_df, source_name)
        edges_df = _process_edges(edges_df, source_name)
        
        all_nodes.append(nodes_df)
        all_edges.append(edges_df)
    
    # Merge
    nodes = pd.concat(all_nodes, ignore_index=True)
    edges = pd.concat(all_edges, ignore_index=True)
    
    # Dedupe nodes by ID (keep first occurrence)
    nodes = nodes.drop_duplicates(subset=["id"], keep="first")
    
    logger.info(f"Unified KG: {len(nodes)} nodes, {len(edges)} edges")
    
    # Cache
    nodes.to_parquet(CACHE_FILES["kg_nodes"])
    edges.to_parquet(CACHE_FILES["kg_edges"])
    _write_manifest()
    
    return nodes, edges


def load_kg() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the unified KG, building if necessary."""
    return build_kg(force=False)


def get_nodes_by_type(node_type: NodeType) -> pd.DataFrame:
    """Get all nodes of a specific type."""
    nodes, _ = load_kg()
    return nodes[nodes["node_type"] == node_type.value]


def get_node(node_id: str) -> pd.Series | None:
    """Look up a single node by ID."""
    nodes, _ = load_kg()
    matches = nodes[nodes["id"] == node_id]
    return matches.iloc[0] if len(matches) > 0 else None


def search_nodes(query: str, node_type: NodeType | None = None) -> pd.DataFrame:
    """Search nodes by name (case-insensitive substring)."""
    nodes, _ = load_kg()
    mask = nodes["name"].str.lower().str.contains(query.lower(), na=False)
    if node_type:
        mask &= nodes["node_type"] == node_type.value
    return nodes[mask]