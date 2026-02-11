# chebi_mapping.py
"""ChEBI compound mapping using InChI connectivity layers."""

import gzip
import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from config import CACHE_FILES, CHEBI_SOURCES, MAPPING_FILES, RAW_DIR

logger = logging.getLogger(__name__)

# Output files
MAPPER_DIR = CACHE_FILES["chebi"].parent / "compound_mapper"

# ============================================================
# ChEBI download and caching
# ============================================================


def _download_chebi_file(url: str, filename: str) -> Path:
    """Download a ChEBI file if not already present."""
    dest = RAW_DIR / filename
    if dest.exists():
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

def normalize_name(name: str) -> str:
    """Normalize compound name for matching."""
    if pd.isna(name):
        return ""
    s = str(name).strip()

    # Lowercase and normalize whitespace
    s = s.lower()
    s = " ".join(s.split())

    # Normalize dashes and quotes
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("′", "'").replace("'", "'")

    # Normalize Unicode stereochemistry letters
    s = s.replace("ᴅ", "d").replace("ᴸ", "l").replace("ʟ", "l")
    s = s.replace("ᴰ", "d")

    return s

def build_chebi(force: bool = False) -> pd.DataFrame:
    """
    Download and process ChEBI names and structures.

    Returns DataFrame with columns: [compound_id, name, name_lower, smiles, inchi, inchi_key]
    Multiple names per compound_id (one row per name).
    """
    cache_path = CACHE_FILES["chebi"]
    if not force and cache_path.exists():
        logger.info("Loading ChEBI from cache")
        return pd.read_parquet(cache_path)

    # Download and decompress compounds
    compounds_gz = _download_chebi_file(
        CHEBI_SOURCES["compounds"]["url"], CHEBI_SOURCES["compounds"]["filename"]
    )
    compounds_tsv = _decompress_gz(compounds_gz)

    # Download and decompress names
    names_gz = _download_chebi_file(
        CHEBI_SOURCES["names"]["url"], CHEBI_SOURCES["names"]["filename"]
    )
    names_tsv = _decompress_gz(names_gz)

    # Download and decompress structures
    struct_gz = _download_chebi_file(
        CHEBI_SOURCES["structures"]["url"], CHEBI_SOURCES["structures"]["filename"]
    )
    struct_tsv = _decompress_gz(struct_gz)

    # Load compounds
    logger.info(f"Loading {compounds_tsv}")
    comp_cols = ['id', 'ascii_name']
    comp_df = pd.read_csv(compounds_tsv, sep="\t", low_memory=False, usecols=comp_cols)
    comp_df.rename(columns={"id": "compound_id"}, inplace=True)
    comp_df['type'] = 'chebi_preferred_name'

    # Load names
    logger.info(f"Loading {names_tsv}")
    name_cols = ['compound_id', 'ascii_name', 'type']
    names_df = pd.read_csv(names_tsv, sep="\t", low_memory=False, usecols=name_cols)
    names_df = pd.concat([comp_df, names_df], ignore_index=True)
    names_df.reset_index(drop=True, inplace=True)

    names_df = names_df.dropna(subset=["ascii_name"])

    # Normalize names
    names_df['ascii_name'] = names_df['ascii_name'].apply(normalize_name)
    names_df['ascii_name'] = names_df['ascii_name'].str.replace(r'<[^>]+>', '', regex=True)
    
    names_df.rename(columns={"ascii_name": "name"}, inplace=True)
    
    logger.info(f"  Loaded {len(names_df)} name entries")


    # Load structures
    logger.info(f"Loading {struct_tsv}")
    keep_struct_cols = ["compound_id", "smiles", "standard_inchi_key"]
    struct_df = pd.read_csv(struct_tsv, sep="\t", low_memory=False,usecols=keep_struct_cols)
    struct_df.rename(columns={"standard_inchi_key": "inchi_key"}, inplace=True)
    logger.info(f"  Loaded {len(struct_df)} structure entries")

    # Merge
    chebi_df = names_df.merge(struct_df, on="compound_id", how="left")
    logger.info(f"ChEBI merged: {len(chebi_df)} rows, {chebi_df['compound_id'].nunique()} unique compounds")
    chebi_df = chebi_df[pd.notna(chebi_df["inchi_key"])]
    chebi_df['first_block_inchi_key'] = chebi_df['inchi_key'].str.split('-').str[0]

    # Do has UHFFFAOYSA as better name
    idx = chebi_df['inchi_key'].str.contains('UHFFFAOYSA')
    chebi_df.loc[idx, 'type'] = 'unspecific_stereochemistry'

    # rank value of names
    best_names = chebi_df.copy()
    name_rank = {'unspecific_stereochemistry': 1, 'chebi_preferred_name': 2,
        'SYNONYM': 3, 'IUPAC NAME': 4, 'UNIPROT NAME': 1, 'INN': 3, 'BRAND NAME': 4}
    best_names['name_rank'] = best_names['type'].map(name_rank).fillna(4).astype(int)

    best_names['name_length'] = best_names['name'].apply(lambda x: len(x))
    
    best_names.sort_values(by=['name_rank','compound_id'], ascending=[True, True], inplace=True)
    best_names.drop_duplicates(subset=['first_block_inchi_key'], keep='first', inplace=True)
    best_names = best_names[['first_block_inchi_key','name']].rename(columns={'name':'best_name'})
    chebi_df = chebi_df.merge(best_names, on='first_block_inchi_key', how='left')
    chebi_df.drop(columns=['type','first_block_inchi_key'], inplace=True)
    chebi_df['name_for_matching'] = chebi_df['name'].str.replace(r' ', '', regex=False)
    chebi_df['name_for_matching'] = chebi_df['name_for_matching'].str.replace(r'-', '', regex=False)
    chebi_df['name_for_matching'] = chebi_df['name_for_matching'].str.replace(r'_', '', regex=False)
    chebi_df.to_parquet(cache_path)
    return chebi_df


def load_chebi() -> pd.DataFrame:
    """Load ChEBI data, building if necessary."""
    return build_chebi(force=False)
