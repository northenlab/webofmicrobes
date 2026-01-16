"""Exometabolite preference data loading and access."""

import json
import logging
import re
from pathlib import Path
from urllib.request import urlopen, Request
import gzip
from difflib import SequenceMatcher

import pandas as pd

from config import (
    RAW_DIR,
    CACHE_FILES,
    MAPPING_FILES,
    CHEBI_SOURCES,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================
# ChEBI download and loading
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


def build_chebi(force: bool = False) -> pd.DataFrame:
    """
    Download and process ChEBI names and structures.
    
    Returns DataFrame with columns: [compound_id, name, name_norm, smiles]
    Multiple names per compound_id (one row per name).
    """
    cache_path = CACHE_FILES["chebi"]
    if not force and cache_path.exists():
        logger.info("Loading ChEBI from cache")
        return pd.read_parquet(cache_path)
    
    # Download and decompress names
    names_gz = _download_chebi_file(
        CHEBI_SOURCES["names"]["url"],
        CHEBI_SOURCES["names"]["filename"]
    )
    names_tsv = _decompress_gz(names_gz)
    
    # Download and decompress structures
    struct_gz = _download_chebi_file(
        CHEBI_SOURCES["structures"]["url"],
        CHEBI_SOURCES["structures"]["filename"]
    )
    struct_tsv = _decompress_gz(struct_gz)
    
    # Load names
    logger.info(f"Loading {names_tsv}")
    names_df = pd.read_csv(names_tsv, sep="\t", low_memory=False)
    names_df.columns = names_df.columns.str.lower().str.strip()
    
    id_col = next((c for c in names_df.columns if "compound_id" in c), None)
    name_col = next((c for c in names_df.columns if "ascii_name" in c or "name" in c), None)
    
    if not id_col or not name_col:
        raise ValueError(f"Could not find required columns in names.tsv: {names_df.columns.tolist()}")
    
    names_df = names_df[[id_col, name_col]].copy()
    names_df.columns = ["compound_id", "name"]
    names_df = names_df.dropna(subset=["name"])
    
    # Normalize names
    names_df["name_norm"] = names_df["name"].apply(_normalize_name)
    logger.info(f"  Loaded {len(names_df)} name entries")
    
    # Load structures
    logger.info(f"Loading {struct_tsv}")
    struct_df = pd.read_csv(struct_tsv, sep="\t", low_memory=False)
    struct_df.columns = struct_df.columns.str.lower().str.strip()
    
    id_col_s = next((c for c in struct_df.columns if "compound_id" in c), None)
    smiles_col = next((c for c in struct_df.columns if "smiles" in c), None)
    
    if smiles_col:
        struct_df = struct_df[[id_col_s, smiles_col]].copy()
        struct_df.columns = ["compound_id", "smiles"]
        struct_df = struct_df.dropna(subset=["smiles"])
        struct_df = struct_df.drop_duplicates(subset=["compound_id"], keep="first")
        logger.info(f"  Loaded {len(struct_df)} structure entries with SMILES")
    else:
        struct_df = pd.DataFrame(columns=["compound_id", "smiles"])
    
    # Merge
    chebi_df = names_df.merge(struct_df, on="compound_id", how="left")
    logger.info(f"ChEBI merged: {len(chebi_df)} rows, {chebi_df['compound_id'].nunique()} unique compounds")
    
    chebi_df.to_parquet(cache_path)
    return chebi_df


def load_chebi() -> pd.DataFrame:
    """Load ChEBI data, building if necessary."""
    return build_chebi(force=False)


# ============================================================
# Compound name resolution
# ============================================================

def _normalize_name(name: str) -> str:
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
    
    return s


def _single_round_variants(name: str) -> set[str]:
    """Generate one round of name variants."""
    results = set()
    
    # === Hyphen vs no hyphen ===
    # "n-acetyl-glucosamine" <-> "n-acetylglucosamine"
    results.add(name.replace("-", ""))
    
    # === Acid/ate conversions ===
    acid_ate = [
        ("acetic acid", "acetate"), ("lactic acid", "lactate"),
        ("pyruvic acid", "pyruvate"), ("succinic acid", "succinate"),
        ("fumaric acid", "fumarate"), ("malic acid", "malate"),
        ("citric acid", "citrate"), ("glutamic acid", "glutamate"),
        ("aspartic acid", "aspartate"), ("formic acid", "formate"),
        ("propionic acid", "propionate"), ("butyric acid", "butyrate"),
        ("uric acid", "urate"), ("orotic acid", "orotate"),
        ("oxalic acid", "oxalate"), ("pantothenic acid", "pantothenate"),
        ("nicotinic acid", "nicotinate"), ("benzoic acid", "benzoate"),
        ("shikimic acid", "shikimate"), ("quinic acid", "quinate"),
        ("gluconic acid", "gluconate"), ("glucuronic acid", "glucuronate"),
        ("galacturonic acid", "galacturonate"), ("neuraminic acid", "neuraminate"),
        ("sialic acid", "sialate"), ("pipecolic acid", "pipecolate"),
        ("kynurenic acid", "kynurenate"), ("xanthurenic acid", "xanthurenate"),
        ("ketoglutaric acid", "ketoglutarate"), ("aminobutyric acid", "aminobutyrate"),
        ("thioctic acid", "thioctic acid"), ("lipoic acid", "lipoate"),
    ]
    for acid, ate in acid_ate:
        if acid in name:
            results.add(name.replace(acid, ate))
        if ate in name:
            results.add(name.replace(ate, acid))
    
    # Generic -ic acid -> -ate pattern
    m = re.search(r'(\w+)ic acid$', name)
    if m:
        results.add(name.replace(m.group(0), m.group(1) + "ate"))
    
    # === Greek letters ===
    greek = [("alpha", "α"), ("beta", "β"), ("gamma", "γ"), 
             ("delta", "δ"), ("omega", "ω"), ("epsilon", "ε")]
    for word, symbol in greek:
        if word in name:
            results.add(name.replace(word, symbol))
        if symbol in name:
            results.add(name.replace(symbol, word))
    
    # === Stereochemistry prefixes ===
    stereo_prefixes = [
        "d-", "l-", "dl-", "(+)-", "(-)-", "(r)-", "(s)-", "(±)-",
        "alpha-", "beta-", "α-", "β-",
        "cis-", "trans-", "meso-", "allo-",
        "sn-", "myo-", "scyllo-",
    ]
    
    # Remove prefixes
    for prefix in stereo_prefixes:
        if name.startswith(prefix):
            results.add(name[len(prefix):])
    
    # Add D-/L- if no stereo prefix
    if not any(name.startswith(p) for p in stereo_prefixes):
        results.add("d-" + name)
        results.add("l-" + name)
    
    # === N-acetyl variations ===
    for old, new in [
        ("n-acetyl-", "n-acetyl"), ("n-acetyl-", "nacetyl-"),
        ("n-acetyl", "n-acetyl-"), ("nacetyl", "n-acetyl-"),
    ]:
        if old in name:
            results.add(name.replace(old, new))
    
    # === Spelling variations ===
    spelling = [
        ("sulfur", "sulphur"), ("sulfate", "sulphate"), ("sulfide", "sulphide"),
        ("glycero", "glycerol"),
    ]
    for v1, v2 in spelling:
        if v1 in name:
            results.add(name.replace(v1, v2))
        if v2 in name:
            results.add(name.replace(v2, v1))
    
    # === Space/hyphen variations ===
    results.add(name.replace(" ", "-"))
    results.add(name.replace("-", " "))
    
    return results


def _generate_variants(name: str, max_rounds: int = 3, max_variants: int = 500) -> list[str]:
    """Generate variants iteratively so transformations combine."""
    if not name:
        return []
    
    variants = {name}
    for _ in range(max_rounds):
        new_variants = set()
        for v in variants:
            new_variants.update(_single_round_variants(v))
        before = len(variants)
        variants.update(new_variants)
        if len(variants) == before or len(variants) > max_variants:
            break
    
    return list(variants)


def _load_manual_mappings() -> dict[str, str]:
    """Load manual compound name -> ChEBI name mappings."""
    path = MAPPING_FILES.get("compound_manual")
    if path and path.exists():
        data = json.loads(path.read_text())
        return {_normalize_name(k): _normalize_name(v) for k, v in data.items()}
    return {}


def _build_chebi_lookup() -> dict[str, list[tuple[int, str]]]:
    """Build complete name -> [(chebi_id, original_name), ...] lookup."""
    chebi_df = load_chebi()
    lookup = {}
    for _, row in chebi_df.iterrows():
        norm = row["name_norm"]
        if pd.notna(norm):
            if norm not in lookup:
                lookup[norm] = []
            lookup[norm].append((row["compound_id"], row["name"]))
    return lookup


_CHEBI_LOOKUP = None
_CHEBI_ALL_NAMES = None


def _get_chebi_lookup() -> tuple[dict, list]:
    """Get cached ChEBI lookup."""
    global _CHEBI_LOOKUP, _CHEBI_ALL_NAMES
    if _CHEBI_LOOKUP is None:
        logger.info("Building ChEBI lookup...")
        _CHEBI_LOOKUP = _build_chebi_lookup()
        _CHEBI_ALL_NAMES = list(_CHEBI_LOOKUP.keys())
        logger.info(f"  ChEBI lookup: {len(_CHEBI_LOOKUP)} unique normalized names")
    return _CHEBI_LOOKUP, _CHEBI_ALL_NAMES


def _resolve_single_compound(
    name: str,
    lookup: dict,
    all_names: list[str],
    manual_mappings: dict[str, str],
) -> dict:
    """Resolve a single compound name to ChEBI."""
    
    query = _normalize_name(name)
    
    # 1. Manual mapping
    if query in manual_mappings:
        target = manual_mappings[query]
        if target in lookup:
            chebi_id, chebi_name = lookup[target][0]
            return {"match_type": "manual", "chebi_id": int(chebi_id), "chebi_name": chebi_name}
    
    # 2. Exact match
    if query in lookup:
        chebi_id, chebi_name = lookup[query][0]
        return {"match_type": "exact", "chebi_id": int(chebi_id), "chebi_name": chebi_name}
    
    # 3. Variant match
    for variant in _generate_variants(query):
        if variant in lookup:
            chebi_id, chebi_name = lookup[variant][0]
            return {"match_type": "variant", "chebi_id": int(chebi_id), "chebi_name": chebi_name}
    
    # 4. Fuzzy match
    query_words = {w for w in re.split(r'[\s\-]+', query) if len(w) >= 4}
    if query_words:
        candidates = [n for n in all_names if any(w in n for w in query_words)]
    else:
        candidates = []
    
    if len(candidates) > 3000:
        candidates = candidates[:3000]
    
    best_match, best_ratio = None, 0.0
    for candidate in candidates:
        ratio = SequenceMatcher(None, query, candidate).ratio()
        if ratio > best_ratio and ratio >= 0.85:
            best_ratio, best_match = ratio, candidate
    
    if best_match:
        chebi_id, chebi_name = lookup[best_match][0]
        return {"match_type": "fuzzy", "chebi_id": int(chebi_id), "chebi_name": chebi_name}
    
    return {"match_type": "none", "chebi_id": None, "chebi_name": None}


def _load_compound_cache() -> dict[str, dict]:
    """Load cached compound resolutions."""
    cache_path = CACHE_FILES["compound_map"]
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_compound_cache(cache: dict[str, dict]):
    """Save compound resolutions."""
    cache_path = CACHE_FILES["compound_map"]
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolve_compound_names(names: list[str], force: bool = False) -> dict[str, dict]:
    """
    Resolve compound names to ChEBI IDs.
    
    Returns:
        Dict mapping name -> {match_type, chebi_id, chebi_name}
    """
    cache = {} if force else _load_compound_cache()
    manual_mappings = _load_manual_mappings()
    
    # Re-resolve if not in cache OR previously failed
    names_to_resolve = [
        n for n in names 
        if pd.notna(n) and (
            n not in cache or 
            cache.get(n, {}).get("match_type") == "none"
        )
    ]
    
    if not names_to_resolve:
        logger.info(f"  All {len(names)} compounds found in cache")
        return cache
    
    logger.info(f"  Resolving {len(names_to_resolve)} compounds ({len(cache)} cached)")
    
    lookup, all_names = _get_chebi_lookup()
    
    for i, name in enumerate(names_to_resolve):
        if i > 0 and i % 50 == 0:
            logger.info(f"    Resolved {i}/{len(names_to_resolve)}...")
        result = _resolve_single_compound(name, lookup, all_names, manual_mappings)
        cache[name] = result
    
    _save_compound_cache(cache)
    
    # Log summary
    match_counts = {"exact": 0, "variant": 0, "fuzzy": 0, "manual": 0, "none": 0}
    for name in names_to_resolve:
        mtype = cache.get(name, {}).get("match_type", "none")
        match_counts[mtype] = match_counts.get(mtype, 0) + 1
    
    logger.info(f"  Resolution: exact={match_counts['exact']}, variant={match_counts['variant']}, "
                f"fuzzy={match_counts['fuzzy']}, manual={match_counts['manual']}, none={match_counts['none']}")
    
    return cache


# ============================================================
# Exometabolite data loading
# ============================================================

def _load_phylogeny() -> pd.DataFrame:
    """Load phylogeny mapping: isolate -> taxonomy."""
    path = DATA_DIR / "phylogeny.tsv"
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lower().str.strip()
    # Rename 'isolate' to 'organism_id' for consistency
    if "isolate" in df.columns:
        df = df.rename(columns={"isolate": "organism_id"})
    return df


def _load_and_melt_exometabolite(path: Path) -> pd.DataFrame:
    """Load a single exometabolite TSV and melt to long format."""
    df = pd.read_csv(path, sep="\t")
    
    # First column is organism ID
    id_col = df.columns[0]
    
    # Find metadata columns (non-compound columns)
    meta_cols = [id_col]
    if "Time Point (h)" in df.columns:
        meta_cols.append("Time Point (h)")
    
    # Everything else is a compound
    compound_cols = [c for c in df.columns if c not in meta_cols]
    
    # Melt
    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=compound_cols,
        var_name="compound_name",
        value_name="assertion_value"
    )
    
    # Rename
    df_long = df_long.rename(columns={id_col: "organism_id"})
    if "Time Point (h)" in df_long.columns:
        df_long = df_long.rename(columns={"Time Point (h)": "timepoint_h"})
    
    df_long["source_file"] = path.name
    
    return df_long


def _process_exometabolite(df: pd.DataFrame) -> pd.DataFrame:
    """Process melted exometabolite data: resolve compounds, join phylogeny."""
    
    # Resolve compounds to ChEBI
    logger.info("Resolving compound names to ChEBI IDs...")
    unique_compounds = df["compound_name"].dropna().unique().tolist()
    logger.info(f"  Unique compound names: {len(unique_compounds)}")
    
    resolution_map = resolve_compound_names(unique_compounds)
    
    df["chebi_id"] = df["compound_name"].map(
        lambda x: resolution_map.get(x, {}).get("chebi_id")
    )
    df["chebi_name"] = df["compound_name"].map(
        lambda x: resolution_map.get(x, {}).get("chebi_name")
    )
    df["compound_match_type"] = df["compound_name"].map(
        lambda x: resolution_map.get(x, {}).get("match_type", "none")
    )
    df["chebi_id_formatted"] = df["chebi_id"].apply(
        lambda x: f"CHEBI:{int(x)}" if pd.notna(x) else None
    )
    
    # Map assertion values to labels
    def assertion_label(val):
        if pd.isna(val):
            return None
        val = int(val)
        if val < -2:
            return "consumed"
        elif val > 2:
            return "secreted"
        else:
            return "neutral"
    
    df["assertion"] = df["assertion_value"].apply(assertion_label)
    
    # Join phylogeny
    logger.info("Joining phylogeny...")
    phylogeny = _load_phylogeny()
    df = df.merge(phylogeny, on="organism_id", how="left")
    
    return df


def build_exometabolite(force: bool = False) -> pd.DataFrame:
    """Load and process exometabolite data from TSV files."""
    cache_path = CACHE_FILES["exometabolite"]
    
    if not force and cache_path.exists():
        logger.info("Loading exometabolite data from cache")
        return pd.read_parquet(cache_path)
    
    # Ensure ChEBI is built
    build_chebi(force=False)
    
    # Load and melt each source file
    source_files = [
        DATA_DIR / "Biggs_isolates.tsv",
        DATA_DIR / "NLDM_syncom.tsv",
    ]
    
    dfs = []
    for path in source_files:
        if path.exists():
            logger.info(f"Loading {path.name}...")
            df = _load_and_melt_exometabolite(path)
            logger.info(f"  {len(df)} observations from {df['organism_id'].nunique()} organisms")
            dfs.append(df)
        else:
            logger.warning(f"Source file not found: {path}")
    
    if not dfs:
        raise FileNotFoundError("No exometabolite source files found")
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined: {len(df)} observations")
    
    df = _process_exometabolite(df)
    df.to_parquet(cache_path)
    
    return df


def load_exometabolite() -> pd.DataFrame:
    """Load exometabolite data, building if necessary."""
    return build_exometabolite(force=False)


# ============================================================
# Public API
# ============================================================

def get_unmatched_compounds() -> list[str]:
    """Get list of compound names that couldn't be matched to ChEBI."""
    df = load_exometabolite()
    return sorted(df[df["compound_match_type"] == "none"]["compound_name"].dropna().unique().tolist())


def get_organism_prefs(organism_id: str) -> pd.DataFrame:
    """Get all metabolite preferences for an organism."""
    df = load_exometabolite()
    return df[df["organism_id"] == organism_id]


def get_compound_prefs(compound_name: str) -> pd.DataFrame:
    """Get all organism preferences for a compound."""
    df = load_exometabolite()
    return df[
        (df["compound_name"] == compound_name) | 
        (df["chebi_name"] == compound_name)
    ]


def mapping_report() -> dict:
    """Return statistics on ID resolution."""
    df = load_exometabolite()
    match_counts = df["compound_match_type"].value_counts().to_dict()
    return {
        "total_observations": len(df),
        "unique_organisms": df["organism_id"].nunique(),
        "unique_compounds": df["compound_name"].nunique(),
        "compound_match_counts": match_counts,
        "unmatched_compounds": len(get_unmatched_compounds()),
    }


def lookup_chebi_by_name(name: str) -> pd.DataFrame:
    """Look up ChEBI compounds by name (case-insensitive exact match)."""
    chebi_df = load_chebi()
    name_lower = name.lower().strip()
    return chebi_df[chebi_df["name_norm"] == name_lower]