# pref.py
"""Exometabolite preference data loading and access."""

import logging
from pathlib import Path

from rapidfuzz import process, fuzz

import pandas as pd

from config import (
    CACHE_FILES,
    DATA_DIR,
    EXOMETABOLITE_SOURCES
)
from chebi_mapping import load_chebi

logger = logging.getLogger(__name__)


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

def rename_troublesome_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known troublesome compound names for better matching."""
    df['name_for_matching'] = df['compound_name'].str.strip()
    df['name_for_matching'] = df['name_for_matching'].str.lower()
    df['name_for_matching'] = df['name_for_matching'].str.replace(r' ', '', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace(r'-', '', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace(r'_', '', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('[m+h]+','', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('[2m+h]+','', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('[mh]','', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('cystine(dimerofcysteine)','cysteine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('cysteine/cystine(dimerofcysteine)','cysteine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('cysteine/cystine','cysteine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('cysteine/cysteine','cysteine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('cystine/cysteine','cysteine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('isomersisonicotinicacid//nicotinicacid','nicotinicacid', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('threonineisomers(coeluters:threonine,homoserine,allothreonine)','threonine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('leucine/norleucine','leucine', regex=False)
    df['name_for_matching'] = df['name_for_matching'].str.replace('isomers2hydroxycinnamicacid//4coumaricacid','coumaricacid', regex=False)

    return df

def _load_and_melt_exometabolite(path: Path) -> pd.DataFrame:
    """Load a single exometabolite TSV and melt to long format."""
    if isinstance(path, str):
        path = Path(path)

    path_str = str(path)
    if path_str.endswith(".csv"):
        sep = ","
    else:
        sep = "\t"
    df = pd.read_csv(path, sep=sep)
    
    # First column is organism ID
    id_col = df.columns[0]
    
    # Find metadata columns (non-compound columns)
    meta_cols = [id_col]
    sort_by_time = False
    for c in df.columns:
        c_lower = c.lower()
        if c_lower.startswith("time ") or c_lower.startswith("timepoint"):
            meta_cols.append(c)
            sort_by_time = True
            time_col = c

    if sort_by_time:
        logger.info(f"  Sorting by time column: {time_col}")
        logger.info(f"  Meta columns: {meta_cols}")
        df.sort_values(by=[time_col], inplace=True, ascending=False)
        df.drop_duplicates(subset=[id_col], keep="first", inplace=True)
        df.drop(columns=[time_col], inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # Everything else is a compound
    logger.info(f"  Meta columns: {meta_cols}")
    compound_cols = [c for c in df.columns if c not in meta_cols]
    
    # Melt
    df_long = df.melt(
        id_vars=id_col,
        value_vars=compound_cols,
        var_name="compound_name",
        value_name="assertion_value"
    )
    
    # Rename
    df_long = df_long.rename(columns={id_col: "organism_id"})
    df_long["source_file"] = path.name
    
    return df_long

def map_compound_name(name: str, chebi: pd.DataFrame = None, score_cutoff: int = 90) -> dict:
    """
    Map a compound name to its representative compound.
    
    Returns dict with keys: representative_compound, score, match_type
    """
    from rapidfuzz import process, fuzz
    
    if chebi is None:
        chebi = load_chebi()
    
    # Normalize input
    normalized = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    
    # Try exact match
    matches = chebi[chebi['name_for_matching'] == normalized]
    if len(matches) > 0:
        best = matches.sort_values('compound_id').drop_duplicates('best_name').iloc[0]
        return {
            'representative_compound': best['best_name'],
            'score': 1.0,
            'match_type': 'exact'
        }
    
    # Fall back to fuzzy
    chebi_names = chebi['name_for_matching'].unique().tolist()
    result = process.extractOne(normalized, chebi_names, scorer=fuzz.ratio, score_cutoff=score_cutoff)
    
    if result:
        match_name, score, _ = result
        matches = chebi[chebi['name_for_matching'] == match_name]
        best = matches.sort_values('compound_id').drop_duplicates('best_name').iloc[0]
        return {
            'representative_compound': best['best_name'],
            'score': score / 100,
            'match_type': 'fuzzy'
        }
    
    return {
        'representative_compound': None,
        'score': 0.0,
        'match_type': 'none'
    }


def build_exometabolite(force: bool = False,
                        min_consumers: int = 3) -> pd.DataFrame:
    """Load and process exometabolite data from TSV files."""
    cache_path = CACHE_FILES["exometabolite"]
    
    if not force and cache_path.exists():
        logger.info("Loading exometabolite data from cache")
        return pd.read_parquet(cache_path)
    
    chebi = load_chebi()

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

    df = df[df['compound_name'].notna()]
    logger.warning(f"After dropping missing compound names: {len(df)} observations")
    bad_compounds = ['C7H8O3S','_RB_BiocrustExp','Unk_']
    for bad in bad_compounds:
        df = df[df['compound_name'].str.contains(bad) == False]
        logger.warning(f"After dropping '{bad}' compound names: {len(df)} observations")

    df = rename_troublesome_compounds(df)

    chebi_names = chebi['name_for_matching'].unique().tolist()

    compound_map = dict({k:{} for k in df['name_for_matching'].unique()})
    # clean up to use map_compound_name above
    for name in compound_map.keys():
        matches = chebi[chebi['name_for_matching'] == name]
        if not len(matches):
            result = process.extractOne(name, chebi_names, scorer=fuzz.ratio, score_cutoff=90)
            if result:
                match_name, score, _ = result
                matches = chebi[chebi['name_for_matching'] == match_name]
                resolved_names = matches.sort_values('compound_id').drop_duplicates('best_name')
                resolved_name = resolved_names.iloc[0]['best_name']
                compound_map[name]['resolved_name'] = resolved_name
                compound_map[name]['score'] = score / 100
            else:
                compound_map[name]['resolved_name'] = None
                compound_map[name]['score'] = 0
            continue
        resolved_names = matches.sort_values('compound_id').drop_duplicates('best_name')
        resolved_name = resolved_names.iloc[0]['best_name']
        compound_map[name]['resolved_name'] = resolved_name
        compound_map[name]['score'] = 1.0

    exoname_to_chebi = {}
    for k,v in compound_map.items():
        exoname_to_chebi[k] = v['resolved_name']
    df['representative_compound'] = df['name_for_matching'].map(exoname_to_chebi)

    exact = sum(1 for v in compound_map.values() if v['score'] == 1.0)
    fuzzy = sum(1 for v in compound_map.values() if 0 < v['score'] < 1.0)
    unmatched = sum(1 for v in compound_map.values() if v['score'] == 0)
    logger.info(f"Compound matching: {exact} exact, {fuzzy} fuzzy, {unmatched} unmatched")

    df = df[pd.notna(df['representative_compound'])]
    taxonomy = _load_phylogeny()
    df = df.merge(taxonomy, on='organism_id', how='left')

    cache_path = CACHE_FILES["exometabolite"]
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

    # Filter out compounds consumed by fewer than min_consumers organisms
    if min_consumers > 0:
        consumer_counts = df[df["assertion"] == "consumed"].groupby("representative_compound")["organism_id"].nunique()
        valid_compounds = consumer_counts[consumer_counts >= min_consumers].index.tolist()
        df = df[df["representative_compound"].isin(valid_compounds)]
        logger.info(f"After filtering to compounds consumed by at least {min_consumers} organisms: {len(df)} observations")
    df.to_parquet(cache_path)
    return df

    


def load_exometabolite() -> pd.DataFrame:
    """Load exometabolite data, building if necessary."""
    return build_exometabolite(force=False)


# ============================================================
# Public API
# ============================================================



def get_organism_prefs(organism_id: str) -> pd.DataFrame:
    """Get all metabolite preferences for an organism."""
    df = load_exometabolite()
    return df[df["organism_id"] == organism_id]


def get_compound_prefs(compound_name: str) -> pd.DataFrame:
    """Get all organism preferences for a compound."""
    df = load_exometabolite()
    return df[df["representative_compound"] == compound_name]

