#!/usr/bin/env python
"""
genome_spot.py
--------------
Build and lookup module for GenomeSPOT culture condition predictions.

Predictions are from the GenomeSPOT tool:
  Metz et al. (2024) bioRxiv. https://github.com/cultivarium/GenomeSPOT

This module serves two roles:
  1. Build: load the precomputed BacDive predictions TSV and create per-level
     parquet summary files in CACHE_DIR.
  2. Lookup: given a taxonomic name, return aggregated statistics (mean,
     median, std, se, count) for each culture condition.

Culture conditions predicted:
  oxygen                - oxygen tolerance probability (0=strict anaerobe, 1=aerobe)
  temperature_optimum   - optimal growth temperature (°C)
  temperature_min       - minimum growth temperature (°C)
  temperature_max       - maximum growth temperature (°C)
  salinity_optimum      - optimal salinity (% w/v NaCl)
  salinity_min          - minimum salinity (% w/v NaCl)
  salinity_max          - maximum salinity (% w/v NaCl)
  ph_optimum            - optimal pH
  ph_min                - minimum pH
  ph_max                - maximum pH

Usage (build):
    from loaders.genome_spot import build_genome_spot_parquets
    build_genome_spot_parquets(tsv_path, output_dir)

Usage (lookup):
    import loaders.genome_spot as genome_spot
    genome_spot.CACHE_DIR = "./wom/processed_data"
    result = genome_spot.lookup("Pseudomonas", level="genus")
"""

import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ── configurable paths ───────────────────────────────────────────────
CACHE_DIR = "processed_data"

TAXONOMY_LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

CONDITION_COLUMNS = [
    "oxygen",
    "temperature_optimum",
    "temperature_min",
    "temperature_max",
    "salinity_optimum",
    "salinity_min",
    "salinity_max",
    "ph_optimum",
    "ph_min",
    "ph_max",
]

CONDITION_UNITS = {
    "oxygen": "probability (0=strict anaerobe, 1=aerobe)",
    "temperature_optimum": "°C",
    "temperature_min": "°C",
    "temperature_max": "°C",
    "salinity_optimum": "% w/v NaCl",
    "salinity_min": "% w/v NaCl",
    "salinity_max": "% w/v NaCl",
    "ph_optimum": "pH",
    "ph_min": "pH",
    "ph_max": "pH",
}

STAT_CHOICES = ["mean", "median", "std", "se", "count"]

# ── in-memory cache ──────────────────────────────────────────────────
_cache: dict[str, pd.DataFrame] = {}


# ═══════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════

def _parquet_path(level: str) -> str:
    return os.path.join(CACHE_DIR, f"genome_spot_{level}.parquet")


def _summarise_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Group by taxonomic level and compute summary stats for all conditions."""
    idx = TAXONOMY_LEVELS.index(level)
    group_cols = TAXONOMY_LEVELS[: idx + 1]

    # Drop rows missing the grouping key
    valid = df.dropna(subset=[level])

    # Aggregate each condition column independently (different rows may have NaN)
    frames = []
    for col in CONDITION_COLUMNS:
        col_valid = valid.dropna(subset=[col]).groupby(group_cols)
        agg = col_valid[col].agg(
            mean="mean",
            median="median",
            std="std",
            count="count",
        ).add_prefix(f"{col}_")
        agg[f"{col}_se"] = agg[f"{col}_std"] / np.sqrt(agg[f"{col}_count"])
        frames.append(agg)

    result = pd.concat(frames, axis=1).reset_index()
    return result


def build_genome_spot_parquets(tsv_path: str, output_dir: str) -> None:
    """Load the GenomeSPOT BacDive predictions TSV and write per-level parquets.

    Args:
        tsv_path:   Path to genome-spot-all-bacdive-predictions.tsv
        output_dir: Directory to write genome_spot_{level}.parquet files
    """
    print(f"Reading {tsv_path} …")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Lowercase taxonomy columns for consistent matching
    for level in TAXONOMY_LEVELS:
        if level in df.columns:
            df[level] = df[level].str.strip().str.lower()

    os.makedirs(output_dir, exist_ok=True)

    for level in TAXONOMY_LEVELS:
        print(f"  Summarising at {level} level …")
        summary = _summarise_level(df, level)
        out = os.path.join(output_dir, f"genome_spot_{level}.parquet")
        summary.to_parquet(out, index=False)
        print(f"    → {out}  ({len(summary)} rows)")

    print("GenomeSPOT build complete.")


# ═══════════════════════════════════════════════════════════════════
# LOOKUP
# ═══════════════════════════════════════════════════════════════════

def _load_level(level: str) -> pd.DataFrame:
    if level not in _cache:
        path = _parquet_path(level)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found – run build_genome_spot_parquets() first."
            )
        _cache[level] = pd.read_parquet(path)
    return _cache[level]


def _exact_lookup(df: pd.DataFrame, level: str, name: str) -> pd.DataFrame:
    return df[df[level] == name]


def _fuzzy_lookup(df: pd.DataFrame, level: str, name: str, score_cutoff: int = 50):
    """Return (matched_rows, matched_name, score)."""
    candidates = df[level].dropna().unique().tolist()
    result = process.extractOne(
        name, candidates, scorer=fuzz.WRatio, score_cutoff=score_cutoff
    )
    if result is None:
        return pd.DataFrame(), None, 0
    matched_name, score, _ = result
    rows = df[df[level] == matched_name]
    return rows, matched_name, score


def _round(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _condition_stats(row: pd.Series, col: str) -> dict:
    count_raw = row.get(f"{col}_count")
    return {
        "mean": _round(row.get(f"{col}_mean")),
        "median": _round(row.get(f"{col}_median")),
        "std": _round(row.get(f"{col}_std")),
        "se": _round(row.get(f"{col}_se")),
        "count": int(count_raw) if count_raw is not None and pd.notna(count_raw) else None,
        "units": CONDITION_UNITS[col],
    }


def _row_to_dict(row: pd.Series, level: str, condition: str | None = None) -> dict:
    """Convert a summary row to a JSON-friendly dict."""
    idx = TAXONOMY_LEVELS.index(level)
    lineage = {lv: row.get(lv, None) for lv in TAXONOMY_LEVELS[: idx + 1]}

    if condition:
        return {
            "lineage": lineage,
            "condition": condition,
            "stats": _condition_stats(row, condition),
        }

    return {
        "lineage": lineage,
        "conditions": {col: _condition_stats(row, col) for col in CONDITION_COLUMNS},
    }


def _search_all_levels(name: str, condition: str | None = None) -> dict:
    """Search every level for an exact or fuzzy match; return best."""
    name_lower = name.strip().lower()

    # 1) exact match across all levels
    for level in TAXONOMY_LEVELS:
        df = _load_level(level)
        hits = _exact_lookup(df, level, name_lower)
        if not hits.empty:
            records = [_row_to_dict(r, level, condition) for _, r in hits.iterrows()]
            return {
                "query": name,
                "matched_name": name_lower,
                "matched_level": level,
                "match_score": 100,
                "results": records,
            }

    # 2) fuzzy across all levels – pick best score
    best = None
    for level in TAXONOMY_LEVELS:
        df = _load_level(level)
        rows, matched, score = _fuzzy_lookup(df, level, name_lower)
        if matched and (best is None or score > best[2]):
            best = (level, rows, score, matched)

    if best:
        level, rows, score, matched = best
        records = [_row_to_dict(r, level, condition) for _, r in rows.iterrows()]
        return {
            "query": name,
            "matched_name": matched,
            "matched_level": level,
            "match_score": round(score, 2),
            "results": records,
        }

    return {"query": name, "error": "No match found at any taxonomic level."}


def _search_level(name: str, level: str, condition: str | None = None) -> dict:
    """Search a specific level; fall back to scanning all levels."""
    name_lower = name.strip().lower()
    level_lower = level.strip().lower()

    if level_lower not in TAXONOMY_LEVELS:
        return _search_all_levels(name, condition)

    df = _load_level(level_lower)

    # exact
    hits = _exact_lookup(df, level_lower, name_lower)
    if not hits.empty:
        records = [_row_to_dict(r, level_lower, condition) for _, r in hits.iterrows()]
        return {
            "query": name,
            "matched_name": name_lower,
            "matched_level": level_lower,
            "match_score": 100,
            "results": records,
        }

    # fuzzy at requested level
    rows, matched, score = _fuzzy_lookup(df, level_lower, name_lower)
    if matched:
        records = [_row_to_dict(r, level_lower, condition) for _, r in rows.iterrows()]
        return {
            "query": name,
            "matched_name": matched,
            "matched_level": level_lower,
            "match_score": round(score, 2),
            "results": records,
        }

    # fall back: maybe they said genus but it's actually a family, etc.
    fallback = _search_all_levels(name, condition)
    if "error" not in fallback:
        fallback["note"] = (
            f"'{name}' was not found at the '{level_lower}' level. "
            f"Found at '{fallback['matched_level']}' instead."
        )
    return fallback


def lookup(
    query: str,
    level: str | None = None,
    condition: str | None = None,
    stat: str | None = None,
) -> dict:
    """Look up GenomeSPOT culture condition predictions by taxonomic name.

    Args:
        query:     Taxonomic name (fuzzy matching enabled).
        level:     Taxonomic level to search ('domain' … 'species').
                   If None, all levels are searched.
        condition: One of CONDITION_COLUMNS. If None, all conditions returned.
        stat:      One of STAT_CHOICES ('mean', 'median', 'std', 'se', 'count').
                   Only used when condition is also specified; simplifies output
                   to a single value per result.

    Returns:
        dict with match metadata and per-condition statistics.
        std is always included alongside other stats because GenomeSPOT
        predictions carry substantial inter-taxon variability.
    """
    if condition and condition not in CONDITION_COLUMNS:
        return {"error": f"Unknown condition '{condition}'. Valid: {CONDITION_COLUMNS}"}
    if stat and stat not in STAT_CHOICES:
        return {"error": f"Unknown stat '{stat}'. Valid: {STAT_CHOICES}"}

    if level:
        result = _search_level(query, level, condition)
    else:
        result = _search_all_levels(query, condition)

    # When both condition and stat requested, simplify output to single value
    # but always retain std so callers understand the spread
    if stat and condition and "results" in result:
        stat_lower = stat.strip().lower()
        for r in result["results"]:
            stats = r.pop("stats")
            r["value"] = stats.get(stat_lower)
            r["std"] = stats.get("std")   # always include spread
            r["count"] = stats.get("count")
            r["stat"] = stat_lower
            r["units"] = CONDITION_UNITS[condition]
        result["stat_requested"] = stat_lower
        result["condition"] = condition

    return result
