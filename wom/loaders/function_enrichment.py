"""Build and load UniProt-based taxonomic function enrichment tables.

This module creates enrichment-ready parquet files for descriptive enzyme-function
analysis over WoM-relevant taxa and compounds.
"""

import io
import re
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from .config import (
    CACHE_FILES,
    UNIPROT_ACTIVITY_TSV_GZ,
    UNIPROT_GENUS_CACHE_DIR,
    UNIPROT_GENUS_STREAM_URL,
)
from .compound_mapper import load_chebi
from .loaders import load_exometabolite

logger = logging.getLogger(__name__)

TAXONOMY_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Regex patterns for parsing the Catalytic activity text field
_RHEA_REACTION_RE = re.compile(r'\bRhea:(RHEA:\d+)')       # RHEA:NNNNN only, not RHEA-COMP
_CHEBI_RE = re.compile(r'\bChEBI:(CHEBI:\d+)')
_REACTION_NAME_RE = re.compile(r'Reaction=([^;]+);')
_EC_IN_CC_RE = re.compile(r'EC=(\d+(?:\.[0-9-]+){3})')     # e.g. 2.7.1.90 or 3.5.1.-
_LINEAGE_RANK_RE = re.compile(r'\s*\([^)]+\)')              # strips "(genus)", "(domain)", etc.


def _normalize_chebi_id(value: str | None) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("CHEBI:", "")
    return f"CHEBI:{s}"


def _normalize_lineage(raw: str) -> str:
    """Strip rank annotations and normalise separator to '|'.

    UniProt TSV lineage looks like:
        'cellular organisms (no rank), Bacteria (domain), ..., Burkholderia (genus)'
    WoM lineages are pipe-separated taxon names without rank annotations.
    """
    stripped = _LINEAGE_RANK_RE.sub("", raw)          # remove "(genus)" etc.
    return "|".join(p.strip() for p in stripped.split(",") if p.strip())


def _parse_catalytic_activity_text(text: str, accession: str) -> list[dict]:
    """Parse a UniProt Catalytic activity TSV field into reaction rows.

    One protein may have multiple CATALYTIC ACTIVITY: blocks in the field,
    each carrying its own Reaction name, Rhea ID, ChEBI IDs, and EC number.
    """
    rows = []
    blocks = re.split(r'CATALYTIC ACTIVITY:', text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        name_match = _REACTION_NAME_RE.search(block)
        reaction_name = name_match.group(1).strip() if name_match else None

        rhea_matches = _RHEA_REACTION_RE.findall(block)
        rhea_id = rhea_matches[0] if rhea_matches else None

        chebi_ids = _CHEBI_RE.findall(block)      # already 'CHEBI:NNNNN'
        ec_matches = _EC_IN_CC_RE.findall(block)

        ec_for_rows = ec_matches if ec_matches else [None]
        chebi_for_rows = chebi_ids if chebi_ids else [None]

        for ec_number in ec_for_rows:
            for chebi_id in chebi_for_rows:
                rows.append({
                    "primaryAccession": accession,
                    "reaction_name": reaction_name,
                    "chebi_id": chebi_id,
                    "rhea_id": rhea_id,
                    "ec_number": ec_number,
                })

    return rows


def _fetch_genus_bytes(genus: str, force: bool = False) -> bytes:
    """Return compressed UniProt TSV bytes for one genus, using a disk cache.

    On first call the data is downloaded from UniProt and saved to
    UNIPROT_GENUS_CACHE_DIR/{genus}.tsv.gz.  Subsequent calls load from disk,
    making the build restartable without re-downloading completed genera.
    Pass force=True to re-download even if a cached file exists.
    """
    UNIPROT_GENUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = UNIPROT_GENUS_CACHE_DIR / f"{genus.lower()}.tsv.gz"

    if cache_path.exists() and not force:
        return cache_path.read_bytes()

    url = UNIPROT_GENUS_STREAM_URL.format(genus=quote(genus))
    req = Request(url, headers={"User-Agent": "wom/0.1"})
    with urlopen(req) as resp:
        payload = resp.read()

    cache_path.write_bytes(payload)
    return payload


def _accumulate_tsv_chunk(
    chunk: pd.DataFrame,
    protein_chunks: list,
    ec_chunks: list,
    reaction_rows: list,
) -> None:
    """Parse one TSV DataFrame chunk into proteins, EC, and reaction accumulators."""
    lineage_col = "Taxonomic lineage"
    if lineage_col not in chunk.columns:
        return
    if chunk.empty:
        return

    # --- proteins ---
    prot = chunk[["Entry", "Organism (ID)", "Organism", lineage_col]].copy()
    prot["lineage"] = prot[lineage_col].apply(_normalize_lineage)
    prot = prot.rename(columns={
        "Entry": "primaryAccession",
        "Organism (ID)": "ncbi_taxonomy_id",
        "Organism": "isolate",
    }).drop(columns=[lineage_col])
    protein_chunks.append(prot)

    # --- protein → EC (vectorised) ---
    ec_raw = chunk[["Entry", "EC number"]].copy()
    ec_raw = ec_raw[ec_raw["EC number"] != ""]
    ec_raw["EC number"] = ec_raw["EC number"].str.split("; ")
    ec_exploded = ec_raw.explode("EC number")
    ec_exploded = ec_exploded[ec_exploded["EC number"].str.strip() != ""]
    ec_exploded.columns = ["primaryAccession", "ec_number"]
    ec_exploded["ec_number"] = ec_exploded["ec_number"].str.strip()
    ec_chunks.append(ec_exploded)

    # --- reactions (per-row text parsing) ---
    cc_col = "Catalytic activity"
    for _, row in chunk[chunk[cc_col] != ""].iterrows():
        reaction_rows.extend(_parse_catalytic_activity_text(row[cc_col], row["Entry"]))


def _parse_uniprot_activity_tsv(
    wom_genera: list[str],
    tsv_gz_path: Optional[Path] = None,
    chunk_size: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse UniProt TSV data into proteins, protein-EC, and reaction rows.

    If tsv_gz_path is given (a pre-downloaded combined file), reads it in chunks
    with a genus pre-filter.  Otherwise streams per-genus from the UniProt API,
    which avoids the stream endpoint's result-count limit.
    """
    protein_chunks: list[pd.DataFrame] = []
    ec_chunks: list[pd.DataFrame] = []
    reaction_rows: list[dict] = []

    if tsv_gz_path is not None:
        # Large local file — read in chunks and pre-filter to WoM genera
        genus_pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(g) for g in wom_genera) + r')\b',
            re.IGNORECASE,
        )
        with pd.read_csv(
            tsv_gz_path,
            sep="\t",
            compression="gzip",
            chunksize=chunk_size,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
        ) as reader:
            for chunk in reader:
                lineage_col = "Taxonomic lineage"
                if lineage_col not in chunk.columns:
                    continue
                chunk = chunk[chunk[lineage_col].str.contains(genus_pattern, na=False)]
                _accumulate_tsv_chunk(chunk, protein_chunks, ec_chunks, reaction_rows)
    else:
        # Per-genus API streaming — each genus is already scoped, no pre-filter needed
        for i, genus in enumerate(wom_genera, 1):
            logger.info(f"  Fetching UniProt data for genus {i}/{len(wom_genera)}: {genus}")
            try:
                payload = _fetch_genus_bytes(genus)
            except Exception as e:
                logger.warning(f"  Skipping {genus}: {e}")
                continue
            df = pd.read_csv(
                io.BytesIO(payload),
                sep="\t",
                compression="gzip",
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
            )
            _accumulate_tsv_chunk(df, protein_chunks, ec_chunks, reaction_rows)

    proteins_df = (
        pd.concat(protein_chunks, ignore_index=True).drop_duplicates()
        if protein_chunks else
        pd.DataFrame(columns=["primaryAccession", "ncbi_taxonomy_id", "isolate", "lineage"])
    )
    protein_ec_df = (
        pd.concat(ec_chunks, ignore_index=True).drop_duplicates()
        if ec_chunks else
        pd.DataFrame(columns=["primaryAccession", "ec_number"])
    )
    reactions_df = (
        pd.DataFrame(reaction_rows, columns=["primaryAccession", "reaction_name", "chebi_id", "rhea_id", "ec_number"])
        .drop_duplicates()
        if reaction_rows else
        pd.DataFrame(columns=["primaryAccession", "reaction_name", "chebi_id", "rhea_id", "ec_number"])
    )

    logger.info(
        f"TSV parse complete: {len(proteins_df):,} proteins, "
        f"{len(protein_ec_df):,} protein-EC rows, "
        f"{len(reactions_df):,} reaction rows"
    )
    return proteins_df, protein_ec_df, reactions_df


def _match_uniprot_genus_to_wom_taxonomy(
    proteins_df: pd.DataFrame,
    exometabolite_df: pd.DataFrame,
    score_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign WoM taxonomy levels to UniProt proteins by matching on genus name.

    The genus is extracted as the first word of the UniProt organism name
    (standard binomial nomenclature).  An exact case-insensitive lookup is tried
    first; genera that don't match exactly fall back to a fuzzy ratio match on
    genus name alone, which is far more reliable than matching on full lineage
    strings (which diverge due to phylum renames, extra clade nodes, etc.).
    """
    tax_table = exometabolite_df[TAXONOMY_LEVELS].drop_duplicates().copy()
    wom_genus_map = {g.lower(): g for g in tax_table["genus"].dropna().unique()}
    wom_genera_list = list(wom_genus_map.values())

    # Extract genus from the first word of the UniProt organism name
    proteins_df = proteins_df.copy()
    proteins_df["uniprot_genus"] = proteins_df["isolate"].str.split().str[0].str.strip()

    # Exact match (case-insensitive)
    proteins_df["matched_genus"] = proteins_df["uniprot_genus"].str.lower().map(wom_genus_map)

    # Fuzzy fallback for unmatched genera (genus name only — fast and accurate)
    unmatched_genera = proteins_df.loc[proteins_df["matched_genus"].isna(), "uniprot_genus"].dropna().unique()
    fuzzy_map: dict[str, str] = {}
    for ug in unmatched_genera:
        best = process.extractOne(ug, wom_genera_list, scorer=fuzz.ratio)
        if best and best[1] >= score_cutoff:
            fuzzy_map[ug] = best[0]

    mask = proteins_df["matched_genus"].isna() & proteins_df["uniprot_genus"].isin(fuzzy_map)
    proteins_df.loc[mask, "matched_genus"] = proteins_df.loc[mask, "uniprot_genus"].map(fuzzy_map)

    # Build match report (analogous to old lineage_matches_df)
    genus_matches_df = (
        proteins_df[["uniprot_genus", "matched_genus"]]
        .drop_duplicates()
        .assign(match_score=proteins_df["matched_genus"].notna().astype(float) * 100)
    )

    proteins_scope_df = proteins_df[proteins_df["matched_genus"].notna()].copy()
    proteins_scope_df = proteins_scope_df.merge(
        tax_table,
        left_on="matched_genus",
        right_on="genus",
        how="left",
    )
    proteins_scope_df["proteome_id"] = proteins_scope_df["isolate"].fillna(
        proteins_scope_df["ncbi_taxonomy_id"].astype(str)
    )

    n_matched = proteins_scope_df["matched_genus"].nunique()
    n_total = proteins_df["uniprot_genus"].nunique()
    logger.info(f"Genus matching: {n_matched}/{n_total} unique genera matched to WoM taxonomy")

    return proteins_scope_df, genus_matches_df


def _compute_taxon_function_enrichment(protein_ec_tax_df: pd.DataFrame) -> pd.DataFrame:
    """Compute tf-idf-style enrichment for EC numbers within each taxon."""
    all_rows = []

    for level in TAXONOMY_LEVELS:
        level_df = protein_ec_tax_df[protein_ec_tax_df[level].notna()].copy()
        if level_df.empty:
            continue

        docs = level_df[["proteome_id", level]].drop_duplicates()
        n_docs_per_taxon = docs.groupby(level)["proteome_id"].nunique().rename("n_proteomes_taxon")

        ec_docs = level_df[["proteome_id", level, "ec_number"]].drop_duplicates()
        counts = (
            ec_docs
            .groupby([level, "ec_number"])["proteome_id"]
            .nunique()
            .rename("n_proteomes_with_ec")
            .reset_index()
        )

        counts = counts.merge(
            n_docs_per_taxon.reset_index(),
            on=level,
            how="left",
        )

        counts["tf_frequency"] = counts["n_proteomes_with_ec"] / counts["n_proteomes_taxon"]

        n_taxa_level = docs[level].nunique()
        taxa_with_ec = counts.groupby("ec_number")[level].nunique().rename("n_taxa_with_ec")
        counts = counts.merge(taxa_with_ec.reset_index(), on="ec_number", how="left")
        counts["n_taxa_level"] = int(n_taxa_level)

        counts["idf"] = np.log((1.0 + counts["n_taxa_level"]) / (1.0 + counts["n_taxa_with_ec"])) + 1.0
        counts["tfidf_score"] = counts["tf_frequency"] * counts["idf"]

        counts["level"] = level
        counts = counts.rename(columns={level: "taxon"})

        all_rows.append(counts[[
            "level",
            "taxon",
            "ec_number",
            "n_proteomes_taxon",
            "n_proteomes_with_ec",
            "tf_frequency",
            "n_taxa_level",
            "n_taxa_with_ec",
            "idf",
            "tfidf_score",
        ]])

    if not all_rows:
        return pd.DataFrame(columns=[
            "level", "taxon", "ec_number", "n_proteomes_taxon", "n_proteomes_with_ec",
            "tf_frequency", "n_taxa_level", "n_taxa_with_ec", "idf", "tfidf_score",
        ])

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["level", "taxon", "tfidf_score"], ascending=[True, True, False])
    return out


def _compute_taxon_ec_compound_support(reactions_scope_df: pd.DataFrame) -> pd.DataFrame:
    """Compute support counts linking taxa, EC numbers, and WoM compounds."""
    filtered = reactions_scope_df[
        reactions_scope_df["in_model"] &
        reactions_scope_df["ec_number"].notna() &
        reactions_scope_df["representative_compound"].notna()
    ].copy()

    all_rows = []
    for level in TAXONOMY_LEVELS:
        level_df = filtered[filtered[level].notna()].copy()
        if level_df.empty:
            continue

        grouped = (
            level_df
            .groupby([level, "ec_number", "representative_compound"], dropna=False)
            .agg(
                n_reaction_rows=("primaryAccession", "size"),
                n_proteins_support=("primaryAccession", "nunique"),
                n_rhea_ids=("rhea_id", lambda s: s.dropna().nunique()),
            )
            .reset_index()
            .rename(columns={level: "taxon"})
        )
        grouped["level"] = level
        all_rows.append(grouped)

    if not all_rows:
        return pd.DataFrame(columns=[
            "level", "taxon", "ec_number", "representative_compound",
            "n_reaction_rows", "n_proteins_support", "n_rhea_ids",
        ])

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(
        ["level", "taxon", "ec_number", "n_reaction_rows"],
        ascending=[True, True, True, False],
    )
    return out


def build_function_enrichment_tables(
    force: bool = False,
    uniprot_tsv_gz_path: Optional[str | Path] = None,
    lineage_score_cutoff: int = 90,
) -> dict[str, pd.DataFrame]:
    """Build function-enrichment parquet tables used for descriptive taxon queries."""
    required_outputs = [
        CACHE_FILES["function_proteins"],
        CACHE_FILES["function_protein_ec"],
        CACHE_FILES["function_protein_reactions"],
        CACHE_FILES["function_reaction_catalog"],
        CACHE_FILES["taxon_function_enrichment"],
        CACHE_FILES["taxon_function_compound_support"],
    ]

    if not force and all(p.exists() for p in required_outputs):
        logger.info("Loading function enrichment tables from cache")
        return {
            "proteins": pd.read_parquet(CACHE_FILES["function_proteins"]),
            "protein_ec": pd.read_parquet(CACHE_FILES["function_protein_ec"]),
            "protein_reactions": pd.read_parquet(CACHE_FILES["function_protein_reactions"]),
            "reaction_catalog": pd.read_parquet(CACHE_FILES["function_reaction_catalog"]),
            "taxon_function_enrichment": pd.read_parquet(CACHE_FILES["taxon_function_enrichment"]),
            "taxon_function_compound_support": pd.read_parquet(CACHE_FILES["taxon_function_compound_support"]),
        }

    # Load WoM data first — needed for genus pre-filter and lineage matching
    exometabolite_df = load_exometabolite()
    wom_genera = exometabolite_df["genus"].dropna().unique().tolist()

    # Prefer an explicit path, then the configured default, then fall back to per-genus API.
    local_tsv: Optional[Path] = None
    if uniprot_tsv_gz_path:
        local_tsv = Path(uniprot_tsv_gz_path).expanduser()
        if not local_tsv.exists():
            raise FileNotFoundError(f"UniProt TSV not found at provided path: {local_tsv}")
    elif UNIPROT_ACTIVITY_TSV_GZ.exists():
        local_tsv = UNIPROT_ACTIVITY_TSV_GZ

    if local_tsv:
        logger.info(f"Building function enrichment tables from local file: {local_tsv}")
    else:
        logger.info(f"Streaming UniProt data per-genus for {len(wom_genera)} genera")

    proteins_df, protein_ec_df, reactions_df = _parse_uniprot_activity_tsv(
        wom_genera, tsv_gz_path=local_tsv
    )

    proteins_scope_df, genus_matches_df = _match_uniprot_genus_to_wom_taxonomy(
        proteins_df=proteins_df,
        exometabolite_df=exometabolite_df,
        score_cutoff=lineage_score_cutoff,
    )

    scoped_accessions = set(proteins_scope_df["primaryAccession"].dropna().unique())

    protein_ec_scope_df = protein_ec_df[protein_ec_df["primaryAccession"].isin(scoped_accessions)].copy()
    protein_ec_scope_df = protein_ec_scope_df.merge(
        proteins_scope_df[["primaryAccession", "proteome_id", *TAXONOMY_LEVELS]],
        on="primaryAccession",
        how="left",
    )

    reactions_scope_df = reactions_df[reactions_df["primaryAccession"].isin(scoped_accessions)].copy()
    reactions_scope_df = reactions_scope_df.merge(
        proteins_scope_df[["primaryAccession", "proteome_id", *TAXONOMY_LEVELS]],
        on="primaryAccession",
        how="left",
    )

    chebi_df = load_chebi()
    chebi_to_rep = (
        chebi_df[["compound_id", "best_name"]]
        .dropna(subset=["compound_id"])
        .drop_duplicates(subset=["compound_id"], keep="first")
        .set_index("compound_id")["best_name"]
    )

    reactions_scope_df["chebi_id_normalized"] = reactions_scope_df["chebi_id"].apply(_normalize_chebi_id)
    reactions_scope_df["representative_compound"] = reactions_scope_df["chebi_id_normalized"].map(chebi_to_rep)

    model_compounds = set(exometabolite_df["representative_compound"].dropna().unique())
    model_chebi_ids = set(exometabolite_df["chebi_id"].dropna().astype(str).str.strip().unique())

    reactions_scope_df["in_model_by_name"] = reactions_scope_df["representative_compound"].isin(model_compounds)
    reactions_scope_df["in_model_by_chebi"] = reactions_scope_df["chebi_id_normalized"].isin(model_chebi_ids)
    reactions_scope_df["in_model"] = reactions_scope_df["in_model_by_name"] | reactions_scope_df["in_model_by_chebi"]

    taxon_function_enrichment_df = _compute_taxon_function_enrichment(
        protein_ec_scope_df[protein_ec_scope_df["ec_number"].notna()].copy()
    )

    taxon_function_compound_support_df = _compute_taxon_ec_compound_support(reactions_scope_df)

    # Split reactions into a normalized pair: protein-reaction links + reaction-compound catalog
    protein_reactions_df = reactions_scope_df[
        ["primaryAccession", "rhea_id", "ec_number"]
    ].drop_duplicates().reset_index(drop=True)

    reaction_catalog_df = reactions_scope_df[
        ["rhea_id", "chebi_id", "reaction_name", "chebi_id_normalized",
         "representative_compound", "in_model_by_name", "in_model_by_chebi", "in_model"]
    ].drop_duplicates().reset_index(drop=True)

    proteins_scope_df.to_parquet(CACHE_FILES["function_proteins"], index=False)
    protein_ec_scope_df.to_parquet(CACHE_FILES["function_protein_ec"], index=False)
    protein_reactions_df.to_parquet(CACHE_FILES["function_protein_reactions"], index=False)
    reaction_catalog_df.to_parquet(CACHE_FILES["function_reaction_catalog"], index=False)
    taxon_function_enrichment_df.to_parquet(CACHE_FILES["taxon_function_enrichment"], index=False)
    taxon_function_compound_support_df.to_parquet(CACHE_FILES["taxon_function_compound_support"], index=False)

    logger.info(
        "Built function enrichment tables: "
        f"proteins={len(proteins_scope_df):,}, "
        f"protein_ec={len(protein_ec_scope_df):,}, "
        f"protein_reactions={len(protein_reactions_df):,}, "
        f"reaction_catalog={len(reaction_catalog_df):,}, "
        f"taxon_enrichment={len(taxon_function_enrichment_df):,}, "
        f"compound_support={len(taxon_function_compound_support_df):,}"
    )

    return {
        "proteins": proteins_scope_df,
        "protein_ec": protein_ec_scope_df,
        "protein_reactions": protein_reactions_df,
        "reaction_catalog": reaction_catalog_df,
        "taxon_function_enrichment": taxon_function_enrichment_df,
        "taxon_function_compound_support": taxon_function_compound_support_df,
        "genus_matches": genus_matches_df,
    }


def load_taxon_function_enrichment() -> pd.DataFrame:
    """Load taxon-level function enrichment table."""
    path = CACHE_FILES["taxon_function_enrichment"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_from_scratch.py to build enrichment tables."
        )
    return pd.read_parquet(path)


def load_taxon_function_compound_support() -> pd.DataFrame:
    """Load taxon/EC/compound support table."""
    path = CACHE_FILES["taxon_function_compound_support"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_from_scratch.py to build enrichment tables."
        )
    return pd.read_parquet(path)
