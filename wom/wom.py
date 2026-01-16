"""Web of Microbes query API."""

import logging
from functools import lru_cache

import pandas as pd
import numpy as np

from config import CACHE_FILES
from rapidfuzz import fuzz, process  # or thefuzz, or simple difflib

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_exo = None
_bpr_models = None
_kg = None


def _load_exo():
    global _exo
    if _exo is None:
        if not CACHE_FILES["exometabolite"].exists():
            raise RuntimeError("Cache not built. Run: python build.py")
        _exo = pd.read_parquet(CACHE_FILES["exometabolite"])
    return _exo


def _load_models():
    global _bpr_models
    if _bpr_models is None:
        from model import load_bpr_models
        _bpr_models = load_bpr_models()
    return _bpr_models


# ============================================================
# Taxonomy-level queries
# ============================================================

def what_consumes(
    genus: str = None,
    family: str = None,
    order: str = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    What metabolites are consumed by organisms in this taxonomic group?
    
    Aggregates BPR predictions across all matching isolates.
    
    Returns:
        DataFrame with chebi_id, chebi_name, mean_score, n_isolates
    """
    exo = _load_exo()
    models = _load_models()
    
    if "consumed" not in models:
        raise RuntimeError("Consumption model not available")
    
    wrapper = models["consumed"]
    
    # Find matching isolates
    mask = pd.Series(True, index=exo.index)
    if genus:
        mask &= exo["genus"].str.lower() == genus.lower()
    if family:
        mask &= exo["family"].str.lower() == family.lower()
    if order:
        mask &= exo["order"].str.lower() == order.lower()
    
    isolates = exo.loc[mask, "organism_id"].unique()
    isolates = [i for i in isolates if i in wrapper.microbe_to_idx]
    
    if not isolates:
        return pd.DataFrame(columns=["chebi_id", "chebi_name", "mean_score", "n_isolates"])
    
    # Aggregate predictions
    all_scores = []
    for isolate in isolates:
        m_idx = wrapper.microbe_to_idx[isolate]
        scores = wrapper.model.predict_all(m_idx)
        all_scores.append(scores)
    
    mean_scores = np.mean(all_scores, axis=0)
    
    # Build result
    top_indices = np.argsort(mean_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chebi_id = wrapper.idx_to_metabolite[idx]
        chebi_name = _chebi_id_to_name(chebi_id)
        results.append({
            "chebi_id": chebi_id,
            "chebi_name": chebi_name,
            "mean_score": float(mean_scores[idx]),
            "n_isolates": len(isolates),
        })
    
    return pd.DataFrame(results)


def what_secretes(
    genus: str = None,
    family: str = None,
    order: str = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """What metabolites are secreted by organisms in this taxonomic group?"""
    # Same pattern as what_consumes, using "secreted" model
    exo = _load_exo()
    models = _load_models()
    
    if "secreted" not in models:
        raise RuntimeError("Secretion model not available")
    
    wrapper = models["secreted"]
    
    mask = pd.Series(True, index=exo.index)
    if genus:
        mask &= exo["genus"].str.lower() == genus.lower()
    if family:
        mask &= exo["family"].str.lower() == family.lower()
    if order:
        mask &= exo["order"].str.lower() == order.lower()
    
    isolates = exo.loc[mask, "organism_id"].unique()
    isolates = [i for i in isolates if i in wrapper.microbe_to_idx]
    
    if not isolates:
        return pd.DataFrame(columns=["chebi_id", "chebi_name", "mean_score", "n_isolates"])
    
    all_scores = []
    for isolate in isolates:
        m_idx = wrapper.microbe_to_idx[isolate]
        scores = wrapper.model.predict_all(m_idx)
        all_scores.append(scores)
    
    mean_scores = np.mean(all_scores, axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chebi_id = wrapper.idx_to_metabolite[idx]
        results.append({
            "chebi_id": chebi_id,
            "chebi_name": _chebi_id_to_name(chebi_id),
            "mean_score": float(mean_scores[idx]),
            "n_isolates": len(isolates),
        })
    
    return pd.DataFrame(results)

def who_secretes(chebi_id: int = None, compound_name: str = None, top_k: int = 20) -> pd.DataFrame:
    """Which organisms secrete this metabolite?"""
    models = _load_models()
    exo = _load_exo()
    wrapper = models["secreted"]
    
    if chebi_id is None and compound_name:
        chebi_id = _name_to_chebi_id(compound_name)
    
    if chebi_id not in wrapper.metabolite_to_idx:
        return pd.DataFrame(columns=["organism_id", "score", "genus", "family"])
    
    c_idx = wrapper.metabolite_to_idx[chebi_id]
    metabolite_emb = wrapper.model.metabolite_embeddings.weight[c_idx].detach().numpy()
    microbe_embs = wrapper.model.microbe_embeddings.weight.detach().numpy()
    
    scores = microbe_embs @ metabolite_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        org_id = wrapper.idx_to_microbe[idx]
        tax = exo[exo["organism_id"] == org_id].iloc[0] if len(exo[exo["organism_id"] == org_id]) > 0 else {}
        results.append({
            "organism_id": org_id,
            "score": float(scores[idx]),
            "genus": tax.get("genus"),
            "family": tax.get("family"),
        })
    
    return pd.DataFrame(results)
# ============================================================
# Organism-level queries
# ============================================================

def predict_consumption(organism_id: str, top_k: int = 10) -> pd.DataFrame:
    """Predict top consumed metabolites for an organism."""
    models = _load_models()
    wrapper = models["consumed"]
    
    predictions = wrapper.predict(organism_id, top_k=top_k)
    
    return pd.DataFrame([
        {"chebi_id": cid, "chebi_name": _chebi_id_to_name(cid), "score": score}
        for cid, score in predictions
    ])


def predict_secretion(organism_id: str, top_k: int = 10) -> pd.DataFrame:
    """Predict top secreted metabolites for an organism."""
    models = _load_models()
    wrapper = models["secreted"]
    
    predictions = wrapper.predict(organism_id, top_k=top_k)
    
    return pd.DataFrame([
        {"chebi_id": cid, "chebi_name": _chebi_id_to_name(cid), "score": score}
        for cid, score in predictions
    ])


def similar_organisms(organism_id: str, top_k: int = 10, by: str = "consumed") -> pd.DataFrame:
    """Find organisms with similar metabolic profiles."""
    models = _load_models()
    exo = _load_exo()
    wrapper = models[by]
    
    similar = wrapper.similar_microbes(organism_id, top_k=top_k)
    
    results = []
    for other_id, similarity in similar:
        tax = exo[exo["organism_id"] == other_id].iloc[0] if len(exo[exo["organism_id"] == other_id]) > 0 else {}
        results.append({
            "organism_id": other_id,
            "similarity": similarity,
            "genus": tax.get("genus"),
            "family": tax.get("family"),
        })
    
    return pd.DataFrame(results)


# ============================================================
# Metabolite-level queries
# ============================================================

def who_consumes(chebi_id: int = None, compound_name: str = None, top_k: int = 20) -> pd.DataFrame:
    """Which organisms consume this metabolite?"""
    models = _load_models()
    exo = _load_exo()
    wrapper = models["consumed"]
    
    if chebi_id is None and compound_name:
        chebi_id = _name_to_chebi_id(compound_name)
    
    if chebi_id not in wrapper.metabolite_to_idx:
        return pd.DataFrame(columns=["organism_id", "score", "genus", "family"])
    
    c_idx = wrapper.metabolite_to_idx[chebi_id]
    metabolite_emb = wrapper.model.metabolite_embeddings.weight[c_idx].detach().numpy()
    microbe_embs = wrapper.model.microbe_embeddings.weight.detach().numpy()
    
    scores = microbe_embs @ metabolite_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        org_id = wrapper.idx_to_microbe[idx]
        tax = exo[exo["organism_id"] == org_id].iloc[0] if len(exo[exo["organism_id"] == org_id]) > 0 else {}
        results.append({
            "organism_id": org_id,
            "score": float(scores[idx]),
            "genus": tax.get("genus"),
            "family": tax.get("family"),
        })
    
    return pd.DataFrame(results)


# ============================================================
# Community queries
# ============================================================

def cross_feeding_potential(organism_a: str, organism_b: str, top_k: int = 10) -> dict:
    """
    Assess cross-feeding potential between two organisms.
    
    Returns metabolites that A secretes and B consumes, and vice versa.
    """
    models = _load_models()
    
    secreted_model = models.get("secreted")
    consumed_model = models.get("consumed")
    
    if not secreted_model or not consumed_model:
        raise RuntimeError("Both secretion and consumption models required")
    
    # A secretes, B consumes
    a_secretes = secreted_model.predict(organism_a, top_k=50)
    b_consumes = consumed_model.predict(organism_b, top_k=50)
    
    a_secretes_ids = {cid for cid, _ in a_secretes}
    b_consumes_ids = {cid for cid, _ in b_consumes}
    a_to_b = a_secretes_ids & b_consumes_ids
    
    # B secretes, A consumes
    b_secretes = secreted_model.predict(organism_b, top_k=50)
    a_consumes = consumed_model.predict(organism_a, top_k=50)
    
    b_secretes_ids = {cid for cid, _ in b_secretes}
    a_consumes_ids = {cid for cid, _ in a_consumes}
    b_to_a = b_secretes_ids & a_consumes_ids
    
    return {
        "a_feeds_b": [{"chebi_id": cid, "chebi_name": _chebi_id_to_name(cid)} for cid in list(a_to_b)[:top_k]],
        "b_feeds_a": [{"chebi_id": cid, "chebi_name": _chebi_id_to_name(cid)} for cid in list(b_to_a)[:top_k]],
    }


# ============================================================
# Helpers
# ============================================================


def search_taxonomy(
    query: str,
    rank: str = None,
    top_k: int = 5,
    threshold: float = 50.0,
) -> list[dict]:
    """
    Fuzzy search for taxonomic names.
    
    Args:
        query: Search string (e.g., "rhodanobacter", "burkolderia")
        rank: Limit to specific rank (genus, family, order, class, kingdom)
        top_k: Max results
        threshold: Minimum similarity score (0-100)
    
    Returns:
        List of {"name": str, "rank": str, "score": float}
    """
    exo = _load_exo()
    
    results = []
    ranks = [rank] if rank else ["genus", "family", "order", "class", "kingdom"]
    
    for r in ranks:
        names = exo[r].dropna().unique().tolist()
        if not names:
            continue
        # Normalize case for matching
        query_lower = query.lower()
        names_lower = [n.lower() for n in names]
        lower_to_original = dict(zip(names_lower, names))
        matches = process.extract(query_lower, names_lower, scorer=fuzz.WRatio, limit=top_k)
        for name_lower, score, _ in matches:
            original_name = lower_to_original[name_lower]
            if score >= threshold:
                results.append({"name": original_name, "rank": r, "score": score})
    
    # Sort by score, dedupe
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def search_metabolites(
    query: str,
    top_k: int = 5,
    threshold: float = 50.0,
) -> list[dict]:
    exo = _load_exo()
    
    metabolites = exo[["chebi_id", "chebi_name"]].dropna().drop_duplicates()
    name_to_id = dict(zip(metabolites["chebi_name"], metabolites["chebi_id"]))
    names = list(name_to_id.keys())
    
    # Normalize case for matching
    query_lower = query.lower()
    names_lower = [n.lower() for n in names]
    lower_to_original = dict(zip(names_lower, names))
    
    matches = process.extract(query_lower, names_lower, scorer=fuzz.WRatio, limit=top_k)
    
    results = []
    for name_lower, score, _ in matches:
        original_name = lower_to_original[name_lower]
        if score >= threshold:
            results.append({
                "chebi_id": int(name_to_id[original_name]),
                "name": original_name,
                "score": score,
            })
    
    return results


def resolve_taxon(query: str) -> dict | None:
    """
    Attempt to resolve a query to a single taxon.
    
    Returns best match if score > 80, else None.
    """
    matches = search_taxonomy(query, top_k=1, threshold=80.0)
    return matches[0] if matches else None


def resolve_metabolite(query: str) -> dict | None:
    """
    Attempt to resolve a query to a single metabolite.
    
    Returns best match if score > 80, else None.
    """
    matches = search_metabolites(query, top_k=1, threshold=80.0)
    return matches[0] if matches else None

@lru_cache(maxsize=1000)
def _chebi_id_to_name(chebi_id: int) -> str:
    """Look up ChEBI name from ID."""
    exo = _load_exo()
    match = exo[exo["chebi_id"] == chebi_id]["chebi_name"].dropna()
    return match.iloc[0] if len(match) > 0 else None


def _name_to_chebi_id(name: str) -> int:
    """Look up ChEBI ID from name."""
    exo = _load_exo()
    name_lower = name.lower()
    match = exo[exo["chebi_name"].str.lower() == name_lower]["chebi_id"].dropna()
    if len(match) > 0:
        return int(match.iloc[0])
    match = exo[exo["compound_name"].str.lower() == name_lower]["chebi_id"].dropna()
    return int(match.iloc[0]) if len(match) > 0 else None


def list_organisms(genus: str = None, family: str = None) -> list[str]:
    """List available organism IDs, optionally filtered by taxonomy."""
    exo = _load_exo()
    mask = pd.Series(True, index=exo.index)
    if genus:
        mask &= exo["genus"].str.lower() == genus.lower()
    if family:
        mask &= exo["family"].str.lower() == family.lower()
    return sorted(exo.loc[mask, "organism_id"].unique().tolist())


def list_metabolites() -> pd.DataFrame:
    """List all metabolites in the dataset."""
    exo = _load_exo()
    return exo[["chebi_id", "chebi_name", "compound_name"]].drop_duplicates().dropna(subset=["chebi_id"])