"""Query engine that executes query requests against trained models.

Hierarchical architecture:
    QueryEngine -> loads model and executes queries
    Individual handlers for each query type
    All use ModelArtifacts and dataclass queries
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.types import ModelArtifacts, PredictionResult
from core.io import load_model_artifacts
from core.training import compute_taxonomy_embeddings
from core.validation import InputMatchResult, InputValidationSummary
from loaders.compound_mapper import load_chebi, map_compound_name
from loaders.loaders import load_environments, load_exometabolite
from loaders.function_enrichment import (
    load_taxon_function_enrichment,
    load_taxon_function_compound_support,
)
from .query_types import (
    QueryConfig,
    QueryResponse,
    CompoundsForMicrobesQuery,
    MicrobesForCompoundsQuery,
    SelectiveMetabolitesQuery,
    MetaboliteEnvironmentFitnessQuery,
    NicheMetabolitesQuery,
    CompetitionAnalysisQuery,
    RankSubstratesQuery,
    SearchCompoundQuery,
    ListTaxaQuery,
    ModelInfoQuery,
    ListEnvironmentsQuery,
    SearchEnvironmentQuery,
    GetEnvironmentCompoundsQuery,
    ListFunctionEnrichmentTaxaQuery,
    EnrichedFunctionsForTaxonQuery,
    SubstrateGapQuery,
)

logger = logging.getLogger(__name__)


class QueryEngine:
    """Execute queries against a trained model."""

    @staticmethod
    def _infer_interaction_type(model_dir: Path) -> str:
        """Infer interaction type from model directory name."""
        model_name = model_dir.name.lower()
        if "secreted" in model_name:
            return "secreted"
        if "consumed" in model_name:
            return "consumed"
        return "unknown"

    @staticmethod
    def _resolve_model_dir(model_dir: Path, config: QueryConfig) -> Path:
        """Resolve model directory from config when a generic path is provided.

        Resolution rules:
        - If model_dir points to `trained_models/latest`, resolve to
          `trained_models/{model_type}_{interaction_type}` when interaction_type is explicit.
        - If model_dir points to `trained_models`, resolve similarly.
        - Otherwise, treat model_dir as an explicit model path and keep it unchanged.
        """
        model_dir = Path(model_dir)
        model_dir_name = model_dir.name.lower()

        if config.interaction_type in {"consumed", "secreted"}:
            resolved_name = f"{config.model_type}_{config.interaction_type}"

            if model_dir_name == "latest":
                return model_dir.parent / resolved_name

            if model_dir_name == "trained_models":
                return model_dir / resolved_name

        return model_dir
    
    def __init__(self, model_dir: str | Path, config: Optional[QueryConfig] = None):
        """Initialize query engine.
        
        Args:
            model_dir: Directory containing trained model artifacts
            config: Query configuration (uses defaults if None)
        """
        self.config = config or QueryConfig()
        requested_model_dir = Path(model_dir)

        # If interaction type was not specified, infer from requested model path.
        if self.config.interaction_type == "auto":
            self.config.interaction_type = self._infer_interaction_type(requested_model_dir)

        # Resolve generic model paths using model_type + interaction_type.
        self.model_dir = self._resolve_model_dir(requested_model_dir, self.config)

        # Re-infer interaction type from resolved path if still unknown.
        if self.config.interaction_type == "unknown":
            self.config.interaction_type = self._infer_interaction_type(self.model_dir)
        
        # Lazy-loaded
        self._artifacts: Optional[ModelArtifacts] = None
        self._chebi_data: Optional[dict] = None
        self._environment_data = None
        self._taxon_function_enrichment_data: Optional[pd.DataFrame] = None
        self._taxon_function_compound_support_data: Optional[pd.DataFrame] = None
        self._exometabolite_data: Optional[pd.DataFrame] = None

    def _base_metadata(self) -> dict:
        """Common metadata included in all query responses."""
        return {
            "model_dir": str(self.model_dir),
            "interaction_type": self.config.interaction_type,
        }
    
    @property
    def artifacts(self) -> ModelArtifacts:
        """Lazy-load model artifacts."""
        if self._artifacts is None:
            logger.info(f"Loading model from {self.model_dir}")
            self._artifacts = load_model_artifacts(self.model_dir)
            
            # Compute taxonomy embeddings if not already done
            if not self._artifacts.taxonomy_embeddings:
                logger.info("Computing taxonomy embeddings...")
                for level in ["genus", "family", "order", "class", "kingdom"]:
                    try:
                        emb = compute_taxonomy_embeddings(self._artifacts, level)
                        self._artifacts.taxonomy_embeddings[level] = emb
                    except Exception as e:
                        logger.warning(f"Could not compute {level} embeddings: {e}")
        
        return self._artifacts
    
    @property
    def chebi_data(self) -> dict:
        """Lazy-load ChEBI database."""
        if self._chebi_data is None:
            self._chebi_data = load_chebi()
        return self._chebi_data
    
    @property
    def environment_data(self):
        """Lazy-load environment data."""
        if self._environment_data is None:
            self._environment_data = load_environments()
        return self._environment_data

    @property
    def taxon_function_enrichment_data(self) -> pd.DataFrame:
        """Lazy-load taxon-function enrichment table."""
        if self._taxon_function_enrichment_data is None:
            self._taxon_function_enrichment_data = load_taxon_function_enrichment()
        return self._taxon_function_enrichment_data

    @property
    def taxon_function_compound_support_data(self) -> pd.DataFrame:
        """Lazy-load taxon/EC/compound support table."""
        if self._taxon_function_compound_support_data is None:
            self._taxon_function_compound_support_data = load_taxon_function_compound_support()
        return self._taxon_function_compound_support_data

    @property
    def exometabolite_data(self) -> pd.DataFrame:
        """Lazy-load raw exometabolite observation data."""
        if self._exometabolite_data is None:
            self._exometabolite_data = load_exometabolite()
        return self._exometabolite_data

    # =========================================================================
    # Observation Helpers
    # =========================================================================

    _OBS_LEVEL_COL = {
        "isolate": "organism_id",
        "genus": "genus",
        "family": "family",
        "order": "order",
        "class": "class",
        "kingdom": "kingdom",
    }

    def _fetch_observations(
        self,
        taxa: list[tuple[str, str]] | None = None,
        rep_compounds: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch raw observation rows from exometabolite data.

        Args:
            taxa: List of (level, name) tuples to filter by.  None = no taxon filter.
            rep_compounds: Representative compound names to filter by.  None = no compound filter.

        Returns:
            Tidy DataFrame with columns:
                organism_id, compound_name (raw, ungrouped), representative_compound,
                assertion_value, assertion, genus, family, order
        """
        df = self.exometabolite_data

        if taxa:
            mask = pd.Series(False, index=df.index)
            for level, name in taxa:
                col = self._OBS_LEVEL_COL.get(level)
                if col and col in df.columns:
                    mask |= (df[col] == name)
            df = df[mask]

        if rep_compounds:
            rep_set = set(rep_compounds)
            df = df[df["representative_compound"].isin(rep_set)]

        keep_cols = [
            c for c in [
                "organism_id", "compound_name", "representative_compound",
                "assertion_value", "assertion", "genus", "family", "order",
            ]
            if c in df.columns
        ]
        return df[keep_cols].reset_index(drop=True)

    @staticmethod
    def _safe_float(v) -> float | None:
        """Convert to float, returning None for NaN/None."""
        if v is None:
            return None
        try:
            f = float(v)
            return None if pd.isna(f) else f
        except (TypeError, ValueError):
            return None

    def _evidence_strength(
        self,
        n_consumed: int,
        n_neutral: int,
        n_secreted: int,
        n_tested: int,
        interaction_type: str | None,
    ) -> str:
        """Classify evidence strength relative to the predicted interaction direction."""
        if n_tested == 0:
            return "unvalidated"
        target_count = n_consumed if interaction_type == "consumed" else n_secreted if interaction_type == "secreted" else (n_consumed + n_secreted)
        non_target = n_tested - target_count
        if target_count == 0:
            return "contradicts_prediction"
        if non_target == 0:
            return "strong_positive" if target_count >= 3 else "positive"
        return "mixed"

    @staticmethod
    def _build_compact_observations(obs_df) -> dict:
        """Compact {genus: {organism_id: {rep_compound: assertion_value}}} dict.

        Drops redundant family/order/assertion columns (already in taxa_evidence_summary).
        Genus is the outer key for easy LLM navigation at genus query level.
        assertion_value interpretation: ≤ -2 = consumed, ≥ 2 = secreted, else neutral.
        """
        import pandas as pd
        if obs_df is None or obs_df.empty:
            return {}
        result: dict = {}
        for row in obs_df.itertuples(index=False):
            genus = row.genus
            isolate = row.organism_id
            compound = row.representative_compound
            av = row.assertion_value
            value = float(av) if not pd.isna(av) else None
            result.setdefault(genus, {}).setdefault(isolate, {})[compound] = value
        return result

    def _build_observation_metadata(
        self,
        taxa: list[tuple[str, str]],
        top_predicted_compounds: list[str],
        interaction_type: str | None = None,
    ) -> dict:
        """Build the 'observations' metadata block attached to every predictive response.

        Keys are ordered so the most narrative-ready summaries appear first:
          - evidence_narrative_data: bucketed taxon lists by evidence strength (lead with this)
          - taxa_evidence_summary: per-taxon counts, assertion values, isolate IDs
          - compounds_with_raw_name_variants: rep compounds that appear under multiple raw names
          - prediction_conflicts: observation rows that contradict predicted direction
          - untested_top_predictions: top predictions with zero empirical data
          - n_isolates_with_data / n_compounds_observed: quick coverage counts
          - isolate_observations: compact {genus → isolate_id → {compound: assertion_value}} dict

        Args:
            taxa: (level, name) pairs that were queried.
            top_predicted_compounds: Rep compound names from top-K predictions, in rank order.
            interaction_type: 'consumed' or 'secreted' — sets conflict and strength logic.
        """
        obs_df = self._fetch_observations(taxa=taxa, rep_compounds=top_predicted_compounds)

        observed_compounds = set(obs_df["representative_compound"].unique()) if not obs_df.empty else set()
        untested = [c for c in top_predicted_compounds if c not in observed_compounds]

        # ── per-taxon evidence summary ──────────────────────────────────────────
        taxa_evidence_summary: dict = {}
        for level, name in taxa:
            col = self._OBS_LEVEL_COL.get(level)
            if not col or col not in obs_df.columns or obs_df.empty:
                taxa_evidence_summary[name] = {"n_tested": 0, "evidence_strength": "unvalidated"}
                continue

            taxon_obs = obs_df[obs_df[col] == name]
            if taxon_obs.empty:
                taxa_evidence_summary[name] = {"n_tested": 0, "evidence_strength": "unvalidated"}
                continue

            n_consumed = int((taxon_obs["assertion"] == "consumed").sum())
            n_neutral  = int((taxon_obs["assertion"] == "neutral").sum())
            n_secreted = int((taxon_obs["assertion"] == "secreted").sum())
            n_tested   = len(taxon_obs)
            avs = [self._safe_float(v) for v in taxon_obs["assertion_value"].tolist()]
            avs = [v for v in avs if v is not None]

            taxa_evidence_summary[name] = {
                "n_tested":   n_tested,
                "n_consumed": n_consumed,
                "n_neutral":  n_neutral,
                "n_secreted": n_secreted,
                "evidence_strength": self._evidence_strength(
                    n_consumed, n_neutral, n_secreted, n_tested, interaction_type
                ),
                "assertion_values":  avs,
                "tested_isolates":   taxon_obs["organism_id"].unique().tolist(),
                "tested_compounds":  taxon_obs["representative_compound"].unique().tolist(),
            }

        # ── narrative buckets (Idea 3: surface first) ───────────────────────────
        def _bucket(strength: str) -> list[str]:
            return [k for k, v in taxa_evidence_summary.items() if v["evidence_strength"] == strength]

        evidence_narrative_data = {
            "taxa_strong_positive_evidence": _bucket("strong_positive"),
            "taxa_positive_evidence":        _bucket("positive"),
            "taxa_mixed_evidence":           _bucket("mixed"),
            "taxa_contradicting_prediction": _bucket("contradicts_prediction"),
            "taxa_unvalidated":              _bucket("unvalidated"),
            "note": (
                "Use this bucket list as the PRIMARY scaffold for your response narrative. "
                "State observed evidence first for any taxon that has data, then use model "
                "predictions only for taxa in taxa_unvalidated."
            ),
        }

        # ── compound name variants (Idea 4: stereoisomer transparency) ──────────
        compounds_with_raw_name_variants: dict = {}
        if not obs_df.empty and "compound_name" in obs_df.columns:
            for rep in observed_compounds:
                raw_names = obs_df.loc[
                    obs_df["representative_compound"] == rep, "compound_name"
                ].unique().tolist()
                if len(raw_names) > 1:
                    compounds_with_raw_name_variants[rep] = {
                        "n_distinct_raw_names": len(raw_names),
                        "raw_names": sorted(raw_names),
                        "note": (
                            "Multiple distinct raw compound names are grouped under this "
                            "representative. Stereoisomers, diastereomers, or naming variants "
                            "may be present — mention this if stereochemistry is relevant."
                        ),
                    }

        # ── prediction conflicts ─────────────────────────────────────────────────
        conflicts = []
        if interaction_type and not obs_df.empty:
            for _, row in obs_df.iterrows():
                assertion = row.get("assertion", "")
                compound  = row.get("representative_compound", "")
                is_conflict = (
                    (interaction_type == "consumed" and assertion in {"secreted", "neutral"})
                    or (interaction_type == "secreted" and assertion in {"consumed", "neutral"})
                )
                if is_conflict:
                    try:
                        predicted_rank = top_predicted_compounds.index(compound) + 1
                    except ValueError:
                        predicted_rank = None
                    conflicts.append({
                        "compound":           compound,
                        "compound_name_raw":  row.get("compound_name"),
                        "predicted_rank":     predicted_rank,
                        "observed_assertion": assertion,
                        "assertion_value":    self._safe_float(row.get("assertion_value")),
                        "organism_id":        row.get("organism_id"),
                        "genus":              row.get("genus"),
                    })

        # ── return with summary fields first (Idea 3) ───────────────────────────
        return {
            "evidence_narrative_data":       evidence_narrative_data,
            "taxa_evidence_summary":         taxa_evidence_summary,
            "compounds_with_raw_name_variants": compounds_with_raw_name_variants,
            "prediction_conflicts":          conflicts,
            "untested_top_predictions":      untested,
            "n_isolates_with_data":          int(obs_df["organism_id"].nunique()) if not obs_df.empty else 0,
            "n_compounds_observed":          len(observed_compounds),
            "isolate_observations":          self._build_compact_observations(obs_df),
            "isolate_observations_note": (
                "Structure: genus → isolate_id → {representative_compound: assertion_value}. "
                "Interpret assertion_value: ≤ -2 = consumed, ≥ 2 = secreted, else neutral."
            ),
        }

    # =========================================================================
    # Input Validation Helpers
    # =========================================================================
    
    def validate_metabolites(self, metabolite_names: list[str]) -> InputValidationSummary:
        """Validate a list of metabolite names, attempting to match each one."""
        results = []
        
        for name in metabolite_names:
            # Try to map the compound name
            mapping = map_compound_name(name, self.chebi_data)
            
            if mapping.representative_name is None:
                results.append(InputMatchResult(
                    requested=name,
                    matched=None,
                    match_type='not_found',
                    match_score=0.0,
                    reason=f"No match found in ChEBI database"
                ))
            elif mapping.representative_name not in self.artifacts.training_data.metabolite_to_idx:
                results.append(InputMatchResult(
                    requested=name,
                    matched=mapping.representative_name,
                    match_type='not_in_model',
                    match_score=mapping.match_score,
                    reason=f"Matched '{mapping.representative_name}' but not in training data"
                ))
            else:
                results.append(InputMatchResult(
                    requested=name,
                    matched=mapping.representative_name,
                    match_type=mapping.match_type,
                    match_score=mapping.match_score,
                    reason=None
                ))
        
        return InputValidationSummary.from_results(results)
    
    def validate_taxa(self, taxa_queries: list[tuple[str, str]]) -> InputValidationSummary:
        """Validate a list of taxonomic queries (level, name)."""
        results = []
        
        for level, name in taxa_queries:
            if level == "isolate":
                # Check if isolate exists in training data
                if name in self.artifacts.training_data.microbe_to_idx:
                    results.append(InputMatchResult(
                        requested=f"{level}:{name}",
                        matched=f"{level}:{name}",
                        match_type='exact',
                        match_score=1.0,
                        reason=None
                    ))
                else:
                    results.append(InputMatchResult(
                        requested=f"{level}:{name}",
                        matched=None,
                        match_type='not_found',
                        match_score=0.0,
                        reason=f"Isolate '{name}' not in training data"
                    ))
            else:
                # Check if taxon exists at this level
                taxa_at_level = self.artifacts.get_taxa_at_level(level)
                if name in taxa_at_level:
                    results.append(InputMatchResult(
                        requested=f"{level}:{name}",
                        matched=f"{level}:{name}",
                        match_type='exact',
                        match_score=1.0,
                        reason=None
                    ))
                else:
                    results.append(InputMatchResult(
                        requested=f"{level}:{name}",
                        matched=None,
                        match_type='not_found',
                        match_score=0.0,
                        reason=f"{level.capitalize()} '{name}' not found in model"
                    ))
        
        return InputValidationSummary.from_results(results)
    
    # =========================================================================
    # Handler: Compounds for Microbes
    # =========================================================================
    
    def query_compounds_for_microbes(
        self,
        query: CompoundsForMicrobesQuery
    ) -> QueryResponse:
        """Find compounds consumed by given microbes."""
        try:
            # Validate all microbe inputs upfront
            validation = self.validate_taxa(query.microbes)
            
            # If NO microbes matched, fail immediately
            if validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="compounds_for_microbes",
                    results=[],
                    metadata={**self._base_metadata(), "input_validation": validation.to_dict()},
                    error=f"None of the {validation.total_requested} microbes could be matched. Check spelling and availability."
                )
            
            results = []
            
            # Only process successfully matched microbes
            for match_result in validation.matched_items:
                # Parse back the level:name from matched string
                # Note: matched is just the name, we need to find the original query
                level, name = None, None
                for orig_level, orig_name in query.microbes:
                    if match_result.requested == f"{orig_level}:{orig_name}":
                        level, name = orig_level, orig_name
                        break
                
                if level is None:
                    continue
                
                explanation = {} if query.explain else None
                
                if query.explain:
                    # Show matching status
                    explanation["input_matching"] = {
                        "requested": match_result.requested,
                        "matched": match_result.matched,
                        "match_type": match_result.match_type,
                    }
                    
                    # Show exactly what's happening
                    if level == "isolate":
                        idx = self.artifacts.training_data.microbe_to_idx.get(name)
                        explanation["embedding_source"] = f"Direct isolate lookup (not aggregated)"
                        explanation["isolate_index"] = idx
                    else:
                        # Get isolates in this taxon
                        isolates = []
                        for org_id, tax in self.artifacts.training_data.microbe_taxonomy.items():
                            if tax.get_level(level) == name:
                                isolates.append(org_id)
                        
                        explanation["embedding_source"] = f"Average of all isolates in this {level}"
                        explanation["n_isolates_averaged"] = len(isolates)
                        explanation["isolates_in_taxon"] = isolates[:10]  # Show first 10
                    
                    explanation["prediction_method"] = "dot_product: microbe_embedding @ metabolite_embeddings.T"
                    explanation["n_metabolites_scored"] = self.artifacts.training_data.n_metabolites
                
                # Get prediction
                prediction = self.artifacts.predict_with_stats(
                    level=level,
                    name=name,
                    compute_distinctive=query.compute_distinctive,
                    background_level=query.background_level,
                )
                
                if query.explain:
                    explanation["score_statistics"] = {
                        "mean": float(prediction.mean_score),
                        "median": float(prediction.median_score),
                        "std": float(prediction.std_score),
                    }
                    if query.compute_distinctive:
                        explanation["distinctive_computation"] = f"raw_score - mean(all_{query.background_level or level}_scores)"
                        explanation["n_background_taxa"] = len(self.artifacts.taxonomy_embeddings.get(query.background_level or level, {}))
                
                # Filter by min_score if specified
                top_result = prediction.top_k(query.top_k, by="distinctive" if query.compute_distinctive else "raw")
                
                # Build result for this microbe
                microbe_result = {
                    "microbe": {
                        "level": level,
                        "name": name,
                    },
                    "embedding_diversity": self.artifacts._taxon_embedding_diversity(level, name),
                    "compounds": [
                        {
                            "name": top_result.metabolite_names[i],
                            "raw_score": float(top_result.raw_scores[i]),
                            "distinctive_score": float(top_result.distinctive_scores[i]) if top_result.distinctive_scores is not None else None,
                            "rank": int(top_result.ranks[i]) if top_result.ranks is not None else None,
                            "percentile": float(top_result.percentiles[i]) if top_result.percentiles is not None else None,
                            "score_sd": float(top_result.score_sd[i]) if top_result.score_sd is not None else None,
                            "score_se": float(top_result.score_se[i]) if top_result.score_se is not None else None,
                        }
                        for i in range(len(top_result.metabolite_names))
                        if query.min_score is None or top_result.raw_scores[i] >= query.min_score
                    ],
                    "statistics": {
                        "mean_score": prediction.mean_score,
                        "median_score": prediction.median_score,
                        "std_score": prediction.std_score,
                        "n_positive_training": prediction.n_positive_training,
                    }
                }
                
                if query.explain:
                    microbe_result["explanation"] = explanation

                results.append(microbe_result)

            # Collect matched taxa and top compound names for observation lookup
            matched_taxa = []
            top_compounds = []
            for r in results:
                lvl = r["microbe"]["level"]
                nm = r["microbe"]["name"]
                matched_taxa.append((lvl, nm))
                for c in r["compounds"]:
                    if c["name"] not in top_compounds:
                        top_compounds.append(c["name"])

            return QueryResponse(
                success=True,
                query_type="compounds_for_microbes",
                results=results,
                metadata={
                    **self._base_metadata(),
                    "n_microbes_queried": len(query.microbes),
                    "top_k": query.top_k,
                    "observations": self._build_observation_metadata(
                        taxa=matched_taxa,
                        top_predicted_compounds=top_compounds,
                        interaction_type=self.config.interaction_type,
                    ),
                }
            )
        
        except Exception as e:
            logger.error(f"Error in compounds_for_microbes query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="compounds_for_microbes",
                results=[],
                metadata=self._base_metadata(),
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Microbes for Compounds
    # =========================================================================
    
    def query_microbes_for_compounds(
        self,
        query: MicrobesForCompoundsQuery
    ) -> QueryResponse:
        """Find microbes that consume given compounds."""
        try:
            # Validate metabolites upfront
            validation = self.validate_metabolites(query.compounds)
            
            # Fail fast if nothing matched
            if validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="microbes_for_compounds",
                    results=[],
                    metadata={**self._base_metadata(), "input_validation": validation.to_dict()},
                    error=f"None of the {validation.total_requested} compounds could be matched or are in the model"
                )
            
            # Get all taxa at requested level
            taxa = self.artifacts.get_taxa_at_level(query.level)

            # Build compound indices list from matched items
            compound_indices = []
            mapped_names = []
            for match_result in validation.matched_items:
                idx = self.artifacts.training_data.metabolite_to_idx.get(match_result.matched)
                if idx is not None:
                    compound_indices.append(idx)
                    mapped_names.append(match_result.matched)

            # Precompute isolate counts per taxon (used in result and for score_sd)
            taxon_isolate_counts: dict[str, int] = {}
            for org_id, tax in self.artifacts.training_data.microbe_taxonomy.items():
                if org_id not in self.artifacts.training_data.microbe_to_idx:
                    continue
                t = tax.get_level(query.level)
                if t:
                    taxon_isolate_counts[t] = taxon_isolate_counts.get(t, 0) + 1

            # Score each taxon
            taxon_scores = []
            for taxon_name in taxa:
                try:
                    explanation = {} if query.explain else None

                    if query.explain:
                        explanation["taxon"] = taxon_name
                        explanation["level"] = query.level
                        explanation["n_compounds_scored"] = len(compound_indices)
                        explanation["agg_method"] = query.agg_method

                    scores = self.artifacts.predict_for_taxon(query.level, taxon_name)
                    compound_scores = scores[compound_indices]

                    # Aggregate scores
                    if query.agg_method == "mean":
                        agg_score = float(np.mean(compound_scores))
                    elif query.agg_method == "sum":
                        agg_score = float(np.sum(compound_scores))
                    elif query.agg_method == "max":
                        agg_score = float(np.max(compound_scores))
                    else:
                        agg_score = float(np.mean(compound_scores))

                    # Per-compound SD from isolate embedding variance
                    score_sd_arr, _ = self.artifacts._taxon_score_sd(query.level, taxon_name)
                    if score_sd_arr is not None:
                        compound_score_sd = {
                            mapped_names[i]: round(float(score_sd_arr[compound_indices[i]]), 4)
                            for i in range(len(mapped_names))
                        }
                    else:
                        compound_score_sd = None

                    result = {
                        "taxon": taxon_name,
                        "level": query.level,
                        "aggregate_score": agg_score,
                        "n_isolates_in_model": taxon_isolate_counts.get(taxon_name, 0),
                        "embedding_diversity": self.artifacts._taxon_embedding_diversity(query.level, taxon_name),
                        "compound_scores": {
                            mapped_names[i]: float(compound_scores[i])
                            for i in range(len(mapped_names))
                        },
                        "compound_score_sd": compound_score_sd,
                    }
                    
                    if query.explain:
                        explanation["score_range"] = {
                            "min": float(np.min(compound_scores)),
                            "max": float(np.max(compound_scores)),
                            "std": float(np.std(compound_scores))
                        }
                        result["explanation"] = explanation
                    
                    taxon_scores.append(result)
                except Exception as e:
                    logger.warning(f"Could not score {taxon_name}: {e}")
            
            # Sort and filter
            taxon_scores.sort(key=lambda x: x["aggregate_score"], reverse=True)
            
            if query.min_score is not None:
                taxon_scores = [t for t in taxon_scores if t["aggregate_score"] >= query.min_score]
            
            results = taxon_scores[:query.top_k]
            
            # Collect returned taxa for observation lookup
            returned_taxa = [(r["level"], r["taxon"]) for r in results]

            # Build metadata with validation info
            metadata = {
                **self._base_metadata(),
                "compounds_queried": mapped_names,
                "n_compounds": len(mapped_names),
                "level": query.level,
                "agg_method": query.agg_method,
                "n_taxa_scored": len(taxon_scores),
                "input_validation": validation.to_dict(),
                # Observations fetched by compound (no taxon filter) to show all empirical data
                "observations": self._build_observation_metadata(
                    taxa=returned_taxa,
                    top_predicted_compounds=mapped_names,
                    interaction_type=self.config.interaction_type,
                ),
            }

            if validation.should_warn:
                metadata["warnings"] = validation.warnings

            return QueryResponse(
                success=True,
                query_type="microbes_for_compounds",
                results=results,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error in microbes_for_compounds query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="microbes_for_compounds",
                results=[],
                metadata=self._base_metadata(),
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Selective Metabolites
    # =========================================================================
    
    def query_selective_metabolites(
        self,
        query: SelectiveMetabolitesQuery
    ) -> QueryResponse:
        """Find metabolites that favor target over competitors."""
        try:
            # Validate target and competitors
            all_taxa = [query.target] + query.competitors
            validation = self.validate_taxa(all_taxa)
            
            # Fail fast if target or no competitors matched
            if validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="selective_metabolites",
                    results=[],
                    metadata={"input_validation": validation.to_dict()},
                    error=f"Could not match target or competitors: {validation.total_failed} of {validation.total_requested} taxa failed"
                )
            
            # Separate target from competitors in validated results
            target_match = None
            competitor_matches = []
            for match_result in validation.matched_items:
                if match_result.requested == f"{query.target[0]}:{query.target[1]}":
                    target_match = match_result
                else:
                    competitor_matches.append(match_result)
            
            if target_match is None:
                return QueryResponse(
                    success=False,
                    query_type="selective_metabolites",
                    results=[],
                    error="Target microbe could not be matched"
                )
            
            if not competitor_matches:
                return QueryResponse(
                    success=False,
                    query_type="selective_metabolites",
                    results=[],
                    error="No competitor microbes could be matched"
                )
            
            # Parse matched target
            target_level, target_name = target_match.matched.split(":", 1)
            
            # Get target scores
            
            if query.use_distinctive:
                target_pred = self.artifacts.predict_with_stats(
                    level=target_level,
                    name=target_name,
                    compute_distinctive=True
                )
                target_scores = target_pred.distinctive_scores
            else:
                target_scores = self.artifacts.predict_for_taxon(target_level, target_name)
            
            # Get competitor scores
            competitor_scores_list = []
            for match_result in competitor_matches:
                comp_level, comp_name = match_result.matched.split(":", 1)
                if query.use_distinctive:
                    pred = self.artifacts.predict_with_stats(
                        level=comp_level,
                        name=comp_name,
                        compute_distinctive=True
                    )
                    scores = pred.distinctive_scores
                else:
                    scores = self.artifacts.predict_for_taxon(comp_level, comp_name)
                competitor_scores_list.append(scores)
            
            # Compute selectivity
            competitor_mean = np.mean(competitor_scores_list, axis=0)
            
            if query.method == "difference":
                selectivity = target_scores - competitor_mean
            elif query.method == "ratio":
                # Avoid division by zero
                selectivity = target_scores / (competitor_mean + 1e-6)
            else:
                selectivity = target_scores - competitor_mean
            
            # Build results
            idx_to_metabolite = {idx: name for name, idx in self.artifacts.training_data.metabolite_to_idx.items()}
            
            # Filter and sort
            metabolite_results = []
            for i in range(len(selectivity)):
                if query.min_selectivity is not None and selectivity[i] < query.min_selectivity:
                    continue
                
                result = {
                    "metabolite": idx_to_metabolite[i],
                    "selectivity": float(selectivity[i]),
                    "target_score": float(target_scores[i]),
                    "competitor_mean_score": float(competitor_mean[i]),
                }
                
                if query.explain:
                    result["explanation"] = {
                        "score_type": "distinctive" if query.use_distinctive else "raw",
                        "selectivity_method": query.method,
                        "n_competitors": len(query.competitors)
                    }
                
                metabolite_results.append(result)
            
            metabolite_results.sort(key=lambda x: x["selectivity"], reverse=True)
            results = metabolite_results[:query.top_k]
            
            # Collect all taxa (target + competitors) and top compound names for obs lookup
            all_taxa = [(target_level, target_name)]
            for match_result in competitor_matches:
                cl, cn = match_result.matched.split(":", 1)
                all_taxa.append((cl, cn))
            top_metabolites = [r["metabolite"] for r in results]

            # Build metadata with validation info
            metadata = {
                "target": {"level": target_level, "name": target_name},
                "n_competitors": len(competitor_matches),
                "method": query.method,
                "input_validation": validation.to_dict(),
                "observations": self._build_observation_metadata(
                    taxa=all_taxa,
                    top_predicted_compounds=top_metabolites,
                    interaction_type=self.config.interaction_type,
                ),
            }

            if validation.should_warn:
                metadata["warnings"] = validation.warnings

            return QueryResponse(
                success=True,
                query_type="selective_metabolites",
                results=results,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error in selective_metabolites query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="selective_metabolites",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Metabolite Environment Fitness
    # =========================================================================
    
    def query_metabolite_environment_fitness(
        self,
        query: MetaboliteEnvironmentFitnessQuery
    ) -> QueryResponse:
        """Rank microbes by fitness in a metabolite environment."""
        try:
            # Validate metabolites upfront
            metabolite_validation = self.validate_metabolites(query.metabolites)
            
            # Fail fast if no metabolites matched
            if metabolite_validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="metabolite_environment_fitness",
                    results=[],
                    metadata={"input_validation": metabolite_validation.to_dict()},
                    error=f"None of the {metabolite_validation.total_requested} metabolites could be matched"
                )
            
            # Build metabolite indices from matched items
            metabolite_indices = []
            mapped_names = []
            for match_result in metabolite_validation.matched_items:
                idx = self.artifacts.training_data.metabolite_to_idx.get(match_result.matched)
                if idx is not None:
                    metabolite_indices.append(idx)
                    mapped_names.append(match_result.matched)
            
            # Validate microbes if specified
            microbe_validation = None
            if query.microbes:
                microbe_validation = self.validate_taxa(query.microbes)
                
                if microbe_validation.should_fail:
                    return QueryResponse(
                        success=False,
                        query_type="metabolite_environment_fitness",
                        results=[],
                        metadata={
                            "metabolite_validation": metabolite_validation.to_dict(),
                            "microbe_validation": microbe_validation.to_dict()
                        },
                        error=f"None of the {microbe_validation.total_requested} microbes could be matched"
                    )
            
            # Determine which microbes to score
            if query.microbes and microbe_validation:
                # Score specific microbes (only matched ones)
                microbe_scores = []
                for match_result in microbe_validation.matched_items:
                    level, name = match_result.matched.split(":", 1)
                    try:
                        if query.use_distinctive:
                            pred = self.artifacts.predict_with_stats(
                                level=level,
                                name=name,
                                compute_distinctive=True
                            )
                            scores = pred.distinctive_scores
                        else:
                            scores = self.artifacts.predict_for_taxon(level, name)
                        
                        met_scores = scores[metabolite_indices]
                        
                        if query.fitness_method == "sum":
                            fitness = float(np.sum(met_scores))
                        elif query.fitness_method == "mean":
                            fitness = float(np.mean(met_scores))
                        elif query.fitness_method == "weighted":
                            # Weight by score magnitude
                            fitness = float(np.sum(met_scores ** 2))
                        else:
                            fitness = float(np.sum(met_scores))
                        
                        result = {
                            "microbe": name,
                            "level": level,
                            "fitness": fitness,
                            "metabolite_scores": {
                                mapped_names[i]: float(met_scores[i])
                                for i in range(len(mapped_names))
                            }
                        }
                        
                        if query.explain:
                            result["explanation"] = {
                                "score_type": "distinctive" if query.use_distinctive else "raw",
                                "fitness_method": query.fitness_method,
                                "n_metabolites": len(mapped_names)
                            }
                        
                        microbe_scores.append(result)
                    except Exception as e:
                        logger.warning(f"Could not score {name}: {e}")
            else:
                # Score all taxa at given level
                taxa = self.artifacts.get_taxa_at_level(query.level)
                microbe_scores = []
                
                for taxon_name in taxa:
                    try:
                        if query.use_distinctive:
                            pred = self.artifacts.predict_with_stats(
                                level=query.level,
                                name=taxon_name,
                                compute_distinctive=True
                            )
                            scores = pred.distinctive_scores
                        else:
                            scores = self.artifacts.predict_for_taxon(query.level, taxon_name)
                        
                        met_scores = scores[metabolite_indices]
                        
                        if query.fitness_method == "sum":
                            fitness = float(np.sum(met_scores))
                        elif query.fitness_method == "mean":
                            fitness = float(np.mean(met_scores))
                        elif query.fitness_method == "weighted":
                            fitness = float(np.sum(met_scores ** 2))
                        else:
                            fitness = float(np.sum(met_scores))
                        
                        result = {
                            "microbe": taxon_name,
                            "level": query.level,
                            "fitness": fitness,
                            "metabolite_scores": {
                                mapped_names[i]: float(met_scores[i])
                                for i in range(len(mapped_names))
                            }
                        }
                        
                        if query.explain:
                            result["explanation"] = {
                                "score_type": "distinctive" if query.use_distinctive else "raw",
                                "fitness_method": query.fitness_method,
                                "n_metabolites": len(mapped_names)
                            }
                        
                        microbe_scores.append(result)
                    except Exception as e:
                        logger.warning(f"Could not score {taxon_name}: {e}")
            
            # Sort and return top K
            microbe_scores.sort(key=lambda x: x["fitness"], reverse=True)
            results = microbe_scores[:query.top_k]
            
            # Build metadata with validation info
            metadata = {
                "metabolites": mapped_names,
                "n_metabolites": len(mapped_names),
                "fitness_method": query.fitness_method,
                "n_microbes_scored": len(microbe_scores),
                "metabolite_validation": metabolite_validation.to_dict(),
            }
            
            if microbe_validation:
                metadata["microbe_validation"] = microbe_validation.to_dict()
            
            # Collect all warnings
            warnings = []
            if metabolite_validation.should_warn:
                warnings.extend(metabolite_validation.warnings)
            if microbe_validation and microbe_validation.should_warn:
                warnings.extend(microbe_validation.warnings)
            
            if warnings:
                metadata["warnings"] = warnings
            
            return QueryResponse(
                success=True,
                query_type="metabolite_environment_fitness",
                results=results,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error in metabolite_environment_fitness query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="metabolite_environment_fitness",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Niche Metabolites
    # =========================================================================
    
    def query_niche_metabolites(
        self,
        query: NicheMetabolitesQuery
    ) -> QueryResponse:
        """Find unique metabolite preferences (distinctive scores)."""
        try:
            # Validate the single microbe
            validation = self.validate_taxa([query.microbe])
            
            # Fail fast if microbe not found
            if validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="niche_metabolites",
                    results=[],
                    metadata={"input_validation": validation.to_dict()},
                    error=f"Microbe '{query.microbe[1]}' at level '{query.microbe[0]}' could not be matched"
                )
            
            # Get the matched microbe
            match_result = validation.matched_items[0]
            level, name = match_result.matched.split(":", 1)
            
            # Determine background level
            background_level = query.background_level
            if background_level is None and level == "isolate":
                background_level = "genus"
            
            # Get prediction with distinctive scores
            prediction = self.artifacts.predict_with_stats(
                level=level,
                name=name,
                compute_distinctive=True,
                background_level=background_level,
            )
            
            # Get top K by distinctive score
            top = prediction.top_k(query.top_k, by="distinctive")
            
            # Filter by min_distinctive_score
            results = []
            for i in range(len(top.metabolite_names)):
                if query.min_distinctive_score is not None:
                    if top.distinctive_scores[i] < query.min_distinctive_score:
                        continue
                
                result = {
                    "metabolite": top.metabolite_names[i],
                    "distinctive_score": float(top.distinctive_scores[i]),
                    "raw_score": float(top.raw_scores[i]),
                    "rank": int(top.ranks[i]),
                    "percentile": float(top.percentiles[i]),
                    "score_sd": float(top.score_sd[i]) if top.score_sd is not None else None,
                    "score_se": float(top.score_se[i]) if top.score_se is not None else None,
                }
                
                if query.explain:
                    result["explanation"] = {
                        "score_type": "distinctive (always for niche query)",
                        "background_level": background_level,
                        "n_background_taxa": prediction.n_isolates_aggregated,
                        "computation": "raw_score - background_mean"
                    }
                
                results.append(result)
            
            top_metabolites = [r["metabolite"] for r in results]

            metadata = {
                "microbe": {
                    "level": level,
                    "name": name,
                    "embedding_diversity": self.artifacts._taxon_embedding_diversity(level, name),
                },
                "background_level": background_level,
                "background_mean": prediction.background_mean,
                "n_isolates_in_background": prediction.n_isolates_aggregated,
                "input_validation": validation.to_dict(),
                "observations": self._build_observation_metadata(
                    taxa=[(level, name)],
                    top_predicted_compounds=top_metabolites,
                    interaction_type=self.config.interaction_type,
                ),
            }

            if validation.should_warn:
                metadata["warnings"] = validation.warnings

            return QueryResponse(
                success=True,
                query_type="niche_metabolites",
                results=results,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error in niche_metabolites query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="niche_metabolites",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Search Compound
    # =========================================================================
    
    def query_search_compound(
        self,
        query: SearchCompoundQuery
    ) -> QueryResponse:
        """Search for compound name matches."""
        try:
            if query.fuzzy:
                # Use fuzzy matching
                result = map_compound_name(
                    query.query,
                    self.chebi_data,
                    score_cutoff=query.min_score,
                )
                
                if result:
                    results = [{
                        "query_name": result.query_name,
                        "matched_name": result.representative_name,
                        "chebi_id": result.chebi_id,
                        "match_score": result.match_score,
                        "match_type": result.match_type,
                        "in_training_data": result.representative_name in self.artifacts.training_data.metabolite_to_idx,
                    }]
                else:
                    results = []
            else:
                # Exact match only
                if query.query in self.artifacts.training_data.metabolite_to_idx:
                    results = [{
                        "query_name": query.query,
                        "matched_name": query.query,
                        "match_type": "exact",
                        "in_training_data": True,
                    }]
                else:
                    results = []
            
            return QueryResponse(
                success=True,
                query_type="search_compound",
                results=results,
                metadata={
                    "query": query.query,
                    "fuzzy": query.fuzzy,
                }
            )
        
        except Exception as e:
            logger.error(f"Error in search_compound query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="search_compound",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: List Taxa
    # =========================================================================
    
    def query_list_taxa(
        self,
        query: ListTaxaQuery
    ) -> QueryResponse:
        """List available taxa at a given level."""
        try:
            taxa = self.artifacts.get_taxa_at_level(query.level)
            
            if query.include_counts:
                # Count isolates per taxon
                taxon_counts = {}
                for org_id, taxonomy in self.artifacts.training_data.microbe_taxonomy.items():
                    taxon = taxonomy.get_level(query.level)
                    if taxon:
                        taxon_counts[taxon] = taxon_counts.get(taxon, 0) + 1
                
                results = [
                    {
                        "name": taxon,
                        "n_isolates": taxon_counts.get(taxon, 0)
                    }
                    for taxon in taxa
                ]
            else:
                results = [{"name": taxon} for taxon in taxa]
            
            return QueryResponse(
                success=True,
                query_type="list_taxa",
                results=results,
                metadata={
                    "level": query.level,
                    "n_taxa": len(results),
                }
            )
        
        except Exception as e:
            logger.error(f"Error in list_taxa query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="list_taxa",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Competition Analysis
    # =========================================================================
    
    def query_competition_analysis(
        self,
        query: CompetitionAnalysisQuery
    ) -> QueryResponse:
        """Analyze competition between microbes for specific metabolites."""
        try:
            # Validate microbes and metabolites upfront
            microbe_validation = self.validate_taxa(query.microbes)
            metabolite_validation = self.validate_metabolites(query.metabolites)
            
            # Fail fast if no microbes matched
            if microbe_validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="competition_analysis",
                    results=[],
                    metadata={"microbe_validation": microbe_validation.to_dict()},
                    error=f"None of the {microbe_validation.total_requested} microbes could be matched"
                )
            
            # Fail fast if no metabolites matched
            if metabolite_validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="competition_analysis",
                    results=[],
                    metadata={
                        "microbe_validation": microbe_validation.to_dict(),
                        "metabolite_validation": metabolite_validation.to_dict()
                    },
                    error=f"None of the {metabolite_validation.total_requested} metabolites could be matched"
                )
            
            # Get predictions for each matched microbe
            microbe_predictions = {}
            for match_result in microbe_validation.matched_items:
                level, name = match_result.matched.split(":", 1)
                prediction = self.artifacts.predict_with_stats(
                    level=level,
                    name=name,
                    compute_distinctive=query.use_distinctive,
                )
                microbe_predictions[(level, name)] = prediction
            
            # Analyze each matched metabolite
            metabolite_results = []
            wins = {match.matched: 0 for match in microbe_validation.matched_items}
            
            for match_result in metabolite_validation.matched_items:
                matched_name = match_result.matched
                
                # Get index
                idx = self.artifacts.training_data.metabolite_to_idx.get(matched_name)
                if idx is None:
                    # This shouldn't happen after validation, but handle gracefully
                    continue
                
                # Get scores for each microbe
                scores = {}
                for (level, name), prediction in microbe_predictions.items():
                    # Find score for this metabolite in prediction results
                    try:
                        met_names = [self.artifacts.training_data.idx_to_metabolite[i] 
                                    for i in range(self.artifacts.training_data.n_metabolites)]
                        if matched_name in met_names:
                            met_position = met_names.index(matched_name)
                            # Use distinctive or raw based on flag
                            if query.use_distinctive:
                                score = float(prediction.distinctive_scores[met_position])
                            else:
                                score = float(prediction.raw_scores[met_position])
                            scores[f"{level}:{name}"] = score
                    except Exception as e:
                        logger.warning(f"Could not get score for {matched_name} from {level}:{name}: {e}")
                
                if not scores:
                    continue
                
                # Find winner
                winner = max(scores, key=scores.get)
                wins[winner] += 1
                
                # Calculate margin (difference from second place)
                sorted_scores = sorted(scores.values(), reverse=True)
                margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
                
                result = {
                    "metabolite": match_result.requested,  # Show original request
                    "matched": matched_name,
                    "winner": winner,
                    "winner_score": round(scores[winner], 4),
                    "all_scores": {k: round(v, 4) for k, v in scores.items()},
                    "margin": round(margin, 4)
                }
                
                if query.explain:
                    result["explanation"] = {
                        "score_type": "distinctive" if query.use_distinctive else "raw",
                        "input_matching": {
                            "requested": match_result.requested,
                            "matched": match_result.matched,
                            "match_type": match_result.match_type,
                        },
                        "n_microbes": len(microbe_validation.matched_items)
                    }
                
                metabolite_results.append(result)
            
            # Overall winner
            overall_winner = max(wins, key=wins.get) if wins else None
            
            # Collect matched taxa and compound names for observation lookup
            comp_taxa = [(lvl, nm) for lvl, nm in microbe_predictions]
            comp_compounds = [m.matched for m in metabolite_validation.matched_items]

            # Build metadata with validation info
            metadata = {
                "n_microbes": len(microbe_validation.matched_items),
                "n_metabolites": len(metabolite_validation.matched_items),
                "n_metabolites_analyzed": len(metabolite_results),
                "win_counts": wins,
                "overall_winner": overall_winner,
                "microbe_validation": microbe_validation.to_dict(),
                "metabolite_validation": metabolite_validation.to_dict(),
                "observations": self._build_observation_metadata(
                    taxa=comp_taxa,
                    top_predicted_compounds=comp_compounds,
                    interaction_type=self.config.interaction_type,
                ),
            }

            # Collect all warnings
            warnings = []
            if microbe_validation.should_warn:
                warnings.extend(microbe_validation.warnings)
            if metabolite_validation.should_warn:
                warnings.extend(metabolite_validation.warnings)

            if warnings:
                metadata["warnings"] = warnings

            return QueryResponse(
                success=True,
                query_type="competition_analysis",
                results=metabolite_results,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error in competition_analysis query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="competition_analysis",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Rank Substrates
    # =========================================================================
    
    def query_rank_substrates(
        self,
        query: RankSubstratesQuery
    ) -> QueryResponse:
        """Rank specific substrates for each microbe."""
        try:
            # Validate microbes and metabolites upfront
            microbe_validation = self.validate_taxa(query.microbes)
            substrate_validation = self.validate_metabolites(query.substrates)
            
            # Fail fast if no microbes matched
            if microbe_validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="rank_substrates",
                    results=[],
                    metadata={"microbe_validation": microbe_validation.to_dict()},
                    error=f"None of the {microbe_validation.total_requested} microbes could be matched"
                )
            
            # Fail fast if no substrates matched
            if substrate_validation.should_fail:
                return QueryResponse(
                    success=False,
                    query_type="rank_substrates",
                    results=[],
                    metadata={
                        "microbe_validation": microbe_validation.to_dict(),
                        "substrate_validation": substrate_validation.to_dict()
                    },
                    error=f"None of the {substrate_validation.total_requested} substrates could be matched"
                )
            
            # Get rankings for each matched microbe
            microbe_results = []
            for match_result in microbe_validation.matched_items:
                level, name = match_result.matched.split(":", 1)
                
                # Get prediction
                prediction = self.artifacts.predict_with_stats(
                    level=level,
                    name=name,
                    compute_distinctive=query.use_distinctive,
                )
                
                # Score each matched substrate
                substrate_scores = []
                met_names = [self.artifacts.training_data.idx_to_metabolite[i] 
                            for i in range(self.artifacts.training_data.n_metabolites)]
                
                for sub_match in substrate_validation.matched_items:
                    matched_name = sub_match.matched
                    
                    if matched_name in met_names:
                        met_position = met_names.index(matched_name)
                        # Use distinctive or raw based on flag
                        if query.use_distinctive:
                            score = float(prediction.distinctive_scores[met_position])
                        else:
                            score = float(prediction.raw_scores[met_position])
                        
                        sub_result = {
                            "substrate": sub_match.requested,
                            "matched": matched_name,
                            "score": round(score, 4),
                            "score_sd": round(float(prediction.score_sd[met_position]), 4) if prediction.score_sd is not None else None,
                            "score_se": round(float(prediction.score_se[met_position]), 4) if prediction.score_se is not None else None,
                        }
                        
                        if query.explain:
                            sub_result["match_info"] = {
                                "match_type": sub_match.match_type,
                                "match_score": sub_match.match_score if sub_match.match_type == "fuzzy" else None,
                            }
                        
                        substrate_scores.append(sub_result)
                
                # Sort by score
                substrate_scores.sort(key=lambda x: x["score"], reverse=True)
                
                result = {
                    "microbe": {
                        "level": level,
                        "name": name,
                        "embedding_diversity": self.artifacts._taxon_embedding_diversity(level, name),
                    },
                    "rankings": substrate_scores[:query.top_k],
                    "n_scored": len(substrate_scores)
                }
                
                if query.explain:
                    result["explanation"] = {
                        "score_type": "distinctive" if query.use_distinctive else "raw",
                        "n_substrates": len(substrate_validation.matched_items),
                        "microbe_match_info": {
                            "requested": match_result.requested,
                            "matched": match_result.matched,
                            "match_type": match_result.match_type,
                        }
                    }
                
                microbe_results.append(result)
            
            # Collect matched taxa and substrate names for observation lookup
            rs_taxa = []
            for m in microbe_validation.matched_items:
                lvl, nm = m.matched.split(":", 1)
                rs_taxa.append((lvl, nm))
            rs_substrates = [m.matched for m in substrate_validation.matched_items]

            # Build metadata with validation info
            metadata = {
                "n_microbes": len(microbe_validation.matched_items),
                "n_substrates": len(substrate_validation.matched_items),
                "microbe_validation": microbe_validation.to_dict(),
                "substrate_validation": substrate_validation.to_dict(),
                "observations": self._build_observation_metadata(
                    taxa=rs_taxa,
                    top_predicted_compounds=rs_substrates,
                    interaction_type=self.config.interaction_type,
                ),
            }

            # Collect all warnings
            warnings = []
            if microbe_validation.should_warn:
                warnings.extend(microbe_validation.warnings)
            if substrate_validation.should_warn:
                warnings.extend(substrate_validation.warnings)

            if warnings:
                metadata["warnings"] = warnings

            return QueryResponse(
                success=True,
                query_type="rank_substrates",
                results=microbe_results,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error in rank_substrates query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="rank_substrates",
                results=[],
                error=str(e)
            )
    
    # =========================================================================
    # Handler: Model Info
    # =========================================================================
    
    def query_model_info(
        self,
        query: ModelInfoQuery
    ) -> QueryResponse:
        """Get model information and statistics."""
        try:
            info = {
                "model_type": self.artifacts.model_type,
                "embedding_dim": self.artifacts.embedding_dim,
                "n_microbes": self.artifacts.model.n_microbes,
                "n_metabolites": self.artifacts.model.n_metabolites,
                "interaction_type": self.config.interaction_type,
                "model_dir": str(self.model_dir),
            }
            
            if query.include_stats:
                # Training data stats
                n_positives = sum(len(v) for v in self.artifacts.training_data.positives.values())
                info["statistics"] = {
                    "n_interactions": n_positives,
                    "sparsity": 1 - (n_positives / (info["n_microbes"] * info["n_metabolites"])),
                    "avg_metabolites_per_microbe": n_positives / info["n_microbes"] if info["n_microbes"] > 0 else 0,
                }
                
                # Taxonomy info
                info["taxonomy"] = {
                    level: len(self.artifacts.taxonomy_embeddings.get(level, {}))
                    for level in ["genus", "family", "order", "class", "kingdom"]
                }
            
            return QueryResponse(
                success=True,
                query_type="model_info",
                results=[info],
                metadata=self._base_metadata()
            )
        
        except Exception as e:
            logger.error(f"Error in model_info query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="model_info",
                results=[],
                metadata=self._base_metadata(),
                error=str(e)
            )
    
    # =========================================================================
    # Environment Query Methods
    # =========================================================================
    
    def query_list_environments(
        self,
        query: ListEnvironmentsQuery
    ) -> QueryResponse:
        """List available environments with statistics."""
        try:
            env_df = self.environment_data
            
            # Group by environment
            env_groups = env_df.groupby('environment_name')
            
            results = []
            for env_name, group in env_groups:
                total = len(group)
                in_model = group['in_model'].sum()
                in_model_frac = in_model / total if total > 0 else 0
                
                # Filter by min_coverage if specified
                if in_model_frac < query.min_coverage:
                    continue
                
                env_info = {
                    'environment_name': env_name,
                    'n_compounds': total,
                }
                
                if query.include_stats:
                    mapped = group['representative_compound'].notna().sum()
                    env_info.update({
                        'n_mapped': mapped,
                        'n_in_model': in_model,
                        'coverage_fraction': round(in_model_frac, 3),
                        'status': group['environment_status'].iloc[0],
                        'description': group['environment_description'].iloc[0],
                        'doi': group['doi'].iloc[0],
                    })
                
                results.append(env_info)
            
            # Sort by coverage descending
            if query.include_stats:
                results.sort(key=lambda x: x['coverage_fraction'], reverse=True)
            
            return QueryResponse(
                success=True,
                query_type="list_environments",
                results=results,
                metadata={
                    **self._base_metadata(),
                    'n_environments': len(results),
                    'min_coverage_filter': query.min_coverage,
                }
            )
        
        except Exception as e:
            logger.error(f"Error in list_environments query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="list_environments",
                results=[],
                metadata=self._base_metadata(),
                error=str(e)
            )
    
    def query_search_environment(
        self,
        query: SearchEnvironmentQuery
    ) -> QueryResponse:
        """Fuzzy search for environment names."""
        try:
            from rapidfuzz import process, fuzz
            
            env_df = self.environment_data
            env_names = env_df['environment_name'].unique().tolist()
            
            # Fuzzy search
            matches = process.extract(
                query.query.lower(),
                [name.lower() for name in env_names],
                scorer=fuzz.WRatio,
                score_cutoff=query.min_score * 100,
                limit=query.max_results
            )
            
            # Map back to original names
            lower_to_orig = {name.lower(): name for name in env_names}
            
            results = []
            for match_lower, score, _ in matches:
                orig_name = lower_to_orig[match_lower]
                env_group = env_df[env_df['environment_name'] == orig_name]
                
                results.append({
                    'environment_name': orig_name,
                    'match_score': round(score / 100, 3),
                    'n_compounds': len(env_group),
                    'n_in_model': env_group['in_model'].sum(),
                    'description': env_group['environment_description'].iloc[0],
                })
            
            return QueryResponse(
                success=True,
                query_type="search_environment",
                results=results,
                metadata={
                    **self._base_metadata(),
                    'query': query.query,
                    'n_matches': len(results),
                }
            )
        
        except Exception as e:
            logger.error(f"Error in search_environment query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="search_environment",
                results=[],
                metadata=self._base_metadata(),
                error=str(e)
            )
    
    def query_get_environment_compounds(
        self,
        query: GetEnvironmentCompoundsQuery
    ) -> QueryResponse:
        """Get compounds in a specific environment."""
        try:
            env_df = self.environment_data
            
            # Find matching environment
            env_group = env_df[env_df['environment_name'] == query.environment_name]
            
            if len(env_group) == 0:
                # Try fuzzy match
                from rapidfuzz import process, fuzz
                env_names = env_df['environment_name'].unique().tolist()
                best_match = process.extractOne(
                    query.environment_name.lower(),
                    [name.lower() for name in env_names],
                    scorer=fuzz.WRatio
                )
                
                if best_match and best_match[1] >= 80:
                    lower_to_orig = {name.lower(): name for name in env_names}
                    matched_name = lower_to_orig[best_match[0]]
                    env_group = env_df[env_df['environment_name'] == matched_name]
                    logger.info(f"Fuzzy matched '{query.environment_name}' to '{matched_name}'")
                else:
                    return QueryResponse(
                        success=False,
                        query_type="get_environment_compounds",
                        results=[],
                        metadata=self._base_metadata(),
                        error=f"Environment not found: {query.environment_name}"
                    )
            
            # Filter to only compounds in model if requested
            if query.only_in_model:
                env_group = env_group[env_group['in_model']].copy()
            
            # Build results
            results = []
            for _, row in env_group.iterrows():
                compound_info = {
                    'original_name': row['compound'],
                    'representative_compound': row['representative_compound'],
                    'chebi_id': row['chebi_id'],
                    'match_type': row['match_type'],
                    'match_score': round(row['match_score'], 3),
                    'in_model': bool(row['in_model']),
                }
                results.append(compound_info)
            
            # Get environment metadata
            first_row = env_df[env_df['environment_name'] == env_group.iloc[0]['environment_name']].iloc[0]
            
            return QueryResponse(
                success=True,
                query_type="get_environment_compounds",
                results=results,
                metadata={
                    **self._base_metadata(),
                    'environment_name': first_row['environment_name'],
                    'description': first_row['environment_description'],
                    'doi': first_row['doi'],
                    'n_compounds': len(results),
                    'only_in_model': query.only_in_model,
                }
            )
        
        except Exception as e:
            logger.error(f"Error in get_environment_compounds query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="get_environment_compounds",
                results=[],
                metadata=self._base_metadata(),
                error=str(e)
            )

    # =========================================================================
    # Function Enrichment Query Methods
    # =========================================================================

    def _compute_bpr_enzyme_scores(
        self,
        level: str,
        matched_taxon: str,
        ec_numbers: list[str],
        support_df: pd.DataFrame,
    ) -> dict[str, float | None]:
        """Project EC functions into BPR latent space and score by taxon alignment.

        For each EC, computes β_enzyme = mean of BPR item embeddings for its
        linked WoM compounds, then scores as θ_taxon · β_enzyme.  ECs whose
        compounds don't appear in the BPR model receive None.

        Returns a dict mapping ec_number -> score (float) or None.
        """
        try:
            theta = self.artifacts.get_embedding(level, matched_taxon)
        except (ValueError, KeyError):
            return {ec: None for ec in ec_numbers}

        metabolite_to_idx = self.artifacts.training_data.metabolite_to_idx
        metabolite_embeddings = self.artifacts.model.metabolite_embeddings  # (n_metabolites, dim)

        raw_scores: dict[str, float | None] = {}
        for ec in ec_numbers:
            ec_support = support_df[support_df["ec_number"] == ec]
            compound_names = ec_support["representative_compound"].dropna().unique().tolist()
            indices = [
                metabolite_to_idx[c] for c in compound_names if c in metabolite_to_idx
            ]
            if not indices:
                raw_scores[ec] = None
                continue
            beta_enzyme = metabolite_embeddings[indices].mean(axis=0)
            raw_scores[ec] = float(np.dot(theta, beta_enzyme))

        # Normalise across ECs that have scores so values are interpretable
        # relative to the queried set (subtract mean, keep units as dot-product).
        valid = [s for s in raw_scores.values() if s is not None]
        if valid:
            mu = float(np.mean(valid))
            raw_scores = {
                ec: (s - mu if s is not None else None)
                for ec, s in raw_scores.items()
            }

        return raw_scores

    def query_list_function_enrichment_taxa(
        self,
        query: ListFunctionEnrichmentTaxaQuery,
    ) -> QueryResponse:
        """List taxa with available function enrichment support."""
        try:
            df = self.taxon_function_enrichment_data
            level = query.level.lower()

            level_df = df[df["level"] == level].copy()
            if level_df.empty:
                return QueryResponse(
                    success=True,
                    query_type="list_function_enrichment_taxa",
                    results=[],
                    metadata={
                        **self._base_metadata(),
                        "level": level,
                        "n_taxa": 0,
                        "min_proteomes": query.min_proteomes,
                    },
                )

            taxon_summary = (
                level_df
                .groupby("taxon")
                .agg(
                    n_proteomes=("n_proteomes_taxon", "max"),
                    n_enriched_functions=("ec_number", "nunique"),
                    top_tfidf=("tfidf_score", "max"),
                )
                .reset_index()
            )

            taxon_summary = taxon_summary[taxon_summary["n_proteomes"] >= query.min_proteomes]
            taxon_summary = taxon_summary.sort_values(
                ["n_proteomes", "n_enriched_functions", "top_tfidf"],
                ascending=[False, False, False],
            )

            results = []
            for _, row in taxon_summary.iterrows():
                item = {"taxon": row["taxon"]}
                if query.include_counts:
                    item.update({
                        "n_proteomes": int(row["n_proteomes"]),
                        "n_enriched_functions": int(row["n_enriched_functions"]),
                        "top_tfidf": float(row["top_tfidf"]),
                    })
                results.append(item)

            return QueryResponse(
                success=True,
                query_type="list_function_enrichment_taxa",
                results=results,
                metadata={
                    **self._base_metadata(),
                    "level": level,
                    "n_taxa": len(results),
                    "min_proteomes": query.min_proteomes,
                },
            )

        except Exception as e:
            logger.error(f"Error in list_function_enrichment_taxa query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="list_function_enrichment_taxa",
                results=[],
                metadata=self._base_metadata(),
                error=str(e),
            )

    def query_enriched_functions_for_taxon(
        self,
        query: EnrichedFunctionsForTaxonQuery,
    ) -> QueryResponse:
        """Return taxon-specialization ranked EC functions using TF-IDF scoring."""
        try:
            enrichment_df = self.taxon_function_enrichment_data
            level = query.level.lower()
            requested_taxon = query.taxon

            level_df = enrichment_df[enrichment_df["level"] == level].copy()
            if level_df.empty:
                return QueryResponse(
                    success=False,
                    query_type="enriched_functions_for_taxon",
                    results=[],
                    metadata={**self._base_metadata(), "level": level},
                    error=f"No enrichment data available for level '{level}'",
                )

            taxon_df = level_df[level_df["taxon"] == requested_taxon].copy()
            matched_taxon = requested_taxon

            if taxon_df.empty:
                from rapidfuzz import process, fuzz
                candidates = level_df["taxon"].dropna().astype(str).unique().tolist()
                best = process.extractOne(requested_taxon, candidates, scorer=fuzz.WRatio)
                if best and best[1] >= 80:
                    matched_taxon = best[0]
                    taxon_df = level_df[level_df["taxon"] == matched_taxon].copy()
                else:
                    return QueryResponse(
                        success=False,
                        query_type="enriched_functions_for_taxon",
                        results=[],
                        metadata={**self._base_metadata(), "level": level},
                        error=f"Taxon not found at level '{level}': {requested_taxon}",
                    )

            if query.min_tfidf is not None:
                taxon_df = taxon_df[taxon_df["tfidf_score"] >= query.min_tfidf]
            if query.min_frequency is not None:
                taxon_df = taxon_df[taxon_df["tf_frequency"] >= query.min_frequency]

            taxon_df = taxon_df.sort_values("tfidf_score", ascending=False).head(query.top_k)

            # Load compound support data if needed for compounds or BPR scoring
            support_df = None
            if query.include_compounds or query.include_bpr_scores:
                support_df = self.taxon_function_compound_support_data
                support_df = support_df[
                    (support_df["level"] == level) &
                    (support_df["taxon"] == matched_taxon)
                ].copy()

            # Compute BPR alignment scores for all ECs in the result set
            bpr_scores: dict[str, float | None] = {}
            if query.include_bpr_scores and support_df is not None:
                ec_list = taxon_df["ec_number"].dropna().unique().tolist()
                bpr_scores = self._compute_bpr_enzyme_scores(
                    level=level,
                    matched_taxon=matched_taxon,
                    ec_numbers=ec_list,
                    support_df=support_df,
                )

            results = []
            for _, row in taxon_df.iterrows():
                result_item = {
                    "ec_number": row["ec_number"],
                    "tfidf_score": float(row["tfidf_score"]),
                    "tf_frequency": float(row["tf_frequency"]),
                    "idf": float(row["idf"]),
                    "n_proteomes_with_ec": int(row["n_proteomes_with_ec"]),
                    "n_proteomes_taxon": int(row["n_proteomes_taxon"]),
                    "n_taxa_with_ec": int(row["n_taxa_with_ec"]),
                    "n_taxa_level": int(row["n_taxa_level"]),
                }

                if query.include_bpr_scores:
                    score = bpr_scores.get(row["ec_number"])
                    result_item["bpr_alignment_score"] = score

                if query.include_compounds and support_df is not None and not support_df.empty:
                    ec_support = support_df[support_df["ec_number"] == row["ec_number"]]
                    ec_support = ec_support.sort_values(
                        ["n_reaction_rows", "n_proteins_support"],
                        ascending=[False, False],
                    ).head(query.max_compounds_per_function)
                    result_item["associated_compounds"] = [
                        {
                            "representative_compound": srow["representative_compound"],
                            "n_reaction_rows": int(srow["n_reaction_rows"]),
                            "n_proteins_support": int(srow["n_proteins_support"]),
                        }
                        for _, srow in ec_support.iterrows()
                    ]

                results.append(result_item)

            n_proteomes = int(taxon_df["n_proteomes_taxon"].iloc[0]) if len(taxon_df) > 0 else None

            return QueryResponse(
                success=True,
                query_type="enriched_functions_for_taxon",
                results=results,
                metadata={
                    **self._base_metadata(),
                    "level": level,
                    "requested_taxon": requested_taxon,
                    "matched_taxon": matched_taxon,
                    "n_proteomes": n_proteomes,
                    "n_functions_returned": len(results),
                    "top_k": query.top_k,
                    "min_tfidf": query.min_tfidf,
                    "min_frequency": query.min_frequency,
                    "bpr_alignment_scores_included": query.include_bpr_scores,
                },
            )

        except Exception as e:
            logger.error(f"Error in enriched_functions_for_taxon query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="enriched_functions_for_taxon",
                results=[],
                metadata=self._base_metadata(),
                error=str(e),
            )

    def query_substrate_gaps(self, query: SubstrateGapQuery) -> QueryResponse:
        """Cross-reference BPR metabolite preference with annotated enzyme presence.

        Returns orphan enzyme candidates (high BPR, no annotated enzyme) and
        regulated/suppressed substrates (annotated enzyme present, low BPR).
        """
        try:
            level = query.level.lower()

            # Resolve taxon name against enrichment table (consistent with other queries)
            enrichment_df = self.taxon_function_enrichment_data
            level_df = enrichment_df[enrichment_df["level"] == level]
            if level_df.empty:
                return QueryResponse(
                    success=False,
                    query_type="substrate_gaps",
                    results=[],
                    metadata={**self._base_metadata(), "level": level},
                    error=f"No enrichment data available for level '{level}'",
                )

            requested_taxon = query.taxon
            matched_taxon = requested_taxon
            if requested_taxon not in level_df["taxon"].values:
                from rapidfuzz import process, fuzz
                candidates = level_df["taxon"].dropna().astype(str).unique().tolist()
                best = process.extractOne(requested_taxon, candidates, scorer=fuzz.WRatio)
                if best and best[1] >= 80:
                    matched_taxon = best[0]
                else:
                    return QueryResponse(
                        success=False,
                        query_type="substrate_gaps",
                        results=[],
                        metadata={**self._base_metadata(), "level": level},
                        error=f"Taxon not found at level '{level}': {requested_taxon}",
                    )

            # Get n_proteomes for this taxon from enrichment table
            taxon_enrich = level_df[level_df["taxon"] == matched_taxon]
            n_proteomes = int(taxon_enrich["n_proteomes_taxon"].iloc[0]) if len(taxon_enrich) > 0 else None

            # Compute BPR scores for ALL metabolites
            try:
                prediction = self.artifacts.predict_with_stats(level=level, name=matched_taxon)
            except (ValueError, KeyError) as e:
                return QueryResponse(
                    success=False,
                    query_type="substrate_gaps",
                    results=[],
                    metadata={**self._base_metadata(), "level": level, "matched_taxon": matched_taxon},
                    error=f"BPR prediction not available for '{matched_taxon}': {e}",
                )

            raw_scores = prediction.raw_scores
            std = prediction.std_score
            metabolite_names = prediction.metabolite_names

            # Compute z-scores (raw_scores are not mean-centered by predict_with_stats)
            mean = prediction.mean_score
            if std > 0:
                z_scores = (raw_scores - mean) / std
            else:
                z_scores = np.zeros_like(raw_scores)

            # Load enzyme compound support for this taxon
            support_df = self.taxon_function_compound_support_data
            taxon_support = support_df[
                (support_df["level"] == level) & (support_df["taxon"] == matched_taxon)
            ]
            enzyme_compounds = set(taxon_support["representative_compound"].dropna())

            # Count distinct ECs per compound for the regulated section
            ec_counts = (
                taxon_support.groupby("representative_compound")["ec_number"]
                .nunique()
                .to_dict()
            )

            # Classify each metabolite
            orphans = []
            regulated = []
            threshold = query.bpr_z_threshold

            for name, z, raw in zip(metabolite_names, z_scores, raw_scores):
                has_enzyme = name in enzyme_compounds
                if z > threshold and not has_enzyme:
                    orphans.append({
                        "compound": name,
                        "bpr_z_score": round(float(z), 3),
                        "bpr_raw_score": round(float(raw), 3),
                    })
                elif z < -threshold and has_enzyme:
                    regulated.append({
                        "compound": name,
                        "bpr_z_score": round(float(z), 3),
                        "bpr_raw_score": round(float(raw), 3),
                        "n_enzymes": ec_counts.get(name, 0),
                    })

            orphans.sort(key=lambda x: x["bpr_z_score"], reverse=True)
            regulated.sort(key=lambda x: x["bpr_z_score"])

            return QueryResponse(
                success=True,
                query_type="substrate_gaps",
                results={
                    "orphan_candidates": orphans[: query.top_k],
                    "regulated_suppressed": regulated[: query.top_k],
                },
                metadata={
                    **self._base_metadata(),
                    "level": level,
                    "requested_taxon": requested_taxon,
                    "matched_taxon": matched_taxon,
                    "n_proteomes": n_proteomes,
                    "n_metabolites_in_model": len(metabolite_names),
                    "n_compounds_with_enzyme_support": len(enzyme_compounds),
                    "bpr_z_threshold": threshold,
                    "n_orphan_candidates": len(orphans),
                    "n_regulated_suppressed": len(regulated),
                },
            )

        except Exception as e:
            logger.error(f"Error in substrate_gaps query: {e}", exc_info=True)
            return QueryResponse(
                success=False,
                query_type="substrate_gaps",
                results=[],
                metadata=self._base_metadata(),
                error=str(e),
            )
