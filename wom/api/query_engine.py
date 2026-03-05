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

from core.types import ModelArtifacts, PredictionResult
from core.io import load_model_artifacts
from core.training import compute_taxonomy_embeddings
from core.validation import InputMatchResult, InputValidationSummary
from loaders.compound_mapper import load_chebi, map_compound_name
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
                    "compounds": [
                        {
                            "name": top_result.metabolite_names[i],
                            "raw_score": float(top_result.raw_scores[i]),
                            "distinctive_score": float(top_result.distinctive_scores[i]) if top_result.distinctive_scores is not None else None,
                            "rank": int(top_result.ranks[i]) if top_result.ranks is not None else None,
                            "percentile": float(top_result.percentiles[i]) if top_result.percentiles is not None else None,
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
            
            return QueryResponse(
                success=True,
                query_type="compounds_for_microbes",
                results=results,
                metadata={
                    **self._base_metadata(),
                    "n_microbes_queried": len(query.microbes),
                    "top_k": query.top_k,
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
                    
                    result = {
                        "taxon": taxon_name,
                        "level": query.level,
                        "aggregate_score": agg_score,
                        "compound_scores": {
                            mapped_names[i]: float(compound_scores[i])
                            for i in range(len(mapped_names))
                        }
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
            
            # Build metadata with validation info
            metadata = {
                **self._base_metadata(),
                "compounds_queried": mapped_names,
                "n_compounds": len(mapped_names),
                "level": query.level,
                "agg_method": query.agg_method,
                "n_taxa_scored": len(taxon_scores),
                "input_validation": validation.to_dict(),
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
            
            # Build metadata with validation info
            metadata = {
                "target": {"level": target_level, "name": target_name},
                "n_competitors": len(competitor_matches),
                "method": query.method,
                "input_validation": validation.to_dict(),
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
                }
                
                if query.explain:
                    result["explanation"] = {
                        "score_type": "distinctive (always for niche query)",
                        "background_level": background_level,
                        "n_background_taxa": prediction.n_isolates_aggregated,
                        "computation": "raw_score - background_mean"
                    }
                
                results.append(result)
            
            metadata = {
                "microbe": {"level": level, "name": name},
                "background_level": background_level,
                "background_mean": prediction.background_mean,
                "n_isolates_in_background": prediction.n_isolates_aggregated,
                "input_validation": validation.to_dict(),
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
            
            # Build metadata with validation info
            metadata = {
                "n_microbes": len(microbe_validation.matched_items),
                "n_metabolites": len(metabolite_validation.matched_items),
                "n_metabolites_analyzed": len(metabolite_results),
                "win_counts": wins,
                "overall_winner": overall_winner,
                "microbe_validation": microbe_validation.to_dict(),
                "metabolite_validation": metabolite_validation.to_dict(),
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
                            "score": round(score, 4)
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
                        "name": name
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
            
            # Build metadata with validation info
            metadata = {
                "n_microbes": len(microbe_validation.matched_items),
                "n_substrates": len(substrate_validation.matched_items),
                "microbe_validation": microbe_validation.to_dict(),
                "substrate_validation": substrate_validation.to_dict(),
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
