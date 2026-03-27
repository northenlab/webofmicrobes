"""
MCP server for the Web of Microbes (WoM) platform.

Data sources — use these names precisely in responses:
  - WoM BPR / WoM model  : substrate preference predictions (consumed/secreted)
  - Phydon               : growth rate predictions (doubling times in hours)
  - GenomeSPOT           : culture condition predictions (temperature, pH, salinity, oxygen)

These are three independent sources. Never attribute growth rates to GenomeSPOT or
culture conditions to Phydon.
"""

from fastmcp import FastMCP
import requests
import random
import json
import logging
import sys

logger = logging.getLogger(__name__)

import sys


sys.path.insert(0, "./growth_rate")  # Ensure the /app directory is in the Python path for imports

import growth_rate as phydon
import json
phydon.CACHE_DIR = "./growth_rate/cache"  # Set the cache directory to a known location

sys.path.insert(1, "./wom")  # Ensure the /app directory is in the Python path for imports

from loaders import genome_spot as genome_spot_module
genome_spot_module.CACHE_DIR = "./wom/processed_data"

from pathlib import Path
import api
from api.query_engine import QueryEngine
from loaders.compound_mapper import map_compound_name

from api.query_types import (
    QueryConfig,
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

wom_dir = Path(api.__file__).parent.parent
trained_models_dir = wom_dir / "trained_models"
_ENGINE_CACHE: dict[tuple[str, str], QueryEngine] = {}

_TOOL_INTENTS = {
  "compounds_for_microbes": {
    "query_direction": "microbe_to_metabolite",
    "inverse_tool": "microbes_for_compounds",
    "use_when": "User asks what metabolites a microbe is predicted to consume or secrete",
    "includes_observations": True,
  },
  "microbes_for_compounds": {
    "query_direction": "metabolite_to_microbe",
    "inverse_tool": "compounds_for_microbes",
    "use_when": "User asks which microbes are predicted to consume or secrete specific metabolites",
    "includes_observations": True,
  },
  "selective_metabolites": {
    "query_direction": "target_vs_competitor_metabolite_selectivity",
    "inverse_tool": None,
    "use_when": "User asks which metabolites differentiate a target microbe from competitors",
    "includes_observations": True,
  },
  "metabolite_environment_fitness": {
    "query_direction": "environment_to_microbe_ranking",
    "inverse_tool": None,
    "use_when": "User asks which microbes perform best in a metabolite environment",
    "includes_observations": False,
  },
  "niche_metabolites": {
    "query_direction": "microbe_to_distinctive_metabolites",
    "inverse_tool": None,
    "use_when": "User asks for niche-specific metabolite preferences",
    "includes_observations": True,
  },
  "competition_analysis": {
    "query_direction": "metabolite_winner_across_microbes",
    "inverse_tool": None,
    "use_when": "User asks who wins metabolite-by-metabolite competition",
    "includes_observations": True,
  },
  "rank_substrates": {
    "query_direction": "microbe_to_ranked_subset_of_metabolites",
    "inverse_tool": None,
    "use_when": "User asks to rank a specific substrate subset for microbes",
    "includes_observations": True,
  },
  "search_compound": {
    "query_direction": "name_to_model_compound_mapping",
    "inverse_tool": None,
    "use_when": "User input is uncertain, synonym-heavy, or requires fuzzy compound matching",
    "includes_observations": False,
  },
  "list_taxa": {
    "query_direction": "taxonomy_discovery",
    "inverse_tool": None,
    "use_when": "User asks what taxa are available before running predictive queries",
    "includes_observations": False,
  },
  "model_info": {
    "query_direction": "model_metadata_lookup",
    "inverse_tool": None,
    "use_when": "User asks what model/direction is loaded and dataset coverage stats",
    "includes_observations": False,
  },
  "list_environments": {
    "query_direction": "environment_discovery",
    "inverse_tool": None,
    "use_when": "User asks what environments are available or wants to see all environment options",
    "includes_observations": False,
  },
  "search_environment": {
    "query_direction": "environment_name_matching",
    "inverse_tool": None,
    "use_when": "User mentions an environment name but it might not be exact, needs fuzzy matching",
    "includes_observations": False,
  },
  "get_environment_compounds": {
    "query_direction": "environment_to_compound_list",
    "inverse_tool": "microbes_for_compounds",
    "use_when": "User asks about microbes that grow on an environment; first get compounds, then query microbes",
    "includes_observations": False,
  },
  "list_function_enrichment_taxa": {
    "query_direction": "taxonomy_to_available_function_profiles",
    "inverse_tool": None,
    "use_when": "User asks which taxa have annotated enzyme-enrichment profiles",
    "includes_observations": False,
  },
  "enriched_functions_for_taxon": {
    "query_direction": "taxon_to_enriched_enzyme_functions",
    "inverse_tool": None,
    "use_when": "User asks which EC functions are distinctive/enriched for a specific taxon; use include_bpr_scores=True to add BPR metabolic alignment scores alongside TF-IDF",
    "includes_observations": False,
  },
  "explain_compound_group": {
    "query_direction": "compound_name_to_group_members",
    "inverse_tool": None,
    "use_when": "User asks why a compound has an unexpected name, what compounds are grouped together, or which synonyms/stereoisomers share a representative name in WoM",
    "includes_observations": False,
  },
  "analyze_substrate_gaps": {
    "query_direction": "taxon_to_metabolite_enzyme_gaps",
    "inverse_tool": "enriched_functions_for_taxon",
    "use_when": "User asks about orphan enzymes, unexplained substrate preferences, catabolite repression candidates, or metabolites consumed without known enzymatic machinery",
    "includes_observations": False,
  },
}

# Instructions injected into every observation-enriched response so the LLM knows how to
# structure its narrative around the experimental evidence.
_OBSERVATION_GUIDANCE = """
RESPONSE STRUCTURE — follow this order every time observations are present:

═══════════════════════════════════════════════════════════════
STEP 1 — LEAD WITH EXPERIMENTAL EVIDENCE (always write this first)
═══════════════════════════════════════════════════════════════
Read metadata.observations.evidence_narrative_data first. This is your primary narrative scaffold.

Write a coherent paragraph (not a table) that summarises what was ACTUALLY OBSERVED in the lab,
grouped by evidence tier:

  taxa_strong_positive_evidence  → "Strong direct evidence: [genus] — [n_consumed] of [n_tested]
    isolates ([tested_isolates]) consumed this compound (assertion values: [assertion_values])."
  taxa_positive_evidence         → "Positive evidence: [genus] — observed consumption in
    [n_consumed]/[n_tested] isolates."
  taxa_mixed_evidence            → "Mixed evidence: [genus] — some isolates consumed, others neutral."
  taxa_contradicting_prediction  → "⚠ Conflict: [genus] is predicted [interaction_type] but
    [n_tested] tested isolate(s) showed [observed_assertion]. The model score should not be
    trusted here without further validation. Isolates tested: [tested_isolates]."
  taxa_unvalidated               → Do NOT mention these taxa yet — they belong in Step 2.

For each genus with data, cite the specific isolate IDs and assertion values from
taxa_evidence_summary[genus]. Negative assertion_values indicate consumption; positive indicate
secretion. The scale runs from -5 (strong consumption) to +5 (strong secretion).

═══════════════════════════════════════════════════════════════
STEP 2 — MODEL PREDICTIONS FOR UNVALIDATED TAXA (fill the gaps)
═══════════════════════════════════════════════════════════════
For taxa in taxa_unvalidated, state the model prediction and always label it explicitly:
  "Not experimentally tested — model predicts [score/rank]."

If the top model predictions are in taxa_unvalidated, make clear that those rankings are
inferred from patterns across similar organisms, not from direct measurement.

═══════════════════════════════════════════════════════════════
STEP 3 — STEREOISOMER NOTE (when applicable)
═══════════════════════════════════════════════════════════════
If compounds_with_raw_name_variants is non-empty, add a brief note:
  "Note: [rep_compound] appears in the experimental data under [n_distinct_raw_names] distinct
  names ([raw_names]). If stereochemistry matters for your application (e.g. L- vs D-forms,
  alpha vs beta anomers), the grouping may mask biologically meaningful differences."

═══════════════════════════════════════════════════════════════
STEP 4 — SYNTHESIS TABLE (always include this)
═══════════════════════════════════════════════════════════════
Combine Steps 1–3 into a ranked table. Every row must include ALL of these columns:

  | Genus | Substrate | BPR score ± SD | Isolates in model | Evidence tier | Notes |

  - BPR score ± SD: use aggregate_score ± compound_score_sd[compound].
    Format: "5.68 ± 1.20". If compound_score_sd is null (single isolate), write "5.68".
  - Isolates in model: n_isolates_in_model — the number of isolate genomes whose BPR embeddings
    were averaged to produce this genus-level score. Can be large (10s–100s) for common genera.
    This is a model quantity; whether the genus was experimentally tested is shown by Evidence tier.
  - Evidence tier: taxa_evidence_summary[genus]["evidence_strength"]. Write "unvalidated" if absent.
  - Notes: brief annotation — "(observed)", "(predicted, not tested)", or "(⚠ conflicts with observation)"

A reader should never have to wonder which claims are empirical vs. model-only.

═══════════════════════════════════════════════════════════════
THREE CAVEATS — raise proactively whenever relevant
═══════════════════════════════════════════════════════════════
1. IMPLICIT RANKINGS — model predictions fill in the blanks from sparse training data. They
   are good for genera with no data but are unreliable when taxa_contradicting_prediction is
   non-empty. Say so explicitly.
2. TAXONOMIC AGGREGATION — genus-level predictions average over all isolates. If
   n_isolates_with_data is low (0–2) for a large genus, the prediction may not represent the
   specific strain the user cares about. Suggest level='isolate' queries if an organism_id is known.
3. STEREOISOMER GROUPING — WoM maps compounds to their InChIKey first block (connectivity only),
   collapsing stereoisomers. If compounds_with_raw_name_variants is non-empty, mention it when
   stereochemistry could matter. Raw-to-representative compound mappings are listed there.
   For per-isolate assertion values, see isolate_observations (genus → isolate_id → {compound: assertion_value}).
"""



def _get_wom_engine(
  model_type: str = "ordinal_bpr",
  interaction_type: str = "consumed",
) -> tuple[QueryEngine, QueryConfig]:
  """Get cached QueryEngine + QueryConfig for model type and interaction type."""
  model_type = model_type.lower()
  interaction_type = interaction_type.lower()

  valid_model_types = {"ordinal_bpr", "bpr", "implicit_bpr", "lmf", "wrmf", "svd"}
  valid_interactions = {"consumed", "secreted"}

  if model_type not in valid_model_types:
    raise ValueError(
      f"Invalid model_type '{model_type}'. Valid: {sorted(valid_model_types)}"
    )
  if interaction_type not in valid_interactions:
    raise ValueError(
      f"Invalid interaction_type '{interaction_type}'. Valid: {sorted(valid_interactions)}"
    )

  cache_key = (model_type, interaction_type)
  if cache_key not in _ENGINE_CACHE:
    cfg = QueryConfig(
      model_type=model_type,
      model_dir=trained_models_dir / "latest",
      interaction_type=interaction_type,
      output_format='json',
      verbose=True,
      explain=True,
    )
    _ENGINE_CACHE[cache_key] = QueryEngine(cfg.model_dir, cfg)

  engine = _ENGINE_CACHE[cache_key]
  return engine, engine.config


def _serialize_wom_response(
  response,
  tool_name: str,
  model_type: str,
  interaction_type: str,
) -> str:
  """Serialize query response with orchestration metadata for LLM routing."""
  payload = response.to_dict()
  metadata = payload.setdefault("metadata", {})

  tool_intent = _TOOL_INTENTS.get(tool_name, {})
  orchestration: dict = {
    "tool": tool_name,
    "query_direction": tool_intent.get("query_direction"),
    "inverse_tool": tool_intent.get("inverse_tool"),
    "use_when": tool_intent.get("use_when"),
    "model_type": model_type,
    "interaction_type": interaction_type,
    "semantic_interpretation": (
      "higher score suggests stronger predicted secretion propensity"
      if interaction_type == "secreted"
      else "higher score suggests stronger predicted consumption propensity"
    ),
  }

  if tool_intent.get("includes_observations"):
    orchestration["observation_guidance"] = _OBSERVATION_GUIDANCE

  metadata["orchestration"] = orchestration
  serialized = json.dumps(payload, indent=2)
  obs = payload.get("metadata", {}).get("observations", {})
  n_records = len(obs.get("isolate_observations", obs.get("isolate_records", [])))
  guidance_bytes = len((orchestration.get("observation_guidance") or "").encode())
  logger.info(
      "[size] tool=%s total_bytes=%d isolate_observation_entries=%d guidance_bytes=%d",
      tool_name, len(serialized), n_records, guidance_bytes,
  )
  return serialized



mcp = FastMCP("My MCP Server")

# @mcp.tool(
#     name="hello_world_greet",           # Custom tool name for the LLM
#     description="None.", # Custom description
# )
# def greet(name: str) -> str:
#     """Simple connectivity check for the MCP server."""
#     return f"Hello, {name}! Here is a number to remember {random.randint(1, 100)}."

# # Testing for multistep calls
# @mcp.tool(
#     name="secret_code_step1",
#     description="First step of a multistep tool that generates a secret code and returns a code_id.",
# )
# def secret_code_step1() -> str:
#     """First step of a multistep tool that generates a secret code and returns a code_id."""
#     # Generate a random secret code
#     secret_code = "44"

#     return secret_code

# @mcp.tool(
#     name="secret_code_step2",
#     description="Second step of a multistep tool that takes the result from secret_code_step1 and changes the answer",
# )
# def secret_code_step2(input_code: str) -> str:
#     """Second step of a multistep tool that takes the result from secret_code_step1 and changes the answer."""
#     # Generate a random secret code
#     secret_code = int(input_code) + 1
#     return str(secret_code)


# # final step
# @mcp.tool(
#     name="secret_code_final",
#     description="Final step of a multistep tool that takes the result from secret_code_step2 and returns the final secret code",
# )
# def secret_code_final(input_code: str) -> str:
#     """Final step of a multistep tool that takes the result from secret_code_step2 and returns the final secret code."""
#     # Generate a random secret code
#     secret_code = int(input_code) + 1
#     return str(secret_code)

### These are the tools for WOM

@mcp.tool(
    name="compounds_for_microbes",
  description="""DIRECT MODEL PREDICTION: Retrieve raw metabolite interaction preferences and distinctive scores for microbes (consumption OR secretion based on interaction_type).

This function directly accesses trained model predictions. Raw scores are computed via:
  raw_score = microbe_embedding · metabolite_embeddings^T (dot product of learned embeddings)

Distinctive scores reveal metabolite niche preferences:
  distinctive_score = raw_score - mean(raw_scores at background_level)
where background_level taxa are aggregated to show relative preference vs background.

When querying at taxonomic levels (genus, family, etc.), the function aggregates across all
representative isolates and reports: mean_score, median_score, std_score, count of organisms.
This allows assessment of prediction confidence and variation within a taxon.
    
PARAMETERS:
- microbes (required): Microbes to query as comma-separated list. Format: 'genus:Name' or 'isolate:ID'
  Examples: 'genus:Escherichia', 'genus:Burkholderia,isolate:NP20', 'genus:Pseudomonas'
  For taxa, statistics (mean/median/std/count) are computed across training organisms
  
- top (optional, default=20): Number of top compound results to return per microbe
  Valid values: 1-100
  
- compute_distinctive (optional, default=True): Boolean. When True, compute distinctive scores
  showing metabolite specificity: raw_score - mean(background_level_scores)
  Higher values = more specific to this microbe relative to background taxa
  
- background_level (optional): Taxonomic level for computing distinctive scores
  Valid values: 'genus', 'family', 'order', 'class', 'kingdom'
  If not specified, automatically matches the query microbe's level
  
- min_score (optional): Minimum raw score threshold. Only return compounds with scores >= this value

- interaction_type (optional, default='consumed'): Interaction direction to query
  Valid values: 'consumed', 'secreted'
  Use 'secreted' to analyze production/secretion models

- model_type (optional, default='ordinal_bpr'): Model family to query
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether the query succeeded
- results (list): Array of compound predictions with:
  - raw_score: Direct model prediction (dot product of embeddings)
  - distinctive_score: raw_score - background_mean (if compute_distinctive=True)
  - rank: Ranking among all metabolites
  - percentile: Percentile ranking (0-100%)
  - score_sd: Standard deviation of individual isolate scores for this compound
              (reflects how diverse the genus is in its preference for this compound;
              None for isolate-level queries or single-isolate genera)
  - score_se: Standard error of the genus-mean score = score_sd / sqrt(n_isolates)
              (uncertainty in the genus-level prediction from aggregation)
  - statistics: mean_score, median_score, std_score across all compounds
- metadata (dict): Input validation, aggregation stats (n_isolates), computation method

SCORE_SD / SCORE_SE DISPLAY GUIDANCE:
When score_sd is present, report the preference score as "score ± score_sd" or note the
uncertainty explicitly. A high score_sd relative to score means isolates in this genus
disagree strongly about this compound — the genus-level prediction is an average over
diverse phenotypes. A low score_sd means all isolates point the same direction.
Example: score "6.91 ± 2.3" with "Isolates in model = 3" signals the genus mean is uncertain.
Do NOT embed the isolate count inside the score cell — put it in the dedicated "Isolates in model" column.
- error (string): Error message if success=False

EXAMPLE USAGE:
- Query a single genus: microbes='genus:Escherichia', top=10
- Query multiple taxa: microbes='genus:Burkholderia,genus:Pseudomonas', top=5
- With distinctive scores: microbes='genus:Bosea', compute_distinctive=True, background_level='genus'
    """,
)
def compounds_for_microbes(
  microbes: str,
  top: int = 20,
  compute_distinctive: bool = True,
  background_level: str = None,
  min_score: float = None,
  interaction_type: str = "consumed",
  model_type: str = "ordinal_bpr",
) -> str:
    """Query compounds associated with microbes (consumed or secreted).
    
    Args:
        microbes: Comma-separated microbes in format 'level:name' (e.g., 'genus:Escherichia')
        top: Number of top results to return
        compute_distinctive: Whether to compute distinctive scores
        background_level: Taxonomic level for background comparison
        min_score: Minimum score threshold
    
    Returns:
        JSON string with query results
    """
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query = CompoundsForMicrobesQuery.from_string(
            microbes,
            top_k=top,
            compute_distinctive=compute_distinctive,
            background_level=background_level,
            min_score=min_score,
            config=query_config,
        )
        
        response = engine.query_compounds_for_microbes(query)
        
        # Serialize to JSON string so LLM receives structured data
        return _serialize_wom_response(
          response,
          tool_name="compounds_for_microbes",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        # Input validation error
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format microbes as 'genus:Name,isolate:ID' (comma-separated, no spaces)"
        }, indent=2)
    except Exception as e:
        # Unexpected error
        return json.dumps({
            "success": False,
            "error": f"Server error: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="search_compound",
    description="""Search for compound name matches in the database.
    
PARAMETERS:
- query (required): Compound name to search for
  Examples: 'glutamate', 'serine', 'arginine'
  
- fuzzy (optional, default=True): Use fuzzy matching to find approximate matches
  
- min_score (optional, default=0.7): Minimum fuzzy match score (0.0-1.0)
  Higher values = stricter matching

- interaction_type (optional, default='consumed'): Which model direction to search against
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to search against
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of matching compounds with match types and scores
- metadata (dict): Query parameters and metadata
- error (string): Error message if success=False

EXAMPLE USAGE:
- Exact search: query='glutamate', fuzzy=False
- Fuzzy search: query='glutamic acid', fuzzy=True, min_score=0.8
    """,
)
def search_compound(
    query: str,
    fuzzy: bool = True,
    min_score: float = 0.7,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Search for compound names by fuzzy or exact matching."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        search_query = SearchCompoundQuery(
            query=query,
            fuzzy=fuzzy,
            min_score=min_score,
            config=query_config,
        )
        
        response = engine.query_search_compound(search_query)
        return _serialize_wom_response(
          response,
          tool_name="search_compound",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="list_taxa",
    description="""List all available taxa at a given taxonomic level.
    
PARAMETERS:
- level (optional, default='genus'): Taxonomic level to list
  Valid values: 'genus', 'family', 'order', 'class', 'kingdom', 'isolate'
  
- include_counts (optional, default=True): Include number of isolates per taxon

- interaction_type (optional, default='consumed'): Which model direction provides taxa context
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family providing taxa context
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of taxa with optional isolate counts
- metadata (dict): Level and total taxa count
- error (string): Error message if success=False

EXAMPLE USAGE:
- List all genera: level='genus'
- List families with counts: level='family', include_counts=True
- List orders without counts: level='order', include_counts=False
    """,
)
def list_taxa(
    level: str = "genus",
    include_counts: bool = True,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """List available taxa at a given taxonomic level."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        taxa_query = ListTaxaQuery(
            level=level,
            include_counts=include_counts,
            config=query_config,
        )
        
        response = engine.query_list_taxa(taxa_query)
        return _serialize_wom_response(
          response,
          tool_name="list_taxa",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list taxa: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="microbes_for_compounds",
  description="""DIRECT MODEL PREDICTION: Retrieve which microbes are most associated with given compounds (consumption OR secretion based on interaction_type; inverse of compounds_for_microbes).

This function retrieves raw metabolite interaction predictions aggregated at a taxonomic level.
Raw scores are computed via:
  raw_score = microbe_embedding · metabolite_embeddings^T (dot product)

Scores for multiple compounds are aggregated using specified method:
  - 'mean': average of scores across queried compounds
  - 'sum': total accumulated score (better for synergistic interaction potential)
  - 'max': maximum score (identifies best single-compound interaction)

When querying at taxonomic levels, the function reports: mean_score, std_score,
compound-specific scores, aggregation method used. This shows both overall preference
and per-compound interaction patterns.
    
PARAMETERS:
- compounds (required): Compounds to query as comma-separated list
  Examples: 'glutamate', 'serine,arginine', 'lactate,acetate,butyrate'
  
- level (optional, default='genus'): Taxonomic level to aggregate at
  Valid values: 'isolate', 'genus', 'family', 'order', 'class'
  Statistics (mean/std/count) computed across organisms at this level
  
- top (optional, default=10): Number of top microbes to return
  
- agg_method (optional, default='mean'): How to aggregate scores across compounds
  Valid values: 'mean' (average preference), 'sum' (total consumption potential), 'max' (best single compound)
  
- min_score (optional): Minimum score threshold for results after aggregation

- interaction_type (optional, default='consumed'): Interaction direction to query
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to query
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of microbes, pre-sorted by aggregate_score DESCENDING (highest first).
  For interaction_type='consumed', higher aggregate_score = stronger predicted consumption.
  This holds regardless of sign: a score of -2 indicates stronger preference than -8.
  DO NOT re-sort these results — present them in the order returned by the API.
  Fields per entry:
  - taxon: Taxon name
  - level: Taxonomic level
  - aggregate_score: Computed using agg_method across compounds
  - n_isolates_in_model: Number of isolates in BPR training data for this taxon
  - compound_scores: Individual raw BPR scores for each queried compound
  - compound_score_sd: Per-compound SD derived from isolate embedding variance
    (only present when level != 'isolate' and ≥2 isolates exist in the model)
  - mean_score, std_score (if available)
- metadata.observations: Experimental evidence (see observation_guidance)
- error (string): Error message if success=False

SCORE_SD EXPLANATION:
compound_score_sd[cpd] = sqrt(v_cpd^T · Cov(E_isolates) · v_cpd)
where v_cpd is the compound embedding and E_isolates is the matrix of all isolate
embeddings for that genus. Larger SD means isolates in the genus disagree on this
compound — the genus-level prediction should be treated with caution.

TABLE FORMAT — always present results as a table with these columns:
  | Taxon | BPR score ± SD | Isolates in model | Evidence tier | Doubling time |

  - BPR score ± SD: aggregate_score ± compound_score_sd[cpd] (e.g. "5.68 ± 1.20").
    Do NOT embed the isolate count inside the score cell.
  - Isolates in model: n_isolates_in_model — isolate genomes averaged for this taxon's BPR score.
    Can be large (10s–100s) for common genera. Whether the taxon was lab-tested is shown by Evidence tier.
  - Evidence tier: from metadata.observations.taxa_evidence_summary[taxon]["evidence_strength"]
    (show "unvalidated" if taxon not in taxa_evidence_summary)
  - Doubling time: call taxonomy_levels_for_growth_rate separately if needed; format as "mean ± std h"

EXAMPLE USAGE:
- Top genera consuming acetate: compounds='acetate', level='genus', top=10
- Family-level aggregation (sum): compounds='glucose,fructose', level='family', agg_method='sum'
- Isolate-level (no aggregation): compounds='serine', level='isolate'
    """,
)
def microbes_for_compounds(
    compounds: str,
    level: str = "genus",
    top: int = 10,
    agg_method: str = "mean",
  min_score: float = None,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Find microbes that consume given compounds."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = MicrobesForCompoundsQuery.from_string(
            compounds,
            level=level,
            top_k=top,
            agg_method=agg_method,
            min_score=min_score,
            config=query_config,
        )
        
        response = engine.query_microbes_for_compounds(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="microbes_for_compounds",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format compounds as comma-separated names (e.g., 'acetate,lactate')"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="selective_metabolites",
    description="""DERIVED ANALYSIS: Find metabolites that preferentially favor a target microbe over competitors.

METHOD: Computes selectivity scores by comparing target vs competitor predictions:

  Difference method (default):
    selectivity_score = target_raw_score - mean(competitor_raw_scores)
    Interpretation: Positive values favor target; larger = more selective

  Ratio method:
    selectivity_score = target_raw_score / (mean(competitor_raw_scores) + epsilon)
    Interpretation: >1 = target preference; log-scale comparison better for visualization

All scores are model predictions (embeddings dot products). Metabolites ranked by selectivity
show how specific the target's metabolite consumption is relative to competing organisms.
Use this to identify niche/specialist metabolites for the target microbe.
    
PARAMETERS:
- target (required): Target microbe in format 'genus:Name' or 'isolate:ID'
  Example: 'genus:Pseudomonas'
  
- competitors (required): Competitor microbes as comma-separated list (same format)
  Example: 'genus:Burkholderia,genus:Variovorax'
  Must provide ≥1 competitor; all must be match-able in model
  
- top (optional, default=20): Number of top selective metabolites to return
  
- method (optional, default='difference'): Selectivity computation method
  Valid values:
    'difference' = target_score - mean(competitors)
    'ratio' = target_score / mean(competitors)
  Choose 'difference' for linear comparison, 'ratio' for log-scale comparison
  
- min_selectivity (optional): Minimum selectivity score to include in results

- interaction_type (optional, default='consumed'): Interaction direction for score computation
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to use
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded  
- results (list): Array of selective metabolites, ranked by selectivity_score (highest first):
  - metabolite: Compound name
  - selectivity: Computed selectivity score
  - target_score: Raw model score for target
  - competitor_mean_score: Average raw score for competitors
- metadata (dict): Target/competitor info, method used, input validation
- error (string): Error message if success=False

EXAMPLE USAGE:
- Find Pseudomonas-specific compounds: target='genus:Pseudomonas', competitors='genus:Burkholderia,genus:Variovorax'
- Use ratio method for log comparison: target='genus:Bosea', competitors='genus:Rhodococcus', method='ratio'
- Filter by selectivity: target='genus:Rhodomonas', competitors='genus:Escherichia', min_selectivity=1.5
    """,
)
def selective_metabolites(
    target: str,
    competitors: str,
    top: int = 20,
    method: str = "difference",
  min_selectivity: float = None,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Find metabolites selective for target over competitors."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = SelectiveMetabolitesQuery.from_strings(
            target,
            competitors,
            top_k=top,
            method=method,
            min_selectivity=min_selectivity,
            config=query_config,
        )
        
        response = engine.query_selective_metabolites(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="selective_metabolites",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format as 'genus:Name' or 'isolate:ID' (comma-separate competitors)"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="metabolite_environment_fitness",
    description="""Rank microbes by their fitness in a metabolite environment.
    
PARAMETERS:
- metabolites (required): Available metabolites in the environment (comma-separated)
  Example: 'acetate,lactate,glucose'
  
- level (optional, default='genus'): Taxonomic level to rank
  Valid values: 'isolate', 'genus', 'family', 'order', 'class'
  
- top (optional, default=10): Number of top-ranked microbes to return
  
- fitness_method (optional, default='sum'): How to compute fitness from metabolite scores
  Valid values: 'sum' (total consumption potential), 'mean' (average), 'weighted' (sum of squares)
  
- microbes (optional): Restrict ranking to specific microbes (comma-separated 'level:Name')
  If not provided, ranks all microbes at the given level

- interaction_type (optional, default='consumed'): Interaction direction for fitness scoring
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to use
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of ranked microbes with fitness scores and per-metabolite scores
- metadata (dict): Metabolites, method, level, input validation
- error (string): Error message if success=False

EXAMPLE USAGE:
- Best bacteria for acetate+lactate environment: metabolites='acetate,lactate', level='genus', fitness_method='sum'
- Rank specific isolates: metabolites='glucose,fructose', microbes='isolate:A,isolate:B'
    """,
)
def metabolite_environment_fitness(
    metabolites: str,
    level: str = "genus",
    top: int = 10,
    fitness_method: str = "sum",
  microbes: str = None,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Rank microbes by fitness in a metabolite environment."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = MetaboliteEnvironmentFitnessQuery.from_strings(
            metabolites,
            microbes_str=microbes,
            level=level,
            top_k=top,
            fitness_method=fitness_method,
            config=query_config,
        )
        
        response = engine.query_metabolite_environment_fitness(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="metabolite_environment_fitness",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format metabolites as comma-separated names. Microbes as 'genus:Name' or 'isolate:ID'"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="niche_metabolites",
    description="""DERIVED ANALYSIS: Find distinctive/niche metabolites unique to a microbe.

METHOD: Computes distinctive scores via:
  distinctive_score = raw_score - mean(raw_scores at background_level)

Raw scores are model predictions (embedding dot products). Distinctive score subtracts
the average consumption pattern at a background taxonomic level (genus, family, etc.),
showing which metabolites are preferentially consumed by this microbe vs its background.

Higher distinctive scores = metabolites more specific to this microbe's niche.
Low/negative distinctive scores = metabolites consumed broadly across background taxa.

When the queried microbe aggregates multiple isolates (e.g., genus), reports:
  n_isolates_in_background: number of organisms used to compute background mean
  This indicates statistical robustness (larger n = more stable background estimate)
    
PARAMETERS:
- microbe (required): Microbe to analyze in format 'genus:Name' or 'isolate:ID'
  Example: 'genus:Burkholderia'
  Isolates query single organism; taxa aggregate organisms and report statistics
  
- top (optional, default=20): Number of top distinctive metabolites to return
  
- background_level (optional): Taxonomic level for background comparison
  Valid values: 'genus', 'family', 'order', 'class', 'kingdom'
  If not specified for isolates, defaults to 'genus' level
  For taxa, matches the query level by default
  
- min_distinctive_score (optional): Minimum distinctive score to include

- interaction_type (optional, default='consumed'): Interaction direction for score computation
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to use
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of distinctive metabolites ranked by distinctive_score (highest first):
  - metabolite: Compound name
  - distinctive_score: raw_score - background_mean (niche preference)
  - raw_score: Direct model prediction
  - rank: Ranking among all metabolites
  - percentile: Percentile (0-100%)
- metadata (dict): Microbe info, background_level used, background_mean value,
                  n_isolates_in_background (robustness indicator), input validation
- error (string): Error message if success=False

EXAMPLE USAGE:
- Find Burkholderia niche metabolites: microbe='genus:Burkholderia', top=20
- Compare vs family background: microbe='genus:Pseudomonas', background_level='family'
- Filter by distinctiveness threshold: microbe='genus:Bosea', min_distinctive_score=2.0
- Isolate-specific niche: microbe='isolate:NP20', background_level='genus' (compare vs genus)
    """,
)
def niche_metabolites(
    microbe: str,
    top: int = 20,
    background_level: str = None,
  min_distinctive_score: float = None,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Find distinctive (niche) metabolites for a microbe."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = NicheMetabolitesQuery.from_string(
            microbe,
            top_k=top,
            background_level=background_level,
            min_distinctive_score=min_distinctive_score,
            config=query_config,
        )
        
        response = engine.query_niche_metabolites(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="niche_metabolites",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format microbe as 'genus:Name' or 'isolate:ID'"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="competition_analysis",
    description="""DERIVED ANALYSIS: Determine which microbe "wins" each metabolite in competition.

METHOD: For each metabolite, compares scores across all competing microbes:
  1. Retrieve raw_score (or distinctive_score if use_distinctive=True) for each microbe
  2. Identify winner: microbe with highest score for that metabolite
  3. Compute margin: winner_score - second_place_score
        (measures decisiveness; large margin = clear winner, small margin = close competition)
  4. Tally overall win count per microbe across all metabolites

Scores are model predictions (embeddings dot products). The 'winner' is the microbe
most predicted to consume that metabolite, given its training data. Margin reveals
competitive intensity. Use this to understand niche partitioning and metabolite allocation.
    
PARAMETERS:
- microbes (required): Microbes to analyze (comma-separated 'level:Name' or 'isolate:ID')
  Example: 'genus:Burkholderia,genus:Pseudomonas,genus:Variovorax'
  All microbes must be match-able in model
  
- metabolites (required): Metabolites to score (comma-separated compound names)
  Example: 'acetate,lactate,glucose'
  All metabolites must be match-able in model training data
  
- use_distinctive (optional, default=False): Compute competition using distinctive scores
  False (default) = use raw model scores
  True = use distinctive scores (raw - background_mean) for niche-based competition

- interaction_type (optional, default='consumed'): Interaction direction for score computation
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to use
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of metabolites, each showing:
  - metabolite: Compound name (original request)
  - matched: Canonical compound name (after exact/fuzzy matching)
  - winner: Microbe with highest score for this metabolite
  - winner_score: Score value for winning microbe
  - all_scores: Dictionary of all microbes and their scores for this metabolite
  - margin: winner_score - second_place_score (competition intensity)
- metadata (dict): 
  - n_microbes: Number of microbes compared
  - n_metabolites: Number of metabolites analyzed
  - n_metabolites_analyzed: Actual number with valid scores
  - win_counts: Tally of wins per microbe (e.g., {"Burkholderia": 3, "Pseudomonas": 2})
  - overall_winner: Microbe with most wins across all metabolites
  - input_validation: Matching details
- error (string): Error message if success=False

EXAMPLE USAGE:
- Three-way competition: microbes='genus:Burkholderia,genus:Pseudomonas,genus:Variovorax', metabolites='acetate,lactate,pyruvate'
- Niche-based competition: microbes='genus:Bosea,genus:Rhodococcus', metabolites='arginine,lysine', use_distinctive=True
- Head-to-head: microbes='isolate:A,isolate:B', metabolites='glucose,fructose,malate'
    """,
)
def competition_analysis(
    microbes: str,
    metabolites: str,
  use_distinctive: bool = False,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Analyze competition between microbes for metabolites."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = CompetitionAnalysisQuery.from_string(
            microbes,
            metabolites,
            use_distinctive=use_distinctive,
            config=query_config,
        )
        
        response = engine.query_competition_analysis(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="competition_analysis",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format microbes and metabolites as comma-separated lists"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="rank_substrates",
    description="""DERIVED ANALYSIS: Rank a specific set of substrates for each microbe.

METHOD: For each microbe, retrieves scores for queried substrates and ranks them:
  1. Get raw_score (or distinctive_score) for each substrate from model
  2. Sort substrates by score (highest → lowest)
  3. Return top-k substrates per microbe

Scores are model predictions (embeddings dot products). This operation extracts
subset predictions and orders them, revealing metabolite hierarchy for each organism.
Use this to understand microbial preference rankings within a constrained set of
substrates (e.g., substrates available in an environment).

When querying at taxonomic levels, reports per-substrate scores for the aggregated
taxon, revealing how representative species within the taxon consume these substrates.
    
PARAMETERS:
- microbes (required): Microbes to score (comma-separated 'level:Name' or 'isolate:ID')
  Example: 'genus:Pseudomonas,genus:Burkholderia'
  
- substrates (required): Substrates to rank (comma-separated compound names)
  Example: 'acetate,lactate,pyruvate,citrate'
  All substrates must be match-able in model training data
  
- top (optional, default=10): Maximum substrates to return per microbe
  Only ranked substrates are included in results
  
- use_distinctive (optional, default=False): Rank using distinctive scores
  False (default) = rank by raw model scores
  True = rank by distinctive scores (niche-specific preferences)

- interaction_type (optional, default='consumed'): Interaction direction for ranking
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to use
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of microbes, each with:
  - microbe: Name (level:name or isolate:name)
  - level: Taxonomic level
  - rankings: Array of substrates sorted by score (highest first), each with:
    - substrate: Original substrate name requested
    - matched: Canonical substrate name after matching
    - score: Raw or distinctive score value
    - match_info (if explain=True): fuzzy match details
  - n_scored: Total substrates scored (before filtering to top-k)
- metadata (dict):
  - n_microbes: Number of microbes scored
  - n_substrates: Number of substrates provided
  - microbe_validation: Matching and availability details
  - substrate_validation: Matching and availability details
- error (string): Error message if success=False

EXAMPLE USAGE:
- Rank top 5 preferred substrates for isolates: microbes='isolate:A,isolate:B', substrates='acetate,lactate,glucose,butyrate,propionate', top=5
- Niche-based ranking: microbes='genus:Burkholderia', substrates='arginine,lysine,glutamate', use_distinctive=True
- Family-level preferences: microbes='family:Pseudomonadaceae', substrates='TCA_cycle_intermediates,fatty_acids,aromatic_compounds'
    """,
)
def rank_substrates(
    microbes: str,
    substrates: str,
    top: int = 10,
  use_distinctive: bool = False,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Rank specific substrates for given microbes."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = RankSubstratesQuery.from_string(
            microbes,
            substrates,
            top_k=top,
            use_distinctive=use_distinctive,
            config=query_config,
        )
        
        response = engine.query_rank_substrates(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="rank_substrates",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "hint": "Format microbes and substrates as comma-separated lists"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Query failed: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="model_info",
    description="""Get information and statistics about the loaded model.
    
PARAMETERS:
- include_stats (optional, default=True): Include detailed statistics

- interaction_type (optional, default='consumed'): Which interaction model to inspect
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family to inspect
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Single object with:
  - model_type: Name of the recommendation model (BPR, WRMF, SVD, etc.)
  - embedding_dim: Dimensionality of learned embeddings
  - n_microbes: Total microbes in training data
  - n_metabolites: Total metabolites in training data
  - statistics (if include_stats=True):
    - n_interactions: Total positive microbe-metabolite interactions
    - sparsity: Fraction of unknown interactions
    - avg_metabolites_per_microbe: Average compounds per organism
  - taxonomy: Number of taxa at each level
- metadata (dict): Empty
- error (string): Error message if success=False

EXAMPLE USAGE:
- Get full model info: (no parameters)
- Summary only: include_stats=False
    """,
)
def model_info(
  include_stats: bool = True,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Get model information and statistics."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = ModelInfoQuery(
            include_stats=include_stats,
            config=query_config,
        )
        
        response = engine.query_model_info(query_obj)
        return _serialize_wom_response(
          response,
          tool_name="model_info",
          model_type=model_type,
          interaction_type=interaction_type,
        )
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get model info: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="list_environments",
    description="""List all available metabolite environments for querying microbe preferences.
    
ENVIRONMENTS are pre-defined collections of metabolites from specific ecological contexts
(e.g., plant root exudates, culture media). Use this to discover what environments are
available before querying microbe performance on those environments.

PARAMETERS:
- include_stats (optional, default=True): Include coverage statistics
- min_coverage (optional, default=0.0): Filter to environments with at least this fraction
  of compounds present in the model (0.0 to 1.0)

- interaction_type (optional, default='consumed'): Which model to query
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Each environment has:
  - environment_name: Name of the environment
  - n_compounds: Total compounds in environment
  - n_in_model: How many are present in the model
  - coverage_fraction: Fraction available (n_in_model / n_compounds)
  - description: What this environment represents
  - doi: Reference for the environment data
  - status: 'valid' or 'low_coverage'
- metadata (dict): Query details and orchestration info
- error (string): Error message if success=False

EXAMPLE USAGE:
- List all environments: (no parameters)
- Only high-coverage environments: min_coverage=0.8
- Check secretion model environments: interaction_type='secreted'
    """,
)
def list_environments(
    include_stats: bool = True,
    min_coverage: float = 0.0,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """List available metabolite environments."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = ListEnvironmentsQuery(
            include_stats=include_stats,
            min_coverage=min_coverage,
            config=query_config,
        )
        
        response = engine.query_list_environments(query_obj)
        return _serialize_wom_response(
            response,
            tool_name="list_environments",
            model_type=model_type,
            interaction_type=interaction_type,
        )
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list environments: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="search_environment",
    description="""Fuzzy search for environment names to find the correct environment.
    
Use this when you have a partial or approximate environment name and need to find
the exact match. Helpful for user queries like 'avena' -> 'Avena barbata root exudate'.

PARAMETERS:
- query (required): Partial or approximate environment name to search
  Examples: 'avena', 'R2A', 'minimal', 'yeast'

- max_results (optional, default=5): Maximum number of matches to return
- min_score (optional, default=0.6): Minimum fuzzy match score (0.0 to 1.0)

- interaction_type (optional, default='consumed'): Which model to query
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Each match has:
  - environment_name: Exact environment name (use this for get_environment_compounds)
  - match_score: How well it matched the query (0.0 to 1.0)
  - n_compounds: Total compounds in this environment
  - n_in_model: How many are in the model
  - description: What this environment represents
- metadata (dict): Query details and orchestration info
- error (string): Error message if success=False

EXAMPLE USAGE:
- Find Avena environments: query='avena'
- Match R2A media: query='R2A'
- Broad search: query='minimal', max_results=10
    """,
)
def search_environment(
    query: str,
    max_results: int = 5,
    min_score: float = 0.6,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Fuzzy search for environment names."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = SearchEnvironmentQuery(
            query=query,
            max_results=max_results,
            min_score=min_score,
            config=query_config,
        )
        
        response = engine.query_search_environment(query_obj)
        return _serialize_wom_response(
            response,
            tool_name="search_environment",
            model_type=model_type,
            interaction_type=interaction_type,
        )
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to search environments: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="get_environment_compounds",
    description="""Get the list of metabolites that define a specific environment.
    
This is the KEY TOOL for queries like 'Which microbes grow on Avena barbata exudates?'

WORKFLOW:
1. Use this tool to get the compound list for an environment
2. Extract the representative_compound names from the results
3. Pass those compounds to 'microbes_for_compounds' tool to find which microbes consume them

PARAMETERS:
- environment_name (required): Exact environment name
  Get this from 'list_environments' or 'search_environment'
  Examples: 'NLDM', 'R2A', 'Avena barbata root exudate (2018KZ)'

- only_in_model (optional, default=True): Only return compounds present in the model
  Set to False to see all compounds including unmapped ones

- interaction_type (optional, default='consumed'): Which model to check against
  Valid values: 'consumed', 'secreted'

- model_type (optional, default='ordinal_bpr'): Model family
  Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Each compound has:
  - original_name: Name from the environment definition
  - representative_compound: Mapped model name (use this for microbes_for_compounds!)
  - chebi_id: ChEBI identifier
  - match_type: 'exact', 'fuzzy', or 'none'
  - match_score: Mapping confidence (0.0 to 1.0)
  - in_model: Boolean indicating availability
- metadata (dict): Environment description, compound count, orchestration info
- error (string): Error message if success=False

EXAMPLE USAGE:
- Get NLDM compounds: environment_name='NLDM'
- Get all Avena compounds: environment_name='Avena barbata root exudate (2018KZ)'
- Include unmapped: environment_name='R2A', only_in_model=False

FOLLOW-UP PATTERN:
After getting compounds, use 'microbes_for_compounds' with the representative_compound
values to find which microbes consume those metabolites.
    """,
)
def get_environment_compounds(
    environment_name: str,
    only_in_model: bool = True,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Get metabolites in a specific environment."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )
        
        query_obj = GetEnvironmentCompoundsQuery(
            environment_name=environment_name,
            only_in_model=only_in_model,
            config=query_config,
        )
        
        response = engine.query_get_environment_compounds(query_obj)
        return _serialize_wom_response(
            response,
            tool_name="get_environment_compounds",
            model_type=model_type,
            interaction_type=interaction_type,
        )
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get environment compounds: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="list_function_enrichment_taxa",
    description="""List taxa that have UniProt-derived EC function enrichment profiles.

  This tool is for discovery: use it to determine which taxa are available for
  enzyme-function enrichment lookups before calling enriched_functions_for_taxon.

  PARAMETERS:
  - level (optional, default='genus'): Taxonomic level to inspect
    Valid values: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'

  - min_proteomes (optional, default=1): Minimum number of proteomes required
    Filters sparse taxa with weak support

  - include_counts (optional, default=True): Include proteome/function counts

  - interaction_type (optional, default='consumed'): Model interaction context
    Valid values: 'consumed', 'secreted'

  - model_type (optional, default='ordinal_bpr'): Model family context
    Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

  RESPONSE FORMAT:
  - success (bool)
  - results (list): taxa with support metrics
  - metadata (dict): query context + orchestration metadata
  - error (string): present when success=False

  EXAMPLE USAGE:
  - list_function_enrichment_taxa(level='genus', min_proteomes=5)
    """,
)
def list_function_enrichment_taxa(
    level: str = "genus",
    min_proteomes: int = 1,
    include_counts: bool = True,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
 ) -> str:
    """List taxa with function enrichment coverage."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )

        query_obj = ListFunctionEnrichmentTaxaQuery(
            level=level,
            min_proteomes=min_proteomes,
            include_counts=include_counts,
            config=query_config,
        )

        response = engine.query_list_function_enrichment_taxa(query_obj)
        return _serialize_wom_response(
            response,
            tool_name="list_function_enrichment_taxa",
            model_type=model_type,
            interaction_type=interaction_type,
        )

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list function enrichment taxa: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="enriched_functions_for_taxon",
    description="""Return EC functions ranked by taxonomic specialization for a specific taxon.

  Combines two complementary signals:

  1. TF-IDF specialization score (always included):
     - Measures how characteristic an EC function is of THIS taxon relative to all taxa
       in the database — not just whether the gene is present, but whether it is a
       genomic hallmark of this taxonomic group
     - term frequency = fraction of proteomes in the taxon containing the EC
     - inverse document frequency = downweights ECs common across many taxa
     - tfidf_score = tf * idf — high means the EC is a specialization of this taxon
     - CAVEAT: specialization scores are less reliable for taxa with few sequenced isolates.
       Always check n_proteomes_taxon per result and n_proteomes in metadata.
       Genera or families with < ~5 isolates may show inflated or unstable scores.

  2. BPR metabolic alignment (opt-in via include_bpr_scores=True):
     - Projects each enzyme into BPR latent space via its linked WoM substrate compounds
     - β_enzyme = mean of BPR item embeddings for that EC's associated compounds
     - score = θ_taxon · β_enzyme (mean-centred across ECs in the result set)
     - Positive = enzyme aligns with this taxon's metabolic niche; negative = misaligned
     - Requires BPR model to be loaded (adds ~1 s the first time)

  Use tfidf_score to rank genomically specialized enzymes.
  Use bpr_alignment_score to identify which of those enzymes are metabolically active
  given the compounds this taxon actually consumes or secretes.

  Use this when users ask for distinctive enzyme activities/functions for a genus/family/etc.

  PARAMETERS:
  - level (required): Taxonomic level of the query taxon
    Valid values: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'

  - taxon (required): Taxon name at that level

  - top_k (optional, default=25): Number of functions to return

  - min_tfidf (optional): Filter by minimum specialization score
  - min_frequency (optional): Filter by minimum within-taxon frequency (0-1)

  - include_compounds (optional, default=True): Include top WoM compounds linked to each EC
  - max_compounds_per_function (optional, default=5): Limit compounds listed per EC

  - include_bpr_scores (optional, default=False): Add BPR metabolic alignment score to each EC
    When True, each result gains a 'bpr_alignment_score' field (float or null if no linked
    compounds appear in the BPR model)

  - interaction_type (optional, default='consumed'): Model interaction context
    Valid values: 'consumed', 'secreted'

  - model_type (optional, default='ordinal_bpr'): Model family context
    Valid values: 'ordinal_bpr', 'bpr', 'implicit_bpr', 'lmf', 'wrmf', 'svd'

  RESPONSE FORMAT:
  - success (bool)
  - results (list): EC numbers ranked by specialization score; each entry includes
    tfidf_score, tf_frequency, idf, n_proteomes_with_ec, n_proteomes_taxon,
    and optionally bpr_alignment_score and associated_compounds
  - metadata (dict): requested/matched taxon, n_proteomes (isolate count for the taxon),
    filter settings, bpr_alignment_scores_included
  - error (string): present when success=False

  EXAMPLE USAGE:
  - enriched_functions_for_taxon(level='genus', taxon='Pseudomonas', top_k=30)
  - enriched_functions_for_taxon(level='genus', taxon='Burkholderia', include_bpr_scores=True)
    """,
)
def enriched_functions_for_taxon(
    level: str,
    taxon: str,
    top_k: int = 25,
    min_tfidf: float | None = None,
    min_frequency: float | None = None,
    include_compounds: bool = True,
    max_compounds_per_function: int = 5,
    include_bpr_scores: bool = False,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
 ) -> str:
    """Get enriched EC functions for a taxon."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )

        query_obj = EnrichedFunctionsForTaxonQuery(
            level=level,
            taxon=taxon,
            top_k=top_k,
            min_tfidf=min_tfidf,
            min_frequency=min_frequency,
            include_compounds=include_compounds,
            max_compounds_per_function=max_compounds_per_function,
            include_bpr_scores=include_bpr_scores,
            config=query_config,
        )

        response = engine.query_enriched_functions_for_taxon(query_obj)
        return _serialize_wom_response(
            response,
            tool_name="enriched_functions_for_taxon",
            model_type=model_type,
            interaction_type=interaction_type,
        )

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get enriched functions for taxon: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="explain_compound_group",
    description="""Explain how a compound name is grouped and which representative name WoM chose for it.

  WoM maps every compound to a single "representative name" drawn from the ChEBI
  chemical ontology.  Two compounds share a representative name when their InChIKey
  connectivity block (the first 14-character segment) is identical — meaning they
  are structural isomers or stereoisomers of the same backbone.

  The representative name is chosen by a simple, deterministic rubric — it is NOT
  intended to favor one form of a compound over another biologically:
    1. Names in the "Goldilocks zone" (5-60 characters) are preferred over very
       short abbreviations and very long IUPAC strings
    2. Within that, type priority: UniProt name > ChEBI preferred name > synonym >
       IUPAC name > brand name
    3. Shorter name wins ties; lower ChEBI ID (more established entry) breaks
       remaining ties

  Use this tool when:
  - A compound appears under an unexpected name in results
  - You want to know which synonyms or stereoisomers are collapsed into one entry
  - You want to confirm whether a specific name is in the WoM model

  PARAMETERS:
  - compound (required): Any name for the compound (exact match attempted first,
    then fuzzy)

  RESPONSE FORMAT:
  - success (bool)
  - matched_input (dict): how the input name was resolved (name, chebi_id, match_type, score)
  - representative_name (str): the name WoM uses for this compound group
  - in_wom_model (bool): whether representative_name appears in the BPR training data
  - grouping_key (str): first-block InChIKey shared by all group members
  - group_size (int): total number of names (synonyms + stereoisomers) in the group
  - group_members (list): all names in the group — each entry has:
      name, chebi_id, name_type, inchi_key, is_representative
  - rubric_explanation (str): plain-English summary of why this representative was chosen
    """,
)
def explain_compound_group(compound: str) -> str:
    """Return all names in a compound group and explain the representative name choice."""
    try:
        engine, _ = _get_wom_engine()
        chebi = engine.chebi_data

        # Map the input name to a ChEBI entry
        mapping = map_compound_name(compound, chebi)
        if mapping.representative_name is None:
            return json.dumps({
                "success": False,
                "error": f"Could not match '{compound}' to any ChEBI entry",
            }, indent=2)

        # Find the first-block InChIKey for the matched entry
        matched_rows = chebi[chebi["compound_id"] == mapping.chebi_id]
        if matched_rows.empty:
            # Fall back: find by best_name
            matched_rows = chebi[chebi["best_name"] == mapping.representative_name]

        if matched_rows.empty:
            return json.dumps({
                "success": False,
                "error": f"Matched to '{mapping.representative_name}' but could not locate group in ChEBI data",
            }, indent=2)

        grouping_key = matched_rows.iloc[0]["first_block_inchi_key"]

        # Get all members of the group
        group_df = chebi[chebi["first_block_inchi_key"] == grouping_key].copy()
        group_df = group_df[group_df["name"] != ""].drop_duplicates(subset=["name", "compound_id"])
        group_df = group_df.sort_values(
            ["name_rank", "type_secondary", "name_length", "compound_id"],
            ascending=[True, True, True, True],
        )

        rep_name = mapping.representative_name
        in_wom = rep_name in engine.artifacts.training_data.metabolite_to_idx

        # Build group member list
        members = []
        for _, row in group_df.iterrows():
            members.append({
                "name": row["name"],
                "chebi_id": row["compound_id"],
                "name_type": row.get("name_type", None),
                "inchi_key": row.get("inchi_key", None),
                "is_representative": row["name"] == rep_name,
            })

        # Build rubric explanation
        rep_row = group_df[group_df["name"] == rep_name]
        if not rep_row.empty:
            r = rep_row.iloc[0]
            name_len = len(rep_name)
            in_goldilocks = 5 <= name_len <= 60
            type_label = r.get("name_type", "unknown type")
            rubric_parts = [
                f"The representative '{rep_name}' was chosen because:",
                f"  • name type: {type_label} (type priority rank {int(r['name_rank'])})",
                f"  • length: {name_len} chars — {'within' if in_goldilocks else 'outside'} the preferred 5-60 char Goldilocks zone",
                f"  • ChEBI ID: {r['compound_id']} (lower = more established entry)",
                "",
                "Selection order (lower wins): UniProt name → ChEBI preferred → synonym → IUPAC → brand name;",
                "then names in the 5-60 char range; then shortest; then lowest ChEBI ID.",
                "",
                "This rubric is deterministic and name-agnostic — it does not reflect biological preference.",
            ]
            rubric_explanation = "\n".join(rubric_parts)
        else:
            rubric_explanation = (
                "Representative name chosen by: name type priority (UniProt > preferred > synonym > IUPAC > brand), "
                "then Goldilocks length (5-60 chars), then shortest, then lowest ChEBI ID."
            )

        return json.dumps({
            "success": True,
            "matched_input": {
                "query": compound,
                "resolved_to": mapping.representative_name,
                "chebi_id": mapping.chebi_id,
                "match_type": mapping.match_type,
                "match_score": round(mapping.match_score, 3),
            },
            "representative_name": rep_name,
            "in_wom_model": in_wom,
            "grouping_key": grouping_key,
            "grouping_basis": "first 14-character block of InChIKey (connectivity — same for all stereoisomers)",
            "group_size": len(members),
            "group_members": members,
            "rubric_explanation": rubric_explanation,
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to explain compound group: {str(e)}",
        }, indent=2)


@mcp.tool(
    name="taxonomy_levels_for_growth_rate",
    description="""Look up predicted microbial doubling times from Phydon by taxonomy.

SOURCE: Phydon — a phylogenetic model that predicts growth rates from genomic sequence.
Do NOT call this "GenomeSPOT"; GenomeSPOT is a separate tool for culture conditions
(temperature, pH, salinity, oxygen).

PARAMETERS:
- query (required): Organism name to search (fuzzy matching enabled)
  Examples: 'Pseudomonas', 'Escherichia coli', 'Burkholderia'

- level (optional, default='genus'): Taxonomic level to query
  Valid values: 'domain', 'phylum', 'class', 'order', 'family', 'genus'

RESPONSE: All statistics returned in one call — median, mean, std, min, max, count
(count = number of genomes in Phydon that informed the estimate).

DISPLAY GUIDANCE — when showing growth rate data in any table or narrative:
- Always report: mean ± std (n=count, range min–max h)
- Format example: "2.63 ± 0.8 h (n=14, range 1.1–4.2 h)"
- High std relative to mean signals ecophysiological diversity within the taxon
- Low count (n<5) means the estimate may be unreliable — note it explicitly
- Always attribute to "Phydon", not "GenomeSPOT"

EXAMPLES:
- taxonomy_levels_for_growth_rate(query='Pseudomonas')
- taxonomy_levels_for_growth_rate(query='Pseudomonadaceae', level='family')
    """,
)
def taxonomy_levels_for_growth_rate(query: str, level: str = 'genus') -> str:
    """Query growth rate predictions from Phydon. Returns all stats in one call."""
    try:
        all_stats: dict = {}
        matched_meta: dict = {}
        for stat in ("median", "mean", "std", "min", "max", "count"):
            r = phydon.lookup(query=query, level=level, stat=stat)
            if r is None or (isinstance(r, dict) and r.get("error")):
                return json.dumps({"error": f"No Phydon data found for '{query}' at {level} level"})
            if not matched_meta:
                matched_meta = {
                    "query": r.get("query"),
                    "matched_name": r.get("matched_name"),
                    "matched_level": r.get("matched_level"),
                    "match_score": r.get("match_score"),
                }
            results = r.get("results", [])
            if results:
                all_stats[stat] = results[0].get("value")

        return json.dumps({
            **matched_meta,
            "source": "Phydon",
            "unit": "hours (doubling time)",
            "statistics": all_stats,
            "display_hint": (
                f"Report as: {all_stats.get('mean')} ± {all_stats.get('std')} h "
                f"(n={int(all_stats.get('count', 0))}, "
                f"range {all_stats.get('min')}–{all_stats.get('max')} h)"
            ),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Phydon growth rate lookup failed: {str(e)}"})


@mcp.tool(
    name="culture_conditions_for_taxon",
    description="""Look up predicted culture conditions for any microbial taxon using GenomeSPOT.

Returns aggregated statistics (mean, median, std, se, count) for all ten
culture condition predictions across every genome in the matched taxon group.

PARAMETERS:
- query (required): Organism name to search (fuzzy matching enabled)
  Examples: 'Pseudomonas', 'Thermotoga', 'Archaeoglobus fulgidus'

- level (optional, default='genus'): Taxonomic level to query
  Valid values: 'domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'

CONDITIONS RETURNED (with mean, median, std, se, count each):
- oxygen               Probability of oxygen tolerance (0=strict anaerobe, 1=aerobe)
- temperature_optimum  Optimal growth temperature (°C)
- temperature_min      Minimum growth temperature (°C)
- temperature_max      Maximum growth temperature (°C)
- salinity_optimum     Optimal salinity (% w/v NaCl)
- salinity_min         Minimum salinity (% w/v NaCl)
- salinity_max         Maximum salinity (% w/v NaCl)
- ph_optimum           Optimal pH
- ph_min               Minimum pH
- ph_max               Maximum pH

IMPORTANT: These are ML predictions from genomic sequence, not measured values.
std is always returned — use it to gauge reliability. High std means the taxon
is ecophysiologically diverse; low std indicates a more uniform lifestyle.
count shows how many sequenced isolates informed the estimate.

EXAMPLES:
- culture_conditions_for_taxon(query='Pseudomonas', level='genus')
- culture_conditions_for_taxon(query='Thermoproteota', level='phylum')
    """,
)
def culture_conditions_for_taxon(query: str, level: str = "genus") -> str:
    """Query all GenomeSPOT culture condition predictions for a taxon."""
    try:
        response = genome_spot_module.lookup(query=query, level=level)
        if response is None or (isinstance(response, dict) and response.get("error")):
            return json.dumps({
                "error": f"No GenomeSPOT data found for '{query}' at {level} level"
            })
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"error": f"GenomeSPOT lookup failed: {str(e)}"})


@mcp.tool(
    name="lookup_culture_condition",
    description="""Look up a single GenomeSPOT culture condition prediction for a taxon.

Use this when you want focused statistics for one condition (e.g. optimal pH or
temperature) rather than the full ten-condition report from culture_conditions_for_taxon.

PARAMETERS:
- query (required): Organism name to search (fuzzy matching enabled)

- condition (required): Which condition to retrieve
  Valid values:
    'oxygen'               - oxygen tolerance probability (0=strict anaerobe, 1=aerobe)
    'temperature_optimum'  - optimal growth temperature (°C)
    'temperature_min'      - minimum growth temperature (°C)
    'temperature_max'      - maximum growth temperature (°C)
    'salinity_optimum'     - optimal salinity (% w/v NaCl)
    'salinity_min'         - minimum salinity (% w/v NaCl)
    'salinity_max'         - maximum salinity (% w/v NaCl)
    'ph_optimum'           - optimal pH
    'ph_min'               - minimum pH
    'ph_max'               - maximum pH

- level (optional, default='genus'): Taxonomic level to query
  Valid values: 'domain', 'phylum', 'class', 'order', 'family', 'genus', 'species'

- stat (optional, default='median'): Primary statistic to highlight
  Valid values: 'mean', 'median', 'std', 'se', 'count'
  NOTE: std is always returned regardless of this parameter — predictions carry
  substantial inter-taxon variability and callers should always consider spread.

EXAMPLES:
- lookup_culture_condition(query='Pseudomonas', condition='temperature_optimum')
- lookup_culture_condition(query='Halobacteria', condition='salinity_optimum', level='class', stat='mean')
- lookup_culture_condition(query='Methanobacterium', condition='oxygen', level='genus')
    """,
)
def lookup_culture_condition(
    query: str,
    condition: str,
    level: str = "genus",
    stat: str = "median",
) -> str:
    """Query a single GenomeSPOT culture condition for a taxon."""
    try:
        response = genome_spot_module.lookup(
            query=query, level=level, condition=condition, stat=stat
        )
        if response is None or (isinstance(response, dict) and response.get("error")):
            return json.dumps({
                "error": f"No GenomeSPOT data found for '{query}' / '{condition}' at {level} level"
            })
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"error": f"GenomeSPOT condition lookup failed: {str(e)}"})


@mcp.tool(
    name="analyze_substrate_gaps",
    description="""Cross-reference BPR metabolite preference with annotated enzyme presence to find biological gaps.

  Uses two independent signals that do NOT share data:
  - BPR preference score: learned from exometabolite uptake/secretion experiments
  - Enzyme annotation: derived from UniProt catalytic activity + ChEBI reaction data

  Divergence between these signals is biologically informative:

  ORPHAN ENZYME CANDIDATES (high BPR preference, no annotated enzyme):
    The taxon strongly prefers this substrate in exometabolome data, but no homolog-based
    enzyme annotation exists for its catabolism. Possible interpretations:
    - Novel enzyme family with no characterized homologs (true orphan)
    - Known enzyme with poor sequence similarity to characterized examples
    - Gap in the UniProt/ChEBI reaction database for this organism's lifestyle
    These are hypotheses, not confirmed orphans — verify with genome mining.

  REGULATED/SUPPRESSED SUBSTRATES (annotated enzyme present, low/negative BPR):
    The taxon has the genetic capacity to catabolize this compound but the exometabolome
    shows low or no consumption. Possible interpretations:
    - Catabolite repression (CCR): a preferred substrate suppresses expression of these enzymes
    - Genome streamlining: genes retained but rarely expressed (e.g., SAR11-type ecology)
    - Conditional utilization: only expressed under specific environmental conditions
    n_enzymes shows how many distinct EC numbers are annotated for that compound.

  PARAMETERS:
  - taxon (required): Taxon name to analyze
  - level (optional, default='genus'): Taxonomic level
    Valid values: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'
  - top_k (optional, default=10): Max results per section (orphans and regulated each)
  - bpr_z_threshold (optional, default=1.0): Z-score cutoff for high/low BPR preference
    1.0 = top/bottom ~16% of compounds for this taxon; increase to be more selective
  - interaction_type (optional, default='consumed'): 'consumed' or 'secreted'
  - model_type (optional, default='ordinal_bpr'): Model family

  RESPONSE FORMAT:
  - results (dict) with two sections:
    - orphan_candidates: list of {compound, bpr_z_score, bpr_raw_score}
      sorted by bpr_z_score descending (most preferred first)
    - regulated_suppressed: list of {compound, bpr_z_score, bpr_raw_score, n_enzymes}
      sorted by bpr_z_score ascending (most suppressed first)
  - metadata: matched_taxon, n_proteomes, n_metabolites_in_model,
    n_compounds_with_enzyme_support, bpr_z_threshold, total counts per section

  EXAMPLE USAGE:
  - analyze_substrate_gaps(taxon='Novosphingobium')
  - analyze_substrate_gaps(taxon='SAR11', level='family', bpr_z_threshold=1.5)
    """,
)
def analyze_substrate_gaps(
    taxon: str,
    level: str = "genus",
    top_k: int = 10,
    bpr_z_threshold: float = 1.0,
    interaction_type: str = "consumed",
    model_type: str = "ordinal_bpr",
) -> str:
    """Find metabolite-enzyme gaps for a taxon."""
    try:
        engine, query_config = _get_wom_engine(
            model_type=model_type,
            interaction_type=interaction_type,
        )

        query_obj = SubstrateGapQuery(
            level=level,
            taxon=taxon,
            top_k=top_k,
            bpr_z_threshold=bpr_z_threshold,
            config=query_config,
        )

        response = engine.query_substrate_gaps(query_obj)
        return _serialize_wom_response(
            response,
            tool_name="analyze_substrate_gaps",
            model_type=model_type,
            interaction_type=interaction_type,
        )

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to analyze substrate gaps: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="calculator_mean",
    description="Calculate the mean (average) of a list of numbers.",
)
def calculator_mean(numbers: list[float]) -> str:
    if not numbers:
        return json.dumps({"success": False, "error": "No numbers provided."})
    result = sum(numbers) / len(numbers)
    return json.dumps({"success": True, "mean": result, "input": numbers})


@mcp.tool(
    name="calculator_sum",
    description="Calculate the sum of a list of numbers.",
)
def calculator_sum(numbers: list[float]) -> str:
    if not numbers:
        return json.dumps({"success": False, "error": "No numbers provided."})
    result = sum(numbers)
    return json.dumps({"success": True, "sum": result, "input": numbers})


@mcp.tool(
    name="calculator_divide",
    description="Divide a numerator by a denominator.",
)
def calculator_divide(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return json.dumps({"success": False, "error": "Cannot divide by zero."})
    result = numerator / denominator
    return json.dumps({"success": True, "result": result, "numerator": numerator, "denominator": denominator})

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)