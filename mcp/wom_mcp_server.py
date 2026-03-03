"""
MCP server for the Web of Microbes growth-rate service.

Purpose: expose lightweight tools that help agents discover taxonomy levels
and interact with the growth-rate API used to look up predicted microbial
doubling times.
"""

from fastmcp import FastMCP
import requests
import random
import json
import sys

import sys


sys.path.insert(0, "./growth_rate")  # Ensure the /app directory is in the Python path for imports

import growth_rate as phydon
import json
phydon.CACHE_DIR = "./growth_rate/cache"  # Set the cache directory to a known location

sys.path.insert(1, "./wom")  # Ensure the /app directory is in the Python path for imports

from pathlib import Path
import api
from api.query_engine import QueryEngine

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
)

wom_dir = Path(api.__file__).parent.parent
model_dir = wom_dir / "trained_models" / "bpr_consumed"

# Create config
config = QueryConfig(
    model_dir=model_dir,
    output_format='json',
    verbose=True,
    explain=True,
)



mcp = FastMCP("My MCP Server")

@mcp.tool(
    name="hello_world_greet",           # Custom tool name for the LLM
    description="None.", # Custom description
)
def greet(name: str) -> str:
    """Simple connectivity check for the MCP server."""
    return f"Hello, {name}! Here is a number to remember {random.randint(1, 100)}."

# Testing for multistep calls
@mcp.tool(
    name="secret_code_step1",
    description="First step of a multistep tool that generates a secret code and returns a code_id.",
)
def secret_code_step1() -> str:
    """First step of a multistep tool that generates a secret code and returns a code_id."""
    # Generate a random secret code
    secret_code = "44"

    return secret_code

@mcp.tool(
    name="secret_code_step2",
    description="Second step of a multistep tool that takes the result from secret_code_step1 and changes the answer",
)
def secret_code_step2(input_code: str) -> str:
    """Second step of a multistep tool that takes the result from secret_code_step1 and changes the answer."""
    # Generate a random secret code
    secret_code = int(input_code) + 1
    return str(secret_code)


# final step
@mcp.tool(
    name="secret_code_final",
    description="Final step of a multistep tool that takes the result from secret_code_step2 and returns the final secret code",
)
def secret_code_final(input_code: str) -> str:
    """Final step of a multistep tool that takes the result from secret_code_step2 and returns the final secret code."""
    # Generate a random secret code
    secret_code = int(input_code) + 1
    return str(secret_code)

### These are the tools for WOM

@mcp.tool(
    name="compounds_for_microbes",
    description="""DIRECT MODEL PREDICTION: Retrieve raw metabolite preferences and distinctive scores for microbes.

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

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether the query succeeded
- results (list): Array of compound predictions with:
  - raw_score: Direct model prediction
  - distinctive_score: raw_score - background_mean (if compute_distinctive=True)
  - rank: Ranking among all metabolites
  - percentile: Percentile ranking (0-100%)
  - statistics: mean_score, median_score, std_score if aggregating taxa
- metadata (dict): Input validation, aggregation stats (n_isolates), computation method
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
    min_score: float = None
) -> str:
    """Query compounds consumed by microbes.
    
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
        engine = QueryEngine(model_dir, config)
        
        query = CompoundsForMicrobesQuery.from_string(
            microbes,
            top_k=top,
            compute_distinctive=compute_distinctive,
            background_level=background_level,
            min_score=min_score,
            config=config,
        )
        
        response = engine.query_compounds_for_microbes(query)
        
        # Serialize to JSON string so LLM receives structured data
        return json.dumps(response.to_dict(), indent=2)
    
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
    min_score: float = 0.7
) -> str:
    """Search for compound names by fuzzy or exact matching."""
    try:
        engine = QueryEngine(model_dir, config)
        
        search_query = SearchCompoundQuery(
            query=query,
            fuzzy=fuzzy,
            min_score=min_score,
            config=config,
        )
        
        response = engine.query_search_compound(search_query)
        return json.dumps(response.to_dict(), indent=2)
    
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
    include_counts: bool = True
) -> str:
    """List available taxa at a given taxonomic level."""
    try:
        engine = QueryEngine(model_dir, config)
        
        taxa_query = ListTaxaQuery(
            level=level,
            include_counts=include_counts,
            config=config,
        )
        
        response = engine.query_list_taxa(taxa_query)
        return json.dumps(response.to_dict(), indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list taxa: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="microbes_for_compounds",
    description="""DIRECT MODEL PREDICTION: Retrieve which microbes consume given compounds (inverse of compounds_for_microbes).

This function retrieves raw metabolite consumption predictions aggregated at a taxonomic level.
Raw scores are computed via:
  raw_score = microbe_embedding · metabolite_embeddings^T (dot product)

Scores for multiple compounds are aggregated using specified method:
  - 'mean': average of scores across queried compounds
  - 'sum': total accumulated score (better for synergistic consumption)
  - 'max': maximum score (identifies best single compound consumption)

When querying at taxonomic levels, the function reports: mean_score, std_score,
compound-specific scores, aggregation method used. This shows both overall preference
and per-compound consumption patterns.
    
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

RESPONSE FORMAT:
Returns JSON with:
- success (bool): Whether query succeeded
- results (list): Array of ranked microbes with:
  - aggregate_score: Computed using agg_method across compounds
  - compound_scores: Individual raw scores for each queried compound
  - mean_score, std_score (if available)
  - organism_count: Number of organisms in taxon (when aggregating)
- metadata (dict): Input validation, aggregation method, statistics
- error (string): Error message if success=False

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
    min_score: float = None
) -> str:
    """Find microbes that consume given compounds."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = MicrobesForCompoundsQuery.from_string(
            compounds,
            level=level,
            top_k=top,
            agg_method=agg_method,
            min_score=min_score,
            config=config,
        )
        
        response = engine.query_microbes_for_compounds(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
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
    min_selectivity: float = None
) -> str:
    """Find metabolites selective for target over competitors."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = SelectiveMetabolitesQuery.from_strings(
            target,
            competitors,
            top_k=top,
            method=method,
            min_selectivity=min_selectivity,
            config=config,
        )
        
        response = engine.query_selective_metabolites(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
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
    microbes: str = None
) -> str:
    """Rank microbes by fitness in a metabolite environment."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = MetaboliteEnvironmentFitnessQuery.from_strings(
            metabolites,
            microbes_str=microbes,
            level=level,
            top_k=top,
            fitness_method=fitness_method,
            config=config,
        )
        
        response = engine.query_metabolite_environment_fitness(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
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
    min_distinctive_score: float = None
) -> str:
    """Find distinctive (niche) metabolites for a microbe."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = NicheMetabolitesQuery.from_string(
            microbe,
            top_k=top,
            background_level=background_level,
            min_distinctive_score=min_distinctive_score,
            config=config,
        )
        
        response = engine.query_niche_metabolites(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
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
    use_distinctive: bool = False
) -> str:
    """Analyze competition between microbes for metabolites."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = CompetitionAnalysisQuery.from_string(
            microbes,
            metabolites,
            use_distinctive=use_distinctive,
            config=config,
        )
        
        response = engine.query_competition_analysis(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
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
    use_distinctive: bool = False
) -> str:
    """Rank specific substrates for given microbes."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = RankSubstratesQuery.from_string(
            microbes,
            substrates,
            top_k=top,
            use_distinctive=use_distinctive,
            config=config,
        )
        
        response = engine.query_rank_substrates(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
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
def model_info(include_stats: bool = True) -> str:
    """Get model information and statistics."""
    try:
        engine = QueryEngine(model_dir, config)
        
        query_obj = ModelInfoQuery(
            include_stats=include_stats,
            config=config,
        )
        
        response = engine.query_model_info(query_obj)
        return json.dumps(response.to_dict(), indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get model info: {str(e)}"
        }, indent=2)
    

@mcp.tool(
    name="taxonomy_levels_for_growth_rate",
    description="""Look up predicted microbial growth rates (doubling times) by taxonomy.
    
PARAMETERS:
- query (required): Organism name to search (fuzzy matching enabled)
  Examples: 'Pseudomonas', 'Escherichia coli', 'genus:Burkholderia'

- level (optional, default='genus'): Taxonomic level to query
  Valid values: 'domain', 'phylum', 'class', 'order', 'family', 'genus'

- stat (optional, default='median'): Aggregation statistic
  Valid values: 'mean', 'median', 'min', 'max', 'std'

RESPONSE: Hours (doubling time in hours) or error message

EXAMPLES:
- Look up genus growth rate: query='Pseudomonas', stat='median'
- Get range of growth rates: query='Burkholderia', stat='min' then stat='max'
- Family-level lookup: query='Pseudomonadaceae', level='family'
    """,
)
def taxonomy_levels_for_growth_rate(query: str, level: str = 'genus', stat: str = 'median') -> str:
    """Query growth rate predictions from Phydon database.
    
    Returns doubling time in hours with statistics.
    """
    try:
        response = phydon.lookup(query=query, level=level, stat=stat)
        if response is None or (isinstance(response, dict) and response.get("error")):
            return json.dumps({"error": f"No growth rate data found for '{query}' at {level} level"})
        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Growth rate lookup failed: {str(e)}"})


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)