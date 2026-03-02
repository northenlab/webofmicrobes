#!/usr/bin/env python
"""
Command-line interface for metabolite preference predictions.

Unified CLI supporting simple and complex queries. Outputs JSON by default.

Basic Examples:
    # What compounds does this microbe consume?
    wom compounds --microbes "genus:Burkholderia" --top 20
    
    # What microbes consume these compounds?
    wom microbes --compounds "glutamate,serine" --level genus --top 10
    
    # Search for a compound
    wom search "glutamic acid"
    
    # List available genera
    wom list --level genus
    
    # Model information
    wom info

Complex Examples:
    # Which metabolites favor target over competitors?
    wom selective --target "genus:Pseudomonas" \
        --competitors "genus:Burkholderia,genus:Variovorax" --top 10
    
    # Which microbes thrive in this metabolite environment?
    wom fitness --metabolites "arginine,lysine,glutamate" --level genus --top 10
    
    # What makes this microbe unique compared to its genus?
    wom niche --microbe "isolate:RhodococcusOAS809" --top 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def output_response(response, format="json"):
    """Output query response with validation warnings highlighted."""
    # Display validation warnings to stderr first (before any output)
    if response.metadata and "warnings" in response.metadata:
        warnings = response.metadata["warnings"]
        if warnings:
            print("\n⚠️  INPUT VALIDATION WARNINGS:", file=sys.stderr)
            for warning in warnings:
                print(f"   {warning}", file=sys.stderr)
            print("   See metadata.input_validation for details", file=sys.stderr)
            print()
    
    if format == "json":
        print(json.dumps(response.to_dict(), indent=2))
    elif format == "table":
        if not response.success:
            print(f"Error: {response.error}", file=sys.stderr)
            return
        
        if not response.results:
            print("No results", file=sys.stderr)
            return
        
        # Simple table output
        import pandas as pd
        df = pd.DataFrame(response.results)
        print(df.to_string(index=False))


def execute_query(command: str, params: dict, engine: QueryEngine, config: QueryConfig = None):
    """
    Execute a query command with parameters.
    
    This function is the shared entry point for both CLI and API.
    
    Args:
        command: Command name (e.g., "compounds", "microbes", "competition")
        params: Dictionary of command parameters
        engine: QueryEngine instance
        config: Optional QueryConfig (creates default if None)
    
    Returns:
        QueryResponse object
    
    Raises:
        ValueError: If command is unknown or parameters are invalid
    """
    if config is None:
        config = QueryConfig()
    
    # Route to appropriate handler
    if command == "compounds":
        query = CompoundsForMicrobesQuery.from_string(
            params.get("microbes", ""),
            top_k=params.get("top", 20),
            compute_distinctive=params.get("compute_distinctive", True),
            background_level=params.get("background_level"),
            min_score=params.get("min_score"),
            config=config,
        )
        query.explain = params.get("explain", False)
        return engine.query_compounds_for_microbes(query)
    
    elif command == "microbes":
        query = MicrobesForCompoundsQuery.from_string(
            params.get("compounds", ""),
            level=params.get("level", "genus"),
            top_k=params.get("top", 10),
            agg_method=params.get("agg", "mean"),
            min_score=params.get("min_score"),
            explain=params.get("explain", False),
            config=config,
        )
        return engine.query_microbes_for_compounds(query)
    
    elif command == "selective":
        query = SelectiveMetabolitesQuery.from_strings(
            params.get("target", ""),
            params.get("competitors", ""),
            top_k=params.get("top", 20),
            method=params.get("method", "difference"),
            min_selectivity=params.get("min_selectivity"),
            use_distinctive=params.get("use_distinctive", False),
            explain=params.get("explain", False),
            config=config,
        )
        return engine.query_selective_metabolites(query)
    
    elif command == "fitness":
        query = MetaboliteEnvironmentFitnessQuery.from_strings(
            params.get("metabolites", ""),
            microbes_str=params.get("microbes"),
            level=params.get("level", "genus"),
            top_k=params.get("top", 10),
            fitness_method=params.get("fitness_method", "sum"),
            use_distinctive=params.get("use_distinctive", False),
            explain=params.get("explain", False),
            config=config,
        )
        return engine.query_metabolite_environment_fitness(query)
    
    elif command == "niche":
        query = NicheMetabolitesQuery.from_string(
            params.get("microbe", ""),
            top_k=params.get("top", 20),
            background_level=params.get("background_level"),
            min_distinctive_score=params.get("min_distinctive_score"),
            explain=params.get("explain", False),
            config=config,
        )
        return engine.query_niche_metabolites(query)
    
    elif command == "competition":
        query = CompetitionAnalysisQuery.from_string(
            params.get("microbes", ""),
            params.get("metabolites", ""),
            use_distinctive=params.get("use_distinctive", False),
            explain=params.get("explain", False),
            config=config,
        )
        return engine.query_competition_analysis(query)
    
    elif command == "rank-substrates":
        query = RankSubstratesQuery.from_string(
            params.get("microbes", ""),
            params.get("substrates", ""),
            top_k=params.get("top", 10),
            use_distinctive=params.get("use_distinctive", False),
            agg_method=params.get("agg"),
            explain=params.get("explain", False),
            config=config,
        )
        return engine.query_rank_substrates(query)
    
    elif command == "search":
        query = SearchCompoundQuery(
            query=params.get("query", ""),
            fuzzy=params.get("fuzzy", True),
            min_score=params.get("min_score", 0.7),
            config=config,
        )
        return engine.query_search_compound(query)
    
    elif command == "list":
        query = ListTaxaQuery(
            level=params.get("level", "genus"),
            include_counts=params.get("include_counts", True),
            config=config,
        )
        return engine.query_list_taxa(query)
    
    elif command == "info":
        query = ModelInfoQuery(
            include_stats=params.get("include_stats", True),
            config=config,
        )
        return engine.query_model_info(query)
    
    else:
        raise ValueError(f"Unknown command: {command}")


def main():
    parser = argparse.ArgumentParser(
        description="Query metabolite preference predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global options
    parser.add_argument(
        "--model-dir",
        default="trained_models/latest",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--format",
        choices=["json", "table"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps (helps trace algorithm)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # =========================================================================
    # Command: compounds
    # =========================================================================
    compounds_parser = subparsers.add_parser(
        "compounds",
        help="Predict compounds consumed by microbes"
    )
    compounds_parser.add_argument(
        "--microbes",
        required=True,
        help="Microbes to query (format: 'genus:Name,isolate:ID')"
    )
    compounds_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results per microbe"
    )
    compounds_parser.add_argument(
        "--no-distinctive",
        action="store_true",
        help="Disable distinctive score computation"
    )
    compounds_parser.add_argument(
        "--background",
        help="Background level for distinctive scores"
    )
    compounds_parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum raw score threshold"
    )
    compounds_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: microbes
    # =========================================================================
    microbes_parser = subparsers.add_parser(
        "microbes",
        help="Predict microbes that consume compounds"
    )
    microbes_parser.add_argument(
        "--compounds",
        required=True,
        help="Compounds to query (comma-separated)"
    )
    microbes_parser.add_argument(
        "--level",
        default="genus",
        choices=["isolate", "genus", "family", "order", "class", "kingdom"],
        help="Taxonomic level to aggregate"
    )
    microbes_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results"
    )
    microbes_parser.add_argument(
        "--agg",
        choices=["mean", "sum", "max"],
        default="mean",
        help="Score aggregation method"
    )
    microbes_parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum aggregate score threshold"
    )
    microbes_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: selective
    # =========================================================================
    selective_parser = subparsers.add_parser(
        "selective",
        help="Find metabolites that favor target over competitors"
    )
    selective_parser.add_argument(
        "--target",
        required=True,
        help="Target microbe (format: 'genus:Name' or 'isolate:ID')"
    )
    selective_parser.add_argument(
        "--competitors",
        required=True,
        help="Competitor microbes (comma-separated)"
    )
    selective_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results"
    )
    selective_parser.add_argument(
        "--method",
        choices=["difference", "ratio"],
        default="difference",
        help="Selectivity computation method"
    )
    selective_parser.add_argument(
        "--min-selectivity",
        type=float,
        help="Minimum selectivity threshold"
    )
    selective_parser.add_argument(
        "--use-distinctive",
        action="store_true",
        help="Use distinctive scores instead of raw scores"
    )
    selective_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: fitness
    # =========================================================================
    fitness_parser = subparsers.add_parser(
        "fitness",
        help="Rank microbes by fitness in metabolite environment"
    )
    fitness_parser.add_argument(
        "--metabolites",
        required=True,
        help="Available metabolites (comma-separated)"
    )
    fitness_parser.add_argument(
        "--level",
        default="genus",
        choices=["isolate", "genus", "family", "order", "class", "kingdom"],
        help="Taxonomic level (if not specifying --microbes)"
    )
    fitness_parser.add_argument(
        "--microbes",
        help="Specific microbes to score (optional)"
    )
    fitness_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results"
    )
    fitness_parser.add_argument(
        "--method",
        choices=["sum", "mean", "weighted"],
        default="sum",
        help="Fitness computation method"
    )
    fitness_parser.add_argument(
        "--use-distinctive",
        action="store_true",
        help="Use distinctive scores instead of raw scores"
    )
    fitness_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: niche
    # =========================================================================
    niche_parser = subparsers.add_parser(
        "niche",
        help="Find distinctive metabolites for a microbe"
    )
    niche_parser.add_argument(
        "--microbe",
        required=True,
        help="Microbe to query (format: 'genus:Name' or 'isolate:ID')"
    )
    niche_parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results"
    )
    niche_parser.add_argument(
        "--background",
        help="Background level for comparison"
    )
    niche_parser.add_argument(
        "--min-distinctive",
        type=float,
        help="Minimum distinctive score threshold"
    )
    niche_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: competition
    # =========================================================================
    competition_parser = subparsers.add_parser(
        "competition",
        help="Analyze competition between microbes for metabolites"
    )
    competition_parser.add_argument(
        "--microbes",
        "-m",
        required=True,
        help='Competing microbes (e.g., "genus:Pseudomonas,genus:Burkholderia")'
    )
    competition_parser.add_argument(
        "--metabolites",
        "-c",
        required=True,
        help='Metabolites to compete for (comma-separated)'
    )
    competition_parser.add_argument(
        "--use-distinctive",
        action="store_true",
        help="Use distinctive scores instead of raw scores"
    )
    competition_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: rank-substrates
    # =========================================================================
    rank_parser = subparsers.add_parser(
        "rank-substrates",
        help="Rank specific substrates for specific microbes"
    )
    rank_parser.add_argument(
        "--microbes",
        "-m",
        required=True,
        help='Microbes to analyze (e.g., "genus:Bacillus,genus:Pseudomonas")'
    )
    rank_parser.add_argument(
        "--substrates",
        "-s",
        required=True,
        help='Substrates to rank (comma-separated)'
    )
    rank_parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=10,
        help="Number of top substrates per microbe"
    )
    rank_parser.add_argument(
        "--use-distinctive",
        action="store_true",
        help="Use distinctive scores instead of raw scores"
    )
    rank_parser.add_argument(
        "--agg",
        choices=["mean", "median", "min", "max"],
        help="If multiple microbes, how to aggregate (default: show each separately)"
    )
    rank_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show detailed computation steps"
    )
    
    # =========================================================================
    # Command: search
    # =========================================================================
    search_parser = subparsers.add_parser(
        "search",
        help="Search for compound name matches"
    )
    search_parser.add_argument(
        "query",
        help="Compound name to search for"
    )
    search_parser.add_argument(
        "--exact",
        action="store_true",
        help="Exact match only (no fuzzy matching)"
    )
    search_parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum fuzzy match score"
    )
    
    # =========================================================================
    # Command: list
    # =========================================================================
    list_parser = subparsers.add_parser(
        "list",
        help="List available taxa"
    )
    list_parser.add_argument(
        "--level",
        default="genus",
        choices=["genus", "family", "order", "class", "kingdom"],
        help="Taxonomic level"
    )
    list_parser.add_argument(
        "--no-counts",
        action="store_true",
        help="Don't include isolate counts"
    )
    
    # =========================================================================
    # Command: info
    # =========================================================================
    info_parser = subparsers.add_parser(
        "info",
        help="Get model information"
    )
    info_parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Exclude detailed statistics"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Create config
    config = QueryConfig(
        model_dir=args.model_dir,
        output_format=args.format,
        verbose=args.verbose,
        explain=args.explain,
    )
    
    # Create engine
    engine = QueryEngine(args.model_dir, config)
    
    # Execute command
    try:
        if args.command == "compounds":
            query = CompoundsForMicrobesQuery.from_string(
                args.microbes,
                top_k=args.top,
                compute_distinctive=not args.no_distinctive,
                background_level=args.background,
                min_score=args.min_score,
                config=config,
            )
            query.explain = args.explain  # Override from global flag
            response = engine.query_compounds_for_microbes(query)
        
        elif args.command == "microbes":
            query = MicrobesForCompoundsQuery.from_string(
                args.compounds,
                level=args.level,
                top_k=args.top,
                agg_method=args.agg,
                min_score=args.min_score,
                explain=getattr(args, 'explain', False),
                config=config,
            )
            response = engine.query_microbes_for_compounds(query)
        
        elif args.command == "selective":
            query = SelectiveMetabolitesQuery.from_strings(
                args.target,
                args.competitors,
                top_k=args.top,
                method=args.method,
                min_selectivity=args.min_selectivity,
                use_distinctive=getattr(args, 'use_distinctive', False),
                explain=getattr(args, 'explain', False),
                config=config,
            )
            response = engine.query_selective_metabolites(query)
        
        elif args.command == "fitness":
            query = MetaboliteEnvironmentFitnessQuery.from_strings(
                args.metabolites,
                microbes_str=args.microbes,
                level=args.level,
                top_k=args.top,
                fitness_method=args.method,
                use_distinctive=getattr(args, 'use_distinctive', False),
                explain=getattr(args, 'explain', False),
                config=config,
            )
            response = engine.query_metabolite_environment_fitness(query)
        
        elif args.command == "niche":
            query = NicheMetabolitesQuery.from_string(
                args.microbe,
                top_k=args.top,
                background_level=args.background,
                min_distinctive_score=args.min_distinctive,
                explain=getattr(args, 'explain', False),
                config=config,
            )
            response = engine.query_niche_metabolites(query)
        
        elif args.command == "competition":
            query = CompetitionAnalysisQuery.from_string(
                args.microbes,
                args.metabolites,
                use_distinctive=getattr(args, 'use_distinctive', False),
                explain=getattr(args, 'explain', False),
                config=config,
            )
            response = engine.query_competition_analysis(query)
        
        elif args.command == "rank-substrates":
            query = RankSubstratesQuery.from_string(
                args.microbes,
                args.substrates,
                top_k=args.top,
                use_distinctive=getattr(args, 'use_distinctive', False),
                agg_method=getattr(args, 'agg', None),
                explain=getattr(args, 'explain', False),
                config=config,
            )
            response = engine.query_rank_substrates(query)
        
        elif args.command == "search":
            query = SearchCompoundQuery(
                query=args.query,
                fuzzy=not args.exact,
                min_score=args.min_score,
                config=config,
            )
            response = engine.query_search_compound(query)
        
        elif args.command == "list":
            query = ListTaxaQuery(
                level=args.level,
                include_counts=not args.no_counts,
                config=config,
            )
            response = engine.query_list_taxa(query)
        
        elif args.command == "info":
            query = ModelInfoQuery(
                include_stats=not args.no_stats,
                config=config,
            )
            response = engine.query_model_info(query)
        
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
        
        # Output response
        output_response(response, args.format)
        
        # Exit with appropriate code
        sys.exit(0 if response.success else 1)
    
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        error_response = {
            "success": False,
            "error": str(e),
            "command": args.command,
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
