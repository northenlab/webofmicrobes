#!/usr/bin/env python
"""
Complex query commands for BPR consumption model.

These commands handle multi-step reasoning that would otherwise require
multiple wom.py calls and numerical aggregation by an LLM.

Commands:
    selective   - Find metabolites favoring target microbe over competitors
    fitness     - Rank microbes by their affinity for a metabolite environment
    niche       - Find unique metabolite preferences for a microbe vs others

Examples:
    # Which metabolite should I add to promote Pseudomonas over Burkholderia?
    python wom_complex.py selective --target "genus:Pseudomonas" \
        --competitors "genus:Burkholderia,genus:Variovorax" --top 10

    # In an environment with arginine and lysine, which microbes thrive?
    python wom_complex.py fitness --metabolites "arginine,lysine" --top 10

    # In an environment with these microbes, who wins if I add glutamate?
    python wom_complex.py fitness --microbes "genus:Pseudomonas,genus:Burkholderia" \
        --metabolites "glutamate" --top 10
"""

import argparse
import json
import logging
import sys
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Lazy-loaded singleton
_wrapper = None


def _get_wrapper():
    global _wrapper
    if _wrapper is None:
        from wom import _get_wrapper as wom_get_wrapper
        _wrapper = wom_get_wrapper('bpr')
    return _wrapper


def _output(data: dict, format: str = 'json') -> None:
    """Output data in requested format."""
    if format == 'json':
        print(json.dumps(data, indent=2, default=str))
    elif format == 'table':
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            print(df.to_string(index=False))
        elif 'error' in data and data['error']:
            print(f"Error: {data['error']}", file=sys.stderr)
        else:
            print(json.dumps(data, indent=2, default=str))


def _error_response(message: str, query: dict = None) -> dict:
    """Build error response."""
    return {
        'success': False,
        'error': message,
        'query': query,
        'results': []
    }


def _parse_microbe_queries(microbes_str: str) -> list[dict]:
    """
    Parse microbe query string into list of query dicts.

    Format: "genus:Burkholderia,isolate:XYZ,family:Rhizobiaceae"
    """
    queries = []
    for part in microbes_str.split(','):
        part = part.strip()
        if not part:
            continue

        if ':' in part:
            level, name = part.split(':', 1)
            level = level.strip().lower()
            name = name.strip()

            if level not in ['isolate', 'genus', 'family', 'order', 'class']:
                raise ValueError(f"Invalid taxonomic level: {level}")

            queries.append({level: name})
        else:
            # Assume genus if no level specified
            queries.append({'genus': part})

    return queries


def _parse_compounds(compounds_str: str) -> list[str]:
    """Parse comma-separated compound string."""
    return [c.strip() for c in compounds_str.split(',') if c.strip()]


def _get_microbe_compound_scores(queries: list[dict], wrapper) -> pd.DataFrame:
    """
    Get compound preference scores for each microbe query.

    Returns DataFrame with compounds as index, query names as columns.
    """
    from predict import CompoundPredictions

    cp = CompoundPredictions(wrapper)
    cp.rank(queries)
    cp.extrapolate(agg='mean')

    # _extrapolated has compounds as index, query names as columns
    return cp._extrapolated.copy(), cp._query_to_isolates


def _get_compound_microbe_scores(compounds: list[str], level: str, wrapper) -> pd.DataFrame:
    """
    Get microbe preference scores for each compound.

    Returns DataFrame with microbes as index, compounds + aggregated score as columns.
    """
    from predict import MicrobePredictions

    mp = MicrobePredictions(wrapper)
    mp.rank(compounds)
    mp.extrapolate(to=level, agg='mean')

    return mp._extrapolated.copy(), mp.compound_mapping


# =============================================================================
# Command: selective
# =============================================================================

def cmd_selective(args) -> dict:
    """
    Find metabolites that favor target microbe over competitors.

    This answers: "Which metabolite should I add to promote microbe X
    in an environment with competitors Y and Z?"

    Algorithm:
    1. Get compound preferences for target
    2. Get compound preferences for each competitor
    3. Calculate selectivity = target_score - max(competitor_scores)
    4. Rank by selectivity (most selective first)
    5. Optionally exclude metabolites already in environment
    """
    wrapper = _get_wrapper()

    try:
        target_queries = _parse_microbe_queries(args.target)
        competitor_queries = _parse_microbe_queries(args.competitors)
    except ValueError as e:
        return _error_response(str(e))

    if len(target_queries) != 1:
        return _error_response("Exactly one target microbe required")

    if len(competitor_queries) < 1:
        return _error_response("At least one competitor required")

    exclude_compounds = set()
    if args.exclude:
        exclude_compounds = set(_parse_compounds(args.exclude))

    query = {
        'target': target_queries[0],
        'competitors': competitor_queries,
        'exclude': list(exclude_compounds) if exclude_compounds else None,
        'top_k': args.top
    }

    try:
        # Get scores for all microbes (target + competitors)
        all_queries = target_queries + competitor_queries
        scores_df, query_to_isolates = _get_microbe_compound_scores(all_queries, wrapper)

        target_name = list(target_queries[0].values())[0]
        competitor_names = [list(q.values())[0] for q in competitor_queries]

        # Check that all taxa were found
        missing = []
        if target_name not in scores_df.columns:
            missing.append(f"target '{target_name}'")
        for cn in competitor_names:
            if cn not in scores_df.columns:
                missing.append(f"competitor '{cn}'")

        if missing:
            return _error_response(f"Taxa not found in model: {', '.join(missing)}", query)

        # Calculate selectivity
        results_df = pd.DataFrame({
            'compound': scores_df.index,
            'target_score': scores_df[target_name].values,
            'max_competitor_score': scores_df[competitor_names].max(axis=1).values,
            'mean_competitor_score': scores_df[competitor_names].mean(axis=1).values,
        })

        results_df['selectivity'] = results_df['target_score'] - results_df['max_competitor_score']
        results_df['advantage_ratio'] = results_df['target_score'] / (results_df['max_competitor_score'] + 1e-6)

        # Add individual competitor scores for transparency
        for cn in competitor_names:
            results_df[f'{cn}_score'] = scores_df[cn].values

        # Filter excluded compounds
        if exclude_compounds:
            results_df = results_df[~results_df['compound'].str.lower().isin(
                [c.lower() for c in exclude_compounds]
            )]

        # Sort by selectivity (highest first)
        results_df = results_df.sort_values('selectivity', ascending=False)

        # Top results
        top_results = results_df.head(args.top).copy()

        # Round for readability
        score_cols = ['target_score', 'max_competitor_score', 'mean_competitor_score',
                      'selectivity', 'advantage_ratio'] + [f'{cn}_score' for cn in competitor_names]
        for col in score_cols:
            if col in top_results.columns:
                top_results[col] = top_results[col].round(4)

        response = {
            'success': True,
            'error': None,
            'query_type': 'selective',
            'query': query,
            'interpretation': f"Metabolites that {target_name} prefers more than competitors",
            'target_isolates': query_to_isolates.get(target_name, []),
            'competitor_isolates': {cn: query_to_isolates.get(cn, []) for cn in competitor_names},
            'n_compounds_analyzed': len(scores_df),
            'n_excluded': len(exclude_compounds),
            'n_results': len(top_results),
            'results': top_results.to_dict(orient='records'),
            'recommendation': top_results.iloc[0]['compound'] if len(top_results) > 0 else None
        }

        return response

    except Exception as e:
        import traceback
        return _error_response(f"{str(e)}\n{traceback.format_exc()}", query)


# =============================================================================
# Command: fitness
# =============================================================================

def cmd_fitness(args) -> dict:
    """
    Rank microbes by their affinity for a metabolite environment.

    This answers: "In an environment with metabolites X, Y, Z,
    which microbes will thrive?"

    Algorithm:
    1. For each microbe (or all genera if not specified), get scores for each metabolite
    2. Aggregate scores across metabolites (sum or mean)
    3. Rank by fitness score
    """
    wrapper = _get_wrapper()

    metabolites = _parse_compounds(args.metabolites)
    if not metabolites:
        return _error_response("At least one metabolite required")

    # Parse specific microbes if provided
    specific_microbes = None
    if args.microbes:
        try:
            specific_microbes = _parse_microbe_queries(args.microbes)
        except ValueError as e:
            return _error_response(str(e))

    query = {
        'metabolites': metabolites,
        'microbes': specific_microbes,
        'aggregation': args.agg,
        'level': args.level,
        'top_k': args.top
    }

    try:
        from predict import MicrobePredictions

        mp = MicrobePredictions(wrapper)
        mp.rank(metabolites)

        # Check compound mapping for any issues
        compound_mapping = mp.compound_mapping.to_dict(orient='records')

        # Extrapolate to desired taxonomic level
        mp.extrapolate(to=args.level, agg='mean')

        # Get the extrapolated scores (taxa x compounds)
        scores_df = mp._extrapolated.copy()

        # The taxon names are in 'taxon' column, not index
        if 'taxon' not in scores_df.columns:
            return _error_response("Missing 'taxon' column in extrapolated data", query)

        # Set taxon as index for easier filtering
        scores_df = scores_df.set_index('taxon')

        # Get only the compound score columns (from metabolites)
        matched_compounds = [m['representative_compound'] for m in compound_mapping if m['representative_compound']]
        score_cols = [c for c in matched_compounds if c in scores_df.columns]

        if not score_cols:
            return _error_response("No matching compound columns found", query)

        # If specific microbes requested, filter to those
        if specific_microbes:
            microbe_names = [list(q.values())[0] for q in specific_microbes]
            available = [m for m in microbe_names if m in scores_df.index]
            missing = [m for m in microbe_names if m not in scores_df.index]

            if missing:
                return _error_response(f"Microbes not found: {', '.join(missing)}", query)

            scores_df = scores_df.loc[available]

        # Only use compound score columns for aggregation
        scores_numeric = scores_df[score_cols]

        # Aggregate across metabolites
        if args.agg == 'sum':
            fitness_scores = scores_numeric.sum(axis=1)
        elif args.agg == 'max':
            fitness_scores = scores_numeric.max(axis=1)
        else:  # mean
            fitness_scores = scores_numeric.mean(axis=1)

        # Build results
        results_df = pd.DataFrame({
            'microbe': fitness_scores.index,
            'fitness_score': fitness_scores.values,
            'level': args.level
        })

        # Add individual metabolite scores
        for met in score_cols:
            results_df[f'{met}_score'] = scores_numeric[met].values

        # Sort by fitness (highest first)
        results_df = results_df.sort_values('fitness_score', ascending=False)

        # Top results
        top_results = results_df.head(args.top).copy()

        # Round scores
        round_cols = ['fitness_score'] + [f'{met}_score' for met in score_cols]
        for col in round_cols:
            if col in top_results.columns:
                top_results[col] = top_results[col].round(4)

        response = {
            'success': True,
            'error': None,
            'query_type': 'fitness',
            'query': query,
            'interpretation': f"Microbes ranked by predicted growth in environment with: {', '.join(metabolites)}",
            'compound_mapping': compound_mapping,
            'n_microbes_analyzed': len(scores_df),
            'n_results': len(top_results),
            'results': top_results.to_dict(orient='records'),
            'top_microbe': top_results.iloc[0]['microbe'] if len(top_results) > 0 else None
        }

        return response

    except Exception as e:
        import traceback
        return _error_response(f"{str(e)}\n{traceback.format_exc()}", query)


# =============================================================================
# Command: niche
# =============================================================================

def cmd_niche(args) -> dict:
    """
    Find the unique metabolic niche of a microbe.

    This answers: "What metabolites does microbe X prefer that
    most other microbes don't?"

    Algorithm:
    1. Get compound preferences for target microbe
    2. Get average compound preferences across all genera
    3. Calculate uniqueness = target_score - mean_all_score
    4. Return compounds with highest uniqueness (target's unique niche)
    """
    wrapper = _get_wrapper()

    try:
        target_queries = _parse_microbe_queries(args.target)
    except ValueError as e:
        return _error_response(str(e))

    if len(target_queries) != 1:
        return _error_response("Exactly one target microbe required")

    # Optional: specific comparison group
    compare_queries = None
    if args.versus:
        try:
            compare_queries = _parse_microbe_queries(args.versus)
        except ValueError as e:
            return _error_response(str(e))

    query = {
        'target': target_queries[0],
        'versus': compare_queries,
        'top_k': args.top
    }

    try:
        from predict import CompoundPredictions

        target_name = list(target_queries[0].values())[0]

        # Get target scores
        cp = CompoundPredictions(wrapper)
        cp.rank(target_queries)
        cp.extrapolate(agg='mean')
        target_scores = cp._extrapolated[target_name].copy()

        # Get comparison scores (either specific taxa or all genera)
        if compare_queries:
            cp2 = CompoundPredictions(wrapper)
            cp2.rank(compare_queries)
            cp2.extrapolate(agg='mean')
            compare_names = [list(q.values())[0] for q in compare_queries]
            compare_scores = cp2._extrapolated[compare_names].mean(axis=1)
            comparison_desc = f"vs {', '.join(compare_names)}"
        else:
            # Compare against all genera
            all_genera = list(set(
                tax.get('genus') for tax in wrapper.microbe_taxonomy.values()
                if tax.get('genus')
            ))
            all_queries = [{'genus': g} for g in all_genera if g != target_name]

            cp2 = CompoundPredictions(wrapper)
            cp2.rank(all_queries)
            cp2.extrapolate(agg='mean')
            compare_scores = cp2._extrapolated.mean(axis=1)
            comparison_desc = "vs all other genera"

        # Calculate uniqueness
        results_df = pd.DataFrame({
            'compound': target_scores.index,
            'target_score': target_scores.values,
            'comparison_mean': compare_scores.values,
        })

        results_df['uniqueness'] = results_df['target_score'] - results_df['comparison_mean']
        results_df['ratio'] = results_df['target_score'] / (results_df['comparison_mean'] + 1e-6)

        # Sort by uniqueness (most unique first)
        results_df = results_df.sort_values('uniqueness', ascending=False)

        # Top unique preferences
        top_unique = results_df.head(args.top).copy()

        # Also get what target DOESN'T prefer (negative uniqueness)
        bottom_unique = results_df.tail(args.top).copy()
        bottom_unique = bottom_unique.sort_values('uniqueness', ascending=True)

        # Round
        for col in ['target_score', 'comparison_mean', 'uniqueness', 'ratio']:
            top_unique[col] = top_unique[col].round(4)
            bottom_unique[col] = bottom_unique[col].round(4)

        response = {
            'success': True,
            'error': None,
            'query_type': 'niche',
            'query': query,
            'interpretation': f"Unique metabolic preferences of {target_name} {comparison_desc}",
            'target_isolates': cp._query_to_isolates.get(target_name, []),
            'n_compounds_analyzed': len(results_df),
            'unique_preferences': top_unique.to_dict(orient='records'),
            'avoided_compounds': bottom_unique.to_dict(orient='records'),
            'top_unique_compound': top_unique.iloc[0]['compound'] if len(top_unique) > 0 else None
        }

        return response

    except Exception as e:
        import traceback
        return _error_response(f"{str(e)}\n{traceback.format_exc()}", query)


# =============================================================================
# Command: competition
# =============================================================================

def cmd_competition(args) -> dict:
    """
    Analyze competition between microbes for metabolites.

    This answers: "If microbes X, Y, Z are competing for metabolites A, B, C,
    who has the advantage?"

    Algorithm:
    1. Get preference scores for each microbe-metabolite pair
    2. For each metabolite, determine which microbe has highest preference (winner)
    3. Summarize: which microbe wins most metabolites?
    """
    wrapper = _get_wrapper()

    try:
        microbe_queries = _parse_microbe_queries(args.microbes)
    except ValueError as e:
        return _error_response(str(e))

    if len(microbe_queries) < 2:
        return _error_response("At least 2 microbes required for competition analysis")

    metabolites = _parse_compounds(args.metabolites)
    if not metabolites:
        return _error_response("At least one metabolite required")

    query = {
        'microbes': microbe_queries,
        'metabolites': metabolites
    }

    try:
        # Get scores for all microbes
        scores_df, query_to_isolates = _get_microbe_compound_scores(microbe_queries, wrapper)
        microbe_names = [list(q.values())[0] for q in microbe_queries]

        # Use proper compound mapping (prioritizes exact matches)
        from pref import map_compound_name

        available_compounds = list(scores_df.index)
        matched_metabolites = []

        for met in metabolites:
            # Use the same mapping as wom.py (exact matches first)
            mapping = map_compound_name(met)
            if mapping['representative_compound'] and mapping['representative_compound'] in available_compounds:
                matched_metabolites.append({
                    'requested': met,
                    'matched': mapping['representative_compound'],
                    'score': mapping['score'] * 100,  # Convert to percentage
                    'match_type': mapping['match_type']
                })

        if not matched_metabolites:
            return _error_response(f"No metabolites matched. Available: {available_compounds[:10]}", query)

        # Analyze competition for each metabolite
        competition_results = []
        wins = {m: 0 for m in microbe_names}

        for met_info in matched_metabolites:
            met = met_info['matched']
            met_scores = {m: float(scores_df.loc[met, m]) for m in microbe_names if m in scores_df.columns}

            winner = max(met_scores, key=met_scores.get)
            wins[winner] += 1

            competition_results.append({
                'metabolite': met,
                'winner': winner,
                'winner_score': round(met_scores[winner], 4),
                'scores': {m: round(s, 4) for m, s in met_scores.items()},
                'margin': round(met_scores[winner] - sorted(met_scores.values())[-2], 4) if len(met_scores) > 1 else 0
            })

        # Overall winner
        overall_winner = max(wins, key=wins.get)

        response = {
            'success': True,
            'error': None,
            'query_type': 'competition',
            'query': query,
            'interpretation': f"Competition analysis for {len(metabolites)} metabolites",
            'metabolite_matching': matched_metabolites,
            'per_metabolite_results': competition_results,
            'win_counts': wins,
            'overall_winner': overall_winner,
            'conclusion': f"{overall_winner} wins {wins[overall_winner]}/{len(competition_results)} metabolites"
        }

        return response

    except Exception as e:
        import traceback
        return _error_response(f"{str(e)}\n{traceback.format_exc()}", query)


# =============================================================================
# Command: rank-substrates
# =============================================================================

def cmd_rank_substrates(args) -> dict:
    """
    Rank specific substrates for specific microbes.

    This answers: "For Bacillus and Pseudomonas, rank these substrates:
    glucose, alanine, glutamate by preference"

    Algorithm:
    1. Get preference scores for each microbe-substrate pair
    2. For each microbe, rank the substrates by preference
    3. Return top N for each microbe
    """
    wrapper = _get_wrapper()

    try:
        microbe_queries = _parse_microbe_queries(args.microbes)
    except ValueError as e:
        return _error_response(str(e))

    if not microbe_queries:
        return _error_response("At least one microbe required")

    substrates = _parse_compounds(args.substrates)
    if not substrates:
        return _error_response("At least one substrate required")

    query = {
        'microbes': microbe_queries,
        'substrates': substrates,
        'top_k': args.top
    }

    try:
        # Get scores for all microbes
        scores_df, query_to_isolates = _get_microbe_compound_scores(microbe_queries, wrapper)
        microbe_names = [list(q.values())[0] for q in microbe_queries]

        # Map substrates using proper compound mapping
        from pref import map_compound_name

        available_compounds = list(scores_df.index)
        substrate_mapping = []

        for sub in substrates:
            mapping = map_compound_name(sub)
            if mapping['representative_compound'] and mapping['representative_compound'] in available_compounds:
                substrate_mapping.append({
                    'requested': sub,
                    'matched': mapping['representative_compound'],
                    'score': mapping['score'] * 100,
                    'match_type': mapping['match_type']
                })
            else:
                substrate_mapping.append({
                    'requested': sub,
                    'matched': None,
                    'score': 0,
                    'match_type': 'not_found'
                })

        matched_substrates = [s['matched'] for s in substrate_mapping if s['matched']]

        if not matched_substrates:
            return _error_response(f"No substrates matched. Available: {available_compounds[:10]}", query)

        # For each microbe, rank the substrates
        per_microbe_results = {}

        for microbe in microbe_names:
            if microbe not in scores_df.columns:
                per_microbe_results[microbe] = {'error': f'{microbe} not found in model'}
                continue

            # Get scores for this microbe on matched substrates
            microbe_scores = []
            for sub_info in substrate_mapping:
                if sub_info['matched']:
                    score = float(scores_df.loc[sub_info['matched'], microbe])
                    microbe_scores.append({
                        'substrate': sub_info['requested'],
                        'matched_name': sub_info['matched'],
                        'score': round(score, 4)
                    })

            # Sort by score descending
            microbe_scores.sort(key=lambda x: x['score'], reverse=True)

            # Take top N
            per_microbe_results[microbe] = {
                'n_isolates': len(query_to_isolates.get(microbe, [])),
                'rankings': microbe_scores[:args.top]
            }

        response = {
            'success': True,
            'error': None,
            'query_type': 'rank_substrates',
            'query': query,
            'interpretation': f"Substrate preferences for {len(microbe_names)} microbes",
            'substrate_mapping': substrate_mapping,
            'n_substrates_matched': len(matched_substrates),
            'n_substrates_requested': len(substrates),
            'per_microbe_results': per_microbe_results
        }

        return response

    except Exception as e:
        import traceback
        return _error_response(f"{str(e)}\n{traceback.format_exc()}", query)


# =============================================================================
# Command: info
# =============================================================================

def cmd_info(args) -> dict:
    """
    Get information about available complex commands.
    """
    response = {
        'success': True,
        'error': None,
        'query_type': 'info',
        'description': 'Complex query commands for BPR consumption model',
        'available_commands': [
            {
                'command': 'selective',
                'description': 'Find metabolites favoring target microbe over competitors',
                'use_case': 'Which metabolite should I add to promote X over Y?',
                'example': 'python wom_complex.py selective --target "genus:Pseudomonas" --competitors "genus:Burkholderia" --top 10'
            },
            {
                'command': 'fitness',
                'description': 'Rank microbes by affinity for metabolite environment',
                'use_case': 'Which microbes will thrive in environment with X, Y, Z?',
                'example': 'python wom_complex.py fitness --metabolites "arginine,lysine" --top 10'
            },
            {
                'command': 'niche',
                'description': 'Find unique metabolic preferences for a microbe',
                'use_case': 'What does X eat that others dont?',
                'example': 'python wom_complex.py niche --target "genus:Pseudomonas" --top 10'
            },
            {
                'command': 'competition',
                'description': 'Analyze which microbe wins given metabolites',
                'use_case': 'If X and Y compete for these metabolites, who wins?',
                'example': 'python wom_complex.py competition --microbes "genus:Pseudomonas,genus:Burkholderia" --metabolites "glutamate,arginine"'
            },
            {
                'command': 'rank-substrates',
                'description': 'Rank specific substrates for specific microbes',
                'use_case': 'For Bacillus and Pseudomonas, rank glucose, glutamate, alanine by preference',
                'example': 'python wom_complex.py rank-substrates --microbes "genus:Bacillus,genus:Pseudomonas" --substrates "glucose,glutamate,alanine" --top 5'
            },
            {
                'command': 'info',
                'description': 'Show this help information',
                'example': 'python wom_complex.py info'
            }
        ]
    }
    return response


# =============================================================================
# Main argument parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Complex queries for BPR consumption model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -------------------------------------------------------------------------
    # selective command
    # -------------------------------------------------------------------------
    p_selective = subparsers.add_parser(
        'selective',
        help='Find metabolites favoring target over competitors',
        description='Find metabolites that the target microbe prefers more than competitors.'
    )
    p_selective.add_argument(
        '--target', '-t',
        required=True,
        help='Target microbe to promote (e.g., "genus:Pseudomonas")'
    )
    p_selective.add_argument(
        '--competitors', '-c',
        required=True,
        help='Competitor microbes, comma-separated (e.g., "genus:Burkholderia,genus:Variovorax")'
    )
    p_selective.add_argument(
        '--exclude', '-e',
        default=None,
        help='Metabolites to exclude (already in environment), comma-separated'
    )
    p_selective.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    p_selective.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format'
    )
    p_selective.set_defaults(func=cmd_selective)

    # -------------------------------------------------------------------------
    # fitness command
    # -------------------------------------------------------------------------
    p_fitness = subparsers.add_parser(
        'fitness',
        help='Rank microbes by affinity for metabolite environment',
        description='Predict which microbes thrive in a given metabolite environment.'
    )
    p_fitness.add_argument(
        '--metabolites', '-m',
        required=True,
        help='Metabolites in environment, comma-separated (e.g., "arginine,lysine,glutamate")'
    )
    p_fitness.add_argument(
        '--microbes',
        default=None,
        help='Specific microbes to evaluate (optional, defaults to all genera)'
    )
    p_fitness.add_argument(
        '--level', '-l',
        choices=['isolate', 'genus', 'family', 'order', 'class'],
        default='genus',
        help='Taxonomic level (default: genus)'
    )
    p_fitness.add_argument(
        '--agg', '-a',
        choices=['mean', 'sum', 'max'],
        default='sum',
        help='Aggregation across metabolites (default: sum)'
    )
    p_fitness.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    p_fitness.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format'
    )
    p_fitness.set_defaults(func=cmd_fitness)

    # -------------------------------------------------------------------------
    # niche command
    # -------------------------------------------------------------------------
    p_niche = subparsers.add_parser(
        'niche',
        help='Find unique metabolic niche of a microbe',
        description='Find metabolites that a microbe uniquely prefers.'
    )
    p_niche.add_argument(
        '--target', '-t',
        required=True,
        help='Target microbe (e.g., "genus:Pseudomonas")'
    )
    p_niche.add_argument(
        '--versus', '-v',
        default=None,
        help='Compare against specific microbes (optional, defaults to all genera)'
    )
    p_niche.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    p_niche.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format'
    )
    p_niche.set_defaults(func=cmd_niche)

    # -------------------------------------------------------------------------
    # competition command
    # -------------------------------------------------------------------------
    p_competition = subparsers.add_parser(
        'competition',
        help='Analyze competition between microbes for metabolites',
        description='Determine which microbe has advantage for given metabolites.'
    )
    p_competition.add_argument(
        '--microbes', '-m',
        required=True,
        help='Competing microbes, comma-separated (e.g., "genus:Pseudomonas,genus:Burkholderia")'
    )
    p_competition.add_argument(
        '--metabolites', '-c',
        required=True,
        help='Metabolites to compete for, comma-separated'
    )
    p_competition.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format'
    )
    p_competition.set_defaults(func=cmd_competition)

    # -------------------------------------------------------------------------
    # rank-substrates command
    # -------------------------------------------------------------------------
    p_rank = subparsers.add_parser(
        'rank-substrates',
        help='Rank specific substrates for specific microbes',
        description='For each microbe, rank the given substrates by preference.'
    )
    p_rank.add_argument(
        '--microbes', '-m',
        required=True,
        help='Microbes to analyze, comma-separated (e.g., "genus:Bacillus,genus:Pseudomonas")'
    )
    p_rank.add_argument(
        '--substrates', '-s',
        required=True,
        help='Substrates to rank, comma-separated (e.g., "glucose,glutamate,alanine")'
    )
    p_rank.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of top substrates per microbe (default: 10)'
    )
    p_rank.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format'
    )
    p_rank.set_defaults(func=cmd_rank_substrates)

    # -------------------------------------------------------------------------
    # info command
    # -------------------------------------------------------------------------
    p_info = subparsers.add_parser(
        'info',
        help='Show available commands and examples',
        description='Display information about complex query commands.'
    )
    p_info.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format'
    )
    p_info.set_defaults(func=cmd_info)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    result = args.func(args)

    # Output
    _output(result, args.format)

    # Exit code
    if not result.get('success', True):
        sys.exit(1)


if __name__ == '__main__':
    main()
