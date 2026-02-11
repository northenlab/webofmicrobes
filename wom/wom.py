#!/usr/bin/env python
"""
Command-line interface for consumption model predictions.

Designed for use by LLM agents. Outputs JSON by default.

Examples:
    # What microbes eat these compounds?
    python wom.py microbes --compounds "glutamate,serine,guanine" --level genus --top 10
    
    # What compounds does this microbe eat?
    python wom.py compounds --microbes "genus:Burkholderia,isolate:RhodococcusOAS809" --top 20
    
    # Search for a compound name
    python wom.py search-compound "glutamic acid"
    
    # List available genera
    python wom.py list-taxa --level genus
    
    # Get model info
    python wom.py info
    
    # Specify a different model type
    python wom.py --model wrmf microbes --compounds "glutamate" --level genus
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# Lazy-loaded singletons per model type
_wrappers: dict[str, "ModelWrapper"] = {}

# Map CLI model names to (model_class, file_prefix)
_MODEL_REGISTRY = {
    'bpr': ('BPRModel', 'bpr'),
    'wrmf': ('WRMFModel', 'wr_mf'),
    'svd': ('SVDModel', 'svd'),
}


def _get_wrapper(model_name: str = 'bpr', preference_type: str = 'consumed'):
    """Load a model wrapper (lazy singleton)."""
    cache_key = f"{model_name}_{preference_type}"
    
    if cache_key not in _wrappers:
        from model import load_model, BPRModel, WRMFModel, SVDModel
        from config import CACHE_DIR
        
        class_map = {
            'bpr': BPRModel,
            'wrmf': WRMFModel,
            'svd': SVDModel,
        }
        
        if model_name not in class_map:
            raise ValueError(f"Unknown model type: {model_name}. Choose from: {list(class_map.keys())}")
        
        model_class = class_map[model_name]
        file_prefix = _MODEL_REGISTRY[model_name][1]
        model_path = Path(CACHE_DIR) / f"{file_prefix}_{preference_type}"
        
        _wrappers[cache_key] = load_model(model_path, model_class=model_class)
    
    return _wrappers[cache_key]


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
            queries.append({'isolate': part})
    
    return queries


def _parse_compounds(compounds_str: str) -> list[str]:
    """Parse comma-separated compound string."""
    return [c.strip() for c in compounds_str.split(',') if c.strip()]


# =============================================================================
# Command: microbes
# =============================================================================

def cmd_microbes(args) -> dict:
    """Predict which microbes consume the given compounds."""
    from predict import MicrobePredictions
    
    wrapper = _get_wrapper(args.model)
    compounds = _parse_compounds(args.compounds)
    
    query = {
        'compounds': compounds,
        'level': args.level,
        'aggregation': args.agg,
        'top_k': args.top,
        'model': args.model,
    }
    
    try:
        mp = MicrobePredictions(wrapper)
        mp.rank(compounds)
        mp.extrapolate(to=args.level, agg=args.agg)
        result_df = mp.summarize(top_k=args.top)
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'microbes',
            'query': query,
            'compound_mapping': mp.compound_mapping.to_dict(orient='records'),
            'n_results': len(result_df),
            'results': result_df.to_dict(orient='records')
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: compounds
# =============================================================================

def cmd_compounds(args) -> dict:
    """Predict which compounds are consumed by the given microbes."""
    from predict import CompoundPredictions
    
    wrapper = _get_wrapper(args.model)
    
    try:
        queries = _parse_microbe_queries(args.microbes)
    except ValueError as e:
        return _error_response(str(e))
    
    query = {
        'microbes': queries,
        'aggregation': args.agg,
        'top_k': args.top,
        'model': args.model,
    }
    
    try:
        cp = CompoundPredictions(wrapper)
        cp.rank(queries)
        cp.extrapolate(agg=args.agg)
        result_df = cp.summarize(top_k=args.top)
        
        query_resolution = []
        for q in queries:
            key = list(q.values())[0]
            level = list(q.keys())[0]
            isolates = cp._query_to_isolates.get(key, [])
            query_resolution.append({
                'query': q,
                'n_isolates_found': len(isolates),
                'isolates': isolates[:10],
                'isolates_truncated': len(isolates) > 10
            })
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'compounds',
            'query': query,
            'query_resolution': query_resolution,
            'n_results': len(result_df),
            'results': result_df.to_dict(orient='records')
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: search-compound
# =============================================================================

def cmd_search_compound(args) -> dict:
    """Fuzzy search for compound names in the model."""
    from pref import map_compound_name
    from rapidfuzz import process, fuzz
    
    wrapper = _get_wrapper(args.model)
    query_str = args.query
    
    query = {
        'search': query_str,
        'top_k': args.top
    }
    
    try:
        all_compounds = list(wrapper.metabolite_to_idx.keys())
        
        matches = process.extract(
            query_str.lower(),
            [c.lower() for c in all_compounds],
            scorer=fuzz.WRatio,
            limit=args.top
        )
        
        lower_to_orig = {c.lower(): c for c in all_compounds}
        
        results = []
        for match_lower, score, _ in matches:
            orig_name = lower_to_orig[match_lower]
            results.append({
                'compound': orig_name,
                'match_score': round(score, 1)
            })
        
        mapped = map_compound_name(query_str)
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'search_compound',
            'query': query,
            'best_mapping': {
                'representative_compound': mapped['representative_compound'],
                'score': mapped['score'],
                'match_type': mapped['match_type'],
                'in_model': mapped['representative_compound'] in wrapper.metabolite_to_idx if mapped['representative_compound'] else False
            },
            'fuzzy_matches': results
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: search-taxon
# =============================================================================

def cmd_search_taxon(args) -> dict:
    """Fuzzy search for taxon names in the model."""
    from rapidfuzz import process, fuzz
    
    wrapper = _get_wrapper(args.model)
    query_str = args.query
    level = args.level
    
    query = {
        'search': query_str,
        'level': level,
        'top_k': args.top
    }
    
    try:
        levels_to_search = [level] if level else ['genus', 'family', 'order', 'class']
        
        all_taxa = []
        for lv in levels_to_search:
            seen = set()
            for tax in wrapper.microbe_taxonomy.values():
                name = tax.get(lv)
                if name and name not in seen:
                    seen.add(name)
                    all_taxa.append({'name': name, 'level': lv})
        
        if not all_taxa:
            return _error_response("No taxa found at specified level", query)
        
        names = [t['name'] for t in all_taxa]
        matches = process.extract(
            query_str.lower(),
            [n.lower() for n in names],
            scorer=fuzz.WRatio,
            limit=args.top
        )
        
        lower_to_idx = {n.lower(): i for i, n in enumerate(names)}
        
        results = []
        for match_lower, score, _ in matches:
            idx = lower_to_idx[match_lower]
            taxon = all_taxa[idx]
            
            n_isolates = sum(
                1 for tax in wrapper.microbe_taxonomy.values()
                if tax.get(taxon['level']) == taxon['name']
            )
            
            results.append({
                'taxon': taxon['name'],
                'level': taxon['level'],
                'match_score': round(score, 1),
                'n_isolates': n_isolates
            })
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'search_taxon',
            'query': query,
            'results': results
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: list-compounds
# =============================================================================

def cmd_list_compounds(args) -> dict:
    """List all compounds in the model."""
    wrapper = _get_wrapper(args.model)
    
    query = {
        'filter': args.filter,
        'top_k': args.top
    }
    
    try:
        compounds = sorted(wrapper.metabolite_to_idx.keys())
        
        if args.filter:
            filter_lower = args.filter.lower()
            compounds = [c for c in compounds if filter_lower in c.lower()]
        
        total = len(compounds)
        compounds = compounds[:args.top]
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'list_compounds',
            'query': query,
            'total_in_model': len(wrapper.metabolite_to_idx),
            'total_matching': total,
            'n_returned': len(compounds),
            'compounds': compounds
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: list-microbes
# =============================================================================

def cmd_list_microbes(args) -> dict:
    """List microbes/isolates in the model."""
    wrapper = _get_wrapper(args.model)
    
    query = {
        'genus': args.genus,
        'family': args.family,
        'order': args.order,
        'class_': args.class_,
        'top_k': args.top
    }
    
    try:
        results = []
        
        for isolate_id, tax in wrapper.microbe_taxonomy.items():
            if args.genus and tax.get('genus') != args.genus:
                continue
            if args.family and tax.get('family') != args.family:
                continue
            if args.order and tax.get('order') != args.order:
                continue
            if args.class_ and tax.get('class') != args.class_:
                continue
            
            results.append({
                'isolate': isolate_id,
                'genus': tax.get('genus'),
                'family': tax.get('family'),
                'order': tax.get('order'),
                'class': tax.get('class')
            })
        
        total = len(results)
        results = sorted(results, key=lambda x: (x['genus'] or '', x['isolate']))
        results = results[:args.top]
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'list_microbes',
            'query': query,
            'total_in_model': len(wrapper.microbe_taxonomy),
            'total_matching': total,
            'n_returned': len(results),
            'microbes': results
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: list-taxa
# =============================================================================
# =============================================================================
# Command: list-taxa
# =============================================================================

def cmd_list_taxa(args) -> dict:
    """List unique taxa at a given level."""
    wrapper = _get_wrapper(args.model)
    level = args.level
    
    query = {
        'level': level,
        'top_k': args.top
    }
    
    try:
        counts = {}
        parent_info = {}
        
        for isolate_id, tax in wrapper.microbe_taxonomy.items():
            name = tax.get(level)
            if name:
                counts[name] = counts.get(name, 0) + 1
                if name not in parent_info:
                    parent_info[name] = {
                        k: v for k, v in tax.items() 
                        if k != level and v is not None
                    }
        
        results = [
            {
                'taxon': name,
                'level': level,
                'n_isolates': count,
                **parent_info.get(name, {})
            }
            for name, count in sorted(counts.items(), key=lambda x: -x[1])
        ]
        
        total = len(results)
        results = results[:args.top]
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'list_taxa',
            'query': query,
            'total_unique': total,
            'n_returned': len(results),
            'taxa': results
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Command: info
# =============================================================================

def cmd_info(args) -> dict:
    """Get model information and statistics."""
    try:
        wrapper = _get_wrapper(args.model)
        
        taxa_counts = {}
        for level in ['genus', 'family', 'order', 'class']:
            unique = set()
            for tax in wrapper.microbe_taxonomy.values():
                name = tax.get(level)
                if name:
                    unique.add(name)
            taxa_counts[level] = len(unique)
        
        model_type_name = type(wrapper.model).__name__
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'info',
            'model': {
                'type': model_type_name,
                'model_key': args.model,
                'n_isolates': len(wrapper.microbe_to_idx),
                'n_compounds': len(wrapper.metabolite_to_idx),
                'embedding_dim': wrapper.model.embedding_dim,
                'taxa_counts': taxa_counts,
                'has_taxonomy_embeddings': wrapper.taxonomy_embeddings is not None
            },
            'available_models': list(_MODEL_REGISTRY.keys()),
            'available_commands': [
                {
                    'command': 'microbes',
                    'description': 'Predict which microbes consume given compounds',
                    'example': 'python wom.py microbes --compounds "glutamate,serine" --level genus'
                },
                {
                    'command': 'compounds', 
                    'description': 'Predict which compounds a microbe consumes',
                    'example': 'python wom.py compounds --microbes "genus:Burkholderia"'
                },
                {
                    'command': 'search-compound',
                    'description': 'Fuzzy search for compound names',
                    'example': 'python wom.py search-compound "glutamic acid"'
                },
                {
                    'command': 'search-taxon',
                    'description': 'Fuzzy search for taxon names',
                    'example': 'python wom.py search-taxon "burkhold"'
                },
                {
                    'command': 'list-compounds',
                    'description': 'List all compounds in model',
                    'example': 'python wom.py list-compounds --filter "amino"'
                },
                {
                    'command': 'list-microbes',
                    'description': 'List isolates, optionally filtered by taxonomy',
                    'example': 'python wom.py list-microbes --genus Burkholderia'
                },
                {
                    'command': 'list-taxa',
                    'description': 'List unique taxa at a given level',
                    'example': 'python wom.py list-taxa --level genus'
                },
                {
                    'command': 'info',
                    'description': 'Show this model information',
                    'example': 'python wom.py info'
                }
            ]
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e))


# =============================================================================
# Command: raw-scores
# =============================================================================

def cmd_raw_scores(args) -> dict:
    """Get raw scores without aggregation."""
    from predict import MicrobePredictions, CompoundPredictions
    
    wrapper = _get_wrapper(args.model)
    
    if args.mode == 'microbes':
        compounds = _parse_compounds(args.compounds)
        
        query = {
            'mode': 'microbes',
            'compounds': compounds,
            'model': args.model,
        }
        
        try:
            mp = MicrobePredictions(wrapper)
            mp.rank(compounds)
            
            raw_df = mp._raw_scores.copy()
            raw_df = raw_df.reset_index()
            
            response = {
                'success': True,
                'error': None,
                'query_type': 'raw_scores',
                'query': query,
                'compound_mapping': mp.compound_mapping.to_dict(orient='records'),
                'shape': list(mp._raw_scores.shape),
                'columns': list(mp._raw_scores.columns),
                'n_isolates': len(mp._raw_scores),
                'scores': raw_df.to_dict(orient='records')
            }
            
            return response
            
        except Exception as e:
            return _error_response(str(e), query)
    
    else:  # compounds mode
        try:
            queries = _parse_microbe_queries(args.microbes)
        except ValueError as e:
            return _error_response(str(e))
        
        query = {
            'mode': 'compounds',
            'microbes': queries,
            'model': args.model,
        }
        
        try:
            cp = CompoundPredictions(wrapper)
            cp.rank(queries)
            
            raw_df = cp._raw_scores.copy()
            raw_df = raw_df.reset_index()
            
            response = {
                'success': True,
                'error': None,
                'query_type': 'raw_scores',
                'query': query,
                'query_resolution': {
                    key: isolates 
                    for key, isolates in cp._query_to_isolates.items()
                },
                'shape': list(cp._raw_scores.shape),
                'columns': list(cp._raw_scores.columns),
                'n_compounds': len(cp._raw_scores),
                'scores': raw_df.to_dict(orient='records')
            }
            
            return response
            
        except Exception as e:
            return _error_response(str(e), query)


# =============================================================================
# Command: compare
# =============================================================================

def cmd_compare(args) -> dict:
    """Compare preferences between two or more microbes/taxa."""
    from predict import CompoundPredictions
    
    wrapper = _get_wrapper(args.model)
    
    try:
        queries = _parse_microbe_queries(args.microbes)
    except ValueError as e:
        return _error_response(str(e))
    
    if len(queries) < 2:
        return _error_response("Need at least 2 microbes to compare")
    
    query = {
        'microbes': queries,
        'top_k': args.top,
        'model': args.model,
    }
    
    try:
        cp = CompoundPredictions(wrapper)
        cp.rank(queries)
        cp.extrapolate(agg='mean')
        
        df = cp._extrapolated.copy()
        query_cols = [list(q.values())[0] for q in queries]
        
        df['variance'] = df[query_cols].var(axis=1)
        df['range'] = df[query_cols].max(axis=1) - df[query_cols].min(axis=1)
        
        df_sorted = df.sort_values('variance', ascending=False)
        top_diff = df_sorted.head(args.top).copy()
        top_diff = top_diff.reset_index()
        
        df_sorted_agree = df.sort_values(['variance', 'mean_score'], ascending=[True, False])
        top_agree = df_sorted_agree.head(args.top).copy()
        top_agree = top_agree.reset_index()
        
        response = {
            'success': True,
            'error': None,
            'query_type': 'compare',
            'query': query,
            'query_resolution': {
                list(q.values())[0]: cp._query_to_isolates.get(list(q.values())[0], [])
                for q in queries
            },
            'most_different': top_diff[['compound'] + query_cols + ['variance', 'range']].to_dict(orient='records'),
            'most_similar': top_agree[['compound'] + query_cols + ['variance', 'mean_score']].to_dict(orient='records')
        }
        
        return response
        
    except Exception as e:
        return _error_response(str(e), query)


# =============================================================================
# Main argument parser
# =============================================================================

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'table'],
        default='json',
        help='Output format (default: json)'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Consumption model predictions for microbe-metabolite interactions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model', '-M',
        choices=list(_MODEL_REGISTRY.keys()),
        default='bpr',
        help='Model type to use (default: bpr)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # -------------------------------------------------------------------------
    # microbes command
    # -------------------------------------------------------------------------
    p_microbes = subparsers.add_parser(
        'microbes',
        help='Predict which microbes consume given compounds',
        description='Given one or more compounds, rank microbes by their predicted consumption preference.'
    )
    p_microbes.add_argument(
        '--compounds', '-c',
        required=True,
        help='Comma-separated compound names (e.g., "glutamate,serine,guanine")'
    )
    p_microbes.add_argument(
        '--level', '-l',
        choices=['isolate', 'genus', 'family', 'order', 'class'],
        default='genus',
        help='Taxonomic level to aggregate to (default: genus)'
    )
    p_microbes.add_argument(
        '--agg', '-a',
        choices=['mean', 'sum', 'max'],
        default='mean',
        help='Aggregation method (default: mean)'
    )
    p_microbes.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of top results (default: 10)'
    )
    _add_common_args(p_microbes)
    p_microbes.set_defaults(func=cmd_microbes)
    
    # -------------------------------------------------------------------------
    # compounds command
    # -------------------------------------------------------------------------
    p_compounds = subparsers.add_parser(
        'compounds',
        help='Predict which compounds a microbe consumes',
        description='Given one or more microbes, rank compounds by predicted consumption preference.'
    )
    p_compounds.add_argument(
        '--microbes', '-m',
        required=True,
        help='Microbe queries as "level:name" pairs, comma-separated '
             '(e.g., "genus:Burkholderia,isolate:RhodococcusOAS809")'
    )
    p_compounds.add_argument(
        '--agg', '-a',
        choices=['mean', 'sum', 'max'],
        default='mean',
        help='Aggregation method across microbes (default: mean)'
    )
    p_compounds.add_argument(
        '--top', '-n',
        type=int,
        default=20,
        help='Number of top results (default: 20)'
    )
    _add_common_args(p_compounds)
    p_compounds.set_defaults(func=cmd_compounds)
    
    # -------------------------------------------------------------------------
    # search-compound command
    # -------------------------------------------------------------------------
    p_search_compound = subparsers.add_parser(
        'search-compound',
        help='Fuzzy search for compound names',
        description='Find compound names matching a search query.'
    )
    p_search_compound.add_argument(
        'query',
        help='Search string (e.g., "glutamic acid")'
    )
    p_search_compound.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    _add_common_args(p_search_compound)
    p_search_compound.set_defaults(func=cmd_search_compound)
    
    # -------------------------------------------------------------------------
    # search-taxon command
    # -------------------------------------------------------------------------
    p_search_taxon = subparsers.add_parser(
        'search-taxon',
        help='Fuzzy search for taxon names',
        description='Find taxon names matching a search query.'
    )
    p_search_taxon.add_argument(
        'query',
        help='Search string (e.g., "burkhold")'
    )
    p_search_taxon.add_argument(
        '--level', '-l',
        choices=['genus', 'family', 'order', 'class'],
        default=None,
        help='Limit search to specific level (default: all levels)'
    )
    p_search_taxon.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    _add_common_args(p_search_taxon)
    p_search_taxon.set_defaults(func=cmd_search_taxon)
    
    # -------------------------------------------------------------------------
    # list-compounds command
    # -------------------------------------------------------------------------
    p_list_compounds = subparsers.add_parser(
        'list-compounds',
        help='List all compounds in model',
        description='List compounds, optionally filtered by substring.'
    )
    p_list_compounds.add_argument(
        '--filter',
        default=None,
        help='Filter by substring (case-insensitive)'
    )
    p_list_compounds.add_argument(
        '--top', '-n',
        type=int,
        default=100,
        help='Max number to return (default: 100)'
    )
    _add_common_args(p_list_compounds)
    p_list_compounds.set_defaults(func=cmd_list_compounds)
    
    # -------------------------------------------------------------------------
    # list-microbes command
    # -------------------------------------------------------------------------
    p_list_microbes = subparsers.add_parser(
        'list-microbes',
        help='List isolates in model',
        description='List isolates, optionally filtered by taxonomy.'
    )
    p_list_microbes.add_argument(
        '--genus', '-g',
        default=None,
        help='Filter by genus'
    )
    p_list_microbes.add_argument(
        '--family',
        default=None,
        help='Filter by family'
    )
    p_list_microbes.add_argument(
        '--order',
        default=None,
        help='Filter by order'
    )
    p_list_microbes.add_argument(
        '--class', dest='class_',
        default=None,
        help='Filter by class'
    )
    p_list_microbes.add_argument(
        '--top', '-n',
        type=int,
        default=100,
        help='Max number to return (default: 100)'
    )
    _add_common_args(p_list_microbes)
    p_list_microbes.set_defaults(func=cmd_list_microbes)
    
    # -------------------------------------------------------------------------
    # list-taxa command
    # -------------------------------------------------------------------------
    p_list_taxa = subparsers.add_parser(
        'list-taxa',
        help='List unique taxa at a level',
        description='List unique taxa with isolate counts.'
    )
    p_list_taxa.add_argument(
        '--level', '-l',
        choices=['genus', 'family', 'order', 'class'],
        default='genus',
        help='Taxonomic level (default: genus)'
    )
    p_list_taxa.add_argument(
        '--top', '-n',
        type=int,
        default=50,
        help='Max number to return (default: 50)'
    )
    _add_common_args(p_list_taxa)
    p_list_taxa.set_defaults(func=cmd_list_taxa)
    
    # -------------------------------------------------------------------------
    # info command
    # -------------------------------------------------------------------------
    p_info = subparsers.add_parser(
        'info',
        help='Show model information',
        description='Display model statistics and available commands.'
    )
    _add_common_args(p_info)
    p_info.set_defaults(func=cmd_info)
    
    # -------------------------------------------------------------------------
    # raw-scores command
    # -------------------------------------------------------------------------
    p_raw = subparsers.add_parser(
        'raw-scores',
        help='Get raw scores without aggregation',
        description='Returns the full score matrix for detailed analysis.'
    )
    p_raw.add_argument(
        '--mode',
        choices=['microbes', 'compounds'],
        required=True,
        help='Score mode: microbes (score isolates on compounds) or compounds (score compounds for microbes)'
    )
    p_raw.add_argument(
        '--compounds', '-c',
        default=None,
        help='Comma-separated compounds (required for mode=microbes)'
    )
    p_raw.add_argument(
        '--microbes', '-m',
        default=None,
        help='Microbe queries (required for mode=compounds)'
    )
    _add_common_args(p_raw)
    p_raw.set_defaults(func=cmd_raw_scores)
    
    # -------------------------------------------------------------------------
    # compare command
    # -------------------------------------------------------------------------
    p_compare = subparsers.add_parser(
        'compare',
        help='Compare preferences between microbes',
        description='Find compounds where microbes differ or agree most.'
    )
    p_compare.add_argument(
        '--microbes', '-m',
        required=True,
        help='Two or more microbe queries, comma-separated '
             '(e.g., "genus:Burkholderia,genus:Arthrobacter")'
    )
    p_compare.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Number of results per category (default: 10)'
    )
    _add_common_args(p_compare)
    p_compare.set_defaults(func=cmd_compare)
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    result = args.func(args)
    
    _output(result, args.format)
    
    if not result.get('success', True):
        sys.exit(1)


if __name__ == '__main__':
    main()