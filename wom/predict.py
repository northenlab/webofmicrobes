"""Prediction interfaces for the consumption model."""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from model import ModelWrapper
from pref import map_compound_name

logger = logging.getLogger(__name__)


class MicrobePredictions:
    """
    Predict which microbes consume given compounds.
    
    Usage:
        mp = MicrobePredictions(wrapper)
        mp.rank(['glutamate', 'serine', 'guanine'])
        mp.extrapolate(to='genus', agg='mean')
        result = mp.summarize(top_k=10)
    """
    
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper
        self._raw_scores: pd.DataFrame | None = None
        self._extrapolated: pd.DataFrame | None = None
        self._query_compounds: list[str] = []
        self._extrapolate_level: str | None = None
        self._compound_mapping: dict[str, dict] = {}
    
    def rank(self, compounds: list[str]) -> None:
        """
        Get raw scores for all isolates on the given compounds.
        
        Args:
            compounds: List of compound names (will be mapped to representative_compound)
        
        Stores:
            DataFrame of shape (n_isolates, n_compounds)
            Index: isolate IDs
            Columns: representative_compound names
            Values: raw dot-product scores
        """
        valid_compounds = []
        self._compound_mapping = {}
        
        for name in compounds:
            if name in self.wrapper.metabolite_to_idx:
                valid_compounds.append(name)
                self._compound_mapping[name] = {
                    'representative_compound': name,
                    'score': 1.0,
                    'match_type': 'exact'
                }
                continue
            
            mapped = map_compound_name(name)
            self._compound_mapping[name] = mapped
            
            if mapped['representative_compound'] is None:
                logger.warning(f"Could not map compound: {name}")
            elif mapped['representative_compound'] not in self.wrapper.metabolite_to_idx:
                logger.warning(
                    f"Compound '{name}' mapped to '{mapped['representative_compound']}' "
                    f"but that is not in the model"
                )
            else:
                if mapped['match_type'] == 'fuzzy':
                    logger.info(
                        f"Fuzzy matched '{name}' -> '{mapped['representative_compound']}' "
                        f"(score: {mapped['score']:.2f})"
                    )
                valid_compounds.append(mapped['representative_compound'])
        
        seen = set()
        unique_compounds = []
        for c in valid_compounds:
            if c not in seen:
                seen.add(c)
                unique_compounds.append(c)
        
        if not unique_compounds:
            raise ValueError("No valid compounds provided")
        
        self._query_compounds = unique_compounds
        
        compound_indices = [self.wrapper.metabolite_to_idx[c] for c in unique_compounds]
        
        isolate_ids = list(self.wrapper.microbe_to_idx.keys())
        n_isolates = len(isolate_ids)
        n_compounds = len(unique_compounds)
        
        score_matrix = np.zeros((n_isolates, n_compounds))
        for i, isolate_id in enumerate(isolate_ids):
            m_idx = self.wrapper.microbe_to_idx[isolate_id]
            all_scores = self.wrapper.model.predict_scores(m_idx)
            score_matrix[i, :] = all_scores[compound_indices]
        
        self._raw_scores = pd.DataFrame(
            score_matrix,
            index=isolate_ids,
            columns=unique_compounds
        )
        self._raw_scores.index.name = 'isolate'
        self._extrapolated = None
        
        logger.info(f"Ranked {n_isolates} isolates on {n_compounds} compounds")
    
    @property
    def compound_mapping(self) -> pd.DataFrame:
        """Return the compound name mapping as a DataFrame."""
        if not self._compound_mapping:
            return pd.DataFrame(columns=['input', 'representative_compound', 'score', 'match_type'])
        
        rows = [
            {
                'input': k,
                'representative_compound': v['representative_compound'],
                'score': v['score'],
                'match_type': v['match_type']
            }
            for k, v in self._compound_mapping.items()
        ]
        return pd.DataFrame(rows)
    
    def extrapolate(
        self,
        to: Literal['isolate', 'genus', 'family', 'order', 'class'] = 'genus',
        agg: Literal['mean', 'sum', 'max'] = 'mean'
    ) -> None:
        """
        Aggregate isolate scores to a taxonomic level.
        
        Args:
            to: Taxonomic level to aggregate to
            agg: Aggregation method ('mean', 'sum', 'max')
        """
        if self._raw_scores is None:
            raise RuntimeError("Must call rank() first")
        
        df = self._raw_scores.copy()
        compound_cols = self._query_compounds
        
        if to == 'isolate':
            result = df.copy()
            result['taxon'] = result.index
            result = result.reset_index(drop=True)
        else:
            df['_group'] = df.index.map(
                lambda x: self.wrapper.microbe_taxonomy.get(x, {}).get(to)
            )
            
            df = df[df['_group'].notna()]
            
            if df.empty:
                raise ValueError(f"No isolates have taxonomy at level: {to}")
            
            agg_func = {'mean': 'mean', 'sum': 'sum', 'max': 'max'}[agg]
            result = df.groupby('_group')[compound_cols].agg(agg_func)
            result = result.reset_index().rename(columns={'_group': 'taxon'})
        
        result['score_sum'] = result[compound_cols].sum(axis=1)
        result['score_mean'] = result[compound_cols].mean(axis=1)
        result['rank'] = result['score_mean'].rank(ascending=False, method='min').astype(int)
        
        result = result.sort_values('rank')
        
        self._extrapolated = result
        self._extrapolate_level = to
        
        logger.info(f"Extrapolated to {len(result)} {to}-level groups")
    
    def summarize(self, top_k: int = 10) -> pd.DataFrame:
        """
        Return top taxa with metadata.
        
        Args:
            top_k: Number of top results to return
        """
        if self._extrapolated is None:
            raise RuntimeError("Must call extrapolate() first")
        
        result = self._extrapolated.head(top_k).copy()
        
        if self._extrapolate_level == 'isolate':
            result['n_isolates'] = 1
        else:
            counts = {}
            for isolate_id in self.wrapper.microbe_to_idx.keys():
                tax = self.wrapper.microbe_taxonomy.get(isolate_id, {})
                group = tax.get(self._extrapolate_level)
                if group:
                    counts[group] = counts.get(group, 0) + 1
            result['n_isolates'] = result['taxon'].map(counts)
        
        parent_levels = {
            'genus': ['family', 'order', 'class'],
            'family': ['order', 'class'],
            'order': ['class'],
            'class': [],
            'isolate': ['genus', 'family', 'order', 'class']
        }
        
        levels_to_add = parent_levels.get(self._extrapolate_level, [])
        
        for level in levels_to_add:
            parent_map = {}
            for isolate_id, tax in self.wrapper.microbe_taxonomy.items():
                if self._extrapolate_level == 'isolate':
                    taxon_value = isolate_id
                else:
                    taxon_value = tax.get(self._extrapolate_level)
                parent_value = tax.get(level)
                if taxon_value and parent_value:
                    parent_map[taxon_value] = parent_value
            result[level] = result['taxon'].map(parent_map)
        
        meta_cols = ['taxon', 'n_isolates'] + levels_to_add
        meta_cols = [c for c in meta_cols if c in result.columns]
        compound_cols = self._query_compounds
        summary_cols = ['score_sum', 'score_mean', 'rank']
        
        return result[meta_cols + compound_cols + summary_cols]


class CompoundPredictions:
    """
    Predict which compounds are consumed by given microbes.
    
    Usage:
        cp = CompoundPredictions(wrapper)
        cp.rank([
            {'isolate': 'BurkholderiaOAS925'},
            {'genus': 'Arthrobacter'},
        ])
        cp.extrapolate(agg='mean')
        result = cp.summarize(top_k=20)
    """
    
    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper
        self._raw_scores: pd.DataFrame | None = None
        self._extrapolated: pd.DataFrame | None = None
        self._queries: list[dict] = []
        self._query_to_isolates: dict[str, list[str]] = {}
    
    def _query_to_key(self, query: dict) -> str:
        """Convert query dict to a string key for column naming."""
        if 'isolate' in query:
            return query['isolate']
        for level in ['genus', 'family', 'order', 'class']:
            if level in query:
                return query[level]
        raise ValueError(f"Invalid query: {query}")
    
    def _resolve_query(self, query: dict) -> list[str]:
        """Resolve a query to list of isolate IDs."""
        if 'isolate' in query:
            isolate = query['isolate']
            if isolate in self.wrapper.microbe_to_idx:
                return [isolate]
            else:
                return []
        
        for level in ['genus', 'family', 'order', 'class']:
            if level in query:
                target = query[level]
                matches = [
                    isolate_id 
                    for isolate_id, tax in self.wrapper.microbe_taxonomy.items()
                    if tax.get(level) == target
                ]
                return matches
        
        raise ValueError(f"Invalid query: {query}")
    
    def rank(self, queries: list[dict]) -> None:
        """
        Get raw scores for specified microbes on all compounds.
        
        Args:
            queries: List of query dicts, each with one of:
                - {'isolate': 'BurkholderiaOAS925'}
                - {'genus': 'Arthrobacter'}
                - {'family': 'Rhizobiaceae'}
                - {'order': 'Rhizobiales'}
                - {'class': 'Alphaproteobacteria'}
        """
        self._queries = queries
        self._query_to_isolates = {}
        
        all_isolates = []
        for query in queries:
            key = self._query_to_key(query)
            isolates = self._resolve_query(query)
            self._query_to_isolates[key] = isolates
            
            if not isolates:
                logger.warning(f"No isolates found for query: {query}")
            
            all_isolates.extend(isolates)
        
        seen = set()
        unique_isolates = []
        for iso in all_isolates:
            if iso not in seen:
                seen.add(iso)
                unique_isolates.append(iso)
        
        if not unique_isolates:
            raise ValueError("No valid isolates found for any query")
        
        n_compounds = len(self.wrapper.metabolite_to_idx)
        score_matrix = np.zeros((len(unique_isolates), n_compounds))
        
        for i, isolate_id in enumerate(unique_isolates):
            m_idx = self.wrapper.microbe_to_idx[isolate_id]
            score_matrix[i, :] = self.wrapper.model.predict_scores(m_idx)
        
        compound_names = [
            self.wrapper.idx_to_metabolite[i] 
            for i in range(n_compounds)
        ]
        
        self._raw_scores = pd.DataFrame(
            score_matrix.T,
            index=compound_names,
            columns=unique_isolates
        )
        self._raw_scores.index.name = 'compound'
        self._extrapolated = None
        
        logger.info(f"Ranked {n_compounds} compounds for {len(unique_isolates)} isolates from {len(queries)} queries")
    
    def extrapolate(self, agg: Literal['mean', 'sum', 'max'] = 'mean') -> None:
        """
        Aggregate isolate scores back to query level and add summary columns.
        
        Args:
            agg: Aggregation method for isolates within a query ('mean', 'sum', 'max')
        """
        if self._raw_scores is None:
            raise RuntimeError("Must call rank() first")
        
        result = pd.DataFrame(index=self._raw_scores.index)
        
        for query in self._queries:
            key = self._query_to_key(query)
            isolates = self._query_to_isolates[key]
            
            if not isolates:
                result[key] = np.nan
            elif len(isolates) == 1:
                result[key] = self._raw_scores[isolates[0]]
            else:
                isolate_data = self._raw_scores[isolates]
                if agg == 'mean':
                    result[key] = isolate_data.mean(axis=1)
                elif agg == 'sum':
                    result[key] = isolate_data.sum(axis=1)
                elif agg == 'max':
                    result[key] = isolate_data.max(axis=1)
        
        query_cols = [self._query_to_key(q) for q in self._queries]
        result['mean_score'] = result[query_cols].mean(axis=1)
        result['rank'] = result['mean_score'].rank(ascending=False, method='min').astype(int)
        
        result = result.sort_values('rank')
        
        self._extrapolated = result
        
        logger.info(f"Extrapolated to {len(self._queries)} query groups")
    
    def summarize(self, top_k: int = 20) -> pd.DataFrame:
        """
        Return top compounds.
        
        Args:
            top_k: Number of top results to return
        """
        if self._extrapolated is None:
            raise RuntimeError("Must call extrapolate() first")
        
        result = self._extrapolated.head(top_k).copy()
        result = result.reset_index()
        
        query_cols = [self._query_to_key(q) for q in self._queries]
        summary_cols = ['mean_score', 'rank']
        
        return result[['compound'] + query_cols + summary_cols]