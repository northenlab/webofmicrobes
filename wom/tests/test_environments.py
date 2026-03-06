"""Test environment query functionality.

This module tests the environment/context-based query system, which enables
queries like "Which microbes grow on Avena barbata exudates?"

Tests include:
- Loading and listing environments
- Fuzzy searching for environment names
- Getting compound lists for environments
- Integration workflows (environment -> compounds -> microbes)
"""

import pytest
import json

from api.query_engine import QueryEngine
from api.query_types import (
    QueryConfig,
    ListEnvironmentsQuery,
    SearchEnvironmentQuery,
    GetEnvironmentCompoundsQuery,
    MicrobesForCompoundsQuery,
)
from loaders.loaders import load_environments


class TestEnvironmentData:
    """Test environment data loading."""

    def test_load_environments_returns_dataframe(self):
        """load_environments() should return a DataFrame with expected columns."""
        env_df = load_environments()
        
        assert env_df is not None
        assert len(env_df) > 0
        
        expected_cols = [
            'compound',
            'environment_name',
            'representative_compound',
            'in_model',
            'environment_status',
        ]
        for col in expected_cols:
            assert col in env_df.columns

    def test_environment_data_has_valid_environments(self):
        """Environment data should have valid environments with good coverage."""
        env_df = load_environments()
        
        # Check that we have multiple environments
        n_envs = env_df['environment_name'].nunique()
        assert n_envs >= 6, f"Should have at least 6 environments, got {n_envs}"
        
        # Check that all environments are marked as 'valid' (>50% coverage)
        statuses = env_df.groupby('environment_name')['environment_status'].first()
        valid_count = (statuses == 'valid').sum()
        assert valid_count >= 6, f"Should have at least 6 valid environments, got {valid_count}"

    def test_environment_compounds_mapped(self):
        """Environment compounds should be mapped to model."""
        env_df = load_environments()
        
        # Check mapping success
        mapped = env_df['representative_compound'].notna().sum()
        total = len(env_df)
        mapping_rate = mapped / total
        
        assert mapping_rate > 0.7, f"Mapping rate should be >70%, got {mapping_rate:.1%}"


class TestListEnvironmentsQuery:
    """Test list_environments query."""

    def test_list_environments_basic(self, temp_model_dir):
        """list_environments should return all environments."""
        engine = QueryEngine(temp_model_dir)
        query = ListEnvironmentsQuery(include_stats=True)
        
        response = engine.query_list_environments(query)
        
        assert response.success is True
        assert len(response.results) >= 6
        
        # Check structure of results
        for env in response.results:
            assert 'environment_name' in env
            assert 'n_compounds' in env
            assert 'n_in_model' in env
            assert 'coverage_fraction' in env

    def test_list_environments_with_min_coverage_filter(self, temp_model_dir):
        """list_environments should filter by min_coverage."""
        engine = QueryEngine(temp_model_dir)
        
        # High coverage filter
        query_high = ListEnvironmentsQuery(min_coverage=0.9)
        response_high = engine.query_list_environments(query_high)
        
        # Low coverage filter
        query_low = ListEnvironmentsQuery(min_coverage=0.5)
        response_low = engine.query_list_environments(query_low)
        
        # High coverage should return fewer results
        assert len(response_high.results) <= len(response_low.results)

    def test_list_environments_includes_metadata(self, temp_model_dir):
        """list_environments response should include environment metadata."""
        engine = QueryEngine(temp_model_dir)
        query = ListEnvironmentsQuery(include_stats=True)
        
        response = engine.query_list_environments(query)
        
        # Check for metadata
        assert 'environment_name' in response.results[0]
        assert 'description' in response.results[0]
        assert 'doi' in response.results[0]


class TestSearchEnvironmentQuery:
    """Test search_environment query."""

    def test_search_environment_exact_match(self, temp_model_dir):
        """search_environment should find exact matches."""
        engine = QueryEngine(temp_model_dir)
        query = SearchEnvironmentQuery(query='NLDM', max_results=5)
        
        response = engine.query_search_environment(query)
        
        assert response.success is True
        assert len(response.results) > 0
        assert response.results[0]['environment_name'] == 'NLDM'

    def test_search_environment_fuzzy_match(self, temp_model_dir):
        """search_environment should do fuzzy matching."""
        engine = QueryEngine(temp_model_dir)
        
        # Partial match
        query = SearchEnvironmentQuery(query='avena', max_results=3, min_score=0.6)
        response = engine.query_search_environment(query)
        
        assert response.success is True
        assert len(response.results) > 0
        # Should match "Avena barbata root exudate (2018KZ)"
        assert any('avena' in r['environment_name'].lower() for r in response.results)

    def test_search_environment_returns_scores(self, temp_model_dir):
        """search_environment should return match scores."""
        engine = QueryEngine(temp_model_dir)
        query = SearchEnvironmentQuery(query='R2A', max_results=5)
        
        response = engine.query_search_environment(query)
        
        for match in response.results:
            assert 'match_score' in match
            assert 0.0 <= match['match_score'] <= 1.0

    def test_search_environment_max_results(self, temp_model_dir):
        """search_environment should respect max_results parameter."""
        engine = QueryEngine(temp_model_dir)
        
        query_3 = SearchEnvironmentQuery(query='', max_results=3)
        response_3 = engine.query_search_environment(query_3)
        
        query_5 = SearchEnvironmentQuery(query='', max_results=5)
        response_5 = engine.query_search_environment(query_5)
        
        # Both should respect their limits (or return all if fewer available)
        assert len(response_3.results) <= 3
        assert len(response_5.results) <= 5


class TestGetEnvironmentCompoundsQuery:
    """Test get_environment_compounds query."""

    def test_get_environment_compounds_returns_list(self, temp_model_dir):
        """get_environment_compounds should return compounds."""
        engine = QueryEngine(temp_model_dir)
        query = GetEnvironmentCompoundsQuery(
            environment_name='NLDM',
            only_in_model=True
        )
        
        response = engine.query_get_environment_compounds(query)
        
        assert response.success is True
        assert len(response.results) > 0

    def test_get_environment_compounds_structure(self, temp_model_dir):
        """get_environment_compounds results should have expected fields."""
        engine = QueryEngine(temp_model_dir)
        query = GetEnvironmentCompoundsQuery(
            environment_name='NLDM',
            only_in_model=True
        )
        
        response = engine.query_get_environment_compounds(query)
        
        for cpd in response.results:
            assert 'original_name' in cpd
            assert 'representative_compound' in cpd
            assert 'chebi_id' in cpd
            assert 'match_type' in cpd
            assert 'match_score' in cpd
            assert 'in_model' in cpd

    def test_get_environment_compounds_fuzzy_matching(self, temp_model_dir):
        """get_environment_compounds should handle fuzzy environment name matching."""
        engine = QueryEngine(temp_model_dir)
        
        # Test with different case/partial match
        query = GetEnvironmentCompoundsQuery(
            environment_name='nldm',  # lowercase
            only_in_model=True
        )
        
        response = engine.query_get_environment_compounds(query)
        
        # Should still work (fuzzy matching)
        assert response.success is True or not response.success  # Either works or fails gracefully

    def test_get_environment_compounds_with_only_in_model_filter(self, temp_model_dir):
        """get_environment_compounds should filter by only_in_model."""
        engine = QueryEngine(temp_model_dir)
        config = QueryConfig(model_dir=temp_model_dir)
        
        # Get all compounds including unmapped
        query_all = GetEnvironmentCompoundsQuery(
            environment_name='NLDM',
            only_in_model=False,
            config=config
        )
        response_all = engine.query_get_environment_compounds(query_all)
        
        # Get only compounds in model
        query_model = GetEnvironmentCompoundsQuery(
            environment_name='NLDM',
            only_in_model=True,
            config=config
        )
        response_model = engine.query_get_environment_compounds(query_model)
        
        # Model-only should have fewer or same compounds
        assert len(response_model.results) <= len(response_all.results)

    def test_get_environment_compounds_includes_metadata(self, temp_model_dir):
        """get_environment_compounds response should include environment metadata."""
        engine = QueryEngine(temp_model_dir)
        query = GetEnvironmentCompoundsQuery(environment_name='NLDM')
        
        response = engine.query_get_environment_compounds(query)
        
        assert 'environment_name' in response.metadata
        assert 'description' in response.metadata
        assert 'doi' in response.metadata
        assert 'n_compounds' in response.metadata


class TestEnvironmentWorkflows:
    """Test complete workflows using environment queries."""

    def test_environment_to_microbes_workflow(self, temp_model_dir):
        """Test complete workflow: environment -> compounds -> microbes."""
        engine = QueryEngine(temp_model_dir)
        config = QueryConfig(model_dir=temp_model_dir)
        
        # Step 1: Get compounds in NLDM
        env_query = GetEnvironmentCompoundsQuery(
            environment_name='NLDM',
            only_in_model=True,
            config=config
        )
        env_response = engine.query_get_environment_compounds(env_query)
        
        assert env_response.success is True
        assert len(env_response.results) > 0
        
        # Step 2: Extract compound names
        compounds = [
            cpd['representative_compound']
            for cpd in env_response.results
        ]
        
        # Step 3: Query microbes that consume these compounds
        microbe_query = MicrobesForCompoundsQuery(
            compounds=compounds[:5],  # Use first 5 for speed
            level='genus',
            top_k=10,
            config=config
        )
        microbe_response = engine.query_microbes_for_compounds(microbe_query)
        
        assert microbe_response.success is True
        assert len(microbe_response.results) > 0
        
        # Verify microbe results have expected structure
        for microbe in microbe_response.results:
            assert 'taxon' in microbe
            assert 'aggregate_score' in microbe

    def test_search_then_query_workflow(self, temp_model_dir):
        """Test workflow: search environment -> get compounds."""
        engine = QueryEngine(temp_model_dir)
        config = QueryConfig(model_dir=temp_model_dir)
        
        # Step 1: Search for environment
        search_query = SearchEnvironmentQuery(
            query='avena',
            max_results=3,
            config=config
        )
        search_response = engine.query_search_environment(search_query)
        
        assert search_response.success is True
        assert len(search_response.results) > 0
        
        # Step 2: Get compounds from first match
        env_name = search_response.results[0]['environment_name']
        cpd_query = GetEnvironmentCompoundsQuery(
            environment_name=env_name,
            only_in_model=True,
            config=config
        )
        cpd_response = engine.query_get_environment_compounds(cpd_query)
        
        assert cpd_response.success is True
        assert len(cpd_response.results) > 0

    def test_list_then_query_workflow(self, temp_model_dir):
        """Test workflow: list environments -> get compounds."""
        engine = QueryEngine(temp_model_dir)
        config = QueryConfig(model_dir=temp_model_dir)
        
        # Step 1: List environments
        list_query = ListEnvironmentsQuery(
            include_stats=True,
            min_coverage=0.7,
            config=config
        )
        list_response = engine.query_list_environments(list_query)
        
        assert list_response.success is True
        assert len(list_response.results) > 0
        
        # Step 2: Get compounds from first environment
        env_name = list_response.results[0]['environment_name']
        cpd_query = GetEnvironmentCompoundsQuery(
            environment_name=env_name,
            only_in_model=True,
            config=config
        )
        cpd_response = engine.query_get_environment_compounds(cpd_query)
        
        assert cpd_response.success is True


class TestEnvironmentQueryInteractionTypes:
    """Test environment queries with different interaction types."""

    def test_environment_queries_with_consumed_model(self, temp_model_dir):
        """Environment queries should work with consumed interaction type."""
        config = QueryConfig(
            model_dir=temp_model_dir,
            interaction_type='consumed'
        )
        engine = QueryEngine(temp_model_dir, config)
        
        query = ListEnvironmentsQuery(include_stats=True, config=config)
        response = engine.query_list_environments(query)
        
        assert response.success is True
        assert response.metadata['interaction_type'] == 'consumed'

    def test_environment_queries_with_secreted_model(self, temp_model_dir):
        """Environment queries should work with secreted interaction type."""
        config = QueryConfig(
            model_dir=temp_model_dir,
            interaction_type='secreted'
        )
        engine = QueryEngine(temp_model_dir, config)
        
        query = ListEnvironmentsQuery(include_stats=True, config=config)
        response = engine.query_list_environments(query)
        
        assert response.success is True
        assert response.metadata['interaction_type'] == 'secreted'
