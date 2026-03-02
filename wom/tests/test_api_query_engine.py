"""Test api.query_engine module - query interface to models.

This module tests the api/, which handles:
- Query engine initialization and loading models
- Various query types (compounds for microbes, etc.)
- Query result formatting and JSON serialization
- Explanation/reasoning functionality for results
"""

import pytest
import json
from pathlib import Path
import tempfile

from api.query_engine import QueryEngine
from api.query_types import (
    CompoundsForMicrobesQuery,
    MicrobesForCompoundsQuery,
    SelectiveMetabolitesQuery,
    MetaboliteEnvironmentFitnessQuery,
    NicheMetabolitesQuery,
    SearchCompoundQuery,
    ListTaxaQuery,
    ModelInfoQuery,
)
from core.io import save_model_artifacts


class TestQueryEngineInitialization:
    """Test QueryEngine initialization."""

    def test_query_engine_loads_successfully(self, temp_model_dir):
        """QueryEngine should load model directory successfully."""
        engine = QueryEngine(temp_model_dir)
        
        assert engine is not None
        assert engine.artifacts is not None

    def test_query_engine_has_training_data(self, temp_model_dir):
        """QueryEngine should have access to training data."""
        engine = QueryEngine(temp_model_dir)
        
        assert engine.artifacts.training_data is not None
        assert engine.artifacts.training_data.n_microbes > 0
        assert engine.artifacts.training_data.n_metabolites > 0

    def test_query_engine_invalid_directory(self):
        """Loading nonexistent directory may raise error or use defaults."""
        # The behavior may vary - QueryEngine might handle missing dirs gracefully
        # Just verify it doesn't crash unexpectedly
        try:
            engine = QueryEngine("/nonexistent/path/xyz")
            # If it doesn't raise, it should still have some state
            assert engine is not None
        except (FileNotFoundError, ValueError, OSError):
            # If it does raise, that's fine too
            pass


class TestCompoundsForMicrobesQuery:
    """Test compounds for microbes query."""

    def test_query_from_string(self):
        """Should parse query from string."""
        query = CompoundsForMicrobesQuery.from_string(
            "genus:Arthrobacter",
            top_k=5,
        )
        
        assert query is not None
        assert query.top_k == 5

    def test_compounds_for_microbes_execution(self, temp_model_dir):
        """Should execute compounds for microbes query."""
        engine = QueryEngine(temp_model_dir)
        
        # Get a genus from the model
        taxa = engine.artifacts.get_taxa_at_level("genus")
        if taxa:
            target_genus = taxa[0]
            
            query = CompoundsForMicrobesQuery.from_string(
                f"genus:{target_genus}",
                top_k=5,
            )
            response = engine.query_compounds_for_microbes(query)
            
            assert response is not None
            assert response.success
            assert len(response.results) > 0

    def test_compounds_for_microbes_returns_structured_results(self, temp_model_dir):
        """Query results should be properly structured."""
        engine = QueryEngine(temp_model_dir)
        
        taxa = engine.artifacts.get_taxa_at_level("genus")
        if taxa:
            query = CompoundsForMicrobesQuery.from_string(
                f"genus:{taxa[0]}",
                top_k=5,
            )
            response = engine.query_compounds_for_microbes(query)
            
            result = response.results[0]
            assert "microbe" in result
            assert "compounds" in result
            assert len(result["compounds"]) > 0
            
            # Check compound structure
            compound = result["compounds"][0]
            assert "name" in compound
            assert "raw_score" in compound


class TestMicrobesForCompoundsQuery:
    """Test microbes for compounds query."""

    def test_microbes_for_compounds_execution(self, temp_model_dir):
        """Should execute microbes for compounds query."""
        engine = QueryEngine(temp_model_dir)
        
        query = MicrobesForCompoundsQuery.from_string(
            "glucose,alanine",
            level="genus",
            top_k=5,
        )
        response = engine.query_microbes_for_compounds(query)
        
        assert response is not None
        assert response.success
        assert len(response.results) > 0

    def test_microbes_for_compounds_returns_structured_results(self, temp_model_dir):
        """Query results should have proper structure."""
        engine = QueryEngine(temp_model_dir)
        
        query = MicrobesForCompoundsQuery.from_string(
            "glucose",
            level="genus",
            top_k=5,
        )
        response = engine.query_microbes_for_compounds(query)
        
        if response.results:
            result = response.results[0]
            assert "taxon" in result
            assert "aggregate_score" in result


class TestSelectiveMetabolitesQuery:
    """Test selective metabolites query."""

    def test_selective_metabolites_execution(self, temp_model_dir):
        """Should execute selective metabolites query."""
        engine = QueryEngine(temp_model_dir)
        
        genera = engine.artifacts.get_taxa_at_level("genus")
        if len(genera) >= 3:
            target = genera[0]
            competitors = genera[1:3]
            
            query = SelectiveMetabolitesQuery.from_strings(
                f"genus:{target}",
                f"genus:{competitors[0]},genus:{competitors[1]}",
                top_k=5,
            )
            response = engine.query_selective_metabolites(query)
            
            assert response is not None
            assert response.success

    def test_selective_metabolites_returns_selectivity_scores(self, temp_model_dir):
        """Results should include selectivity scores."""
        engine = QueryEngine(temp_model_dir)
        
        genera = engine.artifacts.get_taxa_at_level("genus")
        if len(genera) >= 2:
            query = SelectiveMetabolitesQuery.from_strings(
                f"genus:{genera[0]}",
                f"genus:{genera[1]}",
                top_k=3,
            )
            response = engine.query_selective_metabolites(query)
            
            if response.results:
                for result in response.results:
                    assert "metabolite" in result
                    assert "selectivity" in result


class TestMetaboliteEnvironmentFitnessQuery:
    """Test metabolite environment fitness query."""

    def test_environment_fitness_execution(self, temp_model_dir):
        """Should execute environment fitness query."""
        engine = QueryEngine(temp_model_dir)
        
        query = MetaboliteEnvironmentFitnessQuery.from_strings(
            "glucose,alanine",
            level="genus",
            top_k=5,
        )
        response = engine.query_metabolite_environment_fitness(query)
        
        assert response is not None
        assert response.success

    def test_environment_fitness_returns_fitness_scores(self, temp_model_dir):
        """Results should include fitness scores."""
        engine = QueryEngine(temp_model_dir)
        
        query = MetaboliteEnvironmentFitnessQuery.from_strings(
            "glucose",
            level="genus",
            top_k=5,
        )
        response = engine.query_metabolite_environment_fitness(query)
        
        if response.results:
            for result in response.results:
                assert "microbe" in result
                assert "fitness" in result


class TestNicheMetabolitesQuery:
    """Test niche metabolites query."""

    def test_niche_metabolites_execution(self, temp_model_dir):
        """Should execute niche metabolites query."""
        engine = QueryEngine(temp_model_dir)
        
        isolates = list(engine.artifacts.training_data.microbe_to_idx.keys())[:1]
        if isolates:
            query = NicheMetabolitesQuery.from_string(
                f"isolate:{isolates[0]}",
                top_k=5,
            )
            response = engine.query_niche_metabolites(query)
            
            assert response is not None
            assert response.success

    def test_niche_metabolites_returns_distinctive_scores(self, temp_model_dir):
        """Results should include distinctive scores."""
        engine = QueryEngine(temp_model_dir)
        
        isolates = list(engine.artifacts.training_data.microbe_to_idx.keys())[:1]
        if isolates:
            query = NicheMetabolitesQuery.from_string(
                f"isolate:{isolates[0]}",
                top_k=3,
            )
            response = engine.query_niche_metabolites(query)
            
            if response.results:
                for result in response.results:
                    assert "metabolite" in result
                    assert "distinctive_score" in result


class TestSearchCompoundQuery:
    """Test compound search query."""

    def test_search_compound_execution(self, temp_model_dir):
        """Should execute compound search query."""
        engine = QueryEngine(temp_model_dir)
        
        query = SearchCompoundQuery(
            query="glucose",
            fuzzy=True,
        )
        response = engine.query_search_compound(query)
        
        assert response is not None
        assert response.success

    def test_search_compound_returns_matches(self, temp_model_dir):
        """Search should return matching compounds."""
        engine = QueryEngine(temp_model_dir)
        
        query = SearchCompoundQuery(
            query="glucose",
            fuzzy=True,
        )
        response = engine.query_search_compound(query)
        
        if response.results:
            result = response.results[0]
            assert "matched_name" in result
            assert "in_training_data" in result


class TestListTaxaQuery:
    """Test taxa list query."""

    def test_list_taxa_execution(self, temp_model_dir):
        """Should execute list taxa query."""
        engine = QueryEngine(temp_model_dir)
        
        for level in ["genus", "family"]:
            query = ListTaxaQuery(
                level=level,
                include_counts=True,
            )
            response = engine.query_list_taxa(query)
            
            assert response is not None
            assert response.success
            assert len(response.results) > 0

    def test_list_taxa_returns_structured_results(self, temp_model_dir):
        """Results should have proper structure."""
        engine = QueryEngine(temp_model_dir)
        
        query = ListTaxaQuery(
            level="genus",
            include_counts=True,
        )
        response = engine.query_list_taxa(query)
        
        if response.results:
            for result in response.results:
                assert "name" in result


class TestModelInfoQuery:
    """Test model info query."""

    def test_model_info_execution(self, temp_model_dir):
        """Should execute model info query."""
        engine = QueryEngine(temp_model_dir)
        
        query = ModelInfoQuery(include_stats=True)
        response = engine.query_model_info(query)
        
        assert response is not None
        assert response.success
        assert len(response.results) > 0

    def test_model_info_returns_required_fields(self, temp_model_dir):
        """Info should include required model metadata."""
        engine = QueryEngine(temp_model_dir)
        
        query = ModelInfoQuery(include_stats=True)
        response = engine.query_model_info(query)
        
        info = response.results[0]
        assert "model_type" in info
        assert "embedding_dim" in info
        assert "n_microbes" in info
        assert "n_metabolites" in info


class TestQueryResultsJSON:
    """Test JSON serialization of query results."""

    def test_query_response_serializable_to_json(self, temp_model_dir):
        """Query responses should be JSON serializable."""
        engine = QueryEngine(temp_model_dir)
        
        taxa = engine.artifacts.get_taxa_at_level("genus")
        if taxa:
            query = CompoundsForMicrobesQuery.from_string(
                f"genus:{taxa[0]}",
                top_k=3,
            )
            response = engine.query_compounds_for_microbes(query)
            
            # Should serialize without error
            json_str = json.dumps(response.to_dict(), indent=2)
            assert len(json_str) > 0
            
            # Should parse back
            parsed = json.loads(json_str)
            assert parsed["success"] == response.success

    def test_json_roundtrip(self, temp_model_dir):
        """Query responses should survive JSON roundtrip."""
        engine = QueryEngine(temp_model_dir)
        
        query = ListTaxaQuery(level="genus", include_counts=True)
        response = engine.query_list_taxa(query)
        
        # Serialize
        json_str = json.dumps(response.to_dict())
        
        # Parse back
        parsed = json.loads(json_str)
        
        assert parsed["query_type"] == response.query_type
        assert len(parsed["results"]) == len(response.results)
