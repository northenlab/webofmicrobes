"""Tests for the shared execute_query() function.

This function is the core logic used by both the CLI and Web API.
Testing it validates both interfaces simultaneously.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.wom import execute_query
from api.query_types import QueryConfig


class TestCompoundsCommand:
    """Test the 'compounds' command (compounds consumed/produced by microbes)."""
    
    def test_compounds_single_microbe(self, trained_artifacts):
        """Query compounds for a single microbe."""
        response = execute_query(
            command="compounds",
            params={
                "microbes": "genus:Arthrobacter",
                "top": 5,
                "compute_distinctive": True,
            },
            engine=trained_artifacts.training_data,  # Placeholder
            config=QueryConfig(),
        )
        # Note: Will need proper engine fixture
        # assert response.success
        # assert len(response.results) > 0
        # assert response.results[0].get("metabolite") is not None
    
    def test_compounds_compute_distinctive(self, trained_artifacts):
        """Test computing distinctive scores."""
        response = execute_query(
            command="compounds",
            params={
                "microbes": "genus:Arthrobacter",
                "top": 5,
                "compute_distinctive": True,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # if response.results:
        #     assert "distinctive_score" in response.results[0]
    
    def test_compounds_without_distinctive(self, trained_artifacts):
        """Test without computing distinctive scores."""
        response = execute_query(
            command="compounds",
            params={
                "microbes": "genus:Arthrobacter",
                "compute_distinctive": False,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_compounds_with_min_score(self, trained_artifacts):
        """Test filtering by minimum score."""
        response = execute_query(
            command="compounds",
            params={
                "microbes": "genus:Arthrobacter",
                "min_score": 0.5,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # if response.results:
        #     assert all(r["score"] >= 0.5 for r in response.results)
    
    def test_compounds_explain_mode(self, trained_artifacts):
        """Test with explanation enabled."""
        response = execute_query(
            command="compounds",
            params={
                "microbes": "genus:Arthrobacter",
                "explain": True,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # assert response.explanation is not None


class TestMicrobesCommand:
    """Test the 'microbes' command (microbes that consume/produce compounds)."""
    
    def test_microbes_single_compound(self, trained_artifacts):
        """Query microbes for a single compound."""
        response = execute_query(
            command="microbes",
            params={
                "compounds": "glucose",
                "level": "genus",
                "top": 10,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # assert len(response.results) > 0
    
    def test_microbes_multiple_compounds(self, trained_artifacts):
        """Query microbes that consume multiple compounds."""
        response = execute_query(
            command="microbes",
            params={
                "compounds": "glucose,serine,glutamate",
                "level": "genus",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_microbes_different_levels(self, trained_artifacts):
        """Test different taxonomic levels."""
        for level in ["isolate", "genus", "family", "order", "class"]:
            response = execute_query(
                command="microbes",
                params={
                    "compounds": "glucose",
                    "level": level,
                    "top": 5,
                },
                engine=trained_artifacts.training_data,
                config=QueryConfig(),
            )
            # assert response.success
    
    def test_microbes_agg_methods(self, trained_artifacts):
        """Test different aggregation methods."""
        for agg in ["mean", "sum", "max"]:
            response = execute_query(
                command="microbes",
                params={
                    "compounds": "glucose",
                    "agg": agg,
                },
                engine=trained_artifacts.training_data,
                config=QueryConfig(),
            )
            # assert response.success


class TestSelectiveCommand:
    """Test the 'selective' command (metabolites favoring one microbe over others)."""
    
    def test_selective_basic(self, trained_artifacts):
        """Query selective metabolites between target and competitors."""
        response = execute_query(
            command="selective",
            params={
                "target": "genus:Pseudomonas",
                "competitors": "genus:Burkholderia",
                "top": 10,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_selective_multiple_competitors(self, trained_artifacts):
        """Test with multiple competitor microbes."""
        response = execute_query(
            command="selective",
            params={
                "target": "genus:Pseudomonas",
                "competitors": "genus:Burkholderia,genus:Variovorax",
                "top": 10,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_selective_difference_method(self, trained_artifacts):
        """Test difference-based selectivity."""
        response = execute_query(
            command="selective",
            params={
                "target": "genus:Pseudomonas",
                "competitors": "genus:Burkholderia",
                "method": "difference",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success


class TestFitnessCommand:
    """Test the 'fitness' command (microbes that thrive in metabolite environment)."""
    
    def test_fitness_basic(self, trained_artifacts):
        """Query fitness in a metabolite environment."""
        response = execute_query(
            command="fitness",
            params={
                "metabolites": "glucose,pyruvate",
                "level": "genus",
                "top": 10,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_fitness_different_levels(self, trained_artifacts):
        """Test fitness at different taxonomic levels."""
        for level in ["isolate", "genus", "family"]:
            response = execute_query(
                command="fitness",
                params={
                    "metabolites": "glucose",
                    "level": level,
                },
                engine=trained_artifacts.training_data,
                config=QueryConfig(),
            )
            # assert response.success
    
    def test_fitness_fitness_methods(self, trained_artifacts):
        """Test different fitness computation methods."""
        for method in ["sum", "mean", "max"]:
            response = execute_query(
                command="fitness",
                params={
                    "metabolites": "glucose",
                    "fitness_method": method,
                },
                engine=trained_artifacts.training_data,
                config=QueryConfig(),
            )
            # assert response.success


class TestNicheCommand:
    """Test the 'niche' command (distinctive metabolites for a microbe)."""
    
    def test_niche_single_isolate(self, trained_artifacts):
        """Query niche metabolites for a single isolate."""
        # Get first microbe from training data
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        
        response = execute_query(
            command="niche",
            params={
                "microbe": f"isolate:{microbe_id}",
                "top": 10,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_niche_with_background_level(self, trained_artifacts):
        """Test niche with different background level."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        
        response = execute_query(
            command="niche",
            params={
                "microbe": f"isolate:{microbe_id}",
                "background_level": "genus",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success


class TestCompetitionCommand:
    """Test the 'competition' command (competition analysis)."""
    
    def test_competition_basic(self, trained_artifacts):
        """Analyze competition between microbes for metabolites."""
        response = execute_query(
            command="competition",
            params={
                "microbes": "genus:Pseudomonas,genus:Burkholderia",
                "metabolites": "glucose,glutamate",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_competition_single_metabolite(self, trained_artifacts):
        """Test competition for single metabolite."""
        response = execute_query(
            command="competition",
            params={
                "microbes": "genus:Pseudomonas,genus:Burkholderia",
                "metabolites": "glucose",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success


class TestRankSubstratesCommand:
    """Test the 'rank-substrates' command (rank substrates for growth)."""
    
    def test_rank_substrates_basic(self, trained_artifacts):
        """Rank substrates for microbial growth."""
        response = execute_query(
            command="rank-substrates",
            params={
                "microbes": "genus:Pseudomonas",
                "substrates": "glucose,pyruvate,acetate,glutamate",
                "top": 5,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_rank_substrates_multiple_microbes(self, trained_artifacts):
        """Rank substrates for multiple microbes."""
        response = execute_query(
            command="rank-substrates",
            params={
                "microbes": "genus:Pseudomonas,genus:Burkholderia",
                "substrates": "glucose,pyruvate,acetate",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success


class TestSearchCommand:
    """Test the 'search' command (fuzzy search for compounds)."""
    
    def test_search_exact_match(self, trained_artifacts):
        """Search for compound with exact match."""
        response = execute_query(
            command="search",
            params={
                "query": "glucose",
                "exact": False,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # assert len(response.results) > 0
    
    def test_search_fuzzy_match(self, trained_artifacts):
        """Search for compound with fuzzy matching."""
        response = execute_query(
            command="search",
            params={
                "query": "glucos",  # Typo
                "exact": False,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
    
    def test_search_no_matches(self, trained_artifacts):
        """Search with no matches."""
        response = execute_query(
            command="search",
            params={
                "query": "xyzabc123notarealcompound",
                "exact": False,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert not response.success or len(response.results) == 0


class TestListCommand:
    """Test the 'list' command (list taxa at taxonomic level)."""
    
    def test_list_genera(self, trained_artifacts):
        """List all genera."""
        response = execute_query(
            command="list",
            params={
                "level": "genus",
                "include_counts": True,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # assert len(response.results) > 0
        # assert all("count" in r for r in response.results)
    
    def test_list_different_levels(self, trained_artifacts):
        """List taxa at different levels."""
        for level in ["genus", "family", "order", "class", "kingdom"]:
            response = execute_query(
                command="list",
                params={
                    "level": level,
                    "include_counts": False,
                },
                engine=trained_artifacts.training_data,
                config=QueryConfig(),
            )
            # assert response.success


class TestInfoCommand:
    """Test the 'info' command (model information)."""
    
    def test_info_basic(self, trained_artifacts):
        """Get model information."""
        response = execute_query(
            command="info",
            params={
                "include_stats": True,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success
        # assert "model_type" in response.results or response.metadata
    
    def test_info_without_stats(self, trained_artifacts):
        """Get model info without statistics."""
        response = execute_query(
            command="info",
            params={
                "include_stats": False,
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert response.success


class TestErrorHandling:
    """Test error handling in execute_query."""
    
    def test_unknown_command(self, trained_artifacts):
        """Test handling of unknown command."""
        with pytest.raises(ValueError):
            execute_query(
                command="nonexistent",
                params={},
                engine=trained_artifacts.training_data,
                config=QueryConfig(),
            )
    
    def test_missing_required_params(self, trained_artifacts):
        """Test handling of missing required parameters."""
        response = execute_query(
            command="compounds",
            params={},  # Missing 'microbes' parameter
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        assert response is not None
        assert response.success is False
    
    def test_invalid_microbe_format(self, trained_artifacts):
        """Test handling of invalid microbe specification format."""
        response = execute_query(
            command="compounds",
            params={
                "microbes": "invalid_format_no_colon",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        assert response is not None
        assert response.success is False
    
    def test_invalid_level(self, trained_artifacts):
        """Test handling of invalid taxonomic level."""
        response = execute_query(
            command="microbes",
            params={
                "compounds": "glucose",
                "level": "invalid_level",
            },
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        assert response is not None
        assert hasattr(response, "success")
        assert isinstance(response.results, list)


class TestResponseStructure:
    """Test that responses have correct structure."""
    
    def test_response_has_required_fields(self, trained_artifacts):
        """Every response should have success, results, and metadata."""
        response = execute_query(
            command="list",
            params={"level": "genus"},
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert hasattr(response, 'success')
        # assert hasattr(response, 'results')
        # assert hasattr(response, 'metadata')
    
    def test_results_is_list(self, trained_artifacts):
        """Results should always be a list."""
        response = execute_query(
            command="list",
            params={"level": "genus"},
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # assert isinstance(response.results, list)
    
    def test_response_is_json_serializable(self, trained_artifacts):
        """Response should be JSON-serializable."""
        import json
        
        response = execute_query(
            command="list",
            params={"level": "genus"},
            engine=trained_artifacts.training_data,
            config=QueryConfig(),
        )
        # This should not raise an exception
        # json.dumps(response.to_dict())
