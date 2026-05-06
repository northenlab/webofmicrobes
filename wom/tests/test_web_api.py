"""Tests for the Web API (scripts/wom_api.py).

Tests the Flask REST API endpoints:
- /api/health - health check
- /api/schema - API schema
- /api/query - query execution
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.wom_api import app, get_engine, API_SCHEMA


@pytest.fixture
def client():
    """Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_engine():
    """Mock query engine."""
    engine = MagicMock()
    return engine


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
    
    def test_health_check_response_structure(self, client):
        """Test health response has correct structure."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check_has_version(self, client):
        """Test health response includes version."""
        response = client.get("/api/health")
        data = json.loads(response.data)
        
        assert "version" in data


class TestSchemaEndpoint:
    """Test the API schema endpoint."""
    
    def test_schema_endpoint_success(self, client):
        """Test that schema endpoint returns 200."""
        response = client.get("/api/schema")
        
        assert response.status_code == 200
    
    def test_schema_has_all_commands(self, client):
        """Test schema includes all commands."""
        response = client.get("/api/schema")
        data = json.loads(response.data)
        
        assert "commands" in data
        commands = data["commands"]
        
        # Ensure schema command keys match API_SCHEMA source of truth
        expected_commands = set(API_SCHEMA["commands"].keys())
        assert set(commands.keys()) == expected_commands
    
    def test_schema_command_has_description(self, client):
        """Test each command in schema has description."""
        response = client.get("/api/schema")
        data = json.loads(response.data)
        
        for cmd, spec in data["commands"].items():
            assert "description" in spec
            assert isinstance(spec["description"], str)
            assert len(spec["description"]) > 0
    
    def test_schema_command_has_parameters(self, client):
        """Test each command in schema has parameters."""
        response = client.get("/api/schema")
        data = json.loads(response.data)
        
        for cmd, spec in data["commands"].items():
            assert "parameters" in spec
            assert isinstance(spec["parameters"], dict)
    
    def test_schema_command_has_example(self, client):
        """Test each command in schema has example."""
        response = client.get("/api/schema")
        data = json.loads(response.data)
        
        for cmd, spec in data["commands"].items():
            assert "example" in spec
            assert "command" in spec["example"]


class TestQueryEndpoint:
    """Test the query execution endpoint."""
    
    def test_query_success_status_code(self, client, mock_engine):
        """Test successful query returns 200."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "success": True,
                "results": [],
            }
            mock_engine.query_list_taxa.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({"command": "list", "level": "genus"}),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_missing_command_field(self, client):
        """Test query without 'command' field returns 400."""
        response = client.post(
            "/api/query",
            data=json.dumps({"level": "genus"}),  # Missing 'command'
            content_type="application/json",
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert "command" in data["error"].lower()
    
    def test_query_no_json_body(self, client):
        """Test query without JSON body returns 400."""
        response = client.post(
            "/api/query",
            data=None,
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
    
    def test_query_compounds_command(self, client, mock_engine):
        """Test /api/query with 'compounds' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "success": True,
                "results": [{"metabolite": "glucose", "score": 0.8}],
            }
            mock_engine.query_compounds_for_microbes.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "compounds",
                    "microbes": "genus:Pseudomonas",
                    "top": 10,
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
    
    def test_query_microbes_command(self, client, mock_engine):
        """Test /api/query with 'microbes' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {
                "success": True,
                "results": [{"microbe": "Pseudomonas", "score": 0.9}],
            }
            mock_engine.query_microbes_for_compounds.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "microbes",
                    "compounds": "glucose",
                    "level": "genus",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
    
    def test_query_selective_command(self, client, mock_engine):
        """Test /api/query with 'selective' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_selective_metabolites.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "selective",
                    "target": "genus:Pseudomonas",
                    "competitors": "genus:Burkholderia",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_fitness_command(self, client, mock_engine):
        """Test /api/query with 'fitness' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_metabolite_environment_fitness.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "fitness",
                    "metabolites": "glucose",
                    "level": "genus",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_niche_command(self, client, mock_engine):
        """Test /api/query with 'niche' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_niche_metabolites.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "niche",
                    "microbe": "genus:Pseudomonas",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_competition_command(self, client, mock_engine):
        """Test /api/query with 'competition' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_competition_analysis.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "competition",
                    "microbes": "genus:Pseudomonas,genus:Burkholderia",
                    "metabolites": "glucose",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_rank_substrates_command(self, client, mock_engine):
        """Test /api/query with 'rank-substrates' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_rank_substrates.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "rank-substrates",
                    "microbes": "genus:Bacillus",
                    "substrates": "glucose,pyruvate",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_search_command(self, client, mock_engine):
        """Test /api/query with 'search' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_search_compound.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "search",
                    "query": "glucose",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_list_command(self, client, mock_engine):
        """Test /api/query with 'list' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": []}
            mock_engine.query_list_taxa.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "list",
                    "level": "genus",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_info_command(self, client, mock_engine):
        """Test /api/query with 'info' command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_response = MagicMock()
            mock_response.to_dict.return_value = {"success": True, "results": {}}
            mock_engine.query_model_info.return_value = mock_response
            
            response = client.post(
                "/api/query",
                data=json.dumps({"command": "info"}),
                content_type="application/json",
            )
            
            assert response.status_code == 200
    
    def test_query_invalid_command(self, client, mock_engine):
        """Test /api/query with invalid command."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            response = client.post(
                "/api/query",
                data=json.dumps({"command": "invalid_command"}),
                content_type="application/json",
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert data["success"] is False
    
    def test_query_server_error_handling(self, client, mock_engine):
        """Test /api/query handles server exceptions."""
        with patch('scripts.wom_api.get_engine', return_value=mock_engine):
            mock_engine.query_list_taxa.side_effect = Exception("Unexpected error")
            
            response = client.post(
                "/api/query",
                data=json.dumps({
                    "command": "list",
                    "level": "genus",
                }),
                content_type="application/json",
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data["success"] is False


class TestHTTPMethods:
    """Test HTTP method restrictions."""
    
    def test_health_get_only(self, client):
        """Test /api/health only accepts GET."""
        response = client.post("/api/health")
        assert response.status_code == 405  # Method Not Allowed
    
    def test_schema_get_only(self, client):
        """Test /api/schema only accepts GET."""
        response = client.post("/api/schema")
        assert response.status_code == 405
    
    def test_query_post_only(self, client):
        """Test /api/query only accepts POST."""
        response = client.get("/api/query")
        assert response.status_code == 405


class TestContentType:
    """Test content type handling."""
    
    def test_query_requires_json_content_type(self, client):
        """Test /api/query rejects non-JSON payloads (returns 400)."""
        response = client.post(
            "/api/query",
            data="command=list&level=genus",
            content_type="application/x-www-form-urlencoded",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
    
    def test_response_is_json(self, client):
        """Test responses are JSON."""
        response = client.get("/api/health")
        assert response.content_type == "application/json"


class TestCORS:
    """Test CORS headers."""
    
    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses."""
        response = client.get("/api/health")
        
        # CORS headers should be set by flask_cors
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.status_code == 200
    
    def test_cors_allow_any_origin(self, client):
        """Test CORS allows requests from any origin."""
        response = client.options(
            "/api/health",
            headers={"Origin": "http://example.com"}
        )

        assert "Access-Control-Allow-Origin" in response.headers
        # flask_cors allows all origins, may return * or the specific origin
        assert response.headers["Access-Control-Allow-Origin"] in {"*", "http://example.com"}


class TestResponseSerialization:
    """Test response serialization."""
    
    def test_response_is_valid_json(self, client):
        """Test all responses are valid JSON."""
        for endpoint in ["/api/health", "/api/schema"]:
            response = client.get(endpoint)
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"{endpoint} did not return valid JSON")
    
    def test_response_utf8_encoding(self, client):
        """Test responses are UTF-8 encoded."""
        response = client.get("/api/health")
        assert "charset" not in response.content_type or "utf-8" in response.content_type.lower()
