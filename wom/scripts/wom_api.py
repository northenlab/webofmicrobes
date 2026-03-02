#!/usr/bin/env python
"""
Flask REST API for metabolite preference predictions.

Uses the same QueryEngine as the CLI for consistency.

Usage:
    python wom_api.py                    # Run on localhost:5000
    python wom_api.py --port 8080        # Custom port
    python wom_api.py --host 0.0.0.0     # Listen on all interfaces

API Endpoints:
    POST /api/query
        Execute any query command
        Body: JSON with command and parameters
    
    GET /api/schema
        Get API schema with all available commands
    
    GET /api/health
        Health check

Example:
    curl -X POST http://localhost:5000/api/query \
      -H "Content-Type: application/json" \
      -d '{"command": "microbes", "compounds": "glutamate,serine", "level": "genus", "top": 10}'
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.query_engine import QueryEngine
from api.query_types import QueryConfig
from scripts.wom import execute_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global query engine (lazy-loaded)
_engine = None


def get_engine():
    """Get or create query engine."""
    global _engine
    if _engine is None:
        model_dir = app.config.get("MODEL_DIR", "trained_models/latest")
        _engine = QueryEngine(model_dir)
    return _engine


# =============================================================================
# API Schema
# =============================================================================

API_SCHEMA = {
    "version": "1.0",
    "commands": {
        "compounds": {
            "description": "Predict compounds consumed by microbes",
            "parameters": {
                "microbes": {"type": "string", "required": True, "description": "Microbes to query (format: 'genus:Name,isolate:ID')"},
                "top": {"type": "integer", "default": 20, "description": "Number of top results per microbe"},
                "compute_distinctive": {"type": "boolean", "default": True, "description": "Compute distinctive scores"},
                "background_level": {"type": "string", "optional": True, "description": "Background level for distinctive scores"},
                "min_score": {"type": "float", "optional": True, "description": "Minimum raw score threshold"},
            },
            "example": {
                "command": "compounds",
                "microbes": "genus:Burkholderia",
                "top": 20
            }
        },
        "microbes": {
            "description": "Predict microbes that consume compounds",
            "parameters": {
                "compounds": {"type": "string", "required": True, "description": "Compounds to query (comma-separated)"},
                "level": {"type": "string", "default": "genus", "choices": ["isolate", "genus", "family", "order", "class"], "description": "Taxonomic level"},
                "top": {"type": "integer", "default": 10, "description": "Number of top results"},
                "agg_method": {"type": "string", "default": "mean", "choices": ["mean", "sum", "max"], "description": "Score aggregation method"},
                "min_score": {"type": "float", "optional": True, "description": "Minimum aggregate score"},
            },
            "example": {
                "command": "microbes",
                "compounds": "glutamate,serine",
                "level": "genus",
                "top": 10
            }
        },
        "selective": {
            "description": "Find metabolites that favor target over competitors",
            "parameters": {
                "target": {"type": "string", "required": True, "description": "Target microbe"},
                "competitors": {"type": "string", "required": True, "description": "Competitor microbes (comma-separated)"},
                "top": {"type": "integer", "default": 20, "description": "Number of top results"},
                "method": {"type": "string", "default": "difference", "choices": ["difference", "ratio"], "description": "Selectivity method"},
                "min_selectivity": {"type": "float", "optional": True, "description": "Minimum selectivity threshold"},
            },
            "example": {
                "command": "selective",
                "target": "genus:Pseudomonas",
                "competitors": "genus:Burkholderia,genus:Variovorax",
                "top": 10
            }
        },
        "fitness": {
            "description": "Rank microbes by fitness in metabolite environment",
            "parameters": {
                "metabolites": {"type": "string", "required": True, "description": "Available metabolites (comma-separated)"},
                "level": {"type": "string", "default": "genus", "choices": ["isolate", "genus", "family", "order", "class"], "description": "Taxonomic level"},
                "microbes": {"type": "string", "optional": True, "description": "Specific microbes to score"},
                "top": {"type": "integer", "default": 10, "description": "Number of top results"},
                "fitness_method": {"type": "string", "default": "sum", "choices": ["sum", "mean", "weighted"], "description": "Fitness computation method"},
            },
            "example": {
                "command": "fitness",
                "metabolites": "arginine,lysine,glutamate",
                "level": "genus",
                "top": 10
            }
        },
        "niche": {
            "description": "Find distinctive metabolites for a microbe",
            "parameters": {
                "microbe": {"type": "string", "required": True, "description": "Microbe to query"},
                "top": {"type": "integer", "default": 20, "description": "Number of top results"},
                "background_level": {"type": "string", "optional": True, "description": "Background level for comparison"},
                "min_distinctive_score": {"type": "float", "optional": True, "description": "Minimum distinctive score"},
            },
            "example": {
                "command": "niche",
                "microbe": "genus:Burkholderia",
                "top": 20
            }
        },
        "search": {
            "description": "Search for compound name matches",
            "parameters": {
                "query": {"type": "string", "required": True, "description": "Compound name to search for"},
                "fuzzy": {"type": "boolean", "default": True, "description": "Enable fuzzy matching"},
                "min_score": {"type": "float", "default": 0.7, "description": "Minimum fuzzy match score"},
            },
            "example": {
                "command": "search",
                "query": "glutamic acid"
            }
        },
        "list": {
            "description": "List available taxa",
            "parameters": {
                "level": {"type": "string", "default": "genus", "choices": ["genus", "family", "order", "class", "kingdom"], "description": "Taxonomic level"},
                "include_counts": {"type": "boolean", "default": True, "description": "Include isolate counts"},
            },
            "example": {
                "command": "list",
                "level": "genus"
            }
        },
        "info": {
            "description": "Get model information",
            "parameters": {
                "include_stats": {"type": "boolean", "default": True, "description": "Include detailed statistics"},
            },
            "example": {
                "command": "info"
            }
        }
    }
}


# =============================================================================
# Routes
# =============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0"})


@app.route("/api/schema", methods=["GET"])
def schema():
    """Return API schema."""
    return jsonify(API_SCHEMA)


@app.route("/api/query", methods=["POST"])
def query():
    """Execute a query."""
    try:
        # Handle invalid JSON (empty body or malformed JSON)
        try:
            data = request.get_json()
        except Exception as e:
            # get_json() raises BadRequest for invalid/empty JSON or wrong content-type
            logger.debug(f"Invalid JSON in request: {e}")
            return jsonify({
                "success": False,
                "error": "Invalid JSON in request body"
            }), 400
        
        if not data or "command" not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'command' field in request body"
            }), 400
        
        command = data["command"]
        engine = get_engine()
        config = QueryConfig()
        
        # Use shared execute_query function from CLI
        try:
            response = execute_query(command, data, engine, config)
            return jsonify(response.to_dict())
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "available_commands": list(API_SCHEMA["commands"].keys())
            }), 400
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def main():
    parser = argparse.ArgumentParser(description="Run WOM API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--model-dir", default="trained_models/latest", help="Model directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Configure app
    app.config["MODEL_DIR"] = args.model_dir
    
    logger.info(f"Starting WOM API server on {args.host}:{args.port}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Endpoints:")
    logger.info(f"  GET  {args.host}:{args.port}/api/health")
    logger.info(f"  GET  {args.host}:{args.port}/api/schema")
    logger.info(f"  POST {args.host}:{args.port}/api/query")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
