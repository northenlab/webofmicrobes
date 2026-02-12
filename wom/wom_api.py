#!/usr/bin/env python
"""
Flask API for consumption model predictions.

Mirrors the CLI interface in wom.py as a single POST endpoint.

Usage:
    python app.py                  # Run dev server on port 5000
    python app.py --port 8080      # Custom port
    python app.py --host 0.0.0.0   # Listen on all interfaces

API:
    POST /api/query
    Content-Type: application/json
    
    {
        "command": "microbes",
        "compounds": "glutamate,serine,guanine",
        "level": "genus",
        "top": 10
    }

    GET /api/schema
    Returns available commands and their parameters.
"""

import argparse
import json
import logging
import sys
from types import SimpleNamespace

from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# =============================================================================
# Schema: describes available commands and their parameters for introspection
# =============================================================================

COMMAND_SCHEMA = {
    "microbes": {
        "description": "Predict which microbes consume the given compounds",
        "parameters": {
            "compounds": {
                "type": "string",
                "required": True,
                "description": "Comma-separated compound names (e.g., 'glutamate,serine,guanine')"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "level": {
                "type": "string",
                "required": False,
                "default": "genus",
                "choices": ["isolate", "genus", "family", "order", "class"],
                "description": "Taxonomic level to aggregate to"
            },
            "agg": {
                "type": "string",
                "required": False,
                "default": "mean",
                "choices": ["mean", "sum", "max"],
                "description": "Aggregation method"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of top results to return"
            }
        },
        "example": {
            "command": "microbes",
            "compounds": "glutamate,serine",
            "level": "genus",
            "top": 10
        }
    },
    "compounds": {
        "description": "Predict which compounds are consumed by given microbes",
        "parameters": {
            "microbes": {
                "type": "string",
                "required": True,
                "description": "Microbe queries as 'level:name' pairs, comma-separated (e.g., 'genus:Burkholderia,isolate:RhodococcusOAS809')"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "agg": {
                "type": "string",
                "required": False,
                "default": "mean",
                "choices": ["mean", "sum", "max"],
                "description": "Aggregation method across isolates within a query"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 20,
                "description": "Number of top results to return"
            }
        },
        "example": {
            "command": "compounds",
            "microbes": "genus:Burkholderia",
            "top": 20
        }
    },
    "search-compound": {
        "description": "Fuzzy search for compound names in the model",
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "Search string (e.g., 'glutamic acid')"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of results to return"
            }
        },
        "example": {
            "command": "search-compound",
            "query": "glutamic acid"
        }
    },
    "search-taxon": {
        "description": "Fuzzy search for taxon names in the model",
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "Search string (e.g., 'burkhold')"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "level": {
                "type": "string",
                "required": False,
                "default": None,
                "choices": ["genus", "family", "order", "class"],
                "description": "Limit search to a specific taxonomic level (default: search all levels)"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of results to return"
            }
        },
        "example": {
            "command": "search-taxon",
            "query": "burkhold",
            "level": "genus"
        }
    },
    "list-compounds": {
        "description": "List all compounds in the model, optionally filtered by substring",
        "parameters": {
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "filter": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Filter compounds by substring (case-insensitive)"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 100,
                "description": "Max number to return"
            }
        },
        "example": {
            "command": "list-compounds",
            "filter": "amino"
        }
    },
    "list-microbes": {
        "description": "List isolates in the model, optionally filtered by taxonomy",
        "parameters": {
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "genus": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Filter by genus"
            },
            "family": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Filter by family"
            },
            "order": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Filter by order"
            },
            "class_": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Filter by class"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 100,
                "description": "Max number to return"
            }
        },
        "example": {
            "command": "list-microbes",
            "genus": "Burkholderia"
        }
    },
    "list-taxa": {
        "description": "List unique taxa at a given taxonomic level with isolate counts",
        "parameters": {
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "level": {
                "type": "string",
                "required": False,
                "default": "genus",
                "choices": ["genus", "family", "order", "class"],
                "description": "Taxonomic level"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 50,
                "description": "Max number to return"
            }
        },
        "example": {
            "command": "list-taxa",
            "level": "genus"
        }
    },
    "info": {
        "description": "Get model information, statistics, and available commands",
        "parameters": {
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            }
        },
        "example": {
            "command": "info"
        }
    },
    "raw-scores": {
        "description": "Get raw prediction scores without aggregation",
        "parameters": {
            "mode": {
                "type": "string",
                "required": True,
                "choices": ["microbes", "compounds"],
                "description": "Score mode: 'microbes' scores isolates on compounds, 'compounds' scores compounds for microbes"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "compounds": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Comma-separated compounds (required when mode='microbes')"
            },
            "microbes": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Microbe queries (required when mode='compounds')"
            }
        },
        "example": {
            "command": "raw-scores",
            "mode": "microbes",
            "compounds": "glutamate,serine"
        }
    },
    "compare": {
        "description": "Compare consumption preferences between two or more microbes/taxa, showing compounds where they differ or agree most",
        "parameters": {
            "microbes": {
                "type": "string",
                "required": True,
                "description": "Two or more microbe queries, comma-separated (e.g., 'genus:Burkholderia,genus:Arthrobacter')"
            },
            "model": {
                "type": "string",
                "required": False,
                "default": "bpr",
                "choices": ["bpr", "wrmf", "svd"],
                "description": "Model type to use"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of results per category (most different, most similar)"
            }
        },
        "example": {
            "command": "compare",
            "microbes": "genus:Burkholderia,genus:Arthrobacter",
            "top": 10
        }
    }
}



# =============================================================================
# Schema: complex commands
# =============================================================================

COMPLEX_COMMAND_SCHEMA = {
    "selective": {
        "description": "Find metabolites that favor a target microbe over competitors",
        "parameters": {
            "target": {
                "type": "string",
                "required": True,
                "description": "Target microbe to promote (e.g., 'genus:Pseudomonas')"
            },
            "competitors": {
                "type": "string",
                "required": True,
                "description": "Competitor microbes, comma-separated (e.g., 'genus:Burkholderia,genus:Variovorax')"
            },
            "exclude": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Metabolites to exclude (already in environment), comma-separated"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of results"
            }
        },
        "example": {
            "command": "selective",
            "target": "genus:Pseudomonas",
            "competitors": "genus:Burkholderia,genus:Variovorax",
            "top": 10
        }
    },
    "fitness": {
        "description": "Rank microbes by their affinity for a metabolite environment",
        "parameters": {
            "metabolites": {
                "type": "string",
                "required": True,
                "description": "Metabolites in environment, comma-separated (e.g., 'arginine,lysine')"
            },
            "microbes": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Specific microbes to evaluate (optional, defaults to all genera)"
            },
            "level": {
                "type": "string",
                "required": False,
                "default": "genus",
                "choices": ["isolate", "genus", "family", "order", "class"],
                "description": "Taxonomic level"
            },
            "agg": {
                "type": "string",
                "required": False,
                "default": "sum",
                "choices": ["mean", "sum", "max"],
                "description": "Aggregation across metabolites"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of results"
            }
        },
        "example": {
            "command": "fitness",
            "metabolites": "arginine,lysine",
            "top": 10
        }
    },
    "niche": {
        "description": "Find unique metabolic preferences for a microbe vs others",
        "parameters": {
            "target": {
                "type": "string",
                "required": True,
                "description": "Target microbe (e.g., 'genus:Pseudomonas')"
            },
            "versus": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Compare against specific microbes (optional, defaults to all genera)"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of results"
            }
        },
        "example": {
            "command": "niche",
            "target": "genus:Pseudomonas",
            "top": 10
        }
    },
    "competition": {
        "description": "Analyze which microbe wins for given metabolites",
        "parameters": {
            "microbes": {
                "type": "string",
                "required": True,
                "description": "Competing microbes, comma-separated (e.g., 'genus:Pseudomonas,genus:Burkholderia')"
            },
            "metabolites": {
                "type": "string",
                "required": True,
                "description": "Metabolites to compete for, comma-separated"
            }
        },
        "example": {
            "command": "competition",
            "microbes": "genus:Pseudomonas,genus:Burkholderia",
            "metabolites": "glutamate,arginine"
        }
    },
    "rank-substrates": {
        "description": "Rank specific substrates for specific microbes by preference",
        "parameters": {
            "microbes": {
                "type": "string",
                "required": True,
                "description": "Microbes to analyze, comma-separated (e.g., 'genus:Bacillus,genus:Pseudomonas')"
            },
            "substrates": {
                "type": "string",
                "required": True,
                "description": "Substrates to rank, comma-separated (e.g., 'glucose,glutamate,alanine')"
            },
            "top": {
                "type": "integer",
                "required": False,
                "default": 10,
                "description": "Number of top substrates per microbe"
            }
        },
        "example": {
            "command": "rank-substrates",
            "microbes": "genus:Bacillus,genus:Pseudomonas",
            "substrates": "glucose,glutamate,alanine",
            "top": 5
        }
    }
}



# =============================================================================
# Default values per command (used to fill in missing optional parameters)
# =============================================================================

COMMAND_DEFAULTS = {
    "microbes": {"model": "bpr", "level": "genus", "agg": "mean", "top": 10},
    "compounds": {"model": "bpr", "agg": "mean", "top": 20},
    "search-compound": {"model": "bpr", "top": 10},
    "search-taxon": {"model": "bpr", "level": None, "top": 10},
    "list-compounds": {"model": "bpr", "filter": None, "top": 100},
    "list-microbes": {"model": "bpr", "genus": None, "family": None, "order": None, "class_": None, "top": 100},
    "list-taxa": {"model": "bpr", "level": "genus", "top": 50},
    "info": {"model": "bpr"},
    "raw-scores": {"model": "bpr", "compounds": None, "microbes": None},
    "compare": {"model": "bpr", "top": 10},
}


# =============================================================================
# Defaults: complex commands
# =============================================================================

COMPLEX_COMMAND_DEFAULTS = {
    "selective": {"exclude": None, "top": 10, "format": "json"},
    "fitness": {"microbes": None, "level": "genus", "agg": "sum", "top": 10, "format": "json"},
    "niche": {"versus": None, "top": 10, "format": "json"},
    "competition": {"format": "json"},
    "rank-substrates": {"top": 10, "format": "json"},
}


# =============================================================================
# Map command names to the cmd_* functions in wom.py
# =============================================================================

# We map hyphenated API command names to the wom.py function names
_COMMAND_TO_FUNC = {
    "microbes": "cmd_microbes",
    "compounds": "cmd_compounds",
    "search-compound": "cmd_search_compound",
    "search-taxon": "cmd_search_taxon",
    "list-compounds": "cmd_list_compounds",
    "list-microbes": "cmd_list_microbes",
    "list-taxa": "cmd_list_taxa",
    "info": "cmd_info",
    "raw-scores": "cmd_raw_scores",
    "compare": "cmd_compare",
}

_COMPLEX_COMMAND_TO_FUNC = {
    "selective": "cmd_selective",
    "fitness": "cmd_fitness",
    "niche": "cmd_niche",
    "competition": "cmd_competition",
    "rank-substrates": "cmd_rank_substrates",
}

def _build_namespace(command: str, params: dict, defaults_map: dict = None) -> SimpleNamespace:
    if defaults_map is None:
        defaults_map = COMMAND_DEFAULTS
    defaults = defaults_map.get(command, {})
    """
    Build a SimpleNamespace that mimics argparse output for a given command.
    
    Merges provided params with defaults, coerces types, and adds the
    'format' attribute that wom.py's cmd_* functions expect (though they
    don't use it internally — they just return dicts).
    """
    # defaults = COMMAND_DEFAULTS.get(command, {})
    
    merged = {}
    merged.update(defaults)
    merged.update({k: v for k, v in params.items() if k != "command"})
    
    # Coerce 'top' to int if present
    if "top" in merged and merged["top"] is not None:
        try:
            merged["top"] = int(merged["top"])
        except (ValueError, TypeError):
            raise ValueError(f"'top' must be an integer, got: {merged['top']}")
    
    # Add format (not used by cmd_* functions but present in argparse namespace)
    merged["format"] = "json"
    
    return SimpleNamespace(**merged)


def _validate_request(data: dict, schema: dict = None) -> tuple:
    if schema is None:
        schema = COMMAND_SCHEMA

    if not data:
        raise ValueError("Request body is empty. Send JSON with at least a 'command' field.")

    command = data.get("command")
    if not command:
        raise ValueError(
            f"Missing 'command' field. Available commands: {list(schema.keys())}"
        )

    if command not in schema:
        raise ValueError(
            f"Unknown command: '{command}'. Available commands: {list(schema.keys())}"
        )

    cmd_schema = schema[command]
    for param_name, param_info in cmd_schema["parameters"].items():
        if param_info.get("required") and param_name not in data:
            raise ValueError(
                f"Command '{command}' requires parameter '{param_name}': "
                f"{param_info['description']}"
            )

    for param_name, param_info in cmd_schema["parameters"].items():
        if param_name in data and "choices" in param_info:
            value = data[param_name]
            if value is not None and value not in param_info["choices"]:
                raise ValueError(
                    f"Invalid value '{value}' for parameter '{param_name}'. "
                    f"Choices: {param_info['choices']}"
                )

    params = {k: v for k, v in data.items() if k != "command"}
    return command, params

# =============================================================================
# Routes
# =============================================================================

@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Main query endpoint. Accepts JSON body with a 'command' field
    and command-specific parameters.
    """
    data = request.get_json(silent=True)
    
    if data is None:
        return jsonify({
            "success": False,
            "error": "Invalid or missing JSON body. Send Content-Type: application/json with a JSON object.",
            "query": None,
            "results": []
        }), 400
    
    # Validate
    try:
        command, params = _validate_request(data)
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "query": data,
            "results": []
        }), 400
    
    # Build namespace adapter
    try:
        args = _build_namespace(command, params)
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "query": data,
            "results": []
        }), 400
    
    # Import and call the corresponding wom.py function
    import wom
    func_name = _COMMAND_TO_FUNC[command]
    func = getattr(wom, func_name)
    
    result = func(args)
    
    status_code = 200 if result.get("success", True) else 500
    return jsonify(result), status_code


# @app.route("/api/schema", methods=["GET"])
# def api_schema():
#     """
#     Return the full command schema for introspection.
    
#     An LLM agent can call this to discover available commands
#     and their parameters.
#     """
#     return jsonify({
#         "success": True,
#         "endpoint": "POST /api/query",
#         "content_type": "application/json",
#         "description": (
#             "Web of Microbes prediction API. Send a JSON body with a 'command' field "
#             "and command-specific parameters. All commands return JSON with a 'success' "
#             "boolean, 'error' (null on success), 'query' echo, and command-specific results."
#         ),
#         "commands": COMMAND_SCHEMA
#     })


# @app.route("/api/health", methods=["GET"])
# def api_health():
#     """Simple health check endpoint."""
#     return jsonify({
#         "status": "ok",
#         "service": "web-of-microbes"
#     })


# =============================================================================
# Routes: complex commands (NEW)
# =============================================================================

@app.route("/api/complex", methods=["POST"])
def api_complex():
    """
    Complex multi-step query endpoint. Accepts JSON body with a 'command' field
    and command-specific parameters.
    """
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "success": False,
            "error": "Invalid or missing JSON body.",
            "query": None,
            "results": []
        }), 400

    try:
        command, params = _validate_request(data, schema=COMPLEX_COMMAND_SCHEMA)
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "query": data,
            "results": []
        }), 400

    try:
        args = _build_namespace(command, params, defaults_map=COMPLEX_COMMAND_DEFAULTS)
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "query": data,
            "results": []
        }), 400

    import wom_complex
    func_name = _COMPLEX_COMMAND_TO_FUNC[command]
    func = getattr(wom_complex, func_name)

    result = func(args)

    status_code = 200 if result.get("success", True) else 500
    return jsonify(result), status_code


@app.route("/api/complex/schema", methods=["GET"])
def api_complex_schema():
    """Return the complex command schema for introspection."""
    return jsonify({
        "success": True,
        "endpoint": "POST /api/complex",
        "content_type": "application/json",
        "description": (
            "Complex multi-step queries for the Web of Microbes model. "
            "These commands combine multiple internal operations (scoring, "
            "aggregation, comparison, ranking) into a single call."
        ),
        "commands": COMPLEX_COMMAND_SCHEMA
    })


# =============================================================================
# Routes: schema and health (updated)
# =============================================================================

@app.route("/api/schema", methods=["GET"])
def api_schema():
    return jsonify({
        "success": True,
        "endpoint": "POST /api/query",
        "content_type": "application/json",
        "description": (
            "Web of Microbes prediction API. Send a JSON body with a 'command' field "
            "and command-specific parameters."
        ),
        "commands": COMMAND_SCHEMA,
        "complex_commands": {
            "endpoint": "POST /api/complex",
            "schema_url": "/api/complex/schema",
            "description": "Advanced multi-step queries available at /api/complex"
        }
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "service": "web-of-microbes"
    })


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Web of Microbes Flask API")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload the default model on startup instead of lazy-loading on first request"
    )
    args = parser.parse_args()
    
    if args.preload:
        logger.warning("Preloading default model...")
        from wom import _get_wrapper
        _get_wrapper("bpr")
        logger.warning("Model loaded.")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()