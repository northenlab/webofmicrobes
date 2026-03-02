"""
MCP server for the Web of Microbes growth-rate service.

Purpose: expose lightweight tools that help agents discover taxonomy levels
and interact with the growth-rate API used to look up predicted microbial
doubling times.
"""

from fastmcp import FastMCP
import requests
import random
import json
import sys

sys.path.insert(0, "./wom")  # Ensure the /app directory is in the Python path for imports

from api.query_engine import QueryEngine
from api.query_types import QueryConfig

mcp = FastMCP("My MCP Server")

@mcp.tool(
    name="hello_world_greet",           # Custom tool name for the LLM
    description="None.", # Custom description
)
def greet(name: str) -> str:
    """Simple connectivity check for the MCP server."""
    return f"Hello, {name}! Here is a number to remember {random.randint(1, 100)}."

# Testing for multistep calls
@mcp.tool(
    name="secret_code_step1",
    description="First step of a multistep tool that generates a secret code and returns a code_id.",
)
def secret_code_step1() -> str:
    """First step of a multistep tool that generates a secret code and returns a code_id."""
    # Generate a random secret code
    secret_code = "44"

    return secret_code

@mcp.tool(
    name="secret_code_step2",
    description="Second step of a multistep tool that takes the result from secret_code_step1 and changes the answer",
)
def secret_code_step2(input_code: str) -> str:
    """Second step of a multistep tool that takes the result from secret_code_step1 and changes the answer."""
    # Generate a random secret code
    secret_code = int(input_code) + 1
    return str(secret_code)


# final step
@mcp.tool(
    name="secret_code_final",
    description="Final step of a multistep tool that takes the result from secret_code_step2 and returns the final secret code",
)
def secret_code_final(input_code: str) -> str:
    """Final step of a multistep tool that takes the result from secret_code_step2 and returns the final secret code."""
    # Generate a random secret code
    secret_code = int(input_code) + 1
    return str(secret_code)

### These are the tools for WOM

@mcp.tool(
    name="wom_compounds",
    description="""This is an API provided by the Web of Microbes that will return a list of compounds that microbes interact with.
    Here are the parameter descriptions:

    """,
)
def wom_compounds(microbes: str, top: int = 20, compute_distinctive: bool = True, background_level: str = None, min_score: float = None) -> str:
    command = "compounds"

    data = {
        "microbes": microbes,
        "top": top,
        "compute_distinctive": compute_distinctive,
        "background_level": background_level,
        "min_score": min_score,
    }

    engine = get_engine()
    config = QueryConfig()

    response = execute_query(command, data, engine, config)

    return response

    # "parameters": {
    #             "microbes": {"type": "string", "required": True, "description": "Microbes to query (format: 'genus:Name,isolate:ID')"},
    #             "top": {"type": "integer", "default": 20, "description": "Number of top results per microbe"},
    #             "compute_distinctive": {"type": "boolean", "default": True, "description": "Compute distinctive scores"},
    #             "background_level": {"type": "string", "optional": True, "description": "Background level for distinctive scores"},
    #             "min_score": {"type": "float", "optional": True, "description": "Minimum raw score threshold"},
    #         },

    return ""



@mcp.tool(
    name="taxonomy_levels_for_growth_rate",           # Custom tool name for the LLM
    description="This is an API provided by the Web of Microbes growth-rate service that will return ", # Custom description
)
def taxalevels() -> str:
    """List taxonomy levels used by the growth-rate API (for query scoping)."""
    # get it from growth-rate service
    response = requests.get("http://growth-rate:5000/levels")
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=2)
    else:
        return "Error fetching taxonomy levels."


# GROWTH RATE LOOKUPS:
# - You can look up predicted microbial doubling times using the growth rate tool
# - Available levels: domain, phylum, class, order, family, genus
# - Available stats: mean, median, min, max, std
# - Default level is genus and default stat is median if not specified
# - Example: "I'll run: growth_rate query="Pseudomonas" level=genus stat=median"
# - Uses fuzzy matching, so approximate names work
# - Source: Phydon growth rate predictions
# - Doubling times are in hours

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)