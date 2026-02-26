"""
MCP server for the Web of Microbes growth-rate service.

Purpose: expose lightweight tools that help agents discover taxonomy levels
and interact with the growth-rate API used to look up predicted microbial
doubling times.
"""

from fastmcp import FastMCP
import requests
import json

mcp = FastMCP("My MCP Server")

@mcp.tool(
    name="hello world greet",           # Custom tool name for the LLM
    description="None.", # Custom description
)
def greet(name: str) -> str:
    """Simple connectivity check for the MCP server."""
    return f"Hello, {name}!"

@mcp.tool(
    name="taxonomy levels for growth rate",           # Custom tool name for the LLM
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