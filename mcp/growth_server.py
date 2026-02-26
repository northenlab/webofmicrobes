from fastmcp import FastMCP
import requests
import json

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool
def taxalevels() -> str:
    # get it from growth-rate service
    response = requests.get("http://growth-rate:5000/levels")
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=2)
    else:
        return "Error fetching taxonomy levels."

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)