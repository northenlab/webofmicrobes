import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def main():
    # FastMCP HTTP transport mounts the streamable MCP endpoint at /mcp
    url = "http://localhost:8000/mcp"
    
    print(f"Connecting to FastMCP server at {url}...")
    
    try:
        async with streamablehttp_client(url) as transport:
            read_stream, write_stream, _ = transport
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("Successfully connected and initialized!\n")
                
                # ---------------------------------------------------------
                # 1. Verify Tools
                # ---------------------------------------------------------
                print("--- Available Tools ---")
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    print(f"- {tool.name}: {tool.description}")
                print("-" * 23 + "\n")

                # ---------------------------------------------------------
                # 2. Call the "hello world greet" tool
                # ---------------------------------------------------------
                print("Calling 'hello world greet'...")
                greet_result = await session.call_tool(
                    "hello world greet", 
                    arguments={"name": "MCP Client Developer"}
                )
                # MCP returns content as a list of text/image blocks
                greet_text = greet_result.content[0].text
                print(f"Result: {greet_text}\n")

                # ---------------------------------------------------------
                # 3. Call the "taxonomy levels for growth rate" tool
                # ---------------------------------------------------------
                print("Calling 'taxonomy levels for growth rate'...")
                try:
                    taxa_result = await session.call_tool(
                        "taxonomy levels for growth rate", 
                        arguments={}
                    )
                    taxa_text = taxa_result.content[0].text
                    print(f"Result:\n{taxa_text}")
                except Exception as tool_err:
                    print(f"Tool execution failed (is the http://growth-rate:5000 service running?): {tool_err}")

    except Exception as e:
        print(f"\nError connecting to the MCP server: {e}")

if __name__ == "__main__":
    asyncio.run(main())