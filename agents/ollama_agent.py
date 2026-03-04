from pydantic_ai import Agent, Tool, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerStreamableHTTP

import logging
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
system_prompt_filename = './prompts/wom_agent_system_prompt_mcp_version.md'
logger = logging.getLogger(__name__)
# check working directory and if system prompt file exists
if not os.path.isfile(system_prompt_filename):
    logger.error(f"System prompt file not found: {system_prompt_filename}")
    raise FileNotFoundError(f"System prompt file not found: {system_prompt_filename}")
else:
    logger.info(f"System prompt file found: {system_prompt_filename}")
class TestAgent:
    mcp_server = MCPServerStreamableHTTP(url='http://wom-mcp:8000/mcp', max_retries=3) 
    #mcp_server = MCPServerStreamableHTTP(url='http://localhost:8000/mcp', max_retries=3) 
    def __init__(self, model_name: str, base_url: str):
        #self.model = OpenAIModel(model_name=model_name, provider=OpenAIProvider(base_url=base_url))
        self.model = GoogleModel(model_name="gemini-3-flash-preview", provider=GoogleProvider(api_key=os.getenv("GEMINI_API_KEY")))
        #self.model = OpenAIModel(model_name=model_name, provider=OpenAIProvider(base_url=base_url)) # OpenRouter
        self.agent = Agent(
            self.model,
            system_prompt=open(system_prompt_filename, "r").read(),
            mcp_servers=[self.mcp_server],
        )
    # async def run(self, query: str, previous_result=None):
    #     async with self.agent:
    #         if previous_result:
    #             result = await self.agent.run(query, message_history=previous_result.all_messages())
    #         else:
    #             result = await self.agent.run(query)
    #     return result

    def run_sync(self, query: str, previous_result=None):
        with capture_run_messages() as messages:
            agent_response = self.agent.run_sync(query, message_history=previous_result.all_messages() if previous_result else None)

            self.latest_response = agent_response  # Store the latest response for potential future use

            #for msg in messages:
            #    print("Captured message:", msg)  # Debug print to check captured messages

            return agent_response

    def run_streaming_sync(self, query: str, previous_result=None):
        agent_response = self.agent.run_stream_sync(query, message_history=previous_result.all_messages() if previous_result else None)

        # in the meantime, we should stream the response with yield
        for chunk in agent_response.stream_text(delta=True):
            yield chunk

        self.latest_response = agent_response  # Store the latest response for potential future use

        return agent_response

