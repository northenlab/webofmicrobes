import json
import logging
import os

from pydantic_ai import Agent, Tool, capture_run_messages
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic import BaseModel
from pydantic_ai.mcp import MCPServerStreamableHTTP

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
system_prompt_filename = './prompts/wom_agent_system_prompt_mcp_version.md'
logger = logging.getLogger(__name__)
# check working directory and if system prompt file exists
if not os.path.isfile(system_prompt_filename):
    logger.error(f"System prompt file not found: {system_prompt_filename}")
    raise FileNotFoundError(f"System prompt file not found: {system_prompt_filename}")
else:
    logger.info(f"System prompt file found: {system_prompt_filename}")


def _short(value, limit: int = 220) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def _safe_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return repr(value)


def _short_code(value: str, limit: int = 180) -> str:
    compact = " | ".join(line.strip() for line in value.splitlines() if line.strip())
    return _short(compact, limit)


def _summarize_tool_args(tool_name: str, args) -> str:
    if not isinstance(args, dict):
        return _short(_safe_json(args), 180)

    if tool_name == "execute_python_analysis":
        code = args.get("code")
        other_args = {key: value for key, value in args.items() if key != "code"}
        bits = []
        if other_args:
            bits.append(f"args={_short(_safe_json(other_args), 120)}")
        if isinstance(code, str) and code.strip():
            bits.append(f"code={_short_code(code)}")
        if bits:
            return " ".join(bits)

    summarized = dict(args)
    for key in ("code", "python_code", "script"):
        if isinstance(summarized.get(key), str):
            summarized[key] = _short_code(summarized[key])

    return f"args={_short(_safe_json(summarized), 180)}"


def _summarize_tool_result(result) -> str:
    tool_name = getattr(result, "tool_name", "unknown_tool")
    content = getattr(result, "content", None)

    if isinstance(content, dict):
        ok = content.get("ok")
        error = content.get("error")
        stdout = content.get("stdout")
        stderr = content.get("stderr")
        output = content.get("output")
        saved_files = content.get("saved_files")

        bits = []
        if ok is not None:
            bits.append(f"ok={ok}")
        if error:
            bits.append(f"error={_short(str(error), 120)}")
        if stdout:
            bits.append(f"stdout={_short(str(stdout), 120)}")
        if stderr:
            bits.append(f"stderr={_short(str(stderr), 120)}")
        if isinstance(output, dict) and output:
            bits.append(f"output_keys={list(output.keys())}")
        elif output not in (None, {}, []):
            bits.append(f"output={_short(_safe_json(output), 120)}")
        if isinstance(saved_files, list) and saved_files:
            file_names = [os.path.basename(path) for path in saved_files]
            bits.append(f"saved_files={file_names}")

        suffix = f" ({', '.join(bits)})" if bits else ""
        return f"[Tool←] {tool_name}{suffix}"

    if content is not None:
        return f"[Tool←] {tool_name} content={_short(_safe_json(content), 160)}"

    return f"[Tool←] {tool_name}"


def format_stream_event(event):
    event_name = type(event).__name__

    if event_name == "FunctionToolCallEvent":
        part = getattr(event, "part", None)
        tool_name = getattr(part, "tool_name", "unknown_tool")
        args = getattr(part, "args", None)
        return "tool", f"[Tool→] {tool_name} {_summarize_tool_args(tool_name, args)}"

    if event_name == "FunctionToolResultEvent":
        result = getattr(event, "result", None)
        if result is not None:
            return "tool", _summarize_tool_result(result)
        return "tool", "[Tool←] result received"

    if event_name == "PartDeltaEvent":
        delta = getattr(event, "delta", None)
        content_delta = getattr(delta, "content_delta", None)
        if isinstance(content_delta, str) and content_delta:
            return "text", content_delta

    if event_name == "FinalResultEvent":
        return "meta", "\n[Model] final result received"

    return None, None


class TestAgent:
    def __init__(self, model_name: str, base_url: str = "", provider: str = "auto",
                 mcp_url: str = ""):
        mcp_url = mcp_url or os.getenv("MCP_SERVER_URL", "http://wom-mcp:8000/mcp")
        self.mcp_server = MCPServerStreamableHTTP(url=mcp_url, max_retries=3)
        self.mcp_servers = [self.mcp_server]
        provider_name = (provider or "auto").lower()

        if provider_name == "openrouter":
            openrouter_base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            self.model = OpenAIModel(
                model_name=model_name,
                provider=OpenAIProvider(base_url=openrouter_base_url, api_key=openrouter_api_key),
            )
        elif provider_name in {"google", "gemini"}:
            self.model = GoogleModel(
                model_name=model_name or "gemini-3-flash-preview",
                provider=GoogleProvider(api_key=os.getenv("GEMINI_API_KEY")),
            )
        elif provider_name == "litellm":
            litellm_base_url = base_url or os.getenv("LITELLM_PROXY_URL", "http://localhost:4000/v1")
            litellm_api_key = os.getenv("LITELLM_PROXY_API_KEY", "")
            litellm_model = model_name or os.getenv("LITELLM_MODEL_NAME")
            self.model = OpenAIModel(
                model_name=litellm_model,
                provider=OpenAIProvider(base_url=litellm_base_url, api_key=litellm_api_key),
            )
        elif provider_name in {"ollama", "local", "openai-compatible"}:
            self.model = OpenAIModel(
                model_name=model_name,
                provider=OpenAIProvider(base_url=base_url),
            )
        else:
            if os.getenv("OPENROUTER_API_KEY"):
                openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                self.model = OpenAIModel(
                    model_name=model_name,
                    provider=OpenAIProvider(base_url=openrouter_base_url, api_key=os.getenv("OPENROUTER_API_KEY")),
                )
            elif os.getenv("GEMINI_API_KEY"):
                self.model = GoogleModel(
                    model_name=model_name or "gemini-3-flash-preview",
                    provider=GoogleProvider(api_key=os.getenv("GEMINI_API_KEY")),
                )
            elif os.getenv("LITELLM_PROXY_URL"):
                litellm_base_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000/v1")
                litellm_api_key = os.getenv("LITELLM_PROXY_API_KEY", "")
                litellm_model = model_name or os.getenv("LITELLM_MODEL_NAME", "gpt-3.5-turbo")
                self.model = OpenAIModel(
                    model_name=litellm_model,
                    provider=OpenAIProvider(base_url=litellm_base_url, api_key=litellm_api_key),
                )
            else:
                self.model = OpenAIModel(
                    model_name=model_name,
                    provider=OpenAIProvider(base_url=base_url),
                )

        with open(system_prompt_filename, "r") as handle:
            system_prompt = handle.read()

        self.agent = Agent(
            self.model,
            system_prompt=system_prompt,
            mcp_servers=self.mcp_servers,
            model_settings={'temperature': 0.1}
        )

    @staticmethod
    def format_stream_event(event):
        return format_stream_event(event)

    @staticmethod
    def _message_history(previous_result=None):
        if previous_result is None:
            return None
        if hasattr(previous_result, "all_messages"):
            return previous_result.all_messages()
        if isinstance(previous_result, list):
            return previous_result
        return None

    async def run_async(self, query: str, previous_result=None):
        async with self.agent:
            agent_response = await self.agent.run(
                query,
                message_history=self._message_history(previous_result),
            )
        return agent_response

    def run_sync(self, query: str, previous_result=None):
        with capture_run_messages():
            agent_response = self.agent.run_sync(
                query,
                message_history=self._message_history(previous_result),
            )
            self.latest_response = agent_response
            return agent_response

    async def iter_formatted_events(self, query: str, previous_result=None):
        async with self.agent.iter(
            query,
            message_history=self._message_history(previous_result),
        ) as run:
            async for node in run:
                if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            kind, message = self.format_stream_event(event)
                            if kind is not None:
                                yield kind, message

            self.latest_response = run.result

    def run_streaming_sync(self, query: str, previous_result=None):
        agent_response = self.agent.run_stream_sync(
            query,
            message_history=self._message_history(previous_result),
        )

        for chunk in agent_response.stream_text(delta=True):
            yield chunk

        self.latest_response = agent_response
        return agent_response

