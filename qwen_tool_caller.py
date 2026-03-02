"""
Qwen Thinking as tool caller using LlamaIndex high-level APIs.
Uses ReActAgent, FunctionTool, and OpenAILike (any OpenAI-compatible endpoint + custom model names).
Runs inside asyncio so the workflow has a running event loop.

Requires: pip install llama-index llama-index-llms-openai-like
Server: Your OpenAI-compatible endpoint must support tool/function calling
        (e.g. --enable-auto-tool-choice for vLLM) for the chosen model.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
AI_GRID_KEY = os.getenv("AI_GRID_KEY")

# LlamaIndex: OpenAILike accepts custom model names (no OpenAI allowlist)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike

BASE_URL = os.getenv("AI_GRID_BASE_URL", "http://app.ai-grid.io:4000/v1")
MODEL = os.getenv("AI_GRID_TOOL_MODEL", "Qwen3-30B-A3B-Thinking")


# -----------------------------------------------------------------------------
# Tools (FunctionTool wraps Python functions for the agent)
# -----------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """Get the current weather for a given city. Use when the user asks about weather."""
    return f"Weather in {city}: 22°C, partly cloudy (demo)."


def add_numbers(a: float, b: float) -> float:
    """Add two numbers. Use when the user asks to add or sum two numbers."""
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers. Use when the user asks to multiply two numbers."""
    return a * b


# -----------------------------------------------------------------------------
# Async main: ReActAgent requires a running event loop
# -----------------------------------------------------------------------------

async def run_agent():
    llm = OpenAILike(
        model=MODEL,
        api_base=BASE_URL,
        api_key=AI_GRID_KEY or "dummy",
        is_chat_model=True,
        is_function_calling_model=True,
        context_window=32768,
    )

    tools = [
        FunctionTool.from_defaults(fn=get_weather),
        FunctionTool.from_defaults(fn=add_numbers),
        FunctionTool.from_defaults(fn=multiply_numbers),
    ]

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        verbose=True,
    )

    query = "What is 17 multiplied by 4? Then what's the weather in Paris?"
    print("Query:", query)
    print("-" * 60)

    # agent.run() returns a WorkflowHandler (awaitable); must await to get result
    handler = agent.run(user_msg=query)
    result = await handler

    # Result is typically the final response (e.g. AgentWorkflowResult or similar)
    print("-" * 60)
    if hasattr(result, "output"):
        print("Response:", result.output)
    elif hasattr(result, "response"):
        print("Response:", result.response)
    else:
        print("Response:", result)


def main():
    try:
        asyncio.run(run_agent())
    except ValueError as e:
        if "Unknown model" in str(e) or "valid OpenAI model" in str(e):
            print(
                "\nServer or client rejected the model name when using tools. "
                "Ensure your endpoint supports tool/function calling for this model "
                "(e.g. vLLM: --enable-auto-tool-choice). Or set AI_GRID_TOOL_MODEL to a model id the server accepts for tool calls."
            )
            raise
        raise
    except Exception as e:
        if "tool_choice" in str(e).lower() or "400" in str(e):
            print(
                "\nTool calling not supported by the server for this model. "
                "Enable --enable-auto-tool-choice (and --tool-call-parser) on the vLLM/LiteLLM side, or use a model that supports tool calls."
            )
        raise


if __name__ == "__main__":
    main()
