"""
Reusable patch decorators for testing.
"""

from unittest.mock import patch


def patch_openai():
    """Patch ChatOpenAI client."""
    return patch("src.client.openai_client.ChatOpenAI")


def patch_tavily():
    """Patch TavilyClient."""
    return patch("src.tools.execution.web_search_tool.TavilyClient")


def patch_langfuse():
    """Patch Langfuse."""
    return patch("src.graph.agent.Langfuse")


def patch_env(env_vars: dict[str, str]):
    """Patch os.environ with given variables."""
    return patch.dict("os.environ", env_vars, clear=True)


def patch_tool_manager():
    """Patch ToolManager to isolate tool registry."""
    return patch("src.tools.manager.ToolManager")


def patch_openai_client():
    """Patch OpenAIClient class."""
    return patch("src.client.OpenAIClient")
