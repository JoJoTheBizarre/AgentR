import logging
from enum import StrEnum
from typing import Any, ClassVar
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient, AsyncTavilyClient

from .config import EnvConfig
from .exceptions import ToolInitializationError
from .states import Source, SourceType

logger = logging.getLogger(__name__)


class ToolName(StrEnum):
    """Available tool names."""

    WEB_SEARCH = "web_search"
    RESEARCH_TOOL = "research_sub_agent"


# WEB SEARCH TOOL
class SearchInput(BaseModel):
    """Web search input schema."""

    query: str = Field(..., description="Search query", min_length=1, max_length=500)


def _get_tavily_client() -> TavilyClient:
    """Get Tavily client instance."""
    config = EnvConfig()  # type: ignore
    if not config.tavily_api_key:
        raise ToolInitializationError("TAVILY_API_KEY not set")
    return TavilyClient(api_key=config.tavily_api_key)


def _get_async_tavily_client() -> AsyncTavilyClient:
    """Get async Tavily client instance."""
    config = EnvConfig()  # type: ignore
    if not config.tavily_api_key:
        raise ToolInitializationError("TAVILY_API_KEY not set")
    return AsyncTavilyClient(api_key=config.tavily_api_key)


def _format_tavily_response(response: dict[str, Any]) -> list[Source]:
    """Format Tavily API response into Source list."""
    results = response.get("results", [])
    if not isinstance(results, list):
        return []

    return [
        Source(
            source=item.get("url", ""),
            content=item.get("content", ""),
            type=SourceType.WEB,
        )
        for item in results
    ]


def web_search_sync(query: str) -> list[Source]:
    """Synchronous web search."""
    logger.debug(f"Web search: {query[:50]}...")
    client = _get_tavily_client()
    response = client.search(query=query)
    sources = _format_tavily_response(response)
    logger.debug(f"Found {len(sources)} sources")
    return sources


async def web_search_async(query: str) -> list[Source]:
    """Asynchronous web search."""
    logger.debug(f"Async web search: {query[:50]}...")
    client = _get_async_tavily_client()
    response = await client.search(query=query)
    sources = _format_tavily_response(response)
    logger.debug(f"Found {len(sources)} sources")
    return sources


def create_web_search_tool() -> StructuredTool:
    """Create web search tool."""
    return StructuredTool.from_function(
        name=ToolName.WEB_SEARCH,
        description="Search the web for up-to-date information. Returns results with source URLs.",
        args_schema=SearchInput,
        func=web_search_sync,
        coroutine=web_search_async,
    )


# RESEARCH DECISION TOOL
class ShouldResearch(BaseModel):
    """Research decision schema."""

    subtasks: list[str]


def research_subagent(subtasks: list[str]) -> ShouldResearch:
    """Research subagent handoff function."""
    return ShouldResearch(subtasks=subtasks)


def create_research_tool() -> StructuredTool:
    """Create research decision tool."""
    return StructuredTool.from_function(
        name=ToolName.RESEARCH_TOOL,
        description=(
            "Use this tool to delegate research to a sub-agent. Provide independent subtasks "
            "that together answer the user's question."
        ),
        args_schema=ShouldResearch,
        func=research_subagent,
    )


# TOOL MANAGER
class ToolManager:
    """Central tool registry for AgentR."""

    _registry: ClassVar[dict[str, StructuredTool]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls):
        """Initialize tool registry with default tools."""
        if cls._initialized:
            return

        cls._registry[ToolName.WEB_SEARCH] = create_web_search_tool()
        cls._registry[ToolName.RESEARCH_TOOL] = create_research_tool()
        cls._initialized = True
        logger.info("ToolManager initialized")

    @classmethod
    def get_tool(cls, name: ToolName | str) -> StructuredTool:
        """Get tool by name."""
        if not cls._initialized:
            cls.initialize()

        if name not in cls._registry:
            raise KeyError(f"Tool '{name}' not found")

        return cls._registry[name]

    @classmethod
    def clear(cls):
        """Clear registry (for testing)."""
        cls._registry.clear()
        cls._initialized = False
