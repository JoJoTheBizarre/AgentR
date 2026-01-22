import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient, TavilyClient

from src.exceptions import ToolInitializationError
from src.models.states import Source, SourceType

logger = logging.getLogger(__name__)

from ..base.base_tool import BaseTool
from ..names import ToolName


class SearchInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(
        ...,
        description="The search query to execute",
        min_length=1,
        max_length=500,
    )


def get_async_tavily_client() -> AsyncTavilyClient:
    """Initialize and return an async Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable is not set")
        raise ToolInitializationError(
            "TAVILY_API_KEY environment variable is not set. "
        )
    logger.debug("Async Tavily client initialized")
    return AsyncTavilyClient(api_key=api_key)


def get_tavily_client() -> TavilyClient:
    """Initialize and return a Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable is not set")
        raise ToolInitializationError(
            "TAVILY_API_KEY environment variable is not set. "
        )
    logger.debug("Sync Tavily client initialized")
    return TavilyClient(api_key=api_key)


def _format_tavily_response(response: dict[str, Any]) -> list[Source]:
    results = response.get("results", [])
    sources = [
        Source(
            source=item.get("url", ""),
            content=item.get("content", ""),
            type=SourceType.WEB,
        )
        for item in results
    ]
    return sources


def web_search_sync(query: str) -> list[Source]:
    """Synchronous web search using Tavily API."""
    logger.debug(f"Web search sync started: query length {len(query)}")
    try:
        client = get_tavily_client()
        response = client.search(query=query)
        sources = _format_tavily_response(response)
        logger.debug(f"Web search sync completed: {len(sources)} sources found")
        return sources
    except Exception as e:
        logger.error(f"Web search sync failed: {type(e).__name__}")
        raise


async def web_search_async(query: str) -> list[Source]:
    """Asynchronous web search using Tavily API."""
    logger.debug(f"Web search async started: query length {len(query)}")
    try:
        client = get_async_tavily_client()
        response = await client.search(query=query)
        sources = _format_tavily_response(response)
        logger.debug(f"Web search async completed: {len(sources)} sources found")
        return sources
    except Exception as e:
        logger.error(f"Web search async failed: {type(e).__name__}")
        raise


class WebSearchTool(BaseTool):
    """Execution tool for web search using Tavily API."""

    TOOL_NAME = ToolName.WEB_SEARCH
    TOOL_DESCRIPTION = (
        "Search the web for up to date information using natural"
        "language, results are returned with their source URL"
    )

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return self.TOOL_DESCRIPTION

    @property
    def args_schema(self):
        return SearchInput

    @property
    def is_async(self) -> bool:
        return True

    def create_tool(self) -> StructuredTool:
        """Create and return a StructuredTool instance with sync and async support."""
        return StructuredTool.from_function(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=web_search_sync,
            coroutine=web_search_async,
        )

    def get_func(self):
        return web_search_sync

    def get_coroutine(self):
        return web_search_async
