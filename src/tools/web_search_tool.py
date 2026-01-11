import os
from typing import Any

from langchain_core.tools import StructuredTool
from models.states import Source, SourceType
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient, TavilyClient

TOOL_NAME = "web_search"
TOOL_DESCRIPTION = "Search the web using Tavily and return sources with content."


class SearchInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(
        ...,
        description="The search query to execute",
        min_length=1,
        max_length=500,
    )


def get_tavily_client() -> TavilyClient:
    """Initialize and return a Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set. ")
    return TavilyClient(api_key=api_key)


def get_async_tavily_client() -> AsyncTavilyClient:
    """Initialize and return an async Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set. ")
    return AsyncTavilyClient(api_key=api_key)


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
    client = get_tavily_client()
    response = client.search(query=query)
    return _format_tavily_response(response)


async def web_search_async(query: str) -> list[Source]:
    """Asynchronous web search using Tavily API."""
    client = get_async_tavily_client()
    response = await client.search(query=query)
    return _format_tavily_response(response)


def web_search_factory() -> StructuredTool:

    return  StructuredTool.from_function(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=SearchInput,
        func=web_search_sync,
        coroutine=web_search_async,
    )
