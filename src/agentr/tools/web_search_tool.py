"""
Web search tool using Tavily API.

This module provides synchronous and asynchronous web search tools that can be used
with the LLM client. The tools follow OpenAI's tool schema format.
"""

import os
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient, TavilyClient

from agentr.prompts import (
    WEB_SEARCH_TOOL_DESCRIPTION,
    WEB_SEARCH_TOOL_NAME,
)

from .types import InformationResult, InformationSource, SourceType


class SearchInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(..., description="The search query to execute")


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


def _format_tavily_response(response: dict[str, Any]) -> InformationResult:
    results = response.get("results", [])
    sources = [
        InformationSource(
            type=SourceType.WEB_SEARCH,
            source=item.get("url", ""),
            content=item.get("content", ""),
        )
        for item in results
    ]
    return InformationResult(sources=sources)


def web_search_sync(query: str) -> InformationResult:
    """Synchronous web search using Tavily API."""
    client = get_tavily_client()
    response = client.search(query=query)
    return _format_tavily_response(response)


async def web_search_async(query: str) -> InformationResult:
    """Asynchronous web search using Tavily API."""
    client = get_async_tavily_client()
    response = await client.search(query=query)
    return _format_tavily_response(response)


web_search_tool = StructuredTool.from_function(
    name=WEB_SEARCH_TOOL_NAME,
    description=WEB_SEARCH_TOOL_DESCRIPTION,
    args_schema=SearchInput,
    func=web_search_sync,
    coroutine=web_search_async,
)
