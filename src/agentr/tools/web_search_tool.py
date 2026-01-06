"""
Web search tool using Tavily API.

This module provides synchronous and asynchronous web search tools that can be used
with the LLM client. The tools follow OpenAI's tool schema format.
"""

import os
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from tavily import TavilyClient, AsyncTavilyClient
from langchain_core.tools import StructuredTool

from .types import InformationSource, InformationResult

# ---------------------------------------------------------------------------
# Constants for tool
# ---------------------------------------------------------------------------

TOOL_NAME = "web_search"
TOOL_DESCRIPTION = "Search the web using Tavily and return sources with content."

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(
        ...,
        description="The search query to execute",
        min_length=1,
        max_length=500,
    )

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

# Using shared types from .types module:
# - InformationSource: source + content + timestamp
# - InformationResult: container for multiple sources with optional summary

# For backward compatibility, alias the types
WebSearchItem = InformationSource
WebSearchResult = InformationResult

# ---------------------------------------------------------------------------
# Tavily client initialization
# ---------------------------------------------------------------------------

def get_tavily_client() -> TavilyClient:
    """Initialize and return a Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is not set. "
        )
    return TavilyClient(api_key=api_key)

def get_async_tavily_client() -> AsyncTavilyClient:
    """Initialize and return an async Tavily client."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is not set. "
        )
    return AsyncTavilyClient(api_key=api_key)

# ---------------------------------------------------------------------------
# Helper function to normalize Tavily response
# ---------------------------------------------------------------------------

def _format_tavily_response(response: Dict[str, Any]) -> InformationResult:
    results = response.get("results", [])
    sources = [
        InformationSource(
            source=item.get("url", ""),
            content=item.get("content", "")
        )
        for item in results
    ]
    return InformationResult(sources=sources)

# ---------------------------------------------------------------------------
# Sync and async search functions
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# StructuredTool definition
# ---------------------------------------------------------------------------

web_search_tool = StructuredTool.from_function(
    name=TOOL_NAME,
    description=TOOL_DESCRIPTION,
    args_schema=SearchInput,
    func=web_search_sync,
    coroutine=web_search_async,
)
