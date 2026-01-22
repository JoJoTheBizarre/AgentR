"""
API response fixture generators for testing.
"""

from typing import Any


def create_openai_response(
    content: str = "Test response from OpenAI",
    tool_calls: list | None = None,
) -> dict[str, Any]:
    """Create mock OpenAI response."""
    return {
        "content": content,
        "tool_calls": tool_calls or [],
    }


def create_tool_call_response(
    tool_name: str = "web_search",
    args: dict | None = None,
) -> dict[str, Any]:
    """Create mock OpenAI response with tool calls."""
    if args is None:
        args = {"query": "test search query"}

    return {"content": "", "tool_calls": [{"name": tool_name, "args": args}]}


def create_tavily_response(
    query: str = "test search",
    results: list[dict] | None = None,
) -> dict[str, Any]:
    """Create mock Tavily search response."""
    if results is None:
        results = [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "Test content 1",
                "score": 0.9,
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/2",
                "content": "Test content 2",
                "score": 0.8,
            },
        ]

    return {
        "query": query,
        "results": results,
    }


def create_single_source_result(
    title: str = "Test Source",
    url: str = "https://example.com",
    content: str = "Test content",
    score: float = 0.9,
) -> dict[str, Any]:
    """Create a single source result for Tavily."""
    return {
        "title": title,
        "url": url,
        "content": content,
        "score": score,
    }


def create_research_synthesis_response(
    synthesis: str = "Research synthesis based on findings",
    sources: list[dict] | None = None,
) -> dict[str, Any]:
    """Create a mock research synthesis response."""
    if sources is None:
        sources = [
            {
                "source": "https://example.com/1",
                "content": "Content from source 1",
                "type": "web",
            },
            {
                "source": "https://example.com/2",
                "content": "Content from source 2",
                "type": "web",
            },
        ]

    return {
        "synthesis": synthesis,
        "sources": sources,
    }
