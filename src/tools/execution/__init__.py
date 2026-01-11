"""Execution tools (web_search, API calls, database queries)."""

from .web_search_tool import WebSearchTool, web_search_factory

__all__ = ["WebSearchTool", "web_search_factory"]
