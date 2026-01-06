"""Tool implementations for the research agent."""

from .web_search_tool import web_search_tool
from .registry import ToolRegistry
from .types import InformationSource, InformationResult

__all__ = [
    "web_search_tool",
    "ToolRegistry",
    "get_default_registry",
    "InformationSource",
    "InformationResult",
]

# Create a default registry with all available tools
_default_registry = ToolRegistry([web_search_tool])

def get_default_registry() -> ToolRegistry:
    """Return the default registry with all available tools registered."""
    return _default_registry