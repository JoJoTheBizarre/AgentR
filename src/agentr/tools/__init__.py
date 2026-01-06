"""Tool implementations for the research agent."""

from .formatting_tools import create_subtasks_tool, set_synthesis_flag_tool
from .registry import ToolRegistry
from .types import InformationResult, InformationSource
from .web_search_tool import web_search_tool

__all__ = [
    "InformationResult",
    "InformationSource",
    "ToolRegistry",
    "ToolRegistry",
    "create_subtasks_tool",
    "get_default_registry",
    "set_synthesis_flag_tool",
    "set_synthesis_flag_tool",
    "set_synthesis_flag_tool",
    "web_search_tool",
]

_default_registry = ToolRegistry(
    [
        web_search_tool,
        create_subtasks_tool,
        set_synthesis_flag_tool,
    ]
)


def get_default_registry() -> ToolRegistry:
    """Return the default registry with all available tools registered."""
    return _default_registry
