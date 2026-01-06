"""
Tool registry for managing and accessing LLM tools.

This module provides a ToolRegistry class that can store multiple tools,
retrieve them by name, and convert them to OpenAI tool schemas.
"""

from typing import Any, Dict, List, Optional, Union
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class ToolRegistry:
    """
    Registry for managing LLM tools.

    The registry stores tools by name and provides methods to:
    1. Get a tool by name
    2. Get OpenAI-compatible tool schemas for all registered tools
    3. Manage tool lifecycle

    Args:
        tools: Optional list of tools to initialize the registry with.
    """

    def __init__(self, tools: Optional[List[Union[BaseTool, StructuredTool]]] = None):
        self._tools: Dict[str, Union[BaseTool, StructuredTool]] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Union[BaseTool, StructuredTool]) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool to register (must have a 'name' attribute).

        Raises:
            ValueError: If tool lacks a name attribute or if name is already registered.
        """
        if not hasattr(tool, 'name') or not tool.name:
            raise ValueError(f"Tool must have a 'name' attribute: {tool}")

        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")

        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Union[BaseTool, StructuredTool]:
        """
        Get a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The tool object.

        Raises:
            KeyError: If no tool with the given name exists.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]


    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all registered tools to OpenAI tool schema format.

        Returns:
            List of tool definitions in OpenAI tool schema format.
        """
        openai_tools = []
        for tool in self._tools.values():
            openai_tools.append(convert_to_openai_tool(tool))
        return openai_tools

    def get_openai_tool(self, name: str) -> Dict[str, Any]:
        """
        Get OpenAI tool schema for a specific tool by name.

        Args:
            name: Name of the tool.

        Returns:
            Tool definition in OpenAI tool schema format.

        Raises:
            KeyError: If no tool with the given name exists.
        """
        tool = self.get_tool(name)
        return convert_to_openai_tool(tool)

    def get_all_tools(self) -> List[Union[BaseTool, StructuredTool]]:
        """Return list of all registered tool objects."""
        return list(self._tools.values())

    def list_tools(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool with given name exists in registry."""
        return name in self._tools

    def clear(self) -> None:
        """Remove all tools from registry."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools