"""Tool manager for centralized tool registration and retrieval."""

from typing import ClassVar

from langchain_core.tools import StructuredTool

from .base.base_tool import BaseTool
from graph.exceptions import ToolInitializationError


class ToolManager:
    """Central registry for managing tools in AgentR.

    Provides singleton access to registered tools and factory methods
    for creating StructuredTool instances.
    """

    _registry: ClassVar[dict[str, type[BaseTool]]] = {}
    _initialized = False

    @classmethod
    def register_tool(cls, tool_class: type[BaseTool]) -> None:
        """Register a tool class with the manager.

        Args:
            tool_class: Tool class inheriting from BaseTool

        Raises:
            ToolInitializationError: If tool with same name already registered
        """
        tool_instance = tool_class()
        tool_name = tool_instance.name

        if tool_name in cls._registry:
            raise ToolInitializationError(f"Tool '{tool_name}' is already registered")

        cls._registry[tool_name] = tool_class
        print(f"Registered tool: {tool_name}")  # TODO: Replace with proper logging

    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        """Get a tool instance by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool instance

        Raises:
            KeyError: If tool is not registered
        """
        if not cls._initialized:
            cls.initialize_defaults()

        if name not in cls._registry:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {list(cls._registry.keys())}"
            )

        return cls._registry[name]()

    @classmethod
    def get_structured_tool(cls, name: str):
        """Get a LangChain StructuredTool instance by tool name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            StructuredTool instance ready for LLM consumption
        """
        tool_instance = cls.get_tool(name)
        return tool_instance.create_tool()

    @classmethod
    def get_all_tools(cls) -> dict[str, type[BaseTool]]:
        """Get all registered tool classes.

        Returns:
            Dictionary mapping tool names to tool classes
        """
        return cls._registry.copy()

    @classmethod
    def get_all_structured_tools(cls) -> dict[str, StructuredTool]:
        """Get all registered tools as StructuredTool instances.

        Returns:
            Dictionary mapping tool names to StructuredTool instances
        """

        return {
            name: tool_class().create_tool()
            for name, tool_class in cls._registry.items()
        }

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered tools (mainly for testing)."""
        cls._registry.clear()

    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize the manager with default tools.

        This method should be called once during application startup.
        """
        if cls._initialized:
            return

        # Import here to avoid circular imports
        from .decision.research_tool import ResearchDecisionTool
        from .execution.web_search_tool import WebSearchTool

        # Register default tools
        cls.register_tool(ResearchDecisionTool)
        cls.register_tool(WebSearchTool)

        cls._initialized = True
        print("ToolManager initialized with default tools")
        # TODO: Replace with proper logging
