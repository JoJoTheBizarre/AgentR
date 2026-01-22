from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class BaseTool(ABC):
    """Abstract base class for all tools in AgentR.

    Provides consistent interface for tool creation, registration, and usage.
    All tools should inherit from this class and implement required methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool's unique name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool's description for LLM consumption."""
        pass

    @property
    def args_schema(self) -> type[BaseModel] | None:
        """Return the Pydantic model for tool arguments.

        Returns:
            Pydantic model class or None if tool doesn't use structured args
        """
        return None

    @property
    def return_direct(self) -> bool:
        """Whether the tool returns output directly without LLM processing.

        Returns:
            True for tools that return final output, False for intermediate tools
        """
        return False

    @property
    def is_async(self) -> bool:
        """Whether the tool supports async execution.

        Returns:
            True if tool has async implementation, False for sync-only
        """
        return False

    @abstractmethod
    def create_tool(self) -> StructuredTool:
        """Create and return a LangChain StructuredTool instance.

        Returns:
            Configured StructuredTool ready for LLM consumption
        """
        pass

    def get_func(self) -> Any:
        """Return the function to be wrapped by StructuredTool.

        Returns:
            Callable function for sync execution
        """
        raise NotImplementedError(
            "Tool must implement get_func() or override create_tool()"
        )

    def get_coroutine(self) -> Any | None:
        """Return the async function to be wrapped by StructuredTool.

        Returns:
            Async callable function or None if tool doesn't support async
        """
        return None
