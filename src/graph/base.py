from abc import ABC, abstractmethod
from typing import Any

from .nodes import NodeName


class NodeExecutionError(Exception):
    """Exception raised when a node execution fails."""

    def __init__(self, node_name: NodeName, original_exception: Exception) -> None:
        self.node_name = node_name
        self.original_exception = original_exception
        super().__init__(
            f"Node '{node_name.value}' execution failed: {original_exception}"
        )


class BaseNode(ABC):
    """Abstract base class for all graph nodes."""

    @property
    @abstractmethod
    def node_name(self) -> NodeName:
        """Return the name of this node."""
        pass

    @abstractmethod
    def _execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the node's logic. To be implemented by subclasses."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper around _execute that adds node name to error messages."""
        try:
            return self._execute(*args, **kwargs)
        except Exception as e:
            # Re-raise NodeExecutionError to include node name context
            raise NodeExecutionError(self.node_name, e) from e
