from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from .nodes import NodeName


class BaseState(TypedDict):
    message_history: list[BaseMessage]


T = TypeVar("T", bound=BaseState)
S = TypeVar("S", bound=BaseState)


class NodeExecutionError(Exception):
    """Exception raised when a node execution fails."""

    def __init__(self, node_name: NodeName, original_exception: Exception) -> None:
        self.node_name = node_name
        self.original_exception = original_exception
        super().__init__(
            f"Node '{node_name.value}' execution failed: {original_exception}"
        )


class BaseNode(ABC, Generic[T, S]):
    """Abstract base class for all graph nodes."""

    @property
    @abstractmethod
    def node_name(self) -> NodeName:
        """Return the name of this node."""
        raise NotImplementedError

    @abstractmethod
    def _execute(self, state: T, config: RunnableConfig) -> S:
        """Execute the node's logic. To be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, state: T, config: RunnableConfig) -> S:
        """Wrapper around _execute that adds node name to error messages."""
        try:
            return self._execute(state, config)
        except Exception as e:
            raise NodeExecutionError(self.node_name, e) from e
