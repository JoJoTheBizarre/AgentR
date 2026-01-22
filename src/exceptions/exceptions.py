# ===== Agent Exceptions =====
from src.graph.nodes import NodeName


class AgentRError(Exception):
    """Base exception for all AgentR errors."""

    pass


class AgentInitializationError(AgentRError):
    """Exception raised when agent initialization fails."""

    pass


class ClientInitializationError(Exception):
    """Exception raised when LLM client initialization fails."""

    pass


class AgentExecutionError(AgentRError):
    """Base exception for agent execution errors."""

    pass


class ResponseError(AgentExecutionError):
    """Exception raised when agent response is invalid or missing."""

    pass


# ===== Tool Errors =====
class ToolError(Exception):
    """Base exception for tool-related errors."""

    pass


class ToolInitializationError(ToolError):
    """Exception raised when tool initialization fails."""

    pass


# ===== Validation Errors =====
class ValidationError(AgentRError):
    """Exception raised when validation fails."""

    pass


# ===== Node Execution Errors =====
class NodeExecutionError(Exception):
    """Exception raised when a node execution fails."""

    def __init__(self, node_name: NodeName, original_exception: Exception) -> None:
        self.node_name = node_name
        self.original_exception = original_exception
        super().__init__(
            f"Node '{node_name.value}' execution failed: {original_exception}"
        )
