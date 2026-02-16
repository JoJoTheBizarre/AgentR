class AgentRError(Exception):
    """Base exception for all AgentR errors."""

    pass


class AgentInitializationError(AgentRError):
    """Raised when agent initialization fails."""

    pass


class ClientInitializationError(AgentRError):
    """Raised when LLM client initialization fails."""

    pass


class AgentExecutionError(AgentRError):
    """Base exception for agent execution errors."""

    pass


class ResponseError(AgentExecutionError):
    """Raised when agent response is invalid or missing."""

    pass


class ToolError(AgentRError):
    """Base exception for tool-related errors."""

    pass


class ToolInitializationError(ToolError):
    """Raised when tool initialization fails."""

    pass


class ValidationError(AgentRError):
    """Raised when validation fails."""

    pass


class NodeExecutionError(AgentRError):
    """Raised when a node execution fails."""

    def __init__(self, node_name: str, original_exception: Exception):
        self.node_name = node_name
        self.original_exception = original_exception
        super().__init__(f"Node '{node_name}' execution failed: {original_exception}")
