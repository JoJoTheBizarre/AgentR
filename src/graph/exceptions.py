# ===== Agent Exceptions =====
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
