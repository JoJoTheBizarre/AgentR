"""Custom exceptions for the AgentR system."""

from typing import Optional

from .nodes import NodeName


class AgentRError(Exception):
    """Base exception for all AgentR errors."""
    pass


# ===== Agent Initialization Errors =====
class AgentInitializationError(AgentRError):
    """Exception raised when agent initialization fails."""
    pass


class ConfigurationError(AgentInitializationError):
    """Exception raised when configuration is invalid or missing."""
    pass


class ClientInitializationError(AgentInitializationError):
    """Exception raised when LLM client initialization fails."""
    pass


# ===== Agent Execution Errors =====
class AgentExecutionError(AgentRError):
    """Base exception for agent execution errors."""
    pass


class StateError(AgentExecutionError):
    """Exception raised when state is invalid or missing required fields."""

    def __init__(self, message: str, state_field: Optional[str] = None):
        self.state_field = state_field
        if state_field:
            message = f"State error in field '{state_field}': {message}"
        super().__init__(message)


class ResponseError(AgentExecutionError):
    """Exception raised when agent response is invalid or missing."""
    pass


# ===== Graph Execution Errors =====
class GraphExecutionError(AgentExecutionError):
    """Exception raised when graph execution fails."""
    pass


class NodeError(GraphExecutionError):
    """Base exception for node-related errors."""

    def __init__(self, node_name: NodeName, message: str):
        self.node_name = node_name
        super().__init__(f"Node '{node_name.value}' error: {message}")


class NodeInitializationError(NodeError):
    """Exception raised when node initialization fails."""
    pass


class NodeInputError(NodeError):
    """Exception raised when node receives invalid input."""
    pass


class NodeOutputError(NodeError):
    """Exception raised when node produces invalid output."""
    pass


# ===== Tool Errors =====
class ToolError(AgentRError):
    """Base exception for tool-related errors."""
    pass


class ToolInitializationError(ToolError):
    """Exception raised when tool initialization fails."""
    pass


class ToolExecutionError(ToolError):
    """Exception raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' execution error: {message}")


class ToolNotFoundError(ToolError):
    """Exception raised when a requested tool is not found."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found in registry")


# ===== Research Errors =====
class ResearchError(AgentExecutionError):
    """Base exception for research-related errors."""
    pass


class ResearchInitializationError(ResearchError):
    """Exception raised when research initialization fails."""
    pass


class ResearchExecutionError(ResearchError):
    """Exception raised when research execution fails."""
    pass


class ResearchTimeoutError(ResearchError):
    """Exception raised when research times out."""
    pass


# ===== Validation Errors =====
class ValidationError(AgentRError):
    """Exception raised when validation fails."""
    pass


class StateValidationError(ValidationError):
    """Exception raised when state validation fails."""
    pass


class ResponseValidationError(ValidationError):
    """Exception raised when response validation fails."""
    pass