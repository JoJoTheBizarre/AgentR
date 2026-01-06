"""Message types for LLM conversations using Pydantic.

This module provides strongly-typed message classes for LLM conversations,
compatible with OpenAI's Chat Completion API.

Available message types:
- SystemMessage: System instructions
- UserMessage: User queries
- AssistantMessage: Assistant responses (may contain tool calls)
- ToolResultMessage: Results of tool executions
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Enumeration of possible message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """Function call details within a tool call."""

    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Represents a tool call request from the assistant."""

    id: str = Field(..., description="Tool call ID")
    type: Literal["function"] = Field(
        default="function", description="Type of tool call"
    )
    function: FunctionCall = Field(..., description="Function call details")


class BaseMessage(BaseModel):
    """Base class for all message types."""

    role: Role = Field(..., description="Message role")
    content: str | None = Field(None, description="Text content of the message")

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format expected by LLM clients."""
        return self.model_dump(exclude_none=True)


class SystemMessage(BaseMessage):
    """System message providing instructions to the assistant."""

    role: Literal[Role.SYSTEM] = Field(default=Role.SYSTEM)
    content: str = Field(..., description="System instructions")


class UserMessage(BaseMessage):
    """Message from the user."""

    role: Literal[Role.USER] = Field(default=Role.USER)
    content: str = Field(..., description="User query or input")


class AssistantMessage(BaseMessage):
    """Message from the assistant, may contain tool calls."""

    role: Literal[Role.ASSISTANT] = Field(default=Role.ASSISTANT)
    content: str | None = Field(None, description="Assistant's text response")
    tool_calls: list[ToolCall] | None = Field(
        None, description="Tool calls requested by the assistant"
    )

    def has_tool_calls(self) -> bool:
        """Check if this assistant message contains tool calls."""
        return bool(self.tool_calls)

    def get_tool_calls(self) -> list[ToolCall]:
        """Return tool calls if present, otherwise empty list."""
        return self.tool_calls or []


class ToolResultMessage(BaseMessage):
    """Message containing the result of a tool call."""

    role: Literal[Role.TOOL] = Field(default=Role.TOOL)
    content: str = Field(..., description="Result of the tool execution")
    tool_call_id: str = Field(
        ..., description="ID of the tool call this result corresponds to"
    )


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
