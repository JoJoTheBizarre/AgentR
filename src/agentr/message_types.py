"""Message types for LLM conversations using Pydantic.

This module provides strongly-typed message classes for LLM conversations,
compatible with OpenAI's Chat Completion API.

Available message types:
- SystemMessage: System instructions
- UserMessage: User queries
- AssistantMessage: Assistant responses (may contain tool calls)
- ToolResultMessage: Results of tool executions

Example usage:
    >>> from agentr.message_types import *
    >>> sys_msg = SystemMessage(content="You are a helpful assistant.")
    >>> user_msg = UserMessage(content="What's the weather in Paris?")
    >>> tool_call = ToolCall(
    ...     id="call_123",
    ...     type="function",
    ...     function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}')
    ... )
    >>> assistant_msg = AssistantMessage(content=None, tool_calls=[tool_call])
    >>> tool_result = ToolResultMessage(tool_call_id="call_123", content="Sunny, 20°C")
    >>> messages = [sys_msg, user_msg, assistant_msg, tool_result]
    >>> dicts = messages_to_dicts(messages)
    >>> # dicts can be passed to LLMClient.chat()
    >>> roundtrip = dicts_to_messages(dicts)
"""

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """Function call details within a tool call."""

    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Represents a tool call request from the assistant."""

    id: str = Field(..., description="Tool call ID")
    type: Literal["function"] = Field(default="function", description="Type of tool call")
    function: FunctionCall = Field(..., description="Function call details")


class BaseMessage(BaseModel):
    """Base class for all message types."""

    role: str = Field(..., description="Message role: 'system', 'user', 'assistant', or 'tool'")
    content: Optional[str] = Field(None, description="Text content of the message")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format expected by LLM clients."""
        return self.model_dump(exclude_none=True)


class SystemMessage(BaseMessage):
    """System message providing instructions to the assistant."""

    role: Literal["system"] = Field(default="system")
    content: str = Field(..., description="System instructions")


class UserMessage(BaseMessage):
    """Message from the user."""

    role: Literal["user"] = Field(default="user")
    content: str = Field(..., description="User query or input")


class AssistantMessage(BaseMessage):
    """Message from the assistant, may contain tool calls."""

    role: Literal["assistant"] = Field(default="assistant")
    content: Optional[str] = Field(None, description="Assistant's text response")
    tool_calls: Optional[List[ToolCall]] = Field(
        None,
        description="Tool calls requested by the assistant"
    )

    def has_tool_calls(self) -> bool:
        """Check if this assistant message contains tool calls."""
        return bool(self.tool_calls)

    def get_tool_calls(self) -> List[ToolCall]:
        """Return tool calls if present, otherwise empty list."""
        return self.tool_calls or []



class ToolResultMessage(BaseMessage):
    """Message containing the result of a tool call."""

    role: Literal["tool"] = Field(default="tool")
    content: str = Field(..., description="Result of the tool execution")
    tool_call_id: str = Field(..., description="ID of the tool call this result corresponds to")


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage]


def message_to_dict(message: Message) -> Dict[str, Any]:
    """Convert a Message instance to dictionary format."""
    return message.to_dict()


def dict_to_message(data: Dict[str, Any]) -> Message:
    """Convert a dictionary to the appropriate Message type.

    This is a convenience function that tries to infer the correct message type
    from the dictionary structure. Not all fields are validated.
    """
    role = data.get("role")

    if role == "system":
        return SystemMessage(**data)
    elif role == "user":
        return UserMessage(**data)
    elif role == "assistant":
        tool_calls_data = data.get("tool_calls")
        tool_calls = None
        if tool_calls_data:
            tool_calls = []
            for tc in tool_calls_data:
                if "function" in tc and isinstance(tc["function"], dict):
                    tc = tc.copy()
                    tc["function"] = FunctionCall(**tc["function"])
                tool_calls.append(ToolCall(**tc))
        return AssistantMessage(**{**data, "tool_calls": tool_calls})
    elif role == "tool":
        return ToolResultMessage(**data)
    else:
        raise ValueError(f"Unknown message role: {role}")


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert a list of Message instances to list of dictionaries."""
    return [message_to_dict(msg) for msg in messages]


def dicts_to_messages(dicts: List[Dict[str, Any]]) -> List[Message]:
    """Convert a list of dictionaries to list of Message instances."""
    return [dict_to_message(d) for d in dicts]