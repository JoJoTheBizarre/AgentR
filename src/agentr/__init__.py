"""AgentR - A Python agent framework."""

__version__ = "0.1.0"

from agentr.llm_client import LLMClient
from agentr.openai_client import OpenAIClient
from agentr.message_types import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    ToolCall,
    FunctionCall,
    Message,
    message_to_dict,
    messages_to_dicts,
    dict_to_message,
    dicts_to_messages,
)

__all__ = [
    "LLMClient",
    "OpenAIClient",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "ToolCall",
    "FunctionCall",
    "Message",
    "message_to_dict",
    "messages_to_dicts",
    "dict_to_message",
    "dicts_to_messages",
]
