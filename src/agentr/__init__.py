"""AgentR: A research agent framework built on LangGraph."""

from .client.base import LLMClient
from .client.openai import OpenAIClient
from .core.messages import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    ToolCall,
    FunctionCall,
    Message,
    Role,
)
from .core.state import ResearchAgentState
from .agent.graph import ResearchAgent

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
    "Role",
    "ResearchAgentState",
    "ResearchAgent",
]

__version__ = "0.1.0"