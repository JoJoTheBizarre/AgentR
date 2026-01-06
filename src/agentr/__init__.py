"""AgentR: A research agent framework built on LangGraph."""

from . import prompts
from .agent.graph import ResearchAgent
from .client.base import LLMClient
from .client.openai import OpenAIClient
from .core.messages import (
    AssistantMessage,
    FunctionCall,
    Message,
    Role,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from .core.state import ResearchAgentState

__all__ = [
    "AssistantMessage",
    "FunctionCall",
    "LLMClient",
    "Message",
    "OpenAIClient",
    "ResearchAgent",
    "ResearchAgentState",
    "Role",
    "SystemMessage",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "prompts",
]

__version__ = "0.1.0"
