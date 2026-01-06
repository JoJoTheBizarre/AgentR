from typing import TypedDict, Optional, List

from .messages import (
    AssistantMessage,
    Message,
)

# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------

class ResearchAgentState(TypedDict, total=False):
    """State definition for the research agent.

    Attributes:
        query: The original research query from the user
        message_history: List of conversation messages (system, user, assistant, tool results)
        current_response: The latest response from the LLM (may contain tool calls)
    """
    query: str
    message_history: List[Message]
    response: Optional[AssistantMessage]