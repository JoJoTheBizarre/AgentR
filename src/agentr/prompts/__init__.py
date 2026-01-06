"""Prompt management module for the research agent.

This module centralizes all textual content used by the agent:
- System prompts for different node roles
- Tool descriptions for LLM context
- Tool definition constants for the tool framework
"""

from .system_prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from .tool_constants import (
    CREATE_SUBTASKS_TOOL_DESCRIPTION,
    CREATE_SUBTASKS_TOOL_NAME,
    SET_SYNTHESIS_FLAG_TOOL_DESCRIPTION,
    SET_SYNTHESIS_FLAG_TOOL_NAME,
    WEB_SEARCH_TOOL_DESCRIPTION,
    WEB_SEARCH_TOOL_NAME,
)

__all__ = [
    "CREATE_SUBTASKS_TOOL_DESCRIPTION",
    "CREATE_SUBTASKS_TOOL_NAME",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "RESEARCHER_SYSTEM_PROMPT",
    "SET_SYNTHESIS_FLAG_TOOL_DESCRIPTION",
    "SET_SYNTHESIS_FLAG_TOOL_NAME",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "WEB_SEARCH_TOOL_DESCRIPTION",
    "WEB_SEARCH_TOOL_NAME",
]
