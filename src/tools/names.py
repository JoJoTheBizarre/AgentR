from enum import StrEnum


class ToolName(StrEnum):
    """Enumeration of all available tool names in AgentR."""

    WEB_SEARCH = "web_search"
    RESEARCH_TOOL = "research_sub_agent"
