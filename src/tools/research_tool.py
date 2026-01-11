
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class ShouldResearch(BaseModel):
    subtasks: list[str]


def should_research(
    subtasks: list[str],
) -> ShouldResearch:
    return ShouldResearch(
        subtasks=subtasks,
    )


WEB_SEARCH_TOOL_DESCRIPTION = (
    "Use this tool to determine if additional research is needed to answer the "
    "user's query. If research is needed, provide a list of specific subtasks "
    "to be researched."
)

TOOL_NAME = "research_tool"


def research_tool_factory() -> StructuredTool:
    """
    Factory function that creates a structured research-decision tool
    usable by LLMs.
    """
    return StructuredTool.from_function(
        func=should_research,
        name=TOOL_NAME,
        description=WEB_SEARCH_TOOL_DESCRIPTION,
        args_schema=ShouldResearch,
        return_direct=True,
    )
