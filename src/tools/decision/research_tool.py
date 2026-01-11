
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ..base.base_tool import BaseTool


class ShouldResearch(BaseModel):
    subtasks: list[str]


class ResearchDecisionTool(BaseTool):
    """Decision tool for determining if research is needed and planning subtasks."""

    TOOL_NAME = "research_tool"
    TOOL_DESCRIPTION = (
        "Use this tool to determine if additional research is needed to answer the "
        "user's query. If research is needed, provide a list of specific subtasks "
        "to be researched."
    )

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return self.TOOL_DESCRIPTION

    @property
    def args_schema(self):
        return ShouldResearch

    @property
    def return_direct(self) -> bool:
        return True

    def create_tool(self) -> StructuredTool:
        """Create and return a StructuredTool instance."""
        return StructuredTool.from_function(
            func=should_research,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            return_direct=self.return_direct,
        )

    def get_func(self):
        return should_research


def should_research(
    subtasks: list[str],
) -> ShouldResearch:
    return ShouldResearch(
        subtasks=subtasks,
    )




def research_tool_factory() -> StructuredTool:
    """
    Factory function that creates a structured research-decision tool
    usable by LLMs.
    """
    return StructuredTool.from_function(
        func=should_research,
        name=ResearchDecisionTool.TOOL_NAME,
        description=ResearchDecisionTool.TOOL_DESCRIPTION,
        args_schema=ShouldResearch,
        return_direct=True,
    )
