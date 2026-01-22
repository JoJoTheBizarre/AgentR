from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ..base.base_tool import BaseTool
from ..names import ToolName


class ShouldResearch(BaseModel):
    subtasks: list[str]


def research_subagent(
    subtasks: list[str],
) -> ShouldResearch:
    return ShouldResearch(
        subtasks=subtasks,
    )


class ResearchDecisionTool(BaseTool):
    """Decision tool for handling research agent handoff"""

    TOOL_NAME = ToolName.RESEARCH_TOOL
    TOOL_DESCRIPTION = (
        "use this tool to ask help from a research subagent by providing subtasks "
        "to search for, make sure that the subtasks are independent from each other"
        "and provide actionable information to respond to the user's question"
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

    def create_tool(self) -> StructuredTool:
        """Create and return a StructuredTool instance."""
        return StructuredTool.from_function(
            func=research_subagent,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def get_func(self):
        return research_subagent
