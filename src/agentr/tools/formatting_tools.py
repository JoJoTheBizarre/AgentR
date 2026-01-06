"""
Formatting tools for structured LLM output.

These tools act as identity functions: they rely entirely on Pydantic
for validation and simply return the validated input unchanged.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Subtask Creation Tool
# -----------------------------------------------------------------------------


class SubtaskCreationInput(BaseModel):
    """Schema for creating research subtasks."""

    subtasks: list[str] = Field(
        ..., description="List of clear, actionable research subtasks."
    )


def create_subtasks(subtasks: list[str]) -> dict:
    """Identity function: returns validated subtasks."""
    return {"subtasks": subtasks}


create_subtasks_tool = StructuredTool.from_function(
    name="create_subtasks",
    description="Output a structured list of research subtasks.",
    args_schema=SubtaskCreationInput,
    func=create_subtasks,
)

# -----------------------------------------------------------------------------
# Synthesis Flag Tool
# -----------------------------------------------------------------------------


class SynthesisFlagInput(BaseModel):
    """Schema for indicating whether synthesis should begin."""

    should_synthesize: bool = Field(
        ..., description="Whether research is complete and synthesis should begin."
    )


def set_synthesis_flag(should_synthesize: bool) -> dict:
    """Identity function: returns validated synthesis flag."""
    return {"should_synthesize": should_synthesize}


set_synthesis_flag_tool = StructuredTool.from_function(
    name="set_synthesis_flag",
    description="Output a structured flag indicating synthesis readiness.",
    args_schema=SynthesisFlagInput,
    func=set_synthesis_flag,
)
