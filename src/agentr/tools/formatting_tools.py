"""Formatting tools for structured LLM output.

These tools act as identity functions: they rely entirely on Pydantic
for validation and simply return the validated input unchanged.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentr.prompts import (
    CREATE_SUBTASKS_TOOL_DESCRIPTION,
    CREATE_SUBTASKS_TOOL_NAME,
    SET_SYNTHESIS_FLAG_TOOL_DESCRIPTION,
    SET_SYNTHESIS_FLAG_TOOL_NAME,
)


class SubtaskCreationInput(BaseModel):
    """Schema for creating research subtasks."""

    subtasks: list[str] = Field(
        ..., description="List of clear, actionable research subtasks."
    )


def create_subtasks(subtasks: list[str]) -> SubtaskCreationInput:
    """Identity function: returns validated subtasks."""
    return SubtaskCreationInput(subtasks=subtasks)


create_subtasks_tool = StructuredTool.from_function(
    name=CREATE_SUBTASKS_TOOL_NAME,
    description=CREATE_SUBTASKS_TOOL_DESCRIPTION,
    args_schema=SubtaskCreationInput,
    func=create_subtasks,
)


class SynthesisFlagInput(BaseModel):
    """Schema for indicating whether synthesis should begin."""

    should_synthesize: bool = Field(
        ..., description="Whether research is complete and synthesis should begin."
    )


def set_synthesis_flag(should_synthesize: bool) -> SynthesisFlagInput:
    """Identity function: returns validated synthesis flag."""
    return SynthesisFlagInput(should_synthesize=should_synthesize)


set_synthesis_flag_tool = StructuredTool.from_function(
    name=SET_SYNTHESIS_FLAG_TOOL_NAME,
    description=SET_SYNTHESIS_FLAG_TOOL_DESCRIPTION,
    args_schema=SynthesisFlagInput,
    func=set_synthesis_flag,
)
