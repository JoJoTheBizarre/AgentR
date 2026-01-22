"""
Tests for ResearchDecisionTool.
"""

import pytest
from pydantic import ValidationError

from src.tools.decision.research_tool import (
    ResearchDecisionTool,
    ShouldResearch,
    research_subagent,
)
from src.tools.names import ToolName


class TestResearchDecisionTool:
    """Tests for ResearchDecisionTool class."""

    def test_tool_properties(self):
        """Test ResearchDecisionTool properties."""
        tool = ResearchDecisionTool()

        assert tool.name == ToolName.RESEARCH_TOOL
        assert "research subagent" in tool.description.lower()
        assert tool.args_schema == ShouldResearch
        assert tool.is_async is False  # Default
        assert tool.return_direct is False  # Default

    def test_create_tool(self):
        """Test that create_tool returns a StructuredTool."""
        tool = ResearchDecisionTool()
        structured_tool = tool.create_tool()

        assert structured_tool.name == ToolName.RESEARCH_TOOL
        assert structured_tool.description == tool.description
        assert structured_tool.args_schema == tool.args_schema
        assert structured_tool.func is research_subagent
        assert structured_tool.coroutine is None  # Sync only

    def test_get_func(self):
        """Test get_func method."""
        tool = ResearchDecisionTool()
        func = tool.get_func()

        assert func is research_subagent


class TestResearchSubagent:
    """Tests for research_subagent function."""

    def test_research_subagent_basic(self):
        """Test basic functionality of research_subagent."""
        subtasks = ["Research task 1", "Research task 2", "Research task 3"]
        result = research_subagent(subtasks)

        assert isinstance(result, ShouldResearch)
        assert result.subtasks == subtasks

    def test_research_subagent_empty_list(self):
        """Test research_subagent with empty subtasks list."""
        result = research_subagent([])

        assert result.subtasks == []

    def test_research_subagent_single_task(self):
        """Test research_subagent with single task."""
        result = research_subagent(["Single task"])

        assert result.subtasks == ["Single task"]


class TestShouldResearchModel:
    """Tests for ShouldResearch Pydantic model."""

    def test_should_research_validation(self):
        """Test validation of ShouldResearch model."""
        # Valid: list of strings
        valid_data = {"subtasks": ["task1", "task2"]}
        instance = ShouldResearch(**valid_data)
        assert instance.subtasks == ["task1", "task2"]

        # Valid: empty list
        instance = ShouldResearch(subtasks=[])
        assert instance.subtasks == []

        # Invalid: not a list
        with pytest.raises(ValidationError):
            ShouldResearch(subtasks="not a list")

        # Invalid: list contains non-string
        with pytest.raises(ValidationError):
            ShouldResearch(subtasks=["task1", 123])

        # Invalid: missing subtasks field
        with pytest.raises(ValidationError):
            ShouldResearch()  # type: ignore

    def test_should_research_serialization(self):
        """Test serialization of ShouldResearch model."""
        instance = ShouldResearch(subtasks=["a", "b", "c"])
        dict_repr = instance.model_dump()

        assert dict_repr == {"subtasks": ["a", "b", "c"]}

        # Should be able to recreate from dict
        new_instance = ShouldResearch(**dict_repr)
        assert new_instance.subtasks == ["a", "b", "c"]
