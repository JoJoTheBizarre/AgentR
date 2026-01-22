"""
Tests for ToolManager.
"""

import pytest
from langchain_core.tools import StructuredTool

from src.exceptions import ToolInitializationError
from src.tools.base.base_tool import BaseTool
from src.tools.manager import ToolManager
from src.tools.names import ToolName


class TestToolManager:
    """Tests for ToolManager class."""

    def test_register_and_get_tool(self, clean_tool_registry):
        """Test registering and retrieving a tool."""

        # Create a test tool class
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "Test tool description"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "test",
                    name=self.name,
                    description=self.description,
                )

        # Register the tool
        ToolManager.register_tool(TestTool)

        # Retrieve the tool
        tool_instance = ToolManager.get_tool("test_tool")
        assert isinstance(tool_instance, TestTool)
        assert tool_instance.name == "test_tool"

    def test_duplicate_registration_raises_error(self, clean_tool_registry):
        """Test that duplicate tool registration raises ToolInitializationError."""

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "duplicate_tool"

            @property
            def description(self) -> str:
                return "Description"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "test",
                    name=self.name,
                    description=self.description,
                )

        # First registration should succeed
        ToolManager.register_tool(TestTool)

        # Second registration should raise error
        with pytest.raises(ToolInitializationError, match="is already registered"):
            ToolManager.register_tool(TestTool)

    def test_get_nonexistent_tool_raises_keyerror(self, clean_tool_registry):
        """Test that getting a non-existent tool raises KeyError."""
        with pytest.raises(KeyError, match="is not registered"):
            ToolManager.get_tool("nonexistent_tool")

    def test_get_structured_tool(self, clean_tool_registry):
        """Test that get_structured_tool returns a StructuredTool instance."""
        # ToolManager should have default tools initialized
        ToolManager.initialize_defaults()

        structured_tool = ToolManager.get_structured_tool(ToolName.WEB_SEARCH)
        assert isinstance(structured_tool, StructuredTool)
        assert structured_tool.name == ToolName.WEB_SEARCH

    def test_get_all_tools(self, clean_tool_registry):
        """Test that get_all_tools returns all registered tool classes."""

        # Create test tools
        class ToolA(BaseTool):
            @property
            def name(self) -> str:
                return "tool_a"

            @property
            def description(self) -> str:
                return "Tool A"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "a",
                    name=self.name,
                    description=self.description,
                )

        class ToolB(BaseTool):
            @property
            def name(self) -> str:
                return "tool_b"

            @property
            def description(self) -> str:
                return "Tool B"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "b",
                    name=self.name,
                    description=self.description,
                )

        ToolManager.register_tool(ToolA)
        ToolManager.register_tool(ToolB)

        all_tools = ToolManager.get_all_tools()
        assert len(all_tools) == 2
        assert "tool_a" in all_tools
        assert "tool_b" in all_tools
        assert all_tools["tool_a"] == ToolA
        assert all_tools["tool_b"] == ToolB

    def test_get_all_structured_tools(self, clean_tool_registry):
        """Test that get_all_structured_tools returns StructuredTool instances."""
        ToolManager.initialize_defaults()

        all_tools = ToolManager.get_all_structured_tools()
        assert isinstance(all_tools, dict)
        assert len(all_tools) >= 2  # At least WEB_SEARCH and RESEARCH_TOOL
        assert ToolName.WEB_SEARCH in all_tools
        assert ToolName.RESEARCH_TOOL in all_tools

        for tool_name, tool_instance in all_tools.items():
            assert isinstance(tool_instance, StructuredTool)
            assert tool_instance.name == tool_name

    def test_clear_registry(self, clean_tool_registry):
        """Test that clear_registry removes all registered tools."""

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "Description"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "test",
                    name=self.name,
                    description=self.description,
                )

        ToolManager.register_tool(TestTool)
        assert "test_tool" in ToolManager.get_all_tools()

        ToolManager.clear_registry()
        assert len(ToolManager.get_all_tools()) == 0

        # Should be able to register again after clearing
        ToolManager.register_tool(TestTool)
        assert "test_tool" in ToolManager.get_all_tools()

    def test_initialize_defaults(self, clean_tool_registry):
        """Test that initialize_defaults registers the default tools."""
        # Ensure not initialized
        ToolManager._initialized = False
        ToolManager.clear_registry()

        ToolManager.initialize_defaults()

        assert ToolManager._initialized is True
        assert ToolName.WEB_SEARCH in ToolManager.get_all_tools()
        assert ToolName.RESEARCH_TOOL in ToolManager.get_all_tools()

    def test_initialize_defaults_idempotent(self, clean_tool_registry):
        """Test that initialize_defaults is idempotent."""
        ToolManager.initialize_defaults()
        initial_tools = ToolManager.get_all_tools().copy()

        # Call again
        ToolManager.initialize_defaults()
        second_tools = ToolManager.get_all_tools()

        # Should be the same
        assert initial_tools == second_tools

    def test_get_tool_with_toolname_enum(self, clean_tool_registry):
        """Test that get_tool works with ToolName enum values."""
        ToolManager.initialize_defaults()

        # Should work with enum member
        tool = ToolManager.get_tool(ToolName.WEB_SEARCH)
        assert tool.name == ToolName.WEB_SEARCH

        # Should work with string
        tool_str = ToolManager.get_tool("web_search")
        assert tool_str.name == ToolName.WEB_SEARCH

        # Should be the same tool
        assert type(tool) == type(tool_str)
