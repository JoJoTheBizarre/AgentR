"""
Tests for BaseTool abstract class.
"""

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.tools.base.base_tool import BaseTool


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete classes must implement abstract methods."""

        class IncompleteTool(BaseTool):
            # Missing name, description, create_tool
            pass

        with pytest.raises(TypeError):
            IncompleteTool()

    def test_concrete_tool_implementation(self):
        """Test a complete concrete tool implementation."""

        class TestArgs(BaseModel):
            param: str

        class ConcreteTool(BaseTool):
            @property
            def name(self) -> str:
                return "concrete_tool"

            @property
            def description(self) -> str:
                return "A concrete tool implementation"

            @property
            def args_schema(self):
                return TestArgs

            def create_tool(self) -> StructuredTool:
                def tool_func(param: str) -> str:
                    return f"Result: {param}"

                return StructuredTool.from_function(
                    func=tool_func,
                    name=self.name,
                    description=self.description,
                    args_schema=self.args_schema,
                )

            def get_func(self):
                def tool_func(param: str) -> str:
                    return f"Result: {param}"

                return tool_func

        tool = ConcreteTool()

        # Test properties
        assert tool.name == "concrete_tool"
        assert tool.description == "A concrete tool implementation"
        assert tool.args_schema == TestArgs
        assert tool.return_direct is False  # Default
        assert tool.is_async is False  # Default

        # Test create_tool returns StructuredTool
        structured_tool = tool.create_tool()
        assert isinstance(structured_tool, StructuredTool)
        assert structured_tool.name == "concrete_tool"

        # Test get_func returns callable
        func = tool.get_func()
        assert callable(func)
        assert func("test") == "Result: test"

        # Test get_coroutine returns None by default
        assert tool.get_coroutine() is None

    def test_default_args_schema_is_none(self):
        """Test that args_schema defaults to None."""

        class NoArgsTool(BaseTool):
            @property
            def name(self) -> str:
                return "no_args_tool"

            @property
            def description(self) -> str:
                return "Tool without args schema"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "no args",
                    name=self.name,
                    description=self.description,
                )

        tool = NoArgsTool()
        assert tool.args_schema is None

    def test_return_direct_override(self):
        """Test that return_direct can be overridden."""

        class ReturnDirectTool(BaseTool):
            @property
            def name(self) -> str:
                return "return_direct_tool"

            @property
            def description(self) -> str:
                return "Tool that returns directly"

            @property
            def return_direct(self) -> bool:
                return True

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "direct",
                    name=self.name,
                    description=self.description,
                    return_direct=self.return_direct,
                )

        tool = ReturnDirectTool()
        assert tool.return_direct is True

    def test_is_async_override(self):
        """Test that is_async can be overridden."""

        class AsyncTool(BaseTool):
            @property
            def name(self) -> str:
                return "async_tool"

            @property
            def description(self) -> str:
                return "Async tool"

            @property
            def is_async(self) -> bool:
                return True

            def create_tool(self) -> StructuredTool:
                async def async_func():
                    return "async result"

                return StructuredTool.from_function(
                    coroutine=async_func,
                    name=self.name,
                    description=self.description,
                )

            def get_coroutine(self):
                async def async_func():
                    return "async result"

                return async_func

        tool = AsyncTool()
        assert tool.is_async is True
        assert tool.get_coroutine() is not None
        assert callable(tool.get_coroutine())

    def test_get_func_not_implemented_by_default(self):
        """Test that get_func raises NotImplementedError if not overridden."""

        class ToolWithoutGetFunc(BaseTool):
            @property
            def name(self) -> str:
                return "tool"

            @property
            def description(self) -> str:
                return "Tool"

            def create_tool(self) -> StructuredTool:
                return StructuredTool.from_function(
                    func=lambda: "test",
                    name=self.name,
                    description=self.description,
                )

        tool = ToolWithoutGetFunc()

        # get_func should raise NotImplementedError because we didn't override it
        with pytest.raises(NotImplementedError, match="Tool must implement get_func"):
            tool.get_func()

    def test_create_tool_must_be_implemented(self):
        """Test that create_tool is abstract and must be implemented."""

        class MissingCreateTool(BaseTool):
            @property
            def name(self) -> str:
                return "tool"

            @property
            def description(self) -> str:
                return "Tool"

        with pytest.raises(TypeError):
            MissingCreateTool()
