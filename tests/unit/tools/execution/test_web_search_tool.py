"""
Tests for WebSearchTool.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from src.exceptions import ToolInitializationError
from src.models.states import SourceType
from src.tools.execution.web_search_tool import (
    WebSearchTool,
    get_async_tavily_client,
    get_tavily_client,
    web_search_async,
    web_search_sync,
)
from src.tools.names import ToolName


class TestWebSearchTool:
    """Tests for WebSearchTool class."""

    def test_tool_properties(self):
        """Test WebSearchTool properties."""
        tool = WebSearchTool()

        assert tool.name == ToolName.WEB_SEARCH
        assert "Search the web" in tool.description
        assert tool.args_schema is not None
        assert tool.is_async is True
        assert tool.return_direct is False

    def test_create_tool(self):
        """Test that create_tool returns a StructuredTool."""
        tool = WebSearchTool()
        structured_tool = tool.create_tool()

        assert structured_tool.name == ToolName.WEB_SEARCH
        assert structured_tool.description == tool.description
        assert structured_tool.args_schema == tool.args_schema
        # Should have both func and coroutine
        assert structured_tool.func is not None
        assert structured_tool.coroutine is not None

    def test_get_func_and_coroutine(self):
        """Test get_func and get_coroutine methods."""
        tool = WebSearchTool()

        func = tool.get_func()
        coroutine = tool.get_coroutine()

        assert func is web_search_sync
        assert coroutine is web_search_async


class TestTavilyClientFunctions:
    """Tests for Tavily client initialization functions."""

    def test_get_tavily_client_with_key(self, mock_env_vars):
        """Test get_tavily_client when TAVILY_API_KEY is set."""
        with patch("src.tools.execution.web_search_tool.TavilyClient") as mock_client:
            mock_instance = mock_client.return_value
            client = get_tavily_client()

            mock_client.assert_called_once_with(api_key="test-tavily-key")
            assert client is mock_instance

    def test_get_tavily_client_without_key(self):
        """Test get_tavily_client when TAVILY_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ToolInitializationError, match="TAVILY_API_KEY"):
                get_tavily_client()

    def test_get_async_tavily_client_with_key(self, mock_env_vars):
        """Test get_async_tavily_client when TAVILY_API_KEY is set."""
        with patch(
            "src.tools.execution.web_search_tool.AsyncTavilyClient"
        ) as mock_client:
            mock_instance = mock_client.return_value
            client = get_async_tavily_client()

            mock_client.assert_called_once_with(api_key="test-tavily-key")
            assert client is mock_instance

    def test_get_async_tavily_client_without_key(self):
        """Test get_async_tavily_client when TAVILY_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ToolInitializationError, match="TAVILY_API_KEY"):
                get_async_tavily_client()


class TestWebSearchSync:
    """Tests for synchronous web search."""

    def test_web_search_sync_success(self, mock_env_vars):
        """Test successful synchronous web search."""
        # Mock Tavily response
        mock_response = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "Content 1",
                    "score": 0.9,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Content 2",
                    "score": 0.8,
                },
            ]
        }

        # Create a mock Tavily client
        mock_client = Mock()
        mock_client.search.return_value = mock_response

        # Patch get_tavily_client to return our mock client
        with patch(
            "src.tools.execution.web_search_tool.get_tavily_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            sources = web_search_sync("test query")

            # Verify get_tavily_client was called
            mock_get_client.assert_called_once()
            # Verify Tavily client search was called correctly
            mock_client.search.assert_called_once_with(query="test query")

            # Verify response formatting
            assert len(sources) == 2
            assert sources[0] == {
                "source": "https://example.com/1",
                "content": "Content 1",
                "type": SourceType.WEB,
            }
            assert sources[1] == {
                "source": "https://example.com/2",
                "content": "Content 2",
                "type": SourceType.WEB,
            }

    def test_web_search_sync_empty_results(self, mock_env_vars):
        """Test web search with empty results."""
        mock_client = Mock()
        mock_client.search.return_value = {"results": []}

        with patch(
            "src.tools.execution.web_search_tool.get_tavily_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            sources = web_search_sync("test query")

            assert sources == []
            mock_get_client.assert_called_once()
            mock_client.search.assert_called_once_with(query="test query")

    def test_web_search_sync_exception_propagation(self, mock_env_vars):
        """Test that exceptions from Tavily are propagated."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Tavily API error")

        with patch(
            "src.tools.execution.web_search_tool.get_tavily_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            with pytest.raises(Exception, match="Tavily API error"):
                web_search_sync("test query")

            mock_get_client.assert_called_once()
            mock_client.search.assert_called_once_with(query="test query")

    def test_web_search_sync_query_validation(self):
        """Test that query validation works through args schema."""
        tool = WebSearchTool()
        structured_tool = tool.create_tool()

        # Test valid query
        try:
            structured_tool.args_schema.model_validate({"query": "valid query"})
        except ValidationError:
            pytest.fail("Valid query should not raise ValidationError")

        # Test empty query
        with pytest.raises(ValidationError):
            structured_tool.args_schema.model_validate({"query": ""})

        # Test query too long (max_length=500)
        with pytest.raises(ValidationError):
            structured_tool.args_schema.model_validate({"query": "x" * 501})


class TestWebSearchAsync:
    """Tests for asynchronous web search."""

    @pytest.mark.asyncio
    async def test_web_search_async_success(self, mock_env_vars):
        """Test successful asynchronous web search."""
        mock_response = {
            "results": [
                {
                    "title": "Async Result",
                    "url": "https://example.com/async",
                    "content": "Async content",
                    "score": 0.95,
                }
            ]
        }
        # Create async mock client
        mock_client = AsyncMock()
        mock_client.search.return_value = mock_response

        with patch(
            "src.tools.execution.web_search_tool.get_async_tavily_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            sources = await web_search_async("async query")

            mock_get_client.assert_called_once()
            mock_client.search.assert_called_once_with(query="async query")
            assert len(sources) == 1
            assert sources[0] == {
                "source": "https://example.com/async",
                "content": "Async content",
                "type": SourceType.WEB,
            }

    @pytest.mark.asyncio
    async def test_web_search_async_empty_results(self, mock_env_vars):
        """Test async web search with empty results."""
        mock_client = AsyncMock()
        mock_client.search.return_value = {"results": []}

        with patch(
            "src.tools.execution.web_search_tool.get_async_tavily_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            sources = await web_search_async("async query")

            assert sources == []
            mock_get_client.assert_called_once()
            mock_client.search.assert_called_once_with(query="async query")

    @pytest.mark.asyncio
    async def test_web_search_async_exception_propagation(self, mock_env_vars):
        """Test that exceptions from async Tavily are propagated."""
        mock_client = AsyncMock()
        mock_client.search.side_effect = Exception("Async Tavily error")

        with patch(
            "src.tools.execution.web_search_tool.get_async_tavily_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_client
            with pytest.raises(Exception, match="Async Tavily error"):
                await web_search_async("async query")

            mock_get_client.assert_called_once()
            mock_client.search.assert_called_once_with(query="async query")


class TestFormatTavilyResponse:
    """Tests for _format_tavily_response function."""

    def test_format_tavily_response_normal(self):
        """Test normal formatting of Tavily response."""
        from src.tools.execution.web_search_tool import _format_tavily_response

        response = {
            "results": [
                {
                    "url": "https://example.com/1",
                    "content": "Content 1",
                    "title": "Title 1",
                    "score": 0.9,
                },
                {
                    "url": "https://example.com/2",
                    "content": "Content 2",
                    "title": "Title 2",
                    "score": 0.8,
                },
            ]
        }

        sources = _format_tavily_response(response)

        assert len(sources) == 2
        assert sources[0]["source"] == "https://example.com/1"
        assert sources[0]["content"] == "Content 1"
        assert sources[0]["type"] == SourceType.WEB
        assert sources[1]["source"] == "https://example.com/2"
        assert sources[1]["content"] == "Content 2"
        assert sources[1]["type"] == SourceType.WEB

    def test_format_tavily_response_missing_fields(self):
        """Test formatting with missing fields in Tavily response."""
        from src.tools.execution.web_search_tool import _format_tavily_response

        response = {
            "results": [
                {"url": "https://example.com"},  # Missing content
                {"content": "Content only"},  # Missing url
                {},  # Empty dict
            ]
        }

        sources = _format_tavily_response(response)

        assert len(sources) == 3
        # First source: missing content becomes empty string
        assert sources[0]["source"] == "https://example.com"
        assert sources[0]["content"] == ""
        assert sources[0]["type"] == SourceType.WEB
        # Second source: missing url becomes empty string
        assert sources[1]["source"] == ""
        assert sources[1]["content"] == "Content only"
        assert sources[1]["type"] == SourceType.WEB
        # Third source: both empty
        assert sources[2]["source"] == ""
        assert sources[2]["content"] == ""
        assert sources[2]["type"] == SourceType.WEB

    def test_format_tavily_response_no_results_key(self):
        """Test formatting when response has no 'results' key."""
        from src.tools.execution.web_search_tool import _format_tavily_response

        response = {}
        sources = _format_tavily_response(response)

        assert sources == []

    def test_format_tavily_response_results_not_list(self):
        """Test formatting when results is not a list."""
        from src.tools.execution.web_search_tool import _format_tavily_response

        response = {"results": "not a list"}
        sources = _format_tavily_response(response)

        assert sources == []
