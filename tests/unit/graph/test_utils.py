"""
Tests for graph utility functions.
"""

import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from src.exceptions import ValidationError
from src.graph.utils import (
    extract_text_response,
    format_research_synthesis,
    format_single_source,
    is_tool_call,
    parse_research_results,
    validate_source_structure,
)
from src.models.states import Source, SourceType


class TestIsToolCall:
    """Tests for is_tool_call function."""

    def test_is_tool_call_with_tool_calls(self):
        """Test that message with tool_calls returns True."""
        message = AIMessage(
            content="", tool_calls=[{"name": "tool", "args": {}, "id": "test-id"}]
        )
        assert is_tool_call(message) is True

    def test_is_tool_call_without_tool_calls(self):
        """Test that message without tool_calls returns False."""
        message = AIMessage(content="No tool calls")
        assert is_tool_call(message) is False

    def test_is_tool_call_with_empty_tool_calls(self):
        """Test that message with empty tool_calls list returns False."""
        message = AIMessage(content="", tool_calls=[])
        assert is_tool_call(message) is False


class TestExtractTextResponse:
    """Tests for extract_text_response function."""

    def test_extract_text_response_with_string_content(self):
        """Test extraction with string content."""
        message = AIMessage(content="Text response")
        result = extract_text_response(message)
        assert result == "Text response"

    def test_extract_text_response_with_non_string_content(self):
        """Test that non-string content raises TypeError."""
        message = AIMessage(content=["list", "of", "strings"])
        with pytest.raises(TypeError, match="Expected response.content to be str"):
            extract_text_response(message)

    def test_extract_text_response_with_none_content(self):
        """Test that None content raises TypeError."""
        # Mock AIMessage with content=None (cannot instantiate real AIMessage with None)
        message = Mock(spec=AIMessage)
        message.content = None
        with pytest.raises(TypeError, match="Expected response.content to be str"):
            extract_text_response(message)


class TestFormatSingleSource:
    """Tests for format_single_source function."""

    def test_format_single_source_web(self):
        """Test formatting a web source."""
        source: Source = {
            "source": "https://example.com",
            "content": "Example content",
            "type": SourceType.WEB,
        }
        result = format_single_source(0, source)

        assert "Source 1" in result
        assert "Type: web" in result
        assert "Source: https://example.com" in result
        assert "Content: Example content" in result

    def test_format_single_source_document(self):
        """Test formatting a document source."""
        source: Source = {
            "source": "document.pdf",
            "content": "Document content",
            "type": SourceType.DOCUMENT,
        }
        result = format_single_source(1, source)

        assert "Source 2" in result
        assert "Type: document" in result
        assert "Source: document.pdf" in result
        assert "Content: Document content" in result

    def test_format_single_source_multiline_content(self):
        """Test formatting with multiline content (should preserve newlines)."""
        source: Source = {
            "source": "https://example.com",
            "content": "Line 1\nLine 2\nLine 3",
            "type": SourceType.WEB,
        }
        result = format_single_source(2, source)

        assert "Content: Line 1\nLine 2\nLine 3" in result


class TestFormatResearchSynthesis:
    """Tests for format_research_synthesis function."""

    def test_format_research_synthesis_empty(self):
        """Test formatting with empty findings."""
        result = format_research_synthesis([])
        assert "Total Sources Gathered: 0" in result

    def test_format_research_synthesis_with_sources(self):
        """Test formatting with multiple sources."""
        sources = [
            {
                "source": "https://example.com/1",
                "content": "Content 1",
                "type": SourceType.WEB,
            },
            {
                "source": "document.pdf",
                "content": "Content 2",
                "type": SourceType.DOCUMENT,
            },
        ]
        result = format_research_synthesis(sources)

        assert "Total Sources Gathered: 2" in result
        assert "Source 1" in result
        assert "Source 2" in result
        assert "Type: web" in result
        assert "Type: document" in result

    def test_format_research_synthesis_preserves_order(self):
        """Test that sources are formatted in order."""
        sources = [
            {"source": "first", "content": "First", "type": SourceType.WEB},
            {"source": "second", "content": "Second", "type": SourceType.WEB},
        ]
        result = format_research_synthesis(sources)

        # Check order - first source should appear before second
        first_index = result.find("Source 1")
        second_index = result.find("Source 2")
        assert first_index < second_index


class TestValidateSourceStructure:
    """Tests for validate_source_structure function."""

    def test_validate_source_structure_valid(self):
        """Test validation of valid source structure."""
        source = {
            "source": "https://example.com",
            "content": "Content",
            "type": "web",
        }
        # Should not raise any exception
        validate_source_structure(source, 0)

    def test_validate_source_structure_not_dict(self):
        """Test validation fails when source is not a dictionary."""
        with pytest.raises(ValidationError, match="is not a dictionary"):
            validate_source_structure("not a dict", 0)

    def test_validate_source_structure_missing_fields(self):
        """Test validation fails when required fields are missing."""
        source = {"source": "url"}  # Missing content and type
        with pytest.raises(ValidationError, match="missing fields"):
            validate_source_structure(source, 0)

    def test_validate_source_structure_invalid_type(self):
        """Test validation fails when type is invalid."""
        source = {
            "source": "url",
            "content": "content",
            "type": "invalid_type",
        }
        with pytest.raises(ValidationError, match="invalid type"):
            validate_source_structure(source, 0)

    def test_validate_source_structure_non_string_fields(self):
        """Test validation fails when source or content are not strings."""
        source = {
            "source": 123,  # Not a string
            "content": "content",
            "type": "web",
        }
        with pytest.raises(ValidationError, match="must be string"):
            validate_source_structure(source, 0)

        source = {
            "source": "url",
            "content": 456,  # Not a string
            "type": "web",
        }
        with pytest.raises(ValidationError, match="must be string"):
            validate_source_structure(source, 1)


class TestParseResearchResults:
    """Tests for parse_research_results function."""

    def test_parse_research_results_valid_json(self):
        """Test parsing valid JSON string."""
        sources = [
            {"source": "url1", "content": "content1", "type": "web"},
            {"source": "url2", "content": "content2", "type": "document"},
        ]
        json_str = json.dumps(sources)

        result = parse_research_results(json_str)

        assert result == sources

    def test_parse_research_results_invalid_json(self):
        """Test parsing invalid JSON raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid JSON"):
            parse_research_results("{not valid json}")

    def test_parse_research_results_not_list(self):
        """Test parsing JSON that is not a list raises ValidationError."""
        json_str = json.dumps({"not": "a list"})
        with pytest.raises(ValidationError, match="Expected list of sources"):
            parse_research_results(json_str)

    def test_parse_research_results_invalid_source_structure(self):
        """Test parsing JSON with invalid source structure raises ValidationError."""
        sources = [{"source": "url", "content": "content"}]  # Missing type
        json_str = json.dumps(sources)

        with pytest.raises(ValidationError, match="missing fields"):
            parse_research_results(json_str)

    def test_parse_research_results_empty_list(self):
        """Test parsing empty list returns empty list."""
        json_str = json.dumps([])
        result = parse_research_results(json_str)

        assert result == []

    def test_parse_research_results_nested_validation(self):
        """Test that validation is applied to each source."""
        sources = [
            {"source": "url1", "content": "content1", "type": "web"},  # Valid
            {
                "source": 123,
                "content": "content2",
                "type": "web",
            },  # Invalid source field
        ]
        json_str = json.dumps(sources)

        with pytest.raises(ValidationError, match="must be string"):
            parse_research_results(json_str)
