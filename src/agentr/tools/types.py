"""
Shared type definitions for tool inputs and outputs.

This module provides common type definitions that can be used across different tools
to ensure consistent data structures and formatting.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Types of information sources for research findings."""

    WEB_SEARCH = "web_search"
    DOCUMENT = "document"


class InformationSource(BaseModel):
    """
    Represents a source of information with its content and metadata.

    This is the basic building block for tool outputs that need to provide
    citations or references along with content. It can be used by any tool
    that returns information from a source (web pages, documents, databases, etc.).
    """

    type: SourceType = Field(..., description="The type of the information source")
    source: str = Field(
        ..., description="The source identifier (e.g., URL, document path, database ID)"
    )
    content: str = Field(..., description="The content or information from the source")

    def to_llm_context(self) -> str:
        """
        Convert the information source to a string suitable for LLM context.

        Returns:
            Formatted string with source and content, suitable for inclusion
            in LLM prompts or context windows.
        """
        return f"Source ({self.type}): {self.source}\nContent: {self.content}"

    def __str__(self) -> str:
        """String representation of the information source."""
        return self.to_llm_context()


class InformationResult(BaseModel):
    """
    Container for multiple information sources returned by a tool.

    This provides a standardized way for tools to return multiple sources
    of information along with optional summary and metadata.
    """

    sources: list[InformationSource] = Field(
        ..., description="List of information sources returned by the tool"
    )

    def to_llm_context(self) -> str:
        """
        Convert all information sources to a single string for LLM context.

        Returns:
            Concatenated string of all sources' LLM context representations,
            separated by blank lines.
        """
        contexts = [source.to_llm_context() for source in self.sources]
        return "\n\n".join(contexts)

    def __str__(self) -> str:
        """String representation of the information result."""
        return self.to_llm_context()
