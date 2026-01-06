"""
Shared type definitions for tool inputs and outputs.

This module provides common type definitions that can be used across different tools
to ensure consistent data structures and formatting.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class InformationSource(BaseModel):
    """
    Represents a source of information with its content and metadata.

    This is the basic building block for tool outputs that need to provide
    citations or references along with content. It can be used by any tool
    that returns information from a source (web pages, documents, databases, etc.).
    """

    source: str = Field(
        ...,
        description="The source identifier (e.g., URL, document path, database ID)"
    )
    content: str = Field(
        ...,
        description="The content or information from the source"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="When the source was accessed or created (optional)"
    )

    def to_llm_context(self) -> str:
        """
        Convert the information source to a string suitable for LLM context.

        Returns:
            Formatted string with source and content, suitable for inclusion
            in LLM prompts or context windows.
        """
        lines = [
            f"Source: {self.source}",
            f"Content: {self.content}"
        ]
        if self.timestamp:
            lines.append(f"Timestamp: {self.timestamp.isoformat()}")
        return "\n".join(lines)

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
        ...,
        description="List of information sources returned by the tool"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional summary of all sources"
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