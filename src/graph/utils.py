import json
import logging

from langchain_core.messages import AIMessage

from src.exceptions import ValidationError
from src.models.states import Source, SourceType
from src.prompt_templates import RESEARCH_SYNTHESIS_TEMPLATE

logger = logging.getLogger(__name__)


def is_tool_call(response: AIMessage) -> bool:
    """Check if the AIMessage contains a tool call."""
    return bool(response.tool_calls)


def extract_text_response(response: AIMessage) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    raise TypeError(
        f"Expected response.content to be str, got {type(content).__name__}"
    )


def format_single_source(idx: int, source: Source) -> str:
    """Format a single source for synthesis output."""
    return f"""[Source {idx + 1}]
Type: {source['type']}
Source: {source['source']}
Content: {source['content']}
"""


def format_research_synthesis(research_findings: list[Source]) -> str:
    """Format research findings using the synthesis template.

    Args:
        research_findings: List of research sources

    Returns:
        Formatted synthesis string
    """
    formatted_sources = "\n".join(
        format_single_source(i, source) for i, source in enumerate(research_findings)
    )

    return RESEARCH_SYNTHESIS_TEMPLATE.format(
        total_sources=len(research_findings), formatted_sources=formatted_sources
    )


def validate_source_structure(item: dict, index: int) -> None:
    """Validate a single source dictionary structure.

    Args:
        item: Source dictionary to validate
        index: Index of source in list (for error messages)

    Raises:
        ValidationError: If structure is invalid
    """
    if not isinstance(item, dict):
        raise ValidationError(f"Source at index {index} is not a dictionary")

    # Check required fields
    required_fields = {"source", "content", "type"}
    missing = required_fields - set(item.keys())
    if missing:
        raise ValidationError(f"Source at index {index} missing fields: {missing}")

    # Validate type is valid SourceType
    try:
        SourceType(item["type"])
    except ValueError as e:
        logger.error(
            f"Source validation failed at index {index}: invalid type '{item['type']}'"
        )
        raise ValidationError(
            f"Source at index {index} has invalid type: {item['type']}. "
            f"Expected one of: {[e.value for e in SourceType]}"
        ) from e

    # Ensure fields are strings
    if not isinstance(item["source"], str):
        raise ValidationError(f"Source at index {index} 'source' field must be string")
    if not isinstance(item["content"], str):
        raise ValidationError(f"Source at index {index} 'content' field must be string")


def parse_research_results(results_str: str) -> list[dict]:
    """Parse and validate research results JSON.

    Args:
        results_str: JSON string containing research results

    Returns:
        List of validated source dictionaries

    Raises:
        ValidationError: If JSON is invalid or structure doesn't match Source schema
    """
    try:
        parsed = json.loads(results_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in research results: {e.msg}")
        raise ValidationError(f"Invalid JSON in research results: {e}") from e

    if not isinstance(parsed, list):
        logger.error(
            f"Research results validation failed: expected list, got {type(parsed).__name__}"
        )
        raise ValidationError(f"Expected list of sources, got {type(parsed).__name__}")

    validated_sources = []
    for i, item in enumerate(parsed):
        validate_source_structure(item, i)
        validated_sources.append(item)

    return validated_sources
