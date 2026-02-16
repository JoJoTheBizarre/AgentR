import json
import logging
from langchain_core.messages import AIMessage

from .exceptions import ValidationError
from .states import Source, SourceType
from .prompts import RESEARCH_SYNTHESIS_TEMPLATE

logger = logging.getLogger(__name__)


def is_tool_call(response: AIMessage) -> bool:
    """Check if AIMessage contains a tool call."""
    return bool(response.tool_calls)


def extract_text_response(response: AIMessage) -> str:
    """Extract text content from AIMessage."""
    content = response.content
    if isinstance(content, str):
        return content
    raise TypeError(f"Expected str content, got {type(content).__name__}")


def format_single_source(idx: int, source: Source) -> str:
    """Format single source for synthesis."""
    return f"""[Source {idx + 1}]
Type: {source["type"]}
Source: {source["source"]}
Content: {source["content"]}
"""


def format_research_synthesis(research_findings: list[Source]) -> str:
    """Format research findings into synthesis."""
    formatted_sources = "\n".join(
        format_single_source(i, source) for i, source in enumerate(research_findings)
    )

    return RESEARCH_SYNTHESIS_TEMPLATE.format(
        total_sources=len(research_findings), formatted_sources=formatted_sources
    )


def validate_source_structure(item: dict, index: int) -> None:
    """Validate source dictionary structure."""
    if not isinstance(item, dict):
        raise ValidationError(f"Source at index {index} is not a dictionary")

    required_fields = {"source", "content", "type"}
    missing = required_fields - set(item.keys())
    if missing:
        raise ValidationError(f"Source at index {index} missing fields: {missing}")

    try:
        SourceType(item["type"])
    except ValueError as e:
        raise ValidationError(f"Source at index {index} has invalid type: {item['type']}") from e

    if not isinstance(item["source"], str):
        raise ValidationError(f"Source at index {index} 'source' must be string")
    if not isinstance(item["content"], str):
        raise ValidationError(f"Source at index {index} 'content' must be string")


def parse_research_results(results_str: str) -> list[dict]:
    """Parse and validate research results JSON."""
    try:
        parsed = json.loads(results_str)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in research results: {e}") from e

    if not isinstance(parsed, list):
        raise ValidationError(f"Expected list of sources, got {type(parsed).__name__}")

    validated_sources = []
    for i, item in enumerate(parsed):
        validate_source_structure(item, i)
        validated_sources.append(item)

    return validated_sources
